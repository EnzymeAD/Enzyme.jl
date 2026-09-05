
module JIT

using LLVM
using Libdl
import LLVM: TargetMachine

import GPUCompiler
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features

struct CompilerInstance
    jit::LLVM.JuliaOJIT
end

function LLVM.dispose(ci::CompilerInstance)
    dispose(ci.jit)
    return nothing
end

const jit = Ref{CompilerInstance}()
const tm = Ref{TargetMachine}() # for opt pipeline

get_tm() = tm[]
get_jit() = jit[].jit

function absolute_symbol_materialization(name, ptr)
    address = LLVM.API.LLVMOrcJITTargetAddress(reinterpret(UInt, ptr))
    flags = LLVM.API.LLVMJITSymbolFlags(LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
    symbol = LLVM.API.LLVMJITEvaluatedSymbol(address, flags)
    gv = if LLVM.version() >= v"15"
        LLVM.API.LLVMOrcCSymbolMapPair(name, symbol)
    else
        LLVM.API.LLVMJITCSymbolMapPair(name, symbol)
    end
    return LLVM.absolute_symbols(Ref(gv))
end

const hnd_string_map = Dict{String, Ref{Ptr{Cvoid}}}()
const hnd_int_map = Dict{Int, Ref{Ptr{Cvoid}}}()

function fix_ptr_lookup(name)
    if startswith(name, "ejlstr\$") || startswith(name, "ejlptr\$")
        _, fname, arg1 = split(name, "\$")
        if startswith(name, "ejlstr\$")
            hnd_cache = get!(hnd_string_map, arg1) do
                Ref{Ptr{Cvoid}}(C_NULL)
            end
        else
            arg1 =  parse(Int, arg1)
            hnd_cache = get!(hnd_int_map, arg1) do
                Ref{Ptr{Cvoid}}(C_NULL)
            end
            arg1 = reinterpret(Ptr{Cchar}, arg1)
        end
        return @ccall jl_load_and_lookup(arg1::Cstring, fname::Cstring, hnd_cache::Ptr{Cvoid})::Ptr{Cvoid}
    end
    return nothing
end

function define_absolute_symbol(jd, name)
    ptr = LLVM.find_symbol(name)
    if ptr !== C_NULL
        LLVM.define(jd, absolute_symbol_materialization(name, ptr))
        return true
    end
    return false
end

function setup_globals()
    opt_level = Base.JLOptions().opt_level
    if opt_level < 2
        optlevel = LLVM.API.LLVMCodeGenLevelNone
    elseif opt_level == 2
        optlevel = LLVM.API.LLVMCodeGenLevelDefault
    else
        optlevel = LLVM.API.LLVMCodeGenLevelAggressive
    end

    lljit = JuliaOJIT()

    tempTM = LLVM.JITTargetMachine(LLVM.triple(lljit), cpu_name(), cpu_features(); optlevel)
    LLVM.asm_verbosity!(tempTM, true)
    tm[] = tempTM

    jd_main = JITDylib(lljit)

    prefix = LLVM.get_prefix(lljit)
    dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
    LLVM.add!(jd_main, dg)

    jit[] = CompilerInstance(lljit)

    jd_main, lljit
end

function __init__()
    jd_main, lljit = setup_globals()

    if Sys.iswindows() && Int === Int64
        # TODO can we check isGNU?
        define_absolute_symbol(jd_main, mangle(lljit, "___chkstk_ms"))
    end

    hnd = unsafe_load(cglobal(:jl_libjulia_handle, Ptr{Cvoid}))
    for (k, v) in Compiler.JuliaGlobalNameMap
        ptr = unsafe_load(Base.reinterpret(Ptr{Ptr{Cvoid}}, Libdl.dlsym(hnd, k)))
        LLVM.define(
            jd_main,
            absolute_symbol_materialization(mangle(lljit, "ejl_" * k), ptr),
        )
    end

    for (k, v) in Compiler.JuliaEnzymeNameMap
        ptr = Compiler.unsafe_to_ptr(v)
        LLVM.define(
            jd_main,
            absolute_symbol_materialization(mangle(lljit, "ejl_" * k), ptr),
        )
    end

    atexit() do
        dispose(tm[])
    end
end

function move_to_threadsafe(ir)
    LLVM.verify(ir) # try to catch broken modules

    # So 1. serialize the module
    buf = convert(MemoryBuffer, ir)

    # 2. deserialize and wrap by a ThreadSafeModule
    return ThreadSafeContext() do ctx
        mod = parse(LLVM.Module, buf)
        ThreadSafeModule(mod)
    end
end

function prepare!(mod)
    # On Windows, LLVM's GlobalOpt demotes internal functions to `private` linkage,
    # which emits no object symbol. Julia's per-symbol Win64 JIT unwind registrar
    # (create_PRUNTIME_FUNCTION) then skips those functions, so their frames get no
    # RUNTIME_FUNCTION and a fault (e.g. a GC safepoint) landing on one defeats
    # Windows exception dispatch. Promote them back to `internal` here -- the last
    # step before JIT emission, after all optimization -- so they keep a local
    # symbol and get registered. See EnzymeAD/Enzyme.jl#3374.
    if Sys.iswindows()
        for f in functions(mod)
            if !LLVM.isdeclaration(f) && LLVM.linkage(f) == LLVM.API.LLVMPrivateLinkage
                LLVM.linkage!(f, LLVM.API.LLVMInternalLinkage)
            end
        end
    end
    for f in collect(functions(mod))
        ptr = fix_ptr_lookup(LLVM.name(f))
        if ptr === nothing
            continue
        end
        ptr = reinterpret(UInt, ptr)
        ptr = LLVM.ConstantInt(ptr)
        ptr = LLVM.const_inttoptr(ptr, LLVM.PointerType(LLVM.function_type(f)))
        replace_uses!(f, ptr)
        Compiler.eraseInst(mod, f)
    end
    # Bind the symbolic Julia value references (`compiler/relocation.jl`) to the objects'
    # addresses in this process.
    for g in collect(globals(mod))
        Compiler.is_relocation_name(LLVM.name(g)) || continue
        ptr = Compiler.relocation_pointer(LLVM.name(g))

        ptr = reinterpret(UInt, ptr)
        ptr = LLVM.ConstantInt(ptr)
        ptr = LLVM.const_inttoptr(ptr, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[])))
        ptr = LLVM.const_addrspacecast(ptr, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), 10))
        replace_uses!(g, ptr)
        Compiler.eraseInst(mod, g)
    end
end

function add!(mod)
    prepare!(mod)
    lljit = jit[].jit
    jd = LLVM.JITDylib(lljit)
    tsm = move_to_threadsafe(mod)
    LLVM.add!(lljit, jd, tsm)
    return jd
end

function lookup(name)
    lljit = jit[].jit
    LLVM.lookup(lljit, JITDylib(lljit), name)
end

# The address of `name` if the JIT already defines it, else `C_NULL`. `LLVM.lookup` raises
# on an undefined symbol, so probe through the C API and clear the error.
function lookup_or_null(name)::Ptr{Cvoid}
    lljit = jit[].jit
    addr = Ref{LLVM.API.LLVMOrcJITTargetAddress}()
    err = LLVM.API.JLJITLookup(lljit, addr, name, true)
    if err != C_NULL
        LLVM.API.LLVMConsumeError(err)
        return C_NULL
    end
    return reinterpret(Ptr{Cvoid}, addr[] % UInt)
end

function lookup(jd::JITDylib, name)
    LLVM.lookup(jit[].jit, jd, name)
end

end # module
