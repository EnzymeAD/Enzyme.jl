module JIT

using LLVM, LLVM.IR, LLVM.ORC
using Libdl
import LLVM: TargetMachine

import GPUCompiler
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features

struct CompilerInstance
    jit::LLVM.JuliaOJIT
    # The JITDylib that defines the globals of Enzyme. `lookup(name)` also uses it.
    # Before Julia 1.14, this is the one JITDylib that Julia's JIT gives to its users. Enzyme
    # adds each module to it. Since Julia 1.14, this is a JITDylib that only has the globals
    # of Enzyme, and each module has its own JITDylib that resolves names in it (see `dylib`).
    jd::LLVM.JITDylib
end

function LLVM.dispose(ci::CompilerInstance)
    dispose(ci.jit)
    return nothing
end

const jit = Ref{CompilerInstance}()
const tm = Ref{TargetMachine}() # for opt pipeline

get_tm() = tm[]
get_jit() = jit[].jit

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
            arg1 = parse(Int, arg1)
            hnd_cache = get!(hnd_int_map, arg1) do
                Ref{Ptr{Cvoid}}(C_NULL)
            end
            arg1 = reinterpret(Ptr{Cchar}, arg1)
        end
        return @ccall jl_load_and_lookup(arg1::Cstring, fname::Cstring, hnd_cache::Ptr{Cvoid})::Ptr{Cvoid}
    end
    return nothing
end

# Define `name` in the JIT as the address of the symbol with that name in the process.
function define_absolute_symbol(jd, lljit, name::String)
    ptr = LLVM.find_symbol(name)
    if ptr !== C_NULL
        define!(jd, absolute_symbols(mangle(lljit, name) => ptr))
        return true
    end
    return false
end

function setup_globals()
    opt_level = Base.JLOptions().opt_level
    if opt_level < 2
        optlevel = LLVM.CodeGenOptLevel.None
    elseif opt_level == 2
        optlevel = LLVM.CodeGenOptLevel.Default
    else
        optlevel = LLVM.CodeGenOptLevel.Aggressive
    end

    lljit = JuliaOJIT()

    tempTM = LLVM.JITTargetMachine(;
        triple = lljit.triple, cpu = cpu_name(), features = cpu_features(), opt_level = optlevel
    )
    LLVM.asm_verbosity!(tempTM, true)
    tm[] = tempTM

    es = lljit.execution_session

    # `jd_globals` is the JITDylib that defines the globals of Enzyme for each module.
    @static if VERSION >= v"1.14.0-DEV.2171"
        # Each JITDylib that Julia's JIT creates resolves the names of Julia and the symbols
        # of the process (JuliaLang/julia#60988), but not the names in a JITDylib of a
        # different user. `dylib` adds this JITDylib to the link order of each JITDylib that
        # Enzyme creates.
        jd_globals = JITDylib(lljit, "enzyme.globals")
        # Julia compiles the thunks of the `InlineABI` in its own JITDylib, and these thunks
        # refer to the globals of Enzyme. Julia's JITDylib resolves a name in the JITDylibs
        # of its link order in sequence, so it uses `jd_globals` only for a name that Julia
        # does not define.
        LLVM.add_to_link_order!(something(LLVM.lookup_dylib(es, "JuliaOJIT")), jd_globals)
    else
        # Julia's JIT has one JITDylib for all its users.
        jd_globals = lljit.external_dylib
        LLVM.add!(jd_globals, DynamicLibrarySearchGenerator(lljit))
    end

    jit[] = CompilerInstance(lljit, jd_globals)

    return jd_globals, lljit
end

# Return the JITDylib to add one module to.
#
# Since Julia 1.14 (JuliaLang/julia#60988), each call creates a new JITDylib. The new JITDylib
# resolves the names that Julia and the process define, and the globals of Enzyme, because
# the JITDylib of `setup_globals` is in its link order. No other JITDylib
# resolves the names that the module defines. Thus two modules can define the same name. Each
# module must define all other names that it refers to. Julia's codegen gives names that are
# unique only in one module (JuliaLang/julia#60031).
#
# Before Julia 1.14, Julia's JIT has one JITDylib for all its users. There, Julia's codegen
# gives names that are unique in the session.
function dylib()
    @static if VERSION >= v"1.14.0-DEV.2171"
        jd = JITDylib(jit[].jit, "enzyme")
        LLVM.add_to_link_order!(jd, jit[].jd)
        return jd
    else
        return jit[].jd
    end
end

function __init__()
    jd_globals, lljit = setup_globals()

    if Sys.iswindows() && Int === Int64
        # TODO can we check isGNU?
        define_absolute_symbol(jd_globals, lljit, "___chkstk_ms")
    end

    # The well-known Julia values, defined in one go.
    hnd = unsafe_load(cglobal(:jl_libjulia_handle, Ptr{Cvoid}))
    pairs = Pair{LLVMSymbol, Ptr{Cvoid}}[]
    for k in keys(Compiler.JuliaGlobalNameMap)
        ptr = unsafe_load(Base.reinterpret(Ptr{Ptr{Cvoid}}, Libdl.dlsym(hnd, k)))
        push!(pairs, mangle(lljit, "ejl_" * k) => ptr)
    end
    for (k, v) in Compiler.JuliaEnzymeNameMap
        ptr = Compiler.unsafe_to_ptr(Compiler.unbind(v))
        push!(pairs, mangle(lljit, "ejl_" * k) => ptr)
    end
    define!(jd_globals, absolute_symbols(pairs))

    return atexit() do
        dispose(tm[])
    end
end

function move_to_threadsafe(ir)
    LLVM.verify(ir) # try to catch broken modules

    # So 1. serialize the module, and 2. deserialize and wrap by a ThreadSafeModule
    return @dispose buf = convert(MemoryBuffer, ir) ctx = ThreadSafeContext() begin
        mod = parse(LLVM.Module, buf)
        ThreadSafeModule(mod)
    end
end

function prepare!(mod)
    Compiler.modern_write_barriers!(mod)
    # On Windows, LLVM's GlobalOpt demotes internal functions to `private` linkage,
    # which emits no object symbol. Julia's per-symbol Win64 JIT unwind registrar
    # (create_PRUNTIME_FUNCTION) then skips those functions, so their frames get no
    # RUNTIME_FUNCTION and a fault (e.g. a GC safepoint) landing on one defeats
    # Windows exception dispatch. Promote them back to `internal` here -- the last
    # step before JIT emission, after all optimization -- so they keep a local
    # symbol and get registered. See EnzymeAD/Enzyme.jl#3374.
    if Sys.iswindows()
        for f in mod.functions
            if !LLVM.isdeclaration(f) && f.linkage == LLVM.Linkage.Private
                f.linkage = LLVM.Linkage.Internal
            end
        end
    end
    for f in collect(mod.functions)
        ptr = fix_ptr_lookup(f.name)
        if ptr === nothing
            continue
        end
        ptr = reinterpret(UInt, ptr)
        ptr = LLVM.ConstantInt(ptr)
        ptr = LLVM.const_inttoptr(ptr, LLVM.PointerType(f.function_type))
        replace_uses!(f, ptr)
        Compiler.eraseInst(mod, f)
    end
    return
end

function add!(mod)
    prepare!(mod)
    lljit = jit[].jit
    jd = dylib()
    tsm = move_to_threadsafe(mod)
    LLVM.add!(lljit, jd, tsm)
    return jd
end

function lookup(name)
    return LLVM.lookup(jit[].jit, jit[].jd, name)
end

function lookup(jd::JITDylib, name)
    return LLVM.lookup(jit[].jit, jd, name)
end

end # module
