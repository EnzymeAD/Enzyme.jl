
module JIT

using LLVM
using Libdl
import LLVM: TargetMachine

import GPUCompiler
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features

export get_trampoline

struct CompilerInstance
    jit::LLVM.JuliaOJIT
    lctm::Union{LLVM.LazyCallThroughManager,Nothing}
    ism::Union{LLVM.IndirectStubsManager,Nothing}
end

function LLVM.dispose(ci::CompilerInstance)
    if ci.ism !== nothing
        dispose(ci.ism)
    end
    if ci.lctm !== nothing
        dispose(ci.lctm)
    end
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

function fix_ptr_lookup(name)
    if startswith(name, "ejlstr\$") || startswith(name, "ejlptr\$")
        _, fname, arg1 = split(name, "\$")
        if startswith(name, "ejlstr\$")
            ptr = if haskey(hnd_string_map, arg1)
                hnd_string_map[arg1]
            else
                val = Ref{Ptr{Cvoid}}(C_NULL)
                hnd_string_map[arg1] = val
                val
            end

            return ccall(
                :ijl_load_and_lookup,
                Ptr{Cvoid},
                (Cstring, Cstring, Ptr{Cvoid}),
                arg1,
                fname,
                ptr
            )
        else
        end
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

    tempTM = LLVM.JITTargetMachine(LLVM.triple(), cpu_name(), cpu_features(); optlevel)
    LLVM.asm_verbosity!(tempTM, true)
    tm[] = tempTM

    lljit = JuliaOJIT()

    jd_main = JITDylib(lljit)

    prefix = LLVM.get_prefix(lljit)
    dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
    LLVM.add!(jd_main, dg)

    es = ExecutionSession(lljit)
    try
        lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
        ism = LLVM.LocalIndirectStubsManager(triple(lljit))
        jit[] = CompilerInstance(lljit, lctm, ism)
    catch err
        @warn "OrcV2 initialization failed with" err
        jit[] = CompilerInstance(lljit, nothing, nothing)
    end

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

function add_trampoline!(jd, (lljit, lctm, ism), entry, target)
    flags = LLVM.API.LLVMJITSymbolFlags(
        LLVM.API.LLVMJITSymbolGenericFlagsCallable |
        LLVM.API.LLVMJITSymbolGenericFlagsExported,
        0,
    )

    alias = LLVM.API.LLVMOrcCSymbolAliasMapPair(
        mangle(lljit, entry),
        LLVM.API.LLVMOrcCSymbolAliasMapEntry(mangle(lljit, target), flags),
    )

    mu = LLVM.reexports(lctm, ism, jd, [alias])
    LLVM.define(jd, mu)

    LLVM.lookup(lljit, entry)
end

function get_trampoline(job)
    compiler = jit[]
    lljit = compiler.jit
    lctm = compiler.lctm
    ism = compiler.ism

    if lctm === nothing || ism === nothing
        error("Delayed compilation not available.")
    end

    mode = job.config.params.mode
    use_primal = mode == API.DEM_ReverseModePrimal

    # We could also use one dylib per job
    jd = JITDylib(lljit)

    sym = String(gensym(:func))
    _sym = String(gensym(:func))
    addr = add_trampoline!(jd, (lljit, lctm, ism), _sym, sym)

    # 3. add MU that will call back into the compiler
    function materialize(mr)
        # Rename adjointf to match target_sym
        # Really we should do:
        # Create a re-export for a unique name, and a custom materialization unit that makes the deferred decision. E.g. add "foo" -> "my_deferred_decision_sym.1". Then define a CustomMU whose materialization method looks like:
        # 1. Make the runtime decision about what symbol should implement "foo". Let's call this "foo.rt.impl".
        # 2 Add a module defining "foo.rt.impl" to the JITDylib.
        # 2. Call MR.replace(symbolAliases({"my_deferred_decision_sym.1" -> "foo.rt.impl"})).
        GPUCompiler.JuliaContext() do ctx
            mod, adjoint_name, primal_name = Compiler._thunk(job)
            func_name = use_primal ? primal_name : adjoint_name
            other_name = !use_primal ? primal_name : adjoint_name

            func = functions(mod)[func_name]
            LLVM.name!(func, sym)

            if other_name !== nothing
                # Otherwise MR will complain -- we could claim responsibilty,
                # but it would be nicer if _thunk just codegen'd the half
                # we need.
                other_func = functions(mod)[other_name]
                Compiler.eraseInst(mod, other_func)
            end

            tsm = move_to_threadsafe(mod)

            il = LLVM.IRCompileLayer(lljit)
            LLVM.emit(il, mr, tsm)
        end
        return nothing
    end

    function discard(jd, sym) end

    flags = LLVM.API.LLVMJITSymbolFlags(
        LLVM.API.LLVMJITSymbolGenericFlagsCallable |
        LLVM.API.LLVMJITSymbolGenericFlagsExported,
        0,
    )

    symbols = [LLVM.API.LLVMOrcCSymbolFlagsMapPair(mangle(lljit, sym), flags)]

    mu = LLVM.CustomMaterializationUnit(sym, symbols, materialize, discard)
    LLVM.define(jd, mu)
    return addr
end

function add!(mod)
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
    lljit = jit[].jit
    jd = LLVM.JITDylib(lljit)
    tsm = move_to_threadsafe(mod)
    LLVM.add!(lljit, jd, tsm)
    return nothing
end

function lookup(name)
    @static if sizeof(Int) == sizeof(Int64)
	LLVM.lookup(jit[].jit, name)
    else
	UInt(reinterpret(UInt64, LLVM.lookup(jit[].jit, name)))
    end
end

end # module
