module JIT

using LLVM
using Libdl
import LLVM:TargetMachine

import GPUCompiler
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features

@inline function use_ojit()
    return LLVM.has_julia_ojit() && !Sys.iswindows()
end

export get_trampoline

@static if use_ojit()
    struct CompilerInstance
        jit::LLVM.JuliaOJIT
        lctm::Union{LLVM.LazyCallThroughManager, Nothing}
        ism::Union{LLVM.IndirectStubsManager, Nothing}
    end
else
    struct CompilerInstance
        jit::LLVM.LLJIT
        lctm::Union{LLVM.LazyCallThroughManager, Nothing}
        ism::Union{LLVM.IndirectStubsManager, Nothing}
    end
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

function define_absolute_symbol(jd, name)
	ptr = LLVM.find_symbol(name)
	if ptr !== C_NULL
		LLVM.define(jd, absolute_symbol_materialization(name, ptr))
		return true
	end
	return false
end

function __init__()
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
    
    lljit = @static if !use_ojit()
        tempTM = LLVM.JITTargetMachine(LLVM.triple(), cpu_name(), cpu_features(); optlevel)
        LLVM.asm_verbosity!(tempTM, true)

        gdb = haskey(ENV, "ENABLE_GDBLISTENER")
        perf = haskey(ENV, "ENABLE_JITPROFILING")
        if gdb || perf
            ollc = LLVM.ObjectLinkingLayerCreator() do es, triple
                oll = ObjectLinkingLayer(es)
                if gdb
                    register!(oll, GDBRegistrationListener())
                end
                if perf
                    register!(oll, IntelJITEventListener())
                    register!(oll, PerfJITEventListener())
                end
                return oll
            end
            GC.@preserve ollc begin
                builder = LLJITBuilder()
                LLVM.linkinglayercreator!(builder, ollc)
                tmb = TargetMachineBuilder(tempTM)
                LLVM.targetmachinebuilder!(builder, tmb)
                LLJIT(builder)
            end
        else
            LLJIT(;tm=tempTM)
        end
    else
        JuliaOJIT()
    end

    jd_main = JITDylib(lljit)

    prefix = LLVM.get_prefix(lljit)
    dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
    LLVM.add!(jd_main, dg)

	if Sys.iswindows() && Int === Int64
		# TODO can we check isGNU?
		define_absolute_symbol(jd_main, mangle(lljit, "___chkstk_ms"))
	end

    es = ExecutionSession(lljit)
    try
        lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
        ism = LLVM.LocalIndirectStubsManager(triple(lljit))
        jit[] = CompilerInstance(lljit, lctm, ism)
    catch err
        @warn "OrcV2 initialization failed with" err
        jit[] = CompilerInstance(lljit, nothing, nothing)
    end

    hnd = Libdl.dlopen("libjulia")

    for (k, v) in Compiler.JuliaGlobalNameMap
        ptr = unsafe_load(Base.reinterpret(Ptr{Ptr{Cvoid}}, Libdl.dlsym(hnd, k)))
        LLVM.define(jd_main, absolute_symbol_materialization(mangle(lljit, "ejl_"*k), ptr))
    end

    for (k, v) in Compiler.JuliaEnzymeNameMap
        ptr = Compiler.unsafe_to_ptr(v)
        LLVM.define(jd_main, absolute_symbol_materialization(mangle(lljit, "ejl_"*k), ptr))
    end

    atexit() do
        @static if !use_ojit()
           ci = jit[]
           dispose(ci)
        end
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
                LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)

    alias = LLVM.API.LLVMOrcCSymbolAliasMapPair(
                mangle(lljit, entry),
                LLVM.API.LLVMOrcCSymbolAliasMapEntry(
                    mangle(lljit, target), flags))

    mu = LLVM.reexports(lctm, ism, jd, [alias])
    LLVM.define(jd, mu)

    LLVM.lookup(lljit, entry)
end

function get_trampoline(job)
    compiler = jit[]
    lljit = compiler.jit
    lctm  = compiler.lctm
    ism   = compiler.ism

    if lctm === nothing || ism === nothing
        error("Delayed compilation not available.")
    end

    mode = job.config.params.mode
    use_primal = mode == API.DEM_ReverseModePrimal

    # We could also use one dylib per job
    jd = JITDylib(lljit)

    sym = String(gensym(:func))
    _sym = String(gensym(:func))
    addr = add_trampoline!(jd, (lljit, lctm, ism),
                                   _sym, sym)

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
                LLVM.unsafe_delete!(mod, other_func)
            end

            tsm = move_to_threadsafe(mod)

            il = @static if use_ojit()
                LLVM.IRCompileLayer(lljit)
            else
                LLVM.IRTransformLayer(lljit)
            end
            LLVM.emit(il, mr, tsm)
        end
        return nothing
    end

    function discard(jd, sym) end

    flags = LLVM.API.LLVMJITSymbolFlags(
                LLVM.API.LLVMJITSymbolGenericFlagsCallable |
                LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)

    symbols = [
        LLVM.API.LLVMOrcCSymbolFlagsMapPair(
            mangle(lljit, sym), flags),
    ]

    mu = LLVM.CustomMaterializationUnit(sym, symbols,
                                        materialize, discard)
    LLVM.define(jd, mu)
    return addr
end

function add!(mod)
    lljit = jit[].jit
    jd = LLVM.JITDylib(lljit)
    tsm = move_to_threadsafe(mod)
    LLVM.add!(lljit, jd, tsm)
    return nothing
end

function lookup(_, name)
    LLVM.lookup(jit[].jit, name)
end

end # module
