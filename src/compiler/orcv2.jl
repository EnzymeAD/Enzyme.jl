module JIT

using LLVM
import LLVM:TargetMachine

import GPUCompiler
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features


export get_trampoline

struct CompilerInstance
    jit::LLVM.JuliaOJIT
    lctm::Union{LLVM.LazyCallThroughManager, Nothing}
    ism::Union{LLVM.IndirectStubsManager, Nothing}
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
	gv = LLVM.API.LLVMJITCSymbolMapPair(name, symbol)

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

    tempTM = LLVM.JITTargetMachine(LLVM.triple(), cpu_name(), cpu_features(); optlevel)
    LLVM.asm_verbosity!(tempTM, true)

    # gdb = haskey(ENV, "ENABLE_GDBLISTENER")
    # perf = haskey(ENV, "ENABLE_JITPROFILING")
    # if gdb || perf
    #     ollc = LLVM.ObjectLinkingLayerCreator() do es, triple
    #         oll = ObjectLinkingLayer(es)
    #         if gdb
    #             register!(oll, GDBRegistrationListener())
    #         end
    #         if perf
    #             register!(oll, IntelJITEventListener())
    #             register!(oll, PerfJITEventListener())
    #         end
    #         return oll
    #     end
    #     GC.@preserve ollc begin
    #         builder = LLJITBuilder()
    #         LLVM.linkinglayercreator!(builder, ollc)
    #         tmb = TargetMachineBuilder(tempTM)
    #         LLVM.targetmachinebuilder!(builder, tmb)
    #         lljit = LLJIT(builder)
    #     end
    # else
    #     lljit = LLJIT(;tm=tempTM)

    # end
    jljit = JuliaOJIT()

    jd_main = JITDylib(jljit)

    prefix = LLVM.get_prefix(jljit)
    dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
    LLVM.add!(jd_main, dg)

	if Sys.iswindows() && Int === Int64
		# TODO can we check isGNU?
		define_absolute_symbol(jd_main, mangle(jljit, "___chkstk_ms"))
	end

    es = ExecutionSession(jljit)
    try
        lctm = LLVM.LocalLazyCallThroughManager(triple(jljit), es)
        ism = LLVM.LocalIndirectStubsManager(triple(jljit))
        jit[] = CompilerInstance(jljit, lctm, ism)
    catch err
        @warn "OrcV2 initialization failed with" err
        jit[] = CompilerInstance(jljit, nothing, nothing)
    end

    # atexit() do
    #     ci = jit[]
    #     dispose(ci)
    #     dispose(tm[])
    # end
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

function add_trampoline!(jd, (jljit, lctm, ism), entry, target)
    flags = LLVM.API.LLVMJITSymbolFlags(
                LLVM.API.LLVMJITSymbolGenericFlagsCallable |
                LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)

    alias = LLVM.API.LLVMOrcCSymbolAliasMapPair(
                mangle(jljit, entry),
                LLVM.API.LLVMOrcCSymbolAliasMapEntry(
                    mangle(jljit, target), flags))

    mu = LLVM.reexports(lctm, ism, jd, [alias])
    LLVM.define(jd, mu)

    LLVM.lookup(jljit, entry)
end

function get_trampoline(job)
    compiler = jit[]
    jljit = compiler.jit
    lctm  = compiler.lctm
    ism   = compiler.ism

    if lctm === nothing || ism === nothing
        error("Delayed compilation not available.")
    end

    mode = job.config.params.mode
    needs_augmented_primal = mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient

    # We could also use one dylib per job
    jd = JITDylib(jljit)

    adjoint_sym = String(gensym(:adjoint))
    _adjoint_sym = String(gensym(:adjoint))
    adjoint_addr = add_trampoline!(jd, (jljit, lctm, ism),
                                   _adjoint_sym, adjoint_sym)

    if needs_augmented_primal
        primal_sym = String(gensym(:augmented_primal))
        _primal_sym = String(gensym(:augmented_primal))
        primal_addr = add_trampoline!(jd, (jljit, lctm, ism),
                                      _primal_sym, primal_sym)
    else
        primal_sym = nothing
        primal_addr = nothing
    end

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
            adjointf = functions(mod)[adjoint_name]
            LLVM.name!(adjointf, adjoint_sym)
            if needs_augmented_primal
                primalf = functions(mod)[primal_name]
                LLVM.name!(primalf, primal_sym)
            else
                @assert primal_name === nothing
                primalf = nothing
            end

            tsm = move_to_threadsafe(mod)
            il = LLVM.IRTransformLayer(lljit)
            LLVM.emit(il, mr, tsm)
        end

        tsm = move_to_threadsafe(mod)
        il = LLVM.IRCompileLayer(jljit)
        LLVM.emit(il, mr, tsm)
        return nothing
    end

    function discard(jd, sym) end

    flags = LLVM.API.LLVMJITSymbolFlags(
                LLVM.API.LLVMJITSymbolGenericFlagsCallable |
                LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)

    symbols = [
        LLVM.API.LLVMOrcCSymbolFlagsMapPair(
            mangle(jljit, adjoint_sym), flags),
    ]
    if needs_augmented_primal
        push!(symbols, LLVM.API.LLVMOrcCSymbolFlagsMapPair(
            mangle(jljit, primal_sym), flags),)
    end

    mu = LLVM.CustomMaterializationUnit(adjoint_sym, symbols,
                                        materialize, discard)
    LLVM.define(jd, mu)
    return adjoint_addr, primal_addr
end

function add!(mod)
    jljit = jit[].jit
    jd = LLVM.JITDylib(jljit)
    tsm = move_to_threadsafe(mod)
    LLVM.add!(jljit, jd, tsm)
    return nothing
end

function lookup(_, name)
    LLVM.lookup(jit[].jit, name)
end

end # module
