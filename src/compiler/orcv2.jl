module JIT

using LLVM
import LLVM: TargetMachine

import GPUCompiler
import ..Compiler

export get_trampoline

struct CompilerInstance
    jit::LLVM.LLJIT
    lctm::Union{LLVM.LazyCallThroughManager,Nothing}
    ism::Union{LLVM.IndirectStubsManager,Nothing}
end

function LLVM.dispose(ci::CompilerInstance)
    dispose(ci.jit)
    if ci.lctm !== nothing
        dispose(ci.lctm)
    end
    if ci.ism !== nothing
        dispose(ci.ism)
    end
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

    tempTM = LLVM.JITTargetMachine(; optlevel = optlevel)
    LLVM.asm_verbosity!(tempTM, true)
    tm[] = tempTM

    tempTM = LLVM.JITTargetMachine(; optlevel)
    LLVM.asm_verbosity!(tempTM, true)

    if haskey(ENV, "ENABLE_GDBLISTENER")
        ollc = LLVM.ObjectLinkingLayerCreator() do es, triple
            oll = ObjectLinkingLayer(es)
            register!(oll, GDBRegistrationListener())
            return oll
        end

        GC.@preserve ollc begin
            builder = LLJITBuilder()
            LLVM.linkinglayercreator!(builder, ollc)
            tmb = TargetMachineBuilder(tempTM)
            LLVM.targetmachinebuilder!(builder, tmb)
            lljit = LLJIT(builder)
        end
    else
        lljit = LLJIT(; tm = tempTM)
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

    atexit() do
        ci = jit[]
        dispose(ci)
        dispose(tm[])
    end
end

function move_to_threadsafe(ir)
    LLVM.verify(ir) # try to catch broken modules

    # So 1. serialize the module
    buf = convert(MemoryBuffer, ir)

    # 2. deserialize and wrap by a ThreadSafeModule
    return ThreadSafeContext() do ctx
        mod = parse(LLVM.Module, buf; ctx = context(ctx))
        ThreadSafeModule(mod; ctx)
    end
end

function get_trampoline(job)
    compiler = jit[]
    lljit = compiler.jit
    lctm = compiler.lctm
    ism = compiler.ism

    if lctm === nothing || ism === nothing
        error("Delayed compilation not available.")
    end

    # We could also use one dylib per job
    jd = JITDylib(lljit)

    entry_sym = String(gensym(:entry))
    target_sym = String(gensym(:target))
    flags = LLVM.API.LLVMJITSymbolFlags(
        LLVM.API.LLVMJITSymbolGenericFlagsCallable | LLVM.API.LLVMJITSymbolGenericFlagsExported,
        0,
    )
    entry = LLVM.API.LLVMOrcCSymbolAliasMapPair(
        mangle(lljit, entry_sym),
        LLVM.API.LLVMOrcCSymbolAliasMapEntry(mangle(lljit, target_sym), flags),
    )

    mu = LLVM.reexports(lctm, ism, jd, Ref(entry))
    LLVM.define(jd, mu)

    # 2. Lookup address of entry symbol
    addr = LLVM.lookup(lljit, entry_sym)

    # 3. add MU that will call back into the compiler
    sym = LLVM.API.LLVMOrcCSymbolFlagsMapPair(mangle(lljit, target_sym), flags)

    function materialize(mr)
        mod, adjoint_name, primal_name = Compiler._thunk(job)
        adjointf = functions(mod)[adjoint_name]

        # Rename adjointf to match target_sym
        # Really we should do:
        # Create a re-export for a unique name, and a custom materialization unit that makes the deferred decision. E.g. add "foo" -> "my_deferred_decision_sym.1". Then define a CustomMU whose materialization method looks like:
        # 1. Make the runtime decision about what symbol should implement "foo". Let's call this "foo.rt.impl".
        # 2 Add a module defining "foo.rt.impl" to the JITDylib.
        # 2. Call MR.replace(symbolAliases({"my_deferred_decision_sym.1" -> "foo.rt.impl"})).
        LLVM.name!(adjointf, target_sym)
        tsm = move_to_threadsafe(mod)

        il = LLVM.IRTransformLayer(lljit)
        LLVM.emit(il, mr, tsm)

        return nothing
    end

    function discard(jd, sym) end

    mu = LLVM.CustomMaterializationUnit(entry_sym, Ref(sym), materialize, discard)
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
