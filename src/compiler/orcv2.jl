module JIT

using LLVM
import LLVM:TargetMachine

import GPUCompiler
import ..Compiler

export get_trampoline

struct CompilerInstance
    jit::LLVM.LLJIT
    lctm::LLVM.LazyCallThroughManager
    ism::LLVM.IndirectStubsManager
end

const jit = Ref{CompilerInstance}()
const tm = Ref{TargetMachine}() # for opt pipeline

get_tm() = tm[]

function __init__()
    opt_level = Base.JLOptions().opt_level
    if opt_level < 2
        optlevel = LLVM.API.LLVMCodeGenLevelNone
    elseif opt_level == 2
        optlevel = LLVM.API.LLVMCodeGenLevelDefault
    else
        optlevel = LLVM.API.LLVMCodeGenLevelAggressive
    end

    tempTM = LLVM.JITTargetMachine(;optlevel=optlevel)
    LLVM.asm_verbosity!(tempTM, true)
    tm[] = tempTM

    tempTM = LLVM.JITTargetMachine(;optlevel)
    LLVM.asm_verbosity!(tempTM, true)

    lljit = LLJIT(;tm=tempTM)

    jd_main = JITDylib(lljit)

    prefix = LLVM.get_prefix(lljit)
    dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
    LLVM.add!(jd_main, dg)

    es = ExecutionSession(lljit)

    lctm = LLVM.LocalLazyCallThroughManager(triple(lljit), es)
    ism = LLVM.LocalIndirectStubsManager(triple(lljit))

    jit[] = CompilerInstance(lljit, lctm, ism)
    atexit() do
        ci = jit[]
        dispose(ci.ism)
        dispose(ci.lctm)
        dispose(ci.jit)
        dispose(tm[])
    end
end

function move_to_threadsafe(ir)
    LLVM.verify(ir)

    # So 1. serialize the module
    buf = convert(MemoryBuffer, ir)

    # 2. deserialize and wrap by a ThreadSafeModule
    ctx = ThreadSafeContext()
    mod = parse(LLVM.Module, buf; ctx=context(ctx))
    return ThreadSafeModule(mod; ctx)
end

function get_trampoline(job)
    compiler = jit[]
    lljit = compiler.jit
    lctm  = compiler.lctm
    ism   = compiler.ism

    # We could also use one dylib per job
    jd = JITDylib(lljit)

    entry_sym = String(gensym(:entry))
    target_sym = String(gensym(:target))
    flags = LLVM.API.LLVMJITSymbolFlags(
                LLVM.API.LLVMJITSymbolGenericFlagsCallable |
                LLVM.API.LLVMJITSymbolGenericFlagsExported, 0)
    entry = LLVM.API.LLVMOrcCSymbolAliasMapPair(
                mangle(lljit, entry_sym),
                LLVM.API.LLVMOrcCSymbolAliasMapEntry(
                    mangle(lljit, target_sym), flags))

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

    function discard(jd, sym)
    end

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
