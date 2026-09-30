
module JIT

using LLVM, LLVM.IR, LLVM.ORC
using Libdl
import LLVM: TargetMachine

import GPUCompiler
import ..Compiler
import ..Compiler: API, cpu_name, cpu_features

export get_trampoline

struct CompilerInstance
    jit::LLVM.JuliaOJIT
    # the dylib everything is added to, which also defines Enzyme's globals
    jd::LLVM.JITDylib
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
        define!(jd, absolute_symbols(name => ptr))
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

    # before Julia 1.14, Julia's JIT has a single JITDylib that is shared by all its users
    jd_main = @static if VERSION >= v"1.14.0-DEV.2171"
        JITDylib(lljit, "enzyme")
    else
        lljit.external_dylib
    end

    LLVM.add!(jd_main, DynamicLibrarySearchGenerator(lljit))

    es = lljit.execution_session
    try
        lctm = LLVM.LocalLazyCallThroughManager(lljit.triple, es)
        ism = LLVM.LocalIndirectStubsManager(lljit.triple)
        jit[] = CompilerInstance(lljit, jd_main, lctm, ism)
    catch err
        @warn "OrcV2 initialization failed with" err
        jit[] = CompilerInstance(lljit, jd_main, nothing, nothing)
    end

    jd_main, lljit
end

function __init__()
    jd_main, lljit = setup_globals()

    if Sys.iswindows() && Int === Int64
        # TODO can we check isGNU?
        define_absolute_symbol(jd_main, mangle(lljit, "___chkstk_ms"))
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
    define!(jd_main, absolute_symbols(pairs))

    atexit() do
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

function add_trampoline!(jd, (lljit, lctm, ism), entry, target)
    flags = SymbolFlags(callable = true, exported = true)
    mu = lazy_reexports(lctm, ism, jd, [mangle(lljit, entry) => (mangle(lljit, target), flags)])
    define!(jd, mu)

    LLVM.lookup(lljit, jd, entry)
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
    jd = compiler.jd

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
            mod, edges, adjoint_name, primal_name = Compiler._thunk(job)
            func_name = use_primal ? primal_name : adjoint_name
            other_name = !use_primal ? primal_name : adjoint_name

            func = mod.functions[func_name]
            func.name = sym

            if other_name !== nothing
                # Otherwise MR will complain -- we could claim responsibilty,
                # but it would be nicer if _thunk just codegen'd the half
                # we need.
                other_func = mod.functions[other_name]
                Compiler.eraseInst(mod, other_func)
            end

	    prepare!(mod)
            tsm = move_to_threadsafe(mod)

            emit!(lljit.ir_compile_layer, mr, tsm)
        end
        return nothing
    end

    function discard(jd, sym) end

    flags = SymbolFlags(callable = true, exported = true)
    symbols = [mangle(lljit, sym) => flags]

    mu = CustomMaterializationUnit(sym, symbols, materialize, discard)
    define!(jd, mu)
    return addr
end

function add!(mod)
    prepare!(mod)
    lljit = jit[].jit
    jd = jit[].jd
    tsm = move_to_threadsafe(mod)
    LLVM.add!(lljit, jd, tsm)
    return jd
end

function lookup(name)
    return LLVM.lookup(jit[].jit, jit[].jd, name)
end

function lookup(jd::JITDylib, name)
    LLVM.lookup(jit[].jit, jd, name)
end

end # module
