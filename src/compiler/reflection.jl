function get_job(
    @nospecialize(func),
    @nospecialize(A),
    @nospecialize(types);
    run_enzyme::Bool = true,
    mode::API.CDerivativeMode = API.DEM_ReverseModeCombined,
    dupClosure::Bool = false,
    argwrap::Bool = true,
    width::Int = 1,
    modifiedBetween = nothing,
    returnPrimal::Bool = false,
    augmentedInit = false,
    world = nothing,
    ABI = DefaultABI,
    ErrIfFuncWritten = false,
    RuntimeActivity = true,
    kwargs...,
)

    tt = Tuple{map(eltype, types.parameters)...}


    primal, rt = if world isa Nothing
        fspec(Core.Typeof(func), types), Compiler.primal_return_type(mode == API.DEM_ForwardMode ? Forward : Reverse, Core.Typeof(func), tt)
    else
        fspec(Core.Typeof(func), types, world), Compiler.primal_return_type_world(mode == API.DEM_ForwardMode ? Forward : Reverse, world, Core.Typeof(func), tt)
    end

    rt = A{rt}
    target = Compiler.EnzymeTarget()
    if modifiedBetween === nothing
        defaultMod = mode != API.DEM_ReverseModeCombined && mode != API.DEM_ForwardMode
        modifiedBetween = (defaultMod, (defaultMod for _ in types.parameters)...)
    end
    params = Compiler.EnzymeCompilerParams(
        Tuple{(dupClosure ? Duplicated : Const){Core.Typeof(func)},types.parameters...},
        mode,
        width,
        rt,
        run_enzyme,
        argwrap,
        modifiedBetween,
        returnPrimal,
        augmentedInit,
        Compiler.UnknownTapeType,
        ABI,
        ErrIfFuncWritten,
        RuntimeActivity,
    )
    if world isa Nothing
        return Compiler.CompilerJob(
            primal,
            CompilerConfig(target, params; kernel = false),
        )
    else
        return Compiler.CompilerJob(
            primal,
            CompilerConfig(target, params; kernel = false),
            world,
        )
    end
end

function reflect(
    @nospecialize(func),
    @nospecialize(A),
    @nospecialize(types);
    optimize::Bool = true,
    second_stage::Bool = true,
    kwargs...,
)

    job = get_job(func, A, types; kwargs...)
    # Codegen the primal function and all its dependency in one module
    mod, meta = Compiler.codegen(:llvm, job; optimize) #= validate=false =#

    if second_stage
        post_optimze!(mod, JIT.get_tm())
    end

    llvmf = meta.adjointf

    return llvmf, mod
end

struct jl_llvmf_dump
    TSM::LLVM.API.LLVMOrcThreadSafeModuleRef
    F::LLVM.API.LLVMValueRef
end

function enzyme_code_llvm(
    io::IO,
    @nospecialize(func),
    @nospecialize(A),
    @nospecialize(types);
    optimize::Bool = true,
    run_enzyme::Bool = true,
    second_stage::Bool = true,
    raw::Bool = false,
    debuginfo::Symbol = :default,
    dump_module::Bool = false,
    mode = API.DEM_ReverseModeCombined,
)
    JuliaContext() do ctx
        entry_fn, ir = reflect(func, A, types; optimize, run_enzyme, second_stage, mode)
        ts_mod = ThreadSafeModule(ir)
        GC.@preserve ts_mod entry_fn begin
            value = Ref(jl_llvmf_dump(ts_mod.ref, entry_fn.ref))
            str = ccall(
                :jl_dump_function_ir,
                Ref{String},
                (Ptr{jl_llvmf_dump}, Bool, Bool, Ptr{UInt8}),
                value,
                !raw,
                dump_module,
                debuginfo,
            )
        end
        print(io, str)
    end
end
enzyme_code_llvm(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...) =
    enzyme_code_llvm(stdout, func, A, types; kwargs...)

function enzyme_code_native(
    io::IO,
    @nospecialize(func),
    @nospecialize(A),
    @nospecialize(types);
    mode = API.DEM_ReverseModeCombined,
)
    JuliaContext() do ctx
        _, mod = reflect(func, A, types; mode)
        str = String(LLVM.emit(JIT.get_tm(), mod, LLVM.API.LLVMAssemblyFile))
        print(io, str)
    end
end
enzyme_code_native(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...) =
    enzyme_code_native(stdout, func, A, types; kwargs...)

function enzyme_code_typed(
    @nospecialize(func),
    @nospecialize(A),
    @nospecialize(types);
    kwargs...,
)
    job = get_job(func, A, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end
