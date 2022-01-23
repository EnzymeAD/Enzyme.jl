function get_job(@nospecialize(func), @nospecialize(A), @nospecialize(types);
                 run_enzyme::Bool=true, mode::API.CDerivativeMode=API.DEM_ReverseModeCombined, dupClosure::Bool=false, kwargs...)

    primal, adjoint = fspec(func, types)

    rt = Core.Compiler.return_type(primal.f, primal.tt)
    rt = A{rt}

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, split, rt, run_enzyme, dupClosure)
    job    = Compiler.CompilerJob(target, primal, params)

    # Codegen the primal function and all its dependency in one module
    mod, meta = Compiler.codegen(:llvm, job, optimize=optimize, #= validate=false =#)


    if second_stage
        post_optimze!(mod, JIT.get_tm())
    end

    llvmf = meta.adjointf

    return llvmf, mod
end

function enzyme_code_llvm(io::IO, @nospecialize(func), @nospecialize(A), @nospecialize(types);
                          optimize::Bool=true, run_enzyme::Bool=true, second_stage::Bool=true,
                          raw::Bool=false, debuginfo::Symbol=:default, dump_module::Bool=false)
    llvmf, mod = reflect(func, A, types; optimize,run_enzyme, second_stage)

    str = ccall(:jl_dump_function_ir, Ref{String},
                (LLVM.API.LLVMValueRef, Bool, Bool, Ptr{UInt8}),
                llvmf, !raw, dump_module, debuginfo)
    print(io, str)
end
enzyme_code_llvm(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...) = enzyme_code_llvm(stdout, func, A, types; kwargs...)

function enzyme_code_native(io::IO, @nospecialize(func), @nospecialize(A), @nospecialize(types))
    llvmf, mod = reflect(func, A, types)
    str = String(LLVM.emit(JIT.get_tm(), mod, LLVM.API.LLVMAssemblyFile))
    print(io, str)
end
enzyme_code_native(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...) = enzyme_code_native(stdout, func, A, types; kwargs...)

function enzyme_code_typed(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...)
    job = get_job(func, A, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end
