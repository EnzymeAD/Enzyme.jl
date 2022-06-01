function get_job(@nospecialize(func), @nospecialize(A), @nospecialize(types);
                 run_enzyme::Bool=true, mode::API.CDerivativeMode=API.DEM_ReverseModeCombined, dupClosure::Bool=false, argwrap::Bool=true, width::Int64=1, modifiedBetween::Bool=false, returnPrimal::Bool=false, kwargs...)

    primal, adjoint = fspec(Core.Typeof(func), types)

    tt    = Tuple{map(eltype, types.parameters)...}
    rt = Core.Compiler.return_type(func, tt)
    rt = A{rt}
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, mode, width, rt, run_enzyme, dupClosure, argwrap, modifiedBetween, returnPrimal)
    return Compiler.CompilerJob(target, primal, params)
end


function reflect(@nospecialize(func), @nospecialize(A), @nospecialize(types);
                 optimize::Bool=true, second_stage::Bool=true, ctx=nothing, kwargs...)

    job = get_job(func, A, types; kwargs...)
    # Codegen the primal function and all its dependency in one module
    mod, meta = Compiler.codegen(:llvm, job; optimize, ctx #= validate=false =#)

    if second_stage
        post_optimze!(mod, JIT.get_tm())
    end

    llvmf = meta.adjointf

    return llvmf, mod
end

function enzyme_code_llvm(io::IO, @nospecialize(func), @nospecialize(A), @nospecialize(types);
                          optimize::Bool=true, run_enzyme::Bool=true, second_stage::Bool=true,
                          raw::Bool=false, debuginfo::Symbol=:default, dump_module::Bool=false)
    JuliaContext() do ctx
        llvmf, _ = reflect(func, A, types; optimize, run_enzyme, second_stage, ctx)

        str = ccall(:jl_dump_function_ir, Ref{String},
                    (LLVM.API.LLVMValueRef, Bool, Bool, Ptr{UInt8}),
                    llvmf, !raw, dump_module, debuginfo)
        print(io, str)
    end
end
enzyme_code_llvm(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...) = enzyme_code_llvm(stdout, func, A, types; kwargs...)

function enzyme_code_native(io::IO, @nospecialize(func), @nospecialize(A), @nospecialize(types))
    JuliaContext() do ctx
        _, mod = reflect(func, A, types; ctx)
        str = String(LLVM.emit(JIT.get_tm(), mod, LLVM.API.LLVMAssemblyFile))
        print(io, str)
    end
end
enzyme_code_native(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...) = enzyme_code_native(stdout, func, A, types; kwargs...)

function enzyme_code_typed(@nospecialize(func), @nospecialize(A), @nospecialize(types); kwargs...)
    job = get_job(func, A, types; kwargs...)
    GPUCompiler.code_typed(job; kwargs...)
end
