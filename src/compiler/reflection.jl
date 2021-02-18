function reflect(@nospecialize(func), @nospecialize(types);
                 optimize::Bool=true, run_enzyme::Bool=true, second_stage::Bool=true, split::Bool=false)
    primal, adjoint, rt = fspec(func, types)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams()
    job    = Compiler.CompilerJob(target, primal, params)

    # Codegen the primal function and all its dependency in one module
    mod, primalf = Compiler.codegen(:llvm, job, optimize=false, #= validate=false =#)
    linkage!(primalf, LLVM.API.LLVMExternalLinkage)

    # Run pipeline and Enzyme pass
    if optimize
        optimize!(mod)
    end

    if run_enzyme
        annotate!(mod)
        adjointf, augmented_primalf = enzyme!(mod, primalf, adjoint, rt, split)
        if second_stage
            post_optimize!(mod)
        else
            legalize!(mod)
        end
        llvmf = adjointf
    else
        llvmf = primalf
    end


    return llvmf, mod
end

function enzyme_code_llvm(io::IO, @nospecialize(func), @nospecialize(types); 
                          optimize::Bool=true, run_enzyme::Bool=true, second_stage::Bool=true,
                          raw::Bool=false, debuginfo::Symbol=:default, dump_module::Bool=false)
    llvmf, mod = reflect(func, types, optimize=optimize, run_enzyme=run_enzyme, second_stage=second_stage)

    str = ccall(:jl_dump_function_ir, Ref{String},
                (LLVM.API.LLVMValueRef, Bool, Bool, Ptr{UInt8}),
                llvmf, !raw, dump_module, debuginfo)
    print(io, str)
end
enzyme_code_llvm(@nospecialize(func), @nospecialize(types); kwargs...) = enzyme_code_llvm(stdout, func, types; kwargs...)

function enzyme_code_native(io::IO, @nospecialize(func), @nospecialize(types))
    llvmf, mod = reflect(func, types)
    str = String(LLVM.emit(tm[], mod, LLVM.API.LLVMAssemblyFile))
    print(io, str)
end
enzyme_code_native(@nospecialize(func), @nospecialize(types); kwargs...) = enzyme_code_native(stdout, func, types; kwargs...)