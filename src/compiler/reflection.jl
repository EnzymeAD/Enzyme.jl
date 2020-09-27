function enzyme_code_llvm(io::IO, @nospecialize(func), @nospecialize(types); 
                   optimize::Bool=true, run_enzyme::Bool=true, second_stage::Bool=true,
                   raw::Bool=false, debuginfo::Symbol=:default, dump_module::Bool=false)
    primal, adjoint, rt = fspec(func, types)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams()
    job    = Compiler.CompilerJob(target, primal, params)

    # Codegen the primal function and all its dependency in one module
    mod, primalf = Compiler.codegen(:llvm, job, optimize=false, #= validate=false =#)

    # Generate the wrapper, named `enzyme_entry`
    llvmf = wrapper!(mod, primalf, adjoint, rt)

    LLVM.strip_debuginfo!(mod)    
    # Run pipeline and Enzyme pass
    if optimize
        optimize!(mod, llvmf, run_enzyme=run_enzyme)
    end

    str = ccall(:jl_dump_function_ir, Ref{String},
                (LLVM.API.LLVMValueRef, Bool, Bool, Ptr{UInt8}),
                llvmf, !raw, dump_module, debuginfo)
    print(io, str)
end
enzyme_code_llvm(@nospecialize(func), @nospecialize(types); kwargs...) = enzyme_code_llvm(stdout, func, types; kwargs...)