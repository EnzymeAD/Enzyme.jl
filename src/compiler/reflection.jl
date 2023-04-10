function get_job(@nospecialize(func), @nospecialize(A), @nospecialize(types);
        run_enzyme::Bool=true, mode::API.CDerivativeMode=API.DEM_ReverseModeCombined, dupClosure::Bool=false, argwrap::Bool=true, width::Int64=1, modifiedBetween=nothing, returnPrimal::Bool=false, augmentedInit=false, world=nothing, kwargs...)

    tt    = Tuple{map(eltype, types.parameters)...}
    if world === nothing
        world = GPUCompiler.codegen_world_age(Core.Typeof(func), tt)
    end
    
    primal = fspec(Core.Typeof(func), types, world)

    rt = Core.Compiler.return_type(func, tt, world)
    rt = A{rt}
    target = Compiler.EnzymeTarget()
    if modifiedBetween === nothing
        defaultMod = mode != API.DEM_ReverseModeCombined && mode != API.DEM_ForwardMode
        modifiedBetween = (defaultMod, (defaultMod for _ in types.parameters)...)
    end
    params = Compiler.EnzymeCompilerParams(Tuple{(dupClosure ? Duplicated : Const){Core.Typeof(func)}, types.parameters...}, mode, width, remove_innerty(rt), run_enzyme, argwrap, modifiedBetween, returnPrimal, augmentedInit, Compiler.UnknownTapeType)
    return Compiler.CompilerJob(primal, CompilerConfig(target, params; kernel=false), world)
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

# For VERSION >= v"1.9.0-DEV.516"
struct jl_llvmf_dump
    TSM::LLVM.API.LLVMOrcThreadSafeModuleRef
    F::LLVM.API.LLVMValueRef
end

function enzyme_code_llvm(io::IO, @nospecialize(func), @nospecialize(A), @nospecialize(types);
                          optimize::Bool=true, run_enzyme::Bool=true, second_stage::Bool=true,
                          raw::Bool=false, debuginfo::Symbol=:default, dump_module::Bool=false)
    JuliaContext() do ctx
        entry_fn, ir = reflect(func, A, types; optimize, run_enzyme, second_stage, ctx)
        @static if VERSION >= v"1.9.0-DEV.516"
            ts_mod = ThreadSafeModule(ir; ctx)
            if VERSION >= v"1.9.0-DEV.672"
                GC.@preserve ts_mod entry_fn begin
                    value = Ref(jl_llvmf_dump(ts_mod.ref, entry_fn.ref))
                    str = ccall(:jl_dump_function_ir, Ref{String},
                          (Ptr{jl_llvmf_dump}, Bool, Bool, Ptr{UInt8}),
                          value, !raw, dump_module, debuginfo)
                end
            else
                GC.@preserve ts_mod entry_fn begin
                    # N.B. jl_dump_function_ir will `Libc.free` the passed-in pointer
                    value_ptr = reinterpret(Ptr{jl_llvmf_dump},
                                            Libc.malloc(sizeof(jl_llvmf_dump)))
                    unsafe_store!(value_ptr, jl_llvmf_dump(ts_mod.ref, entry_fn.ref))
                    str = ccall(:jl_dump_function_ir, Ref{String},
                          (Ptr{jl_llvmf_dump}, Bool, Bool, Ptr{UInt8}),
                          value_ptr, !raw, dump_module, debuginfo)
                end
            end
        else
            str = ccall(:jl_dump_function_ir, Ref{String},
                  (LLVM.API.LLVMValueRef, Bool, Bool, Ptr{UInt8}),
                  entry_fn, !raw, dump_module, debuginfo)
        end
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
