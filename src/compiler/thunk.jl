abstract type EnzymeABI end
struct ActiveReturn <: EnzymeABI end


struct Thunk{f, RT, TT, Split}
    primal::Ptr{Cvoid}
    adjoint::Ptr{Cvoid}
end

# work around https://github.com/JuliaLang/julia/issues/37778
__normalize(::Type{Base.RefValue{T}}) where T = Ref{T}
__normalize(::Type{Base.RefArray{T}}) where T = Ref{T}
__normalize(T::DataType) = T

@generated function (thunk::Thunk{f, RT, TT})(args...) where {f, RT, TT}
    _args = (:(args[$i]) for i in 1:length(args))
    nargs = map(__normalize, args)
    if RT <: AbstractFloat
        quote
            ccall(thunk.adjoint, $RT, ($(nargs...),$RT), $(_args...), one($RT))
        end
    else 
        quote
            ccall(thunk.adjoint, $RT, ($(nargs...),), $(_args...))
        end
    end
end

function resolver(name, ctx)
    name = unsafe_string(name)
    ptr = try
        ## Step 0: Should have already resolved it iff it was in the
        ##         same module
        ## Step 1: See if it's something known to the execution enging
        # TODO: Do we need to do this?
        # address(jit[], name)

        ## Step 2: Search the program symbols
        #
        # SearchForAddressOfSymbol expects an unmangled 'C' symbol name.
        # Iff we are on Darwin, strip the leading '_' off.
        @static if Sys.isapple()
            if name[1] == '_'
                name = name[2:end]
            end
        end
        LLVM.API.LLVMSearchForAddressOfSymbol(name)
        ## Step 4: Lookup in libatomic
        # TODO: Do we need to do this?
    catch ex
        @error "Enzyme: Lookup failed" jl_name exception=(ex, Base.catch_backtrace())
        C_NULL
    end
    if ptr === C_NULL
        error("Enzyme: Symbol lookup failed. Aborting!")
    end

    return UInt64(reinterpret(UInt, ptr))
end

const cache = Dict{UInt, Dict{UInt, Any}}()

function thunk(f::F,tt::TT=Tuple{},::Val{Split}=Val(false)) where {F<:Core.Function, TT<:Type, Split}
    primal, adjoint = fspec(f, tt)

    # We need to use primal as the key, to lookup the right method
    # but need to mixin the hash of the adjoint to avoid cache collisions
    # This is counter-intuitive since we would expect the cache to be split
    # by the primal, but we want the generated code to be invalidated by
    # invalidations of the primal, which is managed by GPUCompiler.
    local_cache = get!(Dict{Int, Any}, cache, hash(adjoint, UInt64(Split)))

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(adjoint, Split)
    job    = Compiler.CompilerJob(target, primal, params)

    rt = Core.Compiler.return_type(primal.f, primal.tt)

    GPUCompiler.cached_compilation(local_cache, job, _thunk, _link)::Thunk{F,rt,tt,Split}
end

function _link(job, (mod, adjoint_name, primal_name))
    params = job.params
    adjoint = params.adjoint
    split = params.split
    rt = params.rt 

    # Now invoke the JIT
    orc = jit[]

    jitted_mod = compile!(orc, mod, @cfunction(resolver, UInt64, (Cstring, Ptr{Cvoid})))

    adjoint_addr = addressin(orc, jitted_mod, adjoint_name)
    adjoint_ptr  = pointer(adjoint_addr)
    if adjoint_ptr === C_NULL
        throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk, adjoint not found"))
    end
    if primal_name === nothing
        primal_ptr = C_NULL
    else
        primal_addr = addressin(orc, jitted_mod, primal_name)
        primal_ptr  = pointer(primal_addr)
        if primal_ptr === C_NULL
            throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk, primal not found"))
        end
    end

    return Thunk{typeof(adjoint.f), rt, adjoint.tt, split}(primal_ptr, adjoint_ptr)
end

function GPUCompiler.codegen(output::Symbol, job::CompilerJob{<:EnzymeTarget};
                 libraries::Bool=true, deferred_codegen::Bool=true, optimize::Bool=true,
                 strip::Bool=false, validate::Bool=true, only_entry::Bool=false)
    @assert output === :llvm
    split = job.params.split
    adjoint = job.params.adjoint

    mod, primalf = invoke(GPUCompiler.codegen, Tuple{Symbol, CompilerJob}, output, job; libraries, deferred_codegen, optimize, strip, validate, only_entry)

    # Run early pipeline
    optimize!(mod)

    # annotate
    annotate!(mod)

    # Generate the adjoint
    adjointf, augmented_primalf = enzyme!(job, mod, primalf, adjoint, split)

    linkage!(adjointf, LLVM.API.LLVMExternalLinkage)

    if augmented_primalf !== nothing
        linkage!(augmented_primalf, LLVM.API.LLVMExternalLinkage)
    end

    if augmented_primalf === nothing
        return mod, adjointf
    else
        return mod, (adjointf, augmented_primalf)
    end
end

# actual compilation
function _thunk(job)
    params = job.params

    mod, fns = codegen(:llvm, job, optimize=false)

    if fns isa Tuple
        adjointf, augmented_primalf = fns
    else
        adjointf = fns
        augmented_primalf = nothing
    end

    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end

    # Run post optimization pipeline
    post_optimze!(mod)

    return (mod, adjoint_name, primal_name)
end

