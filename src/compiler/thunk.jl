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
    primal, adjoint, rt = fspec(f, tt)

    # We need to use primal as the key, to lookup the right method
    # but need to mixin the hash of the adjoint to avoid cache collisions
    # This is counter-intuitive since we would expect the cache to be split
    # by the primal, but we want the generated code to be invalidated by
    # invalidations of the primal, which is managed by GPUCompiler.
    local_cache = get!(Dict{Int, Any}, cache, hash(adjoint, UInt64(Split)))

    GPUCompiler.cached_compilation(local_cache, _thunk, _link, primal, adjoint=adjoint, rt=rt, split=Split)::Thunk{F,rt,tt,Split}
end

function _link(@nospecialize(primal::FunctionSpec), (mod, adjoint_name, primal_name); adjoint, rt, split)
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

# actual compilation
function _thunk(@nospecialize(primal::FunctionSpec); adjoint, rt, split)
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams()
    job    = Compiler.CompilerJob(target, primal, params)

    # Codegen the primal function and all its dependency in one module
    mod, primalf = Compiler.codegen(:llvm, job, optimize=false, #= validate=false =#)

    # Run Julia pipeline
    optimize!(mod)

    # annotate
    annotate!(mod)

    # Generate the adjoint
    adjointf, augmented_primalf = enzyme!(mod, primalf, adjoint, rt, split)

    linkage!(adjointf, LLVM.API.LLVMExternalLinkage)
    adjoint_name = name(adjointf)

    if augmented_primalf !== nothing
        linkage!(augmented_primalf, LLVM.API.LLVMExternalLinkage)
        primal_name = name(augmented_primalf)
    else
        primal_name = nothing
    end

    # Run post optimization pipeline
    post_optimze!(mod)

    return (mod, adjoint_name, primal_name)
end


