struct Thunk{f, RT, TT}
    ptr::Ptr{Cvoid}
end

# work around https://github.com/JuliaLang/julia/issues/37778
__normalize(::Type{Base.RefValue{T}}) where T = Ref{T}
__normalize(::Type{Base.RefArray{T}}) where T = Ref{T}
__normalize(T::DataType) = T

@generated function (thunk::Thunk{f, RT, TT})(args...) where {f, RT, TT}
    _args = (:(args[$i]) for i in 1:length(args))
    nargs = map(__normalize, args)
    quote
        ccall(thunk.ptr, $RT, ($(nargs...),), $(_args...))
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

function thunk(f::F,tt::TT=Tuple{}) where {F<:Core.Function, TT<:Type}
    primal, adjoint, rt = fspec(f, tt)

    # We need to use primal as the key, to lookup the right method
    # but need to mixin the hash of the adjoint to avoid cache collisions
    # This is counter-intuitive since we would expect the cache to be split
    # by the primal, but we want the generated code to be invalidated by
    # invalidations of the primal, which is managed by GPUCompiler.
    local_cache = get!(Dict{Int, Any}, cache, hash(adjoint))

    GPUCompiler.cached_compilation(local_cache, _thunk, _link, primal, adjoint=adjoint, rt=rt)::Thunk{F,rt,tt}
end

function _link(@nospecialize(primal::FunctionSpec), thunk; kwargs...)
    return thunk
end

# actual compilation
function _thunk(@nospecialize(primal::FunctionSpec); adjoint, rt)
    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams()
    job    = Compiler.CompilerJob(target, primal, params)

    # Codegen the primal function and all its dependency in one module
    mod, primalf = Compiler.codegen(:llvm, job, optimize=false, #= validate=false =#)

    # Generate the wrapper, named `enzyme_entry`
    orc = jit[]
    name = mangle(orc, "enzyme_entry")
    llvmf = wrapper!(mod, primalf, adjoint, rt, name)

    LLVM.strip_debuginfo!(mod)    
    # Run pipeline and Enzyme pass
    optimize!(mod, llvmf)

    # Now invoke the JIT
    jitted_mod = compile!(orc, mod, @cfunction(resolver, UInt64, (Cstring, Ptr{Cvoid})))
    addr = addressin(orc, jitted_mod, name)
    ptr  = pointer(addr)
    if ptr === C_NULL
        throw(GPUCompiler.InternalCompilerError(job, "Failed to compile Enzyme thunk"))
    end

    return Thunk{typeof(adjoint.f), rt, adjoint.tt}(pointer(addr))
end


