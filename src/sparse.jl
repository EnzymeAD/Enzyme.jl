# Sparse derivatives, built on Enzyme's `__enzyme_todense` pointers and its
# automatic sparsity (see `Compiler.lower_sparsification!`).

"""
    PtrVector{T}(ptr::Ptr{T}, len::Int)

A vector of `len` elements stored at `ptr`. It is an `isbits` type, so a pointer
from [`todense`](@ref) that is wrapped in a `PtrVector` stays visible to Enzyme.
The memory behind `ptr` must be kept alive, e.g. with `GC.@preserve`, while the
vector is in use.
"""
struct PtrVector{T} <: AbstractVector{T}
    ptr::Ptr{T}
    len::Int
end
PtrVector(x::DenseVector{T}) where {T} = PtrVector{T}(pointer(x), length(x))

Base.size(v::PtrVector) = (v.len,)
Base.IndexStyle(::Type{<:PtrVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(v::PtrVector, i::Int)
    @boundscheck checkbounds(v, i)
    return unsafe_load(v.ptr, i)
end
Base.@propagate_inbounds function Base.setindex!(v::PtrVector{T}, val, i::Int) where {T}
    @boundscheck checkbounds(v, i)
    unsafe_store!(v.ptr, convert(T, val), i)
    return v
end

# The loader and storer of a `todense` pointer, called with the byte offset of
# the element. The wrappers fix the types Enzyme expects: the loader returns a
# `T` and the storer returns nothing.
struct TodenseLoad{F, T} end
struct TodenseStore{F, T} end
# Fixed arities, so that each specialization is compiled with one argument per
# value (a vararg method would receive a tuple).
for N in 0:8
    args = [Symbol(:arg, i) for i in 1:N]
    @eval @inline (::TodenseLoad{F, T})(offset::Int, $(args...)) where {F, T} =
        convert(T, F(offset, $(args...)))::T
    @eval @inline function (::TodenseStore{F, T})(val::T, offset::Int, $(args...)) where {F, T}
        F(val, offset, $(args...))
        return nothing
    end
end

# `_todense_fnptr(f, tt)` is a pointer to `f(::tt...)`, compiled on its own and
# linked into the module that calls it. Its id is produced by the generated
# function `todense_fnptr_id`, defined in src/late_generated.jl.
function todense_fnptr_id end

function todense_fnptr_generator(world::UInt, source, @nospecialize(self), @nospecialize(ft::Type), @nospecialize(tt::Type))
    @nospecialize
    slotnames = Core.svec(Symbol("#self#"), :f, :tt)
    stub = Core.GeneratedFunctionStub(identity, slotnames, Core.svec())
    TT = tt.parameters[1]
    mi = my_methodinstance(Forward, ft, TT, world)
    mi === nothing && return stub(world, source, :(throw(MethodError(f, $TT, $world))))
    target = Compiler.DefaultCompilerTarget()
    params = Compiler.PrimalCompilerParams(API.DEM_ForwardMode)
    config = GPUCompiler.CompilerConfig(target, params; kernel = false, entry_abi = :specfunc, optimize = false, cleanup = false, validate = false)
    job = GPUCompiler.CompilerJob(mi, config, world)
    # Ids of `GPUCompiler.deferred_codegen_jobs` share the key space with those of
    # `autodiff_deferred`, which are trampoline addresses.
    id = reinterpret(Int, hash(job, 0x7d0c_5f3a_e1b2_9c46))
    GPUCompiler.deferred_codegen_jobs[id] = job
    code = Any[Core.Compiler.ReturnNode(id)]
    ci = Compiler.create_fresh_codeinfo(todense_fnptr_id, source, world, slotnames, code)
    ci.edges = Any[mi]
    return ci
end

@generated function _todense_fnptr_call(::Val{id}) where {id}
    return quote
        Base.@_inline_meta
        ccall("extern deferred_codegen", llvmcall, Ptr{Cvoid}, (Int,), $id)
    end
end

@inline _todense_fnptr(f, tt) = _todense_fnptr_call(Val(todense_fnptr_id(f, tt)))

function check_todense_callback(name, F)
    if !Base.issingletontype(F)
        return :(throw(ArgumentError($("`$name` must be a function without captured variables, got $F"))))
    end
    return nothing
end

function check_todense_args(args)
    if length(args) > 8
        return :(throw(ArgumentError("`todense` supports at most 8 arguments for its loader and storer")))
    end
    for A in args
        if !isbitstype(A)
            return :(throw(ArgumentError($("The arguments of a `todense` loader or storer must be bits types, got $A"))))
        end
    end
    return nothing
end

"""
    todense(::Type{T}, load, store, args...)::Ptr{T}

Return a pointer to a virtual array of `T` whose elements are computed on
access: `unsafe_load(p, i)` evaluates `load(offset, args...)` and
`unsafe_store!(p, v, i)` evaluates `store(v, offset, args...)`, where
`offset = (i - 1) * sizeof(T)` is the byte offset of the element.

`load` and `store` must be functions without captured variables, and `args`
must be bits types (pass objects with `pointer_from_objref`).

The pointer only exists in code compiled by Enzyme, e.g. through
[`todense_call`](@ref), and is typically passed as the shadow of an argument
(wrapped in a [`PtrVector`](@ref)) to [`autodiff_deferred`](@ref). A `todense`
pointer that seeds a derivative with a structured matrix (e.g. a column of the
identity), together with one whose `store` feeds [`sparse_accumulate`](@ref),
lets Enzyme rewrite the loop that contains them to only visit the non-zero
entries of the derivative (see [`sparse_jacobian`](@ref)).

Enzyme finds those entries by solving the conditions in `load` and `store` for
the loop counters. It can solve (in)equalities between the offset and affine
functions of the counters, so compare offsets in bytes (e.g.
`offset == col * sizeof(T)`) rather than dividing them. Where it cannot solve
the conditions, the loop is kept dense, with a warning.
"""
@generated function todense(::Type{T}, load::L, store::S, args::Vararg{Any, N}) where {T, L, S, N}
    err = something(check_todense_callback("load", L), check_todense_callback("store", S), check_todense_args(args), Some(nothing))
    err === nothing || return err
    name = "extern __enzyme_todense." * string(hash(Tuple{T, args...}); base = 16)
    argtys = Expr(:tuple, :(Ptr{Cvoid}), :(Ptr{Cvoid}), args...)
    argvals = [:(args[$i]) for i in 1:N]
    loader = TodenseLoad{L.instance, T}()
    storer = TodenseStore{S.instance, T}()
    return quote
        Base.@_inline_meta
        lp = _todense_fnptr($loader, Tuple{Int, $(args...)})
        sp = _todense_fnptr($storer, Tuple{$T, Int, $(args...)})
        ccall($name, llvmcall, Ptr{$T}, $argtys, lp, sp, $(argvals...))
    end
end

"""
    sparse_accumulate(f, args...)

Call `f(args...)`, and mark it as the accumulation of a sparse derivative for
the automatic sparsity of [`todense`](@ref) loops. `f` must be a function
without captured variables, and `args` must be bits types.

A loop is rewritten to only visit the indices at which it reaches a
`sparse_accumulate` call. Guard the call so that it is only reached for the
non-zero entries, e.g. `iszero(v) || sparse_accumulate(push_entry!, i, j, v, acc)`.
"""
@generated function sparse_accumulate(f::F, args::Vararg{Any, N}) where {F, N}
    err = something(check_todense_callback("f", F), check_todense_args(args), Some(nothing))
    err === nothing || return err
    name = "extern __enzyme_sparse_accumulate_call." * string(hash(Tuple{args...}); base = 16)
    argtys = Expr(:tuple, :(Ptr{Cvoid}), args...)
    argvals = [:(args[$i]) for i in 1:N]
    return quote
        Base.@_inline_meta
        fp = _todense_fnptr(f, Tuple{$(args...)})
        ccall($name, llvmcall, Cvoid, $argtys, fp, $(argvals...))
        return nothing
    end
end

"""
    todense_call(f, args...)

Call `f(args...)` compiled by Enzyme's compiler, which lowers the
[`todense`](@ref) pointers in `f` and makes its loops sparse where possible.
`f` may call [`autodiff_deferred`](@ref). The result of `f` is discarded.
"""
function todense_call(f::F, args::Vararg{Any, N}) where {F, N}
    g = DiscardResult(f)
    thunk = @with Compiler.SPARSITY_COMPILATION => true begin
        Compiler.primal_thunk(g, Tuple{map(Core.Typeof, args)...})
    end
    thunk(Const(g), map(Const, args)...)
    return nothing
end

struct DiscardResult{F}
    f::F
end
@inline function (d::DiscardResult)(args::Vararg{Any, N}) where {N}
    d.f(args...)
    return nothing
end

# Loaders and storers of `sparse_jacobian`, for the column that starts at byte
# offset `coff`: the seed is that column of the identity, and the shadow of the
# output collects the non-zero entries of that column of the Jacobian.
struct JacobianSeed{T} end
(::JacobianSeed{T})(offset::Int, coff::Int) where {T} = T(offset == coff)
struct JacobianZero{T} end
(::JacobianZero{T})(offset::Int, col::Int, acc::Ptr{Cvoid}) where {T} = zero(T)
jacobian_nostore(val, offset::Int, coff::Int) = nothing
function jacobian_store(val::T, offset::Int, col::Int, acc::Ptr{Cvoid}) where {T}
    iszero(val) || sparse_accumulate(jacobian_accumulate!, offset, col, val, acc)
    return nothing
end
@noinline function jacobian_accumulate!(offset::Int, col::Int, val::T, acc::Ptr{Cvoid}) where {T}
    # A sparsified loop reaches the accumulation at every index that may be
    # non-zero, whatever the value there.
    iszero(val) && return nothing
    entries = unsafe_pointer_to_objref(acc)::Vector{Tuple{Int, Int, T}}
    push!(entries, (div(offset, sizeof(T)) + 1, col, val))
    return nothing
end

# The byte offset of the (0-based) element `i`. Enzyme computes the offsets of
# accesses the same way, and without the no-wrap flags a comparison of two
# offsets does not simplify to a comparison of the indices.
@inline offset_of(::Type{T}, i::Int) where {T} =
    Base.llvmcall("%3 = mul nuw nsw i64 %0, %1\nret i64 %3", Int, Tuple{Int, Int}, i, sizeof(T))

function sparse_jacobian_driver(f!::F, y::PtrVector{T}, x::PtrVector{T}, acc::Ptr{Cvoid}) where {F, T}
    n = length(x)
    m = length(y)
    # A 0-based column counter is the canonical induction variable that
    # automatic sparsity requires.
    for c in 0:(n - 1)
        dx = todense(T, JacobianSeed{T}(), jacobian_nostore, offset_of(T, c))
        dy = todense(T, JacobianZero{T}(), jacobian_store, c + 1, acc)
        autodiff_deferred(Forward, Const(f!), Const, Duplicated(y, PtrVector{T}(dy, m)), Duplicated(x, PtrVector{T}(dx, n)))
    end
    return nothing
end

"""
    sparse_jacobian(f!, y::DenseVector{T}, x::DenseVector{T}) where {T<:AbstractFloat}

Compute the Jacobian of `f!(y, x)` with respect to `x` as a `SparseMatrixCSC`,
using forward mode with [`todense`](@ref) seeds. When Enzyme can prove where
each column of the derivative is non-zero, it only computes those entries, so
that the cost scales with the number of non-zeros rather than with
`length(x) * length(y)`.

`f!` is called with `y` and `x` wrapped as [`PtrVector`](@ref)s, so it must
accept any `AbstractVector`. `y` is overwritten with the value of `f!`.

Enzyme can only skip the zero entries of loops that it can analyze: loops with
a single exit (bounds checks add exits, so use `@inbounds`) whose accesses are
affine in the loop counters. Other loops are evaluated densely, with a warning,
and the result is still exact.

```julia
julia> function f!(y, x)
           @inbounds for i in 1:length(x)-1
               y[i] = x[i] * x[i+1]
           end
           @inbounds y[end] = x[end]^2
           return nothing
       end;

julia> Enzyme.sparse_jacobian(f!, zeros(3), [1.0, 2.0, 3.0])
3×3 SparseArrays.SparseMatrixCSC{Float64, Int64} with 5 stored entries:
 2.0  1.0   ⋅
  ⋅   3.0  2.0
  ⋅    ⋅   6.0
```
"""
function sparse_jacobian(f!::F, y::DenseVector{T}, x::DenseVector{T}) where {F, T <: AbstractFloat}
    entries = Tuple{Int, Int, T}[]
    GC.@preserve x y entries begin
        todense_call(sparse_jacobian_driver, f!, PtrVector(y), PtrVector(x), pointer_from_objref(entries))
    end
    # The sparse loops only compute the entries of `y` that they visit.
    f!(y, x)
    I = Int[e[1] for e in entries]
    J = Int[e[2] for e in entries]
    V = T[e[3] for e in entries]
    # Each entry records a store to the shadow of `y`, so a later store to the
    # same element replaces an earlier one (e.g. the partial sums of a
    # reduction into `y[i]`).
    return SparseArrays.sparse(I, J, V, length(y), length(x), (a, b) -> b)
end
