module RecursiveMaps

using EnzymeCore: EnzymeCore, isvectortype, isscalartype
using ..Compiler: guaranteed_const_nongen, guaranteed_nonactive_nongen

### traits defining active leaf types for recursive_map
@inline EnzymeCore.isvectortype(::Type{T}) where {T} = isscalartype(T)
@inline function EnzymeCore.isvectortype(::Type{<:DenseArray{U}}) where {U}
    return isbitstype(U) && isscalartype(U)
end

@inline EnzymeCore.isscalartype(::Type) = false
@inline EnzymeCore.isscalartype(::Type{T}) where {T<:AbstractFloat} = isconcretetype(T)
@inline function EnzymeCore.isscalartype(::Type{Complex{T}}) where {T<:AbstractFloat}
    return isconcretetype(T)
end

### recursive_map: walk arbitrary objects and map a function over scalar and vector leaves
"""
    ys = recursive_map(
        [seen::Union{Nothing,IdDict},]
        f,
        ::Val{Nout}
        xs::NTuple{Nin,T},
        ::Val{copy_if_inactive}=Val(false),
        isinactivetype=guaranteed_const_nongen,
    )::T
    newys = recursive_map(
        [seen::Union{Nothing,IdDict},]
        f,
        ys::NTuple{Nout,T},
        xs::NTuple{Nin,T},
        ::Val{copy_if_inactive}=Val(false),
        isinactivetype=guaranteed_const_nongen,
    )::T

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recurse through `Nin` objects `xs = (x1::T, x2::T, ..., xNin::T)` of the same type, mapping the
function `f` over every differentiable value encountered and building `Nout` new objects
`(y1::T, y2::T, ..., yNout::T)` from the resulting values
`(y1_i, ..., yNout_i) = f(x1_i, ..., xNin_i)`.

The trait `EnzymeCore.isvectortype`(@ref) determines which values are considered
differentiable leaf nodes at which recursion terminates and `f` is invoked. See the
docstring for [`EnzymeCore.isvectortype`](@ref) and the related
[`EnzymeCore.isscalartype`](@ref) for more information.

A tuple of existing objects `ys = (y1::T, ..., yNout::T)` can be passed, in which case the
`ys` are updated "partially-in-place": any parts of the `ys` that are mutable or
non-differentiable are reused in the returned object tuple `newys`, while immutable
differentiable parts are handled out-of-place as if the `ys` were not passed (this can be
seen as a recursive generalization of the BangBang.jl idiom). If `T` itself is a mutable
type, the `ys` are modified in-place and returned, such that `newys === ys`.

The recursion and mapping operates on the structure of `T` as defined by struct fields and
plain array elements, not on the values provided through an iteration or array interface.
For example, given a structured matrix wrapper or sparse array type, this function recurses
into the struct type and the plain arrays held within, rather than operating on the array
that the type notionally represents.

# Arguments

* `seen::Union{IdDict,Nothing}` (optional): Dictionary for tracking object identity as
  needed to construct `y` such that its internal graph of object references is identical to
  that of the `xs`, including cycles (i.e., recursive substructures) and multiple paths to
  the same objects. If not provided, an `IdDict` will be allocated internally if required.

  If `nothing` is provided, object identity is not tracked. In this case, objects with
  multiple references are duplicated such that the `ys`s object reference graph becomes a
  tree, cycles lead to infinite recursion and stack overflow, and `copy_if_inactive == true`
  will likely cause errors. This is useful only in specific cases.

* `f`: Function mapping leaf nodes within the `xs` to the corresponding leaf nodes in the
  `ys`, that is, `(y1_i, ..., yNout_i) = f(x1_i::U, ..., xNin_i::U)::NTuple{Nout,U}`.
  The function `f` must be applicable to the type of every leaf node, and must return a
  tuple of values of the same type as its arguments.

  When an existing object tuple `ys` is passed and contains leaf nodes of a non-isbits
  non-scalar type `U`, `f` should also have a partially-in-place method
  `(newy1_i, ..., newyNout_i) === f(y1_i::U, ..., yNout_i::U, x1_i::U, ..., xNin_i::U)::NTuple{Nout,U}`
  that modifies and reuses any mutable parts of the `yj_i`; in particular, if `U` is a
  mutable type, this method should return `newyj_i === yj_i`. If a non-isbits type `U`
  should always be handled using the out-of-place signature, extend
  [`EnzymeCore.isscalartype`](@ref) such that `isscalartype(U) == true`.

  See [`EnzymeCore.isvectortype`](@ref) and [`EnzymeCore.isscalartype`](@ref) for more
  details about leaf types and scalar types.

* `::Val{Nout}` or `ys::NTuple{Nout,T}`: For out-of-place operation, pass `Val(Nout)` where
  `Nout` is the length of the tuple returned by `f`, that is, the length of the expected
  return value `ys` (this is required; `Nout` never inferred). For partially-in-place
  operation, pass the existing tuple `ys::NTuple{Nout,T}` containing the values to be
  modified.

* `xs::NTuple{N,T}`: Tuple of `N` objects of the same type `T` over which `f` is mapped.

  The first object `x1 = first(xs)` is the reference for graph structure and
  non-differentiable values when constructing the returned object. In particular:
  * When `ys` is not passed, the returned objects take any non-differentiable parts from
    `x1`. (When `ys` is passed, its non-differentiable parts are kept unchanged in the
    returned object, unless they are not initialized, in which case they are taken from
    `x1`.)
  * The graph of object references in `x1` is the one which is reproduced in the returned
    object. For each instance of multiple paths and cycles within `x1`, the same structure
    must be present in the other objects `x2, ..., xN`, otherwise the corresponding values
    in the `ys` would not be uniquely defined. However, `x2, ..., xN` may contain multiple
    paths or cycles that are not present in `x1`; these do not affect the structure of `ys`.
  * If any values within `x1` are not initialized (that is, struct fields are undefined or
    array elements are unassigned), they are left uninitialized in the returned object. If
    any such values are mutable and `ys` is passed, the corresponding value in `y` must not
    already be initialized, since initialized values cannot be nulled. Conversely, for every
    value in `x1` that is initialized, the corresponding values in `x2, ..., xN` must also
    be initialized, such that the corresponding values of the `ys` can be computed (however,
    values in `x2, ..., xN` can be initialized while the corresponding value in `x1` is not;
    such values are ignored.)

* `::Val{copy_if_inactive::Bool}` (optional): When a non-differentiable part of `x1` is
  included in the returned object, either because an object tuple `ys` is not passed or this
  part of the `ys` is not initialized, `copy_if_inactive` determines how: if
  `copy_if_inactive == false`, it is shared as `yj_i = x1_i`; if `copy_if_inactive == true`,
  it is deep-copied, more-or-less as `yj_i = deepcopy(x1_i)` (the difference is that when
  `x1` has several non-differentiable parts, object identity is tracked across the multiple
  deep-copies such that the object reference graph is reproduced also within the inactive
  parts.)

* `isinactivetype` (optional): Callable mapping types to `Bool` to determines whether the
  type should be treated according to `copy_if_inactive` (`true`) or recursed into (`false`).
"""
function recursive_map end

## type alias for unified handling of out-of-place and partially-in-place recursive_map
const YS{Nout,T} = Union{Val{Nout},NTuple{Nout,T}}
@inline hasvalues(::Val{Nout}) where {Nout} = (Nout::Int; false)
@inline hasvalues(::NTuple) = true

## main entry point: set default arguments, allocate IdDict if needed, exit early if possible
function recursive_map(
    f::F,
    ys::YS{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive::Val=Val(false),
    isinactivetype::L=guaranteed_const_nongen,
) where {F,Nout,Nin,T,L}
    newys = if isinactivetype(T)
        recursive_map_inactive(nothing, ys, xs, copy_if_inactive)
    elseif isvectortype(T) || isbitstype(T)
        recursive_map_inner(nothing, f, ys, xs, copy_if_inactive, isinactivetype)
    else
        recursive_map_inner(IdDict(), f, ys, xs, copy_if_inactive, isinactivetype)
    end
    return newys::NTuple{Nout,T}
end

## recursive methods
function recursive_map(
    seen::Union{Nothing,IdDict},
    f::F,
    ys::YS{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive::Val=Val(false),
    isinactivetype::L=guaranteed_const_nongen,
) where {F,Nout,Nin,T,L}
    # determine whether to continue recursion, copy/share, or retrieve from cache
    newys = if isinactivetype(T)
        recursive_map_inactive(seen, ys, xs, copy_if_inactive)
    elseif isbitstype(T)  # no object identity to to track in this branch
        recursive_map_inner(nothing, f, ys, xs, copy_if_inactive, isinactivetype)
    elseif hascache(seen, xs)
        getcached(seen, Val(Nout), xs)
    else
        recursive_map_inner(seen, f, ys, xs, copy_if_inactive, isinactivetype)
    end
    return newys::NTuple{Nout,T}
end

@inline function recursive_map_inner(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    # forward to appropriate handler for leaf vs. mutable vs. immutable type
    @assert !isabstracttype(T)
    @assert isconcretetype(T)
    newys = if isvectortype(T)
        recursive_map_leaf(seen, f, ys, xs)
    elseif ismutabletype(T)
        recursive_map_mutable(seen, f, ys, xs, copy_if_inactive, isinactivetype)
    else
        recursive_map_immutable(seen, f, ys, xs, copy_if_inactive, isinactivetype)
    end
    return newys::NTuple{Nout,T}
end

@inline function recursive_map_mutable(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    @assert ismutabletype(T)
    if !hasvalues(ys) && !(T <: DenseArray) && all(isbitstype, fieldtypes(T))
        # fast path for out-of-place handling when all fields are bitstypes, which rules
        # out undefined fields and circular references
        newys = recursive_map_new(seen, f, ys, xs, copy_if_inactive, isinactivetype)
        maybecache!(seen, newys, xs)
    else
        newys = if hasvalues(ys)
            ys
        else
            x1 = first(xs)
            ntuple(_ -> (@inline; _similar(x1)), Val(Nout))
        end
        maybecache!(seen, newys, xs)
        recursive_map_mutable_inner!(seen, f, newys, ys, xs, copy_if_inactive, isinactivetype)
    end
    return newys::NTuple{Nout,T}
end

@inline function recursive_map_mutable_inner!(
    seen,
    f::F,
    newys::NTuple{Nout,T},
    ys::YS{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive,
    isinactivetype::L,
) where {F,Nout,Nin,T<:DenseArray,L}
    if (Nout == 1) && isbitstype(eltype(T))
        newy = only(newys)
        if hasvalues(ys)
            y = only(ys)
            broadcast!(newy, y, xs...) do y_i, xs_i...
                only(recursive_map(nothing, f, (y_i,), xs_i, copy_if_inactive, isinactivetype))
            end
        else
            broadcast!(newy, xs...) do xs_i...
                only(recursive_map(nothing, f, Val(1), xs_i, copy_if_inactive, isinactivetype))
            end
        end
    else
        @inbounds for i in eachindex(newys..., xs...)
            recursive_map_item!(i, seen, f, newys, ys, xs, copy_if_inactive, isinactivetype)
        end
    end
    return nothing
end

@generated function recursive_map_mutable_inner!(
    seen,
    f::F,
    newys::NTuple{Nout,T},
    ys::YS{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive,
    isinactivetype::L,
) where {F,Nout,Nin,T,L}
    return quote
        @inline
        Base.Cartesian.@nexprs $(fieldcount(T)) i -> @inbounds begin
            recursive_map_item!(i, seen, f, newys, ys, xs, copy_if_inactive, isinactivetype)
        end
        return nothing
    end
end

@inline function recursive_map_immutable(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    @assert !ismutabletype(T)
    nf = fieldcount(T)
    if nf == 0  # nothing to do (also no known way to hit this branch)
        newys = recursive_map_inactive(seen, ys, xs, Val(false))
    else
        newys = if isinitialized(first(xs), nf)  # fast path when all fields are defined
            check_allinitialized(Base.tail(xs), nf)
            recursive_map_new(seen, f, ys, xs, copy_if_inactive, isinactivetype)
        else
            recursive_map_immutable_inner(seen, f, ys, xs, copy_if_inactive, isinactivetype)
        end
        # maybecache! _should_ be a no-op here; call it anyway for consistency
        maybecache!(seen, newys, xs)
    end
    return newys::NTuple{Nout,T}
end

@generated function recursive_map_immutable_inner(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    nf = fieldcount(T)
    return quote
        @inline
        x1, xtail = first(xs), Base.tail(xs)
        fields = Base.@ntuple $Nout _ -> Vector{Any}(undef, $(nf - 1))
        Base.Cartesian.@nexprs $(nf - 1) i -> begin  # unrolled loop over struct fields
            @inbounds if isinitialized(x1, i)
                check_allinitialized(xtail, i)
                newys_i = recursive_map_item(
                    i, seen, f, ys, xs, copy_if_inactive, isinactivetype
                )
                Base.Cartesian.@nexprs $Nout j -> (fields[j][i] = newys_i[j])
            else
                return new_structvs(T, fields, i - 1)
            end
        end
        @assert !isinitialized(x1, $nf)
        return new_structvs(T, fields, $(nf - 1))
    end
end

@generated function recursive_map_new(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    # direct construction of fully initialized non-cyclic structs
    nf = fieldcount(T)
    return quote
        @inline
        Base.Cartesian.@nexprs $nf i -> @inbounds begin
            newys_i = recursive_map_item(i, seen, f, ys, xs, copy_if_inactive, isinactivetype)
        end
        newys = Base.@ntuple $Nout j -> begin
            $(Expr(:splatnew, :T, :(Base.@ntuple $nf i -> newys_i[j])))
        end
        return newys::NTuple{Nout,T}
    end
end

Base.@propagate_inbounds function recursive_map_item!(
    i,
    seen,
    f::F,
    newys::NTuple{Nout,T},
    ys::YS{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive,
    isinactivetype::L,
) where {F,Nout,Nin,T,L}
    if isinitialized(first(xs), i)
        check_allinitialized(Base.tail(xs), i)
        newys_i = recursive_map_item(i, seen, f, ys, xs, copy_if_inactive, isinactivetype)
        setitems!(newys, i, newys_i)
    elseif hasvalues(ys)
        check_allinitialized(ys, i, false)
    end
    return nothing
end

Base.@propagate_inbounds function recursive_map_item(
    i, seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    # recurse into the xs and apply recursive_map to items with index i
    xs_i = getitems(xs, i)
    newys_i = if hasvalues(ys) && isinitialized(first(ys), i)
        check_allinitialized(Base.tail(ys), i)
        ys_i = getitems(ys, i)
        recursive_map_barrier!!(
            seen, f, copy_if_inactive, isinactivetype, Val(Nout), ys_i..., xs_i...
        )
    else
        recursive_map_barrier(seen, f, copy_if_inactive, isinactivetype, Val(Nout), xs_i...)
    end
    return newys_i
end

# function barriers such that abstractly typed items trigger minimal runtime dispatch
function recursive_map_barrier(
    seen, f::F, copy_if_inactive::Val, isinactivetype::L, ::Val{Nout}, xs_i::Vararg{ST,Nin}
) where {F,Nout,Nin,ST,L}
    return recursive_map(
        seen, f, Val(Nout), xs_i, copy_if_inactive, isinactivetype
    )::NTuple{Nout,ST}
end

function recursive_map_barrier!!(  # TODO: hit this when VectorSpace implemented
    seen, f::F, copy_if_inactive, isinactivetype::L, ::Val{Nout}, yxs_i::Vararg{ST,M}
) where {F,Nout,M,ST,L}
    ys_i, xs_i = yxs_i[1:(Nout::Int)], yxs_i[((Nout::Int)+1):end]
    return recursive_map(
        seen, f, ys_i, xs_i, copy_if_inactive, isinactivetype
    )::NTuple{Nout,ST}
end

# specialized methods to optimize the common cases Nout == 1 and Nout == 2
function recursive_map_barrier!!(
    seen, f::F, copy_if_inactive::Val, isinactivetype::L, ::Val{1}, y_i::ST, xs_i::Vararg{ST,Nin}
) where {F,Nin,ST,L}
    return recursive_map(
        seen, f, (y_i,), xs_i, copy_if_inactive, isinactivetype
    )::NTuple{1,ST}
end

function recursive_map_barrier!!(  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
    seen,
    f::F,
    copy_if_inactive::Val,
    isinactivetype::L,
    ::Val{2},
    y1_i::ST,
    y2_i::ST,
    xs_i::Vararg{ST,Nin}
) where {F,Nin,ST,L}
    return recursive_map(
        seen, f, (y1_i, y2_i), xs_i, copy_if_inactive, isinactivetype
    )::NTuple{2,ST}
end

## recursion base case handlers
@inline function recursive_map_leaf(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}
) where {F,Nout,Nin,T}
    # apply the mapped function to leaf values
    if !hasvalues(ys) || isbitstype(T) || isscalartype(T)
        newys = f(xs...)::NTuple{Nout,T}
    else  # !isbitstype(T)
        newys = f(ys..., xs...)::NTuple{Nout,T}
        if ismutabletype(T)
            @assert newys === ys
        end
    end
    maybecache!(seen, newys, xs)
    return newys::NTuple{Nout,T}
end

@inline function recursive_map_inactive(
    _, ys::NTuple{Nout,T}, xs::NTuple{Nin,T}, ::Val{copy_if_inactive}
) where {Nout,Nin,T,copy_if_inactive}
    return ys::NTuple{Nout,T}
end

@inline function recursive_map_inactive(
    seen, ::Val{Nout}, (x1,)::NTuple{Nin,T}, ::Val{copy_if_inactive}
) where {Nout,Nin,T,copy_if_inactive}
    @inline
    y = if copy_if_inactive && !isbitstype(T)
        if isnothing(seen)
            deepcopy(x1)
        else
            Base.deepcopy_internal(x1, seen)
        end
    else
        x1
    end
    return ntuple(_ -> (@inline; y), Val(Nout))::NTuple{Nout,T}
end

### recursive_map!: fully in-place wrapper around recursive_map
"""
    recursive_map!(
        [seen::Union{Nothing,IdDict},]
        f!!,
        ys::NTuple{Nout,T},
        xs::NTuple{Nin,T},
        [::Val{copy_if_inactive},]
    )::Nothing

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recurse through `Nin` objects `xs = (x1::T, x2::T, ..., xNin::T)` of the same type, mapping
the function `f!!` over every differentiable value encountered and updating
`(y1::T, y2::T, ..., yNout::T)`` in-place with the resulting values.

This is a simple wrapper that verifies that `T` is a type where all differentiable values
can be updated in-place, calls `recursive_map`, and verifies that the returned value is
indeed identically the same tuple `ys`. See [`recursive_map`](@ref) for details.
"""
function recursive_map!(
    f!!::F, ys::NTuple{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactives::Vararg{Val,M}
) where {F,Nout,Nin,T,M}
    @assert M <= 1
    check_nonactive(T)
    newys = recursive_map(f!!, ys, xs, copy_if_inactives...)
    @assert newys === ys
    return nothing
end

function recursive_map!(
    seen::Union{Nothing,IdDict},
    f!!::F,
    ys::NTuple{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactives::Vararg{Val,M},
) where {F,Nout,Nin,T,M}
    @assert M <= 1
    check_nonactive(T)
    newys = recursive_map(seen, f!!, ys, xs, copy_if_inactives...)
    @assert newys === ys
    return nothing
end

### recursive_map helpers
@generated function new_structvs(::Type{T}, fields::NTuple{N,Vector{Any}}, nfields_) where {T,N}
    return quote
        @inline
        return Base.@ntuple $N j -> begin
            ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields[j], nfields_)::T
        end
    end
end

@inline _similar(::T) where {T} = ccall(:jl_new_struct_uninit, Any, (Any,), T)::T
@inline _similar(x::T) where {T<:DenseArray} = similar(x)::T
Base.@propagate_inbounds isinitialized(x, i) = isdefined(x, i)
Base.@propagate_inbounds isinitialized(x::DenseArray, i) = isassigned(x, i)
Base.@propagate_inbounds getitem(x, i) = getfield(x, i)
Base.@propagate_inbounds getitem(x::DenseArray, i) = x[i]
Base.@propagate_inbounds setitem!(x, i, v) = setfield_force!(x, i, v)
Base.@propagate_inbounds setitem!(x::DenseArray, i, v) = (x[i] = v; nothing)

Base.@propagate_inbounds function setfield_force!(x::T, i, v) where {T}
    if Base.isconst(T, i)
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x, i - 1, v)
    else
        setfield!(x, i, v)
    end
    return nothing
end

Base.@propagate_inbounds function getitems((x1, xtail...)::Tuple{T,T,Vararg{T,N}}, i) where {T,N}
    return (getitem(x1, i), getitems(xtail, i)...)
end

Base.@propagate_inbounds getitems((x1,)::Tuple{T}, i) where {T} = (getitem(x1, i),)

Base.@propagate_inbounds function setitems!(  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
    (x1, xtail...)::Tuple{T,T,Vararg{T,N}}, i, (v1, vtail...)::Tuple{ST,ST,Vararg{ST,N}}
) where {T,ST,N}
    setitem!(x1, i, v1)
    setitems!(xtail, i, vtail)
    return nothing
end

Base.@propagate_inbounds function setitems!((x1,)::Tuple{T}, i, (v1,)::Tuple{ST}) where {T,ST}
    setitem!(x1, i, v1)
    return nothing
end

## cache (seen) helpers
@inline function iscachedtype(::Type{T}) where {T}
    # cache all mutable types and any non-isbits types that are also leaf types
    return ismutabletype(T) || ((!isbitstype(T)) && isvectortype(T))
end

@inline shouldcache(::IdDict, ::Type{T}) where {T} = iscachedtype(T)
@inline shouldcache(::Nothing, ::Type{T}) where {T} = false

@inline function maybecache!(seen, newys::NTuple{Nout,T}, (x1, xtail...)::NTuple{Nin,T}) where {Nout,Nin,T}
    if shouldcache(seen, T)
        seen[x1] = if (Nout == 1) && (Nin == 1)
            only(newys)
        else  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
            (newys..., xtail...)
        end
    end
    return nothing
end

@inline function hascache(seen, (x1,)::NTuple{Nin,T}) where {Nin,T}
    return shouldcache(seen, T) ? haskey(seen, x1) : false
end

@inline function getcached(seen::IdDict, ::Val{Nout}, (x1, xtail...)::NTuple{Nin,T}) where {Nout,Nin,T}
    newys = if (Nout == 1) && (Nin == 1)
        (seen[x1]::T,)
    else   # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
        cache = seen[x1]::NTuple{(Nout + Nin - 1),T}
        cachedtail = cache[(Nout+1):end]
        check_identical(cachedtail, xtail)  # check compatible layout
        cache[1:Nout]
    end
    return newys::NTuple{Nout,T}
end

## argument validation
Base.@propagate_inbounds function check_initialized(x, i, initialized=true)
    if isinitialized(x, i) != initialized
        throw_initialized()  # TODO: hit this when VectorSpace implemented
    end
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(  # TODO: hit this when VectorSpace implemented
    (x1, xtail...)::Tuple{T,T,Vararg{T,N}}, i, initialized=true
) where {T,N}
    check_initialized(x1, i, initialized)
    check_allinitialized(xtail, i, initialized)
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(
    (x1,)::Tuple{T}, i, initialized=true
) where {T}
    check_initialized(x1, i, initialized)
    return nothing
end

Base.@propagate_inbounds check_allinitialized(::Tuple{}, i, initialized=true) = nothing

@inline function check_identical(u, v)  # TODO: hit this when VectorSpace implemented
    if u !== v
        throw_identical()
    end
    return nothing
end

@inline function check_nonactive(::Type{T}) where {T}
    if !guaranteed_nonactive_nongen(T)
        throw_nonactive()
    end
    return nothing
end

# TODO: hit all of these via check_* when VectorSpace implemented
@noinline function throw_initialized()
    msg = "recursive_map(!) called on objects whose undefined fields/unassigned elements "
    msg *= "don't line up"
    throw(ArgumentError(msg))
end

@noinline function throw_identical()
    msg = "recursive_map(!) called on objects whose layout don't match"
    throw(ArgumentError(msg))
end

@noinline function throw_nonactive()
    msg = "recursive_map! called on objects containing immutable differentiable values"
    throw(ArgumentError(msg))
end

### EnzymeCore.make_zero(!) implementation
function EnzymeCore.make_zero(prev::T, copy_if_inactives::Vararg{Val,M}) where {T,M}
    @assert M <= 1
    new = if iszero(M) && !guaranteed_const_nongen(T) && isvectortype(T)  # fallback
        # guaranteed_const has precedence over isvectortype for consistency with recursive_map
        convert(T, zero(prev))  # convert because zero(prev)::T may fail when eltype(T) is abstract
    else
        only(recursive_map(_make_zero!!, Val(1), (prev,), copy_if_inactives...))
    end
    return new::T
end

function EnzymeCore.make_zero!(val::T, seens::Vararg{IdDict,M}) where {T,M}
    @assert M <= 1
    @assert !isscalartype(T)  # not appropriate for in-place handler
    if iszero(M) && !guaranteed_const_nongen(T) && isvectortype(T)  # fallback
        # isinactivetype has precedence over isvectortype for consistency with recursive_map
        fill!(val, false)
    else
        recursive_map!(seens..., _make_zero!!, (val,), (val,))
    end
    return nothing
end

function _make_zero!!(prev::T) where {T}
    @assert isvectortype(T)  # otherwise infinite loop
    return (EnzymeCore.make_zero(prev)::T,)
end

function _make_zero!!(val::T, _val::T) where {T}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    @assert isvectortype(T)   # otherwise infinite loop
    @assert val === _val
    EnzymeCore.make_zero!(val)
    return (val::T,)
end

# alternative entry point for passing custom IdDict
function EnzymeCore.make_zero(
    ::Type{T}, seen::IdDict, prev::T, copy_if_inactives::Vararg{Val,M}
) where {T,M}
    @assert M <= 1
    return only(recursive_map(seen, _make_zero!!, Val(1), (prev,), copy_if_inactives...))::T
end

end  # module RecursiveMaps
