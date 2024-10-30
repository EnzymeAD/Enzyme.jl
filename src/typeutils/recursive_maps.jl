module RecursiveMaps

using EnzymeCore: EnzymeCore, isvectortype, isscalartype
using ..Compiler: Compiler, guaranteed_const, guaranteed_const_nongen, guaranteed_nonactive,
    guaranteed_nonactive_nongen

### IsInactive: helper for creating consistent inactive/nonactive type checkers
"""
    isinactivetype = IsInactive{runtime::Bool}(extra=(T -> false))
    isinactivetype = IsInactive(isinactivetype::IsInactive, extra)

!!! warning
    Internal type, documented for developer convenience but not covered by semver API
    stability guarantees

Create a callable `isinactivetype` such that `isinactivetype(T) == true` if the type `T` is
non-differentiable, that is, if differentiable values can never be reached from any instance
of the type (that is, the activity state of `T` is `AnyState`).

The callable takes an optional argument `Val(nonactive::Bool)`, such that the full signature
is

```julia
isinactivetype(::Type{T}, ::Val{nonactive}=Val(false))::Bool
```

Setting `nonactive == true` selects for _nonactive_ types, which is a superset of inactive
types that also includes types `T` where every differentiable value can be mutated without
creating a new instance of `T` (that is, the activity state of `T` is either `AnyState` or
`DupState`).

The optional argument `extra` takes a function defining additional types that should be
treated as inactive regardless of their nominal activity state; that is,

```julia
IsInactive{runtime}(extra)(T, args...) == IsInactive{runtime}()(T, args...) || extra(T)
```

The constructor `IsInactive(isinactivetype::IsInactive{runtime}, extra)` can be used to
extend an existing instance `isinactivetype::IsInactive` with an additional `extra`
function, and is more or less equivalent to
`IsInactive{runtime}(T -> isinactivetype.extra(T) || extra(T))`.

The type parameter `runtime` specifies whether the activity state of a type is queried at
runtime every time the callable is invoked (`true`), or if compile-time queries from earlier
calls can be reused (`false`). Runtime querying is necessary to pick up recently added
methods to `EnzymeRules.inactive_type`, but may incur a significant performance penalty and
is usually not needed unless `EnzymeRules.inactive_type` is extended interactively for types
that have previously been passed to an instance of `IsInactive{false}`.
"""
struct IsInactive{runtime,F}
    extra::F
    function IsInactive{runtime}(
        extra::F=(@nospecialize(T) -> (@inline; false))
    ) where {runtime,F}
        return new{runtime::Bool,F}(extra)
    end
end

function IsInactive(isinactivetype::IsInactive{runtime}, extra::F) where {runtime,F}
    combinedextra(::Type{T}) where {T} = (isinactivetype.extra(T) || extra(T))
    return IsInactive{runtime}(combinedextra)
end

@inline function (f::IsInactive{runtime,F})(
    ::Type{T}, ::Val{nonactive}=Val(false)
) where {runtime,F,T,nonactive}
    if runtime
        # evaluate f.extra first, as guaranteed_*_nongen may incur runtime dispatch
        if nonactive
            return f.extra(T) || guaranteed_nonactive_nongen(T, nothing)
        else
            return f.extra(T) || guaranteed_const_nongen(T, nothing)
        end
    else
        # evaluate guaranteed_* first, as these are always known at compile time
        if nonactive
            return guaranteed_nonactive(T) || f.extra(T)
        else
            return guaranteed_const(T) || f.extra(T)
        end
    end
end

### traits defining active leaf types for recursive_map
@inline isdensearraytype(::Type{<:DenseArray}) = true
@inline isdensearraytype(::Type) = false

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
        isinactivetype=IsInactive{false}(),
    )::T
    newys = recursive_map(
        [seen::Union{Nothing,IdDict},]
        f,
        ys::NTuple{Nout,T},
        xs::NTuple{Nin,T},
        ::Val{copy_if_inactive}=Val(false),
        isinactivetype=IsInactive{false}(),
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

* `isinactivetype` (optional): Callable determining which types are considered inactive and
  thus treated according to `copy_if_inactive`. The [`IsInactive`](@ref) type is a
  convenient helper for obtaining a callable with relevant semantics, but any callable that
  maps types to `true` or `false` can be used.
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
    copy_if_inactive::Val=Val{false},
    isinactivetype::L=IsInactive{false}(),
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
    copy_if_inactive::Val=Val{false},
    isinactivetype::L=IsInactive{false}(),
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

@generated function recursive_map_mutable(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    @assert ismutabletype(T)
    iteration_i = quote
        @inbounds if isinitialized(x1, i)
            check_allinitialized(xtail, i)
            newys_i = recursive_map_index(i, seen, f, ys, xs, copy_if_inactive, isinactivetype)
            setitems!(newys, i, newys_i)
        elseif hasvalues(ys)
            check_allinitialized(ys, i, false)
        end
    end
    return quote
        @inline
        if !hasvalues(ys) && !isdensearraytype(T) && all(isbitstype, fieldtypes(T))
            # fast path for out-of-place handling when all fields are bitstypes, which rules
            # out undefined fields and circular references
            newys = recursive_map_new(seen, f, ys, xs, copy_if_inactive, isinactivetype)
            maybecache!(seen, newys, xs)
        else
            x1, xtail = first(xs), Base.tail(xs)
            newys = if hasvalues(ys)
                ys
            else
                Base.@ntuple $Nout _ -> _similar(x1)
            end
            maybecache!(seen, newys, xs)
            if isdensearraytype(T)
                if (Nout == 1) && isbitstype(eltype(T))
                    recursive_map_broadcast!(
                        f, newys, ys, xs, copy_if_inactive, isinactivetype
                    )
                else
                    for i in eachindex(newys..., xs...)
                        $iteration_i
                    end
                end
            else  # unrolled loop over struct fields
                Base.Cartesian.@nexprs $(fieldcount(T)) i -> $iteration_i
            end
        end
        return newys::NTuple{Nout,T}
    end
end

@generated function recursive_map_immutable(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    @assert !ismutabletype(T)
    nf = fieldcount(T)
    return quote
        @inline
        if $nf == 0  # nothing to do (also no known way to hit this branch)
            newys = recursive_map_inactive(nothing, ys, xs, Val(false))
        else
            x1, xtail = first(xs), Base.tail(xs)
            if isinitialized(x1, $nf)  # fast path when all fields are defined
                check_allinitialized(xtail, $nf)
                newys = recursive_map_new(seen, f, ys, xs, copy_if_inactive, isinactivetype)
            else
                Base.Cartesian.@nexprs $Nout j -> (fields_j = Vector{Any}(undef, $(nf - 1)))
                Base.Cartesian.@nexprs $(nf - 1) i -> begin  # unrolled loop over struct fields
                    @inbounds if isinitialized(x1, i)
                        check_allinitialized(xtail, i)
                        newys_i = recursive_map_index(
                            i, seen, f, ys, xs, copy_if_inactive, isinactivetype
                        )
                        Base.Cartesian.@nexprs $Nout j -> (fields_j[i] = newys_i[j])
                    else
                        ndef = i - 1  # rest of tail must be undefined values
                        @goto done    # break out of unrolled loop
                    end
                end
                ndef = $(nf - 1)      # loop didn't break, only last field is undefined
                @label done
                newys = Base.@ntuple $Nout j -> begin
                    ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields_j, ndef)::T
                end
            end
            # maybecache! _should_ be a no-op here; call it anyway for consistency
            maybecache!(seen, newys, xs)
        end
        return newys::NTuple{Nout,T}
    end
end

@generated function recursive_map_new(
    seen, f::F, ys::YS{Nout,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nout,Nin,T,L}
    # direct construction of fully initialized non-cyclic structs
    nf = fieldcount(T)
    return quote
        @inline
        Base.Cartesian.@nexprs $nf i -> begin
            newys_i = @inbounds recursive_map_index(
                i, seen, f, ys, xs, copy_if_inactive, isinactivetype
            )
        end
        newys = Base.@ntuple $Nout j -> begin
            $(Expr(:splatnew, :T, :(Base.@ntuple $nf i -> newys_i[j])))
        end
        return newys::NTuple{Nout,T}
    end
end

@inline function recursive_map_broadcast!(
    f::F, newys::NTuple{1,T}, ys::YS{1,T}, xs::NTuple{Nin,T}, copy_if_inactive, isinactivetype::L
) where {F,Nin,T,L}
    # broadcast recursive_map over array-like inputs with isbits elements
    @assert isdensearraytype(T)
    @assert isbitstype(eltype(T))
    newy = first(newys)
    if hasvalues(ys)
        @assert newys === ys
        broadcast!(
            (newy_i, xs_i...) -> first(recursive_map_barrier!!(
                nothing, f, copy_if_inactive, isinactivetype, Val(1), newy_i, xs_i...
            )),
            newy,
            newy,
            xs...,
        )
    else
        broadcast!(
            (xs_i...,) -> first(recursive_map_barrier(
                nothing, f, copy_if_inactive, isinactivetype, Val(1), xs_i...
            )),
            newy,
            xs...,
        )
    end
    return nothing
end

Base.@propagate_inbounds function recursive_map_index(
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
    seen, f::F, copy_if_inactive::Val, isinactivetype::L, ::Val{1}, yi::ST, xs_i::Vararg{ST,Nin}
) where {F,Nin,ST,L}
    return recursive_map(
        seen, f, (yi,), xs_i, copy_if_inactive, isinactivetype
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
    newys = if !hasvalues(ys) || isbitstype(T) || isscalartype(T)
        f(xs...)::NTuple{Nout,T}
    else  # !isbitstype(T)
        newys_ = f(ys..., xs...)::NTuple{Nout,T}
        if ismutabletype(T)
            @assert newys_ === ys
        end
        newys_
    end
    maybecache!(seen, newys, xs)
    return newys::NTuple{Nout,T}
end

@inline function recursive_map_inactive(
    _, ys::NTuple{Nout,T}, xs::NTuple{Nin,T}, ::Val{copy_if_inactive}
) where {Nout,Nin,T,copy_if_inactive}
    return ys::NTuple{Nout,T}
end

@generated function recursive_map_inactive(
    seen, ::Val{Nout}, xs::NTuple{Nin,T}, ::Val{copy_if_inactive}
) where {Nout,Nin,T,copy_if_inactive}
    return quote
        @inline
        y = if copy_if_inactive && !isbitstype(T)
            Base.deepcopy_internal(first(xs), isnothing(seen) ? IdDict() : seen)
        else
            first(xs)
        end
        return (Base.@ntuple $Nout _ -> y)::NTuple{Nout,T}
    end
end

### recursive_map!: fully in-place wrapper around recursive_map
"""
    recursive_map!(
        [seen::IdDict,]
        f!!,
        ys::NTuple{Nout,T},
        xs::NTuple{Nin,T},
        ::Val{copy_if_inactive}=Val(false),
        isinactivetype::IsInactive=IsInactive{false}(),
    )::Nothing

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recurse through `Nin` objects `xs = (x1::T, x2::T, ..., xNin::T)` of the same type, mapping
the function `f!!` over every differentiable value encountered and updating
`(y1::T, y2::T, ..., yNout::T)`` in-place with the resulting values.

This is a simple wrapper that verifies that `T` is a type where all differentiable values
can be updated in-place (this uses the `nonactive == true` mode of `isinactivetype`, see
[`IsInactive`](@ref) for details), calls `recursive_map`, and verifies that the returned
value is indeed identically the same tuple `ys`. See [`recursive_map`](@ref) for details.

Note that this wrapper only supports instances of [`IsInactive`](@ref) for the
`isinactivetype` argument, as this is the only way we can insure consistency between the
upfront compatibility check and actual behavior. If this is not appropriate, use
`recursive_map` directly.
"""
function recursive_map!(
    f!!::F,
    ys::NTuple{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive::Val=Val(false),
    isinactivetype::IsInactive=IsInactive{false}(),
) where {F,Nout,Nin,T}
    check_nonactive(T, isinactivetype)
    newys = recursive_map(f!!, ys, xs, copy_if_inactive, isinactivetype)
    @assert newys === ys
    return nothing
end

function recursive_map!(
    seen::IdDict,
    f!!::F,
    ys::NTuple{Nout,T},
    xs::NTuple{Nin,T},
    copy_if_inactive::Val=Val(false),
    isinactivetype::IsInactive=IsInactive{false}(),
) where {F,Nout,Nin,T}
    check_nonactive(T, isinactivetype)
    newys = recursive_map(seen, f!!, ys, xs, copy_if_inactive, isinactivetype)
    @assert newys === ys
    return nothing
end

### recursive_map helpers
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

Base.@propagate_inbounds function getitems(xs::Tuple{T,T,Vararg{T,N}}, i) where {T,N}
    return (getitem(first(xs), i), getitems(Base.tail(xs), i)...)
end

Base.@propagate_inbounds getitems(xs::Tuple{T}, i) where {T} = (getitem(only(xs), i),)

Base.@propagate_inbounds function setitems!(  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
    xs::Tuple{T,T,Vararg{T,N}}, i, vs::Tuple{ST,ST,Vararg{ST,N}}
) where {T,ST,N}
    setitem!(first(xs), i, first(vs))
    setitems!(Base.tail(xs), i, Base.tail(vs))
    return nothing
end

Base.@propagate_inbounds function setitems!(xs::Tuple{T}, i, vs::Tuple{ST}) where {T,ST}
    setitem!(only(xs), i, only(vs))
    return nothing
end

## cache (seen) helpers
@inline function iscachedtype(::Type{T}) where {T}
    # cache all mutable types and any non-isbits types that are also leaf types
    return ismutabletype(T) || ((!isbitstype(T)) && isvectortype(T))
end

@inline shouldcache(::IdDict, ::Type{T}) where {T} = iscachedtype(T)
@inline shouldcache(::Nothing, ::Type{T}) where {T} = false

@inline function maybecache!(seen, newys::NTuple{Nout,T}, xs::NTuple{Nin,T}) where {Nout,Nin,T}
    if shouldcache(seen, T)
        if (Nout == 1) && (Nin == 1)
            seen[only(xs)] = only(newys)
        else  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
            seen[first(xs)] = (newys..., Base.tail(xs)...)
        end
    end
    return nothing
end

@inline function hascache(seen, xs::NTuple{Nin,T}) where {Nin,T}
    return shouldcache(seen, T) ? haskey(seen, first(xs)) : false
end

@inline function getcached(seen::IdDict, ::Val{Nout}, xs::NTuple{Nin,T}) where {Nout,Nin,T}
    newys = if (Nout == 1) && (Nin == 1)
        (seen[only(xs)]::T,)
    else   # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
        cache = seen[first(xs)]::NTuple{(Nout + Nin - 1),T}
        cachedtail = cache[(Nout+1):end]
        check_identical(cachedtail, Base.tail(xs))  # check compatible layout
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
    xs::Tuple{T,T,Vararg{T,N}}, i, initialized=true
) where {T,N}
    check_initialized(first(xs), i, initialized)
    check_allinitialized(Base.tail(xs), i, initialized)
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(
    xs::Tuple{T}, i, initialized=true
) where {T}
    check_initialized(only(xs), i, initialized)
    return nothing
end

Base.@propagate_inbounds check_allinitialized(::Tuple{}, i, initialized=true) = nothing

@inline function check_identical(u, v)  # TODO: hit this when VectorSpace implemented
    if u !== v
        throw_identical()
    end
    return nothing
end

@inline function check_nonactive(::Type{T}, isinactivetype::IsInactive) where {T}
    if !isinactivetype(T, Val(true)) #=nonactive=#
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
function EnzymeCore.make_zero(prev::T, args::Vararg{Any,M}) where {T,M}
    new = if iszero(M) && !IsInactive{false}()(T) && isvectortype(T)  # fallback
        # IsInactive has precedence over isvectortype for consistency with recursive handler
        convert(T, zero(prev))  # convert because zero(prev)::T may fail when eltype(T) is abstract
    else
        _make_zero_inner(prev, args...)
    end
    return new::T
end

function EnzymeCore.make_zero!(val::T, args::Vararg{Any,M}) where {T,M}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    if iszero(M) && !IsInactive{false}()(T) && isvectortype(T)  # fallback
        # IsInactive has precedence over isvectortype for consistency with recursive handler
        fill!(val, false)
    else
        _make_zero_inner!(val, args...)
    end
    return nothing
end

@inline function _make_zero_inner(
    prev::T, copy_if_inactive::Val=Val(false), ::Val{runtime_inactive}=Val(false)
) where {T,runtime_inactive}
    isinactivetype = IsInactive{runtime_inactive}()
    news = recursive_map(_make_zero!!, Val(1), (prev,), copy_if_inactive, isinactivetype)
    return only(news)::T
end

@inline function _make_zero_inner!(
    val::T, ::Val{runtime_inactive}=Val(false)
) where {T,runtime_inactive}
    isinactivetype = IsInactive{runtime_inactive}()
    recursive_map!(_make_zero!!, (val,), (val,), Val(false), isinactivetype)
    return nothing
end

@inline function _make_zero_inner!(
    val::T, seen::IdDict, ::Val{runtime_inactive}=Val(false)
) where {T,runtime_inactive}
    isinactivetype = IsInactive{runtime_inactive}()
    recursive_map!(seen, _make_zero!!, (val,), (val,), Val(false), isinactivetype)
    return nothing
end

function _make_zero!!(prev::T) where {T}
    @assert isvectortype(T)  # otherwise infinite loop
    return (EnzymeCore.make_zero(prev),)::Tuple{T}
end

function _make_zero!!(val::T, _val::T) where {T}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    @assert isvectortype(T)   # otherwise infinite loop
    @assert val === _val
    EnzymeCore.make_zero!(val)
    return (val,)::Tuple{T}
end

# alternative entry point for passing custom IdDict
function EnzymeCore.make_zero(
    ::Type{T},
    seen::IdDict,
    prev::T,
    copy_if_inactive::Val=Val(false),
    ::Val{runtime_inactive}=Val(false),
) where {T,runtime_inactive}
    isinactivetype = IsInactive{runtime_inactive}()
    news = recursive_map(seen, _make_zero!!, Val(1), (prev,), copy_if_inactive, isinactivetype)
    return only(news)::T
end

end  # module RecursiveMaps
