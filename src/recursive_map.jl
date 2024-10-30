module RecursiveMap

using EnzymeCore: EnzymeCore, isvectortype, isscalartype
using ..Compiler: ActiveState, active_reg_inner, guaranteed_const_nongen, splatnew

"""
    y = recursive_map(
        [seen::IdDict,]
        f,
        xs::NTuple{N,T},
        ::Val{copy_if_inactive}=Val(false),
    )::T where {T,N,copy_if_inactive}
    newy = recursive_map(
        [seen::IdDict,]
        f,
        (; y, xs)::@NamedTuple{y::T,xs::NTuple{N,T}},
        ::Val{copy_if_inactive}=Val(false),
    )::T where {T,N,copy_if_inactive}

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recurse through `N` objects `xs = (x1::T, x2::T, ..., xN::T)` of the same type, mapping the
function `f` over every differentiable value encountered and building a new object `y::T`
from the resulting values `yi = f(x1i, ..., xNi)`.

The trait `EnzymeCore.isvectortype`(@ref) determines which values are considered
differentiable leaf nodes at which recursion terminates and `f` is invoked. See the
docstring for that and the related [`EnzymeCore.isscalartype`](@ref) for more information.

An existing object `y::T` can be passed by replacing the tuple `xs` with a NamedTuple
`(; y, xs)`, in which case `y` is updated "partially-in-place": any parts of `y` that are
mutable or non-differentiable are reused in the returned object `newy`, while immutable
differentiable parts are handled out-of-place as if `y` were not passed (this can be seen as
a recursive generalization of the BangBang.jl idiom). If `y` itself is mutable, it is
modified in-place and returned, such that `newy === y`.

The recursion and mapping operates on the structure of `T` as defined by struct fields and
plain array elements, not on the values provided through an iteration or array interface.
For example, given a structured matrix wrapper or sparse array type, this function recurses
into the struct type and the plain arrays held within, rather than operating on the array
that the type notionally represents.

# Arguments

* `seen::IdDict` (optional): Dictionary for tracking object identity as needed to construct
  `y` such that its internal graph of object references is identical to that of the `xs`,
  including cycles (i.e., recursive substructures) and multiple paths to the same objects.
  If not provided, an `IdDict` will be allocated internally if required.

* `f`: Function mapping leaf nodes within the `xs` to the corresponding leaf node in `y`,
  that is, `yi = f(x1i::U, ..., xNi::U)::U`. The function `f` must be applicable to the type
  of every leaf node, and must return a value of the same type as its arguments.

  When an existing object `y` is passed and contains leaf nodes of a non-isbits non-scalar 
  type `U`, `f` should also have a partially-in-place method
  `newyi === f(yi::U, x1i::U, ..., xNi::U)::U` that modifies and reuses any mutable parts of
  `yi`; in particular, if `U` is a mutable type, this method should return `newyi === yi`.
  If a non-isbits type `U` should always be handled using the out-of-place signature, extend
  [`EnzymeCore.isscalartype`](@ref) such that `isscalartype(U) == true`.

  See [`EnzymeCore.isvectortype`](@ref) and [`EnzymeCore.isscalartype`)(@ref) for more
  details about leaf types and scalar types.

* `xs::NTuple{N,T}` or `(; y, xs)::@NamedTuple{y::T,xs::NTuple{N,T}}`: Tuple of `N` objects
  of the same type `T`, or NamedTuple combining this Tuple with an existing object `y::T`
  that can be partially or fully reused in the returned object.

  The first object `x1 = first(xs)` is the reference for graph structure and
  non-differentiable values when constructing the returned object. In particular:
  * When `y` is not passed, the returned object takes any non-differentiable parts from
    `x1`. (When `y` is passed, its non-differentiable parts are reused in the returned
    object, unless they are not initialized, in which case they are taken from `x1`.)
  * The graph of object references in `x1` is the one which is reproduced in the returned
    object. For each instance of multiple paths and cycles within `x1`, the same structure
    must be present in the other objects `x2, ..., xN`, otherwise the corresponding values
    in `y` would not be uniquely defined. However, `x2, ..., xN` may contain multiple paths
    or cycles that are not present in `x1`; these do not affect the structure of `y`.
  * If any values within `x1` are not initialized (that is, struct fields are undefined or
    array elements are unassigned), they are left uninitialized in the returned object. If
    any such values are mutable and `y` is passed, the corresponding value in `y` must not
    already be initialized, since initialized values cannot be nulled. Conversely, for every
    value in `x1` that is initialized, the corresponding values in `x2, ..., xN` must also
    be initialized, such that the corresponding value of `y` can be computed (however,
    values in `x2, ..., xN` can be initialized while the corresponding value in `x1` is not;
    such values are ignored.)

* `Val(copy_if_inactive)::Val{::Bool}` (optional): When a non-differentiable part of `x1` is
  included in the returned object, either because an object `y` is not passed or this part
  of `y` is not initialized, `copy_if_inactive` determines how it is included: if
  `copy_if_inactive == false`, it is shared as `yi = x1i`; if `copy_if_inactive == true`, it
  is deep-copied, more-or-less as `yi = deepcopy(x1i)` (the difference is that when `x1` has
  several non-differentiable parts, object identity is tracked across the multiple
  deep-copies such that the object reference graph is reproduced also within the inactive
  parts.)
"""
function recursive_map end

## type aliases, for generic handling of out-of-place and partially-in-place recursive_map
const XTup{N,T} = NTuple{N,T}
const YXTup{N,T} = @NamedTuple{y::T,xs::XTup{N,T}}
const XTupOrYXTup{N,T} = Union{XTup{N,T},YXTup{N,T}}

@inline xtup(xs::XTup) = xs
@inline xtup((; xs)::YXTup) = xs

@static if VERSION < v"1.11-"
    const Arraylike{U} = Array{U}
else
    const Arraylike{U} = Union{Array{U},GenericMemory{kind,U} where {kind}}
end

## main entry point
@inline function recursive_map(
    f::F, yxs::XTupOrYXTup{N,T}, copy_if_inactive::Val=Val(false)
) where {F,N,T}
    # determine whether or not an IdDict is needed for this T
    if isbitstype(T) || (
        guaranteed_const_nongen(T, nothing) && !needscopy(yxs, copy_if_inactive)
    )
        y = recursive_map(nothing, f, yxs, copy_if_inactive)
    else
        y = recursive_map(IdDict(), f, yxs, copy_if_inactive)
    end
    return y::T
end

## recursive methods
@inline function recursive_map(
    seen::Union{Nothing,IdDict},
    f::F,
    yxs::XTupOrYXTup{N,T},
    copy_if_inactive::Val=Val(false),
) where {F,N,T}
    # determine whether to continue recursion, copy/share, or retrieve from cache
    xs = xtup(yxs)
    if guaranteed_const_nongen(T, nothing)
        y = maybecopy(seen, yxs, copy_if_inactive)
    elseif isbitstype(T)  # no need to track identity or pass y in this branch
        y = recursive_map_inner(nothing, f, xs, copy_if_inactive)
    elseif hascache(seen, xs)
        y = getcached(seen, xs)
    else
        y = recursive_map_inner(seen, f, yxs, copy_if_inactive)
    end
    return y::T
end

@inline function recursive_map_inner(
    seen, f::F, yxs::XTupOrYXTup{N,T}, args::Vararg{Any,M}
) where {F,N,T,M}
    # forward to appropriate handler for leaf vs. mutable vs. immutable type
    @assert !isabstracttype(T)
    @assert isconcretetype(T)
    if isvectortype(T)
        y = recursive_map_leaf(seen, f, yxs)
    elseif ismutabletype(T)
        y = recursive_map_mutable(seen, f, yxs, args...)
    else
        y = recursive_map_immutable(seen, f, yxs, args...)
    end
    return y::T
end

@inline function recursive_map_mutable(
    seen, f::F, xs::XTup{N,T}, args::Vararg{Any,M}
) where {F,N,T,M}
    # out-of-place mutable handler: construct y
    @assert ismutabletype(T)
    x1, xtail... = xs
    nf = fieldcount(T)
    if (!(T <: Arraylike)) && all(isbitstype, fieldtypes(T)) && all(i -> isdefined(x1, i), 1:nf)
        # fast path when all fields are defined and all fieldtypes are bitstypes (the latter
        # preventing circular references, which are incompatible with the fast path)
        check_initialized(xtail, 1:nf)
        fieldtup = ntuple(Val(nf)) do i
            @inline
            recursive_map_index(i, seen, f, xs, args...)
        end
        y = splatnew(T, fieldtup)
        cache!(seen, y, xs)
    else  # handle both structs, arrays, and memory through generic helpers
        y = _similar(x1)
        cache!(seen, y, xs)
        @inbounds for i in _eachindex(y, xs...)
            if isinitialized(x1, i)
                check_initialized(xtail, i)
                yi = recursive_map_index(i, seen, f, xs, args...)
                setvalue(y, i, yi)
            end
        end
    end
    return y::T
end

@inline function recursive_map_mutable(
    seen, f!!::F, (; y, xs)::YXTup{N,T}, args::Vararg{Any,M}
) where {F,N,T,M}
    # in-place mutable handler: set/update values in y
    @assert ismutabletype(T)
    cache!(seen, y, xs)
    x1, xtail... = xs
    @inbounds for i in _eachindex(y, xs...)
        # handle both structs, arrays, and memory through generic helpers
        if isinitialized(x1, i)
            check_initialized(xtail, i)
            newyi = recursive_map_index(i, seen, f!!, (; y, xs), args...)
            setvalue(y, i, newyi)
        else
            check_initialized((y,), i, false)
        end
    end
    return y::T
end

@inline function recursive_map_immutable(
    seen, f::F, yxs::XTupOrYXTup{N,T}, args::Vararg{Any,M}
) where {F,N,T,M}
    # immutable handler: construct y/newy
    @assert !ismutabletype(T)
    x1, xtail... = xtup(yxs)
    nf = fieldcount(T)
    if nf == 0  # nothing to do; assume inactive
        y = maybecopy(seen, yxs, args...)
    elseif isdefined(x1, nf)  # fast path when all fields are defined
        check_initialized(xtail, nf)
        fieldtup = ntuple(Val(nf)) do i
            @inline
            recursive_map_index(i, seen, f, yxs, args...)
        end
        y = splatnew(T, fieldtup)
    else
        flds = Vector{Any}(undef, nf)
        @inbounds for i in 1:nf
            if isdefined(x1, i)
                check_initialized(xtail, i)
                flds[i] = recursive_map_index(i, seen, f, yxs, args...)
            else
                nf = i - 1  # rest of tail must be undefined values
                break
            end
        end
        y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nf)
    end
    return y::T
end

Base.@propagate_inbounds function recursive_map_index(
    i, seen, f::F, xs::XTup, args::Vararg{Any,M}
) where {F,M}
    # out-of-place recursive handler: extract value i from each of the xs; call
    # recursive_map to obtain yi
    xis = getvalues(xs, i)
    yi = recursive_map(seen, f, xis, args...)
    return yi::Core.Typeof(first(xis))
end

Base.@propagate_inbounds function recursive_map_index(
    i, seen, f!!::F, (; y, xs)::YXTup, args::Vararg{Any,M}
) where {F,M}
    # partially-in-place recursive handler: extract value i from each of the xs and, if
    # initialized, from y; call recursive_map to obtain newyi
    xis = getvalues(xs, i)
    if isinitialized(y, i)
        yi = getvalue(y, i)
        newyi = recursive_map(seen, f!!, (; y=yi, xs=xis), args...)
    else
        newyi = recursive_map(seen, f!!, xis, args...)
    end
    return newyi::Core.Typeof(first(xis))
end

## leaf handlers
function recursive_map_leaf(seen, f::F, xs::XTup{N,T}) where {F,N,T}
    # out-of-place
    y = f(xs...)
    if !isbitstype(T)
        cache!(seen, y, xs)
    end
    return y::T
end

function recursive_map_leaf(seen, f!!::F, (; y, xs)::YXTup{N,T}) where {F,N,T}
    # partially-in-place
    if isbitstype(T) || isscalartype(T)
        newy = f!!(xs...)
    else  # !isbitstype(T)
        newy = f!!(y, xs...)
        if ismutabletype(T)
            @assert newy === y
        end
    end
    if !isbitstype(T)
        cache!(seen, newy, xs)
    end
    return newy::T
end

## helpers
# vector/scalar trait implementation
@inline EnzymeCore.isvectortype(::Type{T}) where {T} = isscalartype(T)
@inline EnzymeCore.isvectortype(::Type{<:Arraylike{U}}) where {U} = isscalartype(U)

@inline EnzymeCore.isscalartype(::Type{<:AbstractFloat}) = true
@inline EnzymeCore.isscalartype(::Type{<:Complex{<:AbstractFloat}}) = true
@inline EnzymeCore.isscalartype(::Type) = false

# generic handling of mutable structs, arrays, and memory
@inline _similar(::T) where {T} = ccall(:jl_new_struct_uninit, Any, (Any,), T)::T
@inline _similar(x::T) where {T<:Arraylike} = similar(x)::T
@inline _eachindex(xs::T...) where {T} = 1:fieldcount(T)
@inline _eachindex(xs::Arraylike...) = eachindex(xs...)
@inline isinitialized(x, i) = isdefined(x, i)
Base.@propagate_inbounds isinitialized(x::Arraylike, i) = isassigned(x, i)
@inline getvalue(x, i) = getfield(x, i)
Base.@propagate_inbounds getvalue(x::Arraylike, i) = x[i]
@inline setvalue(x, i, v) = setfield_force!(x, i, v)
Base.@propagate_inbounds setvalue(x::Arraylike, i, v) = (x[i] = v; nothing)

Base.@propagate_inbounds function getvalues(xs::XTup{N}, i) where {N}
    return ntuple(Val(N)) do j
        Base.@_propagate_inbounds_meta
        getvalue(xs[j], i)
    end
end

@inline function setfield_force!(y::T, i, newyi) where {T}
    if Base.isconst(T, i)
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, newyi)
    else
        setfield!(y, i, newyi)
    end
    return nothing
end

# generic inactive handler: sharing/copying (out-of-place) or leaving unchanged (in-place)
@inline maybecopy(_, (; y)::YXTup{N,T}, _) where {N,T} = y::T
@inline function maybecopy(seen, xs::XTup{N,T}, copy) where {N,T}
    if needscopy(xs, copy)
        y = Base.deepcopy_internal(first(xs), seen)
    else
        y = first(xs)
    end
    return y::T
end

@inline needscopy(::YXTup, _) = false
@inline needscopy(::XTup{N,T}, ::Val{copy}) where {N,T,copy} = (copy && !isbitstype(T))

# validating cache handlers
@inline function cache!(seen::IdDict, y::T, xs::XTup{N,T}) where {N,T}
    x1, xtail... = xs
    seen[x1] = (y, xtail...)
    return nothing
end

@inline hascache(seen, xs::XTup) = haskey(seen, first(xs))

@inline function getcached(seen::IdDict, xs::XTup{N,T}) where {N,T}
    x1, xtail... = xs
    y, xtail_... = seen[x1]::XTup{N,T}
    check_identical(xtail, xtail_)  # check compatible topology
    return y::T
end

## in-place wrapper
"""
    recursive_map!(
        [seen::IdDict,]
        f!!,
        y::T,
        xs::NTuple{N,T},
        isleaftype=Returns(false),
        ::Val{copy_if_inactive}=Val(false),
    )::Nothing where {T,N,copy_if_inactive}

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recurse through `N` objects `xs = (x1::T, x2::T, ..., xN::T)` of the same type, mapping the
function `f!!` over every differentiable value encountered and updating new mutable object
`y::T` in-place with the resulting values.

This is a wrapper that calls
`recursive_map([seen,] f!!, (; y, xs), isleaftype, Val(copy_if_inactive))`, but only accepts
types `T` that are mutable (or, trivially, entirely non-differentiable), and enforces a
fully in-place update of `y`. See [`recursive_map`](@ref) for details.
"""
function recursive_map!(f!!::F, y::T, xs::XTup{N,T}, args::Vararg{Any,M}) where {F,N,T,M}
    check_notactive(T)
    newy = recursive_map(f!!, (; y, xs), args...)
    @assert newy === y
    return nothing
end

function recursive_map!(
    seen::IdDict, f!!::F, y::T, xs::XTup{N,T}, args::Vararg{Any,M}
) where {F,N,T,M}
    check_notactive(T)
    newy = recursive_map(seen, f!!, (; y, xs), args...)
    @assert newy === y
    return nothing
end

## argument checkers
Base.@propagate_inbounds function check_initialized(xs, indices, value=true)
    for xj in xs
        for i in indices
            if isinitialized(xj, i) != value
                throw_initialized()
            end
        end
    end
    return nothing
end

@inline function check_identical(x1, x2)
    if x1 !== x2
        throw_identical()
    end
    return nothing
end

@inline function check_notactive(::Type{T}) where {T}
    if active_reg_inner(T, (), nothing, Val(true)) == ActiveState  # justActive
        throw_notactive()
    end
    return nothing
end

@noinline function throw_initialized()
    msg = "recursive_map(!) called on objects whose undefined fields/unassigned elements "
    msg *= "don't line up"
    throw(ArgumentError(msg))
end

@noinline function throw_identical()
    msg = "recursive_map(!) called on objects whose topology don't match"
    throw(ArgumentError(msg))
end

@noinline function throw_notactive()
    msg = "recursive_map! called on objects containing immutable differentiable elements"
    throw(ArgumentError(msg))
end

### make_zero(!)
## entry points, with default handling of leaves
function EnzymeCore.make_zero(prev::T, args::Vararg{Any,M}) where {T,M}
    if EnzymeCore.isvectortype(T)
        if length(args) > 0  # pick up custom methods for custom vector types
            new = EnzymeCore.make_zero(prev)
        else  # default implementation
            # convert because zero may produce different eltype when eltype(T) is abstract
            new = convert(T, zero(prev))
        end
    else
        new = recursive_map(make_zero_f!!, (prev,), args...)
    end
    return new::T
end

function EnzymeCore.make_zero!(prev::T) where {T}
    @assert !EnzymeCore.isscalartype(T)  # sanity check
    if EnzymeCore.isvectortype(T)  # default implementation
        fill!(prev, false)
    else
        recursive_map!(make_zero_f!!, prev, (prev,))
    end
    return nothing
end

## low-level interface, for bringing your own IdDict
function EnzymeCore.make_zero(
    ::Type{T}, seen::IdDict, prev::T, args::Vararg{Any,M}
) where {T,M}
    return recursive_map(seen, make_zero_f!!, (prev,), args...)::T
end

function EnzymeCore.make_zero!(prev, seen::IdDict)
    recursive_map!(seen, make_zero_f!!, prev, (prev,))
    return nothing
end

## the mapped function: assert valid leaf type and call back into make_zero(!)
function make_zero_f!!(prev::T) where {T}
    @assert EnzymeCore.isvectortype(T)  # otherwise infinite loop
    return EnzymeCore.make_zero(prev)::T
end

function make_zero_f!!(pout::T, pin::T) where {T}
    @assert !EnzymeCore.isscalartype(T)  # not appropriate for in-place handler
    @assert EnzymeCore.isvectortype(T)   # otherwise infinite loop
    @assert pout === pin
    EnzymeCore.make_zero!(pout)
    return pout::T
end

end  # module RecursiveMap
