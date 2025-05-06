module RecursiveMaps

using EnzymeCore: EnzymeCore, isvectortype, isscalartype
using ..Compiler: guaranteed_const, guaranteed_const_nongen, guaranteed_nonactive,
    guaranteed_nonactive_nongen

### Config type for setting inactive/nonactive options
"""
    config = InactiveConfig(
        extra = (T -> false); copy_if_inactive = Val(false), runtime_inactive = Val(false)
    )
    config = InactiveConfig{copy_if_inactive::Bool, runtime_inactive::Bool}(extra)
    newconfig = InactiveConfig(config::InactiveConfig, extra)

Config type for specifying which parts of objects should be skipped by `recursive_map{!}`.

At a minimum, parts that Enzyme always considers inactive are skipped. An inactive type is a
type for which Enzyme can prove that a differentiable value can never be reached from any
instance of the type.

The optional argument `extra` takes a function defining additional types that should be
skipped regardless of their nominal activity. `extra` should be a plain function
or callable of a singleton type, not a closure or otherwise stateful callable; this is to
ensure that an `InactiveConfig` instance is fully specified by its type.

The parameter `copy_if_inactive` specifies whether `recursive_map{!}` should share (if
`Val(false)`, the default) or deep-copy (if `Val(true)`) inactive/skipped parts from inputs
to outputs.

The parameter `runtime_inactive` specifies whether `recursive_map{!}` should respect runtime
semantics when determining if a type is guaranteed inactive. If `Val(false)`, guaranteed
inactivity is determined by `active_reg_nothrow`, which is a generated function and thus
frozen in the precompilation world age; this means that methods added to
`EnzymeRules.inactive_type` after `Enzyme` precompilation are not respected. If `Val(true)`,
the generated function is not used and changes to the `EnzymeRules.inactive_type` method
table are picked up through invalidation as usual.

Using `runtime_inactive = Val(false)` may be preferred in interactive sessions or if
`EnzymeRules.inactive_type` is extended in downstream packages or package extensions.
However, performance may suffer if the activity of every type cannot be resolved at compile
time, so `runtime_inactive = Val(true)` is preferable when possible and is the default.

The updating constructor `InactiveConfig(config::InactiveConfig, extra)` returns a new
config that extends `config` with an additional `extra` function.
"""
struct InactiveConfig{copy_if_inactive, runtime_inactive, E}
    extra::E
    function InactiveConfig{C, R}(extra::E) where {C, R, E}
        @assert Base.issingletontype(E)
        return new{C::Bool, R::Bool, E}(extra)
    end
end

function InactiveConfig(
        extra::E = (_ -> (@nospecialize; false));
        copy_if_inactive::Val{C} = Val(false), runtime_inactive::Val{R} = Val(false),
    ) where {E, C, R}
    return InactiveConfig{C, R}(extra)
end

function InactiveConfig(config::InactiveConfig{C, R}, extra::E) where {C, R, E}
    @inline combinedextra(::Type{T}) where {T} = (config.extra(T) || extra(T))
    return InactiveConfig{C, R}(combinedextra)
end

function isinactivetype(::Type{T}, config::InactiveConfig{C, false}) where {T, C}
    return guaranteed_const(T) || config.extra(T) # call guaranteed_const first, as this is a constant at runtime
end
function isinactivetype(::Type{T}, config::InactiveConfig{C, true}) where {T, C}
    return config.extra(T) || guaranteed_const_nongen(T, nothing) # call config.extra first, as guaranteed_const_nongen may incur runtime dispatch
end

function isnonactivetype(::Type{T}, config::InactiveConfig{C, false}) where {T, C}
    return guaranteed_nonactive(T) || config.extra(T) # call guaranteed_const first, as this is a constant at runtime
end
function isnonactivetype(::Type{T}, config::InactiveConfig{C, true}) where {T, C}
    return config.extra(T) || guaranteed_nonactive_nongen(T, nothing) # call config.extra first, as guaranteed_nonactive_nongen may incur runtime dispatch
end

### traits defining active leaf types for recursive_map
@inline EnzymeCore.isvectortype(::Type{T}) where {T} = isscalartype(T)
@inline function EnzymeCore.isvectortype(::Type{<:DenseArray{U}}) where {U}
    return isbitstype(U) && isscalartype(U)
end

@inline EnzymeCore.isscalartype(::Type) = false
@inline EnzymeCore.isscalartype(::Type{T}) where {T <: AbstractFloat} = isconcretetype(T)
@inline function EnzymeCore.isscalartype(::Type{Complex{T}}) where {T <: AbstractFloat}
    return isconcretetype(T)
end

### recursive_map: walk arbitrary objects and map a function over scalar and vector leaves
"""
    newy = recursive_map(
        [seen::Union{Nothing, IdDict},]
        f,
        [y::T,]
        xs::NTuple{N, T},
        config::InactiveConfig = InactiveConfig(),
    )::T

Recurse through `N` objects `xs = (x1::T, x2::T, ..., xN::T)` of the same type, mapping the
function `f` over every differentiable value encountered and constructing a new object
`newy::T` from the resulting values `newy_i = f(x1_i, ..., xN_i)`.

The trait [`EnzymeCore.isvectortype`](@ref) determines which values are considered leaf
nodes at which to terminate recursion and invoke `f`. See the docstring for
[`EnzymeCore.isvectortype`](@ref) and the related [`EnzymeCore.isscalartype`](@ref) for more
information.

An existing object `y::T` may be passed, in which case it is updated "partially-in-place":
any parts of `y` that are mutable or non-differentiable are reused in the returned object
`newy`, while immutable differentiable parts are handled out-of-place as if `y` were not
passed. If `T` itself is a mutable type, `y` is modified fully in-place and returned, such
that `newy === y`.

The recursion and mapping operate on the structure of `T` as defined by struct fields and
plain array elements, not on the values provided through iteration or array interfaces. For
example, given a structured matrix wrapper or sparse array type, this function recurses into
the struct type and operates on the plain arrays held within, rather than operating on the
array that the type notionally represents.

# Arguments

* `seen::Union{IdDict, Nothing}` (optional): Dictionary for tracking object identity as
  needed to reproduce the object reference graph topology of the `xs` when constructing `y`,
  including cycles (i.e., recursive substructures) and convergent paths. If not provided, an
  `IdDict` will be allocated internally if required.

  If `nothing` is provided, object identity tracking is turned off. Objects with multiple
  references are then duplicated such that the graph of object references within `newy`
  becomes a tree.  Note that any cycles in the `xs` will result in infinite recursion
  and stack overflow.

* `f`: Function mapping leaf nodes within the `xs` to the corresponding leaf node in `newy`,
  that is, `newy_i = f(x1_i::U, ..., xN_i::U)::U`. The function `f` must be applicable to
  the type of every leaf node, and must return a value of the same type as its arguments.

  When an existing object `y` is provided and contains leaf nodes of a non-isbits non-scalar
  type `U`, `f` should also have a partially-in-place method
  `newy_i = f(y_i::U, x1_i::U, ..., xN_i::U)::U` that modifies and reuses any mutable parts
  of `y_i`; in particular, if `U` is a mutable type, this method should return
  `newy_i === y_i`.

  If a non-isbits leaf type `U` must always be handled using the out-of-place signature,
  define the method `EnzymeCore.isscalartype(::Type{U}) = true`.

  See [`EnzymeCore.isvectortype`](@ref) and [`EnzymeCore.isscalartype`](@ref) for more
  details about leaf types and scalar types.

* `y::T` (optional): Instance from which to reuse mutable and non-differentiable parts when
  mapping (partially) in-place.

* `xs::NTuple{N, T}`: Tuple of `N` objects of the same type `T`.

  The first object `x1 = first(xs)` is the reference for graph structure and
  non-differentiable values when constructing the returned object. In particular:
  * When `y` is not provided, non-differentiable values within `newy` are shared with/copied
    from `x1`.
  * When `y` is provided, its non-differentiable values are kept unchanged, unless they are
    uninitialized, in which case they are shared with/copied from from `x1`.
  * The graph topology of object references in `x1` is the one which is reproduced in the
    returned object. Hence, for each instance of cycles and converging paths within `x1`,
    the same structure must be present in the other objects `x2, ..., xN`, otherwise the
    corresponding values in `newy` would not be uniquely defined. However, `x2, ..., xN` may
    contain additional cycles and converging paths that are not present in `x1`; these do
    not affect the structure of `newy`.
  * If any values within `x1` are not initialized (that is, struct fields are undefined or
    array elements are unassigned), they remain uninitialized in `newy`. If any such values
    are mutable and `y` is provided, the corresponding value in `y` must not already be
    initialized, since initialized values cannot be nulled. Conversely, for every value in
    `x1` that is initialized, the corresponding values in `x2, ..., xN` must also be
    initialized, such that the corresponding value in `newy` can be computed. However,
    `x2, ..., xN` may have initialized values where `x1` has uninitialized values; these
    will remain uninitialized in `newy`.

* `config::InactiveConfig` (optional): Config object detailing how to deal with
  non-differentiable (inactive) parts. The config specifies whether non-differentiable parts
  should be shared or deep-copied from `x1` to `newy`, and whether any additional types
  should be skipped in addition to those Enzyme always considers inactive. See
  [`InactiveConfig`](@ref) for details.
"""
function recursive_map end

const Maybe{T} = Union{Nothing, Some{T}}

## entry points: set default arguments, deal with nothing/Some
function recursive_map(f::F, xs::NTuple, config::InactiveConfig = InactiveConfig()) where {F}
    return recursive_map_main(f, nothing, xs, config)
end

function recursive_map(
        f::F, y::T, xs::NTuple{N, T}, config::InactiveConfig = InactiveConfig()
    ) where {F, N, T}
    return recursive_map_main(f, Some(y), xs, config)
end

function recursive_map(
        seen::Union{Nothing, IdDict},
        f::F,
        xs::NTuple,
        config::InactiveConfig = InactiveConfig(),
    ) where {F}
    return recursive_map_main(seen, f, nothing, xs, config)
end

function recursive_map(
        seen::Union{Nothing, IdDict},
        f::F,
        y::T,
        xs::NTuple{N, T},
        config::InactiveConfig = InactiveConfig(),
    ) where {F, N, T}
    return recursive_map_main(seen, f, Some(y), xs, config)
end

## main dispatcher: allocate IdDict if needed, exit early if possible
function recursive_map_main(
        f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config::InactiveConfig
    ) where {F, N, T}
    newy = if isinactivetype(T, config)
        recursive_map_inactive(nothing, maybe_y, xs, config)
    elseif isvectortype(T) || isbitstype(T)
        recursive_map_inner(nothing, f, maybe_y, xs, config)
    else
        recursive_map_inner(IdDict(), f, maybe_y, xs, config)
    end
    return newy::T
end

## recursive methods
function recursive_map_main(
        seen::Union{Nothing, IdDict},
        f::F,
        maybe_y::Maybe{T},
        xs::NTuple{N, T},
        config::InactiveConfig,
    ) where {F, N, T}
    # determine whether to continue recursion, copy/share, or retrieve from cache
    newy = if isinactivetype(T, config)
        recursive_map_inactive(seen, maybe_y, xs, config)
    elseif isbitstype(T)  # no object identity to to track in this branch
        recursive_map_inner(nothing, f, maybe_y, xs, config)
    elseif hascache(seen, xs)
        getcached(seen, xs)
    else
        recursive_map_inner(seen, f, maybe_y, xs, config)
    end
    return newy::T
end

@inline function recursive_map_inner(
        seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    # forward to appropriate handler for leaf vs. mutable vs. immutable type
    @assert !isabstracttype(T)
    @assert isconcretetype(T)
    newy = if isvectortype(T)
        recursive_map_leaf(seen, f, maybe_y, xs)
    elseif ismutabletype(T)
        recursive_map_mutable(seen, f, maybe_y, xs, config)
    else
        recursive_map_immutable(seen, f, maybe_y, xs, config)
    end
    return newy::T
end

@inline function recursive_map_mutable(
        seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    @assert ismutabletype(T)
    if isnothing(maybe_y) && !(T <: DenseArray) && all(isbitstype, fieldtypes(T))
        # fast path for out-of-place handling when all fields are bitstypes, which rules
        # out undefined fields and circular references
        newy = recursive_map_new(seen, f, nothing, xs, config)
        maybecache!(seen, newy, xs)
    else
        newy = if isnothing(maybe_y)
            _similar(first(xs))
        else
            something(maybe_y)
        end
        maybecache!(seen, newy, xs)
        recursive_map_mutable_inner!(seen, f, newy, maybe_y, xs, config)
    end
    return newy::T
end

@inline function recursive_map_mutable_inner!(
        seen, f::F, newy::T, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T <: DenseArray}
    if isbitstype(eltype(T))
        if isnothing(maybe_y)
            broadcast!(newy, xs...) do xs_i...
                recursive_map_main(nothing, f, nothing, xs_i, config)
            end
        else
            broadcast!(newy, something(maybe_y), xs...) do y_i, xs_i...
                recursive_map_main(nothing, f, Some(y_i), xs_i, config)
            end
        end
    else
        @inbounds for i in eachindex(newy, xs...)
            recursive_map_item!(i, seen, f, newy, maybe_y, xs, config)
        end
    end
    return nothing
end

@generated function recursive_map_mutable_inner!(
        seen, f::F, newy::T, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    return quote
        @inline
        Base.Cartesian.@nexprs $(fieldcount(T)) i -> @inbounds begin
            recursive_map_item!(i, seen, f, newy, maybe_y, xs, config)
        end
        return nothing
    end
end

@inline function recursive_map_immutable(
        seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    @assert !ismutabletype(T)
    nf = fieldcount(T)
    if nf == 0  # nothing to do (also no known way to hit this branch)
        newy = recursive_map_inactive(seen, maybe_y, xs, config)
    else
        newy = if isinitialized(first(xs), nf)  # fast path when all fields are defined
            check_allinitialized(Base.tail(xs), nf)
            recursive_map_new(seen, f, maybe_y, xs, config)
        else
            recursive_map_immutable_inner(seen, f, maybe_y, xs, config)
        end
        # maybecache! _should_ be a no-op here; call it anyway for consistency
        maybecache!(seen, newy, xs)
    end
    return newy::T
end

@generated function recursive_map_immutable_inner(
        seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    nf = fieldcount(T)
    return quote
        @inline
        x1, xtail = first(xs), Base.tail(xs)
        fields = Vector{Any}(undef, $(nf - 1))
        Base.Cartesian.@nexprs $(nf - 1) i -> begin  # unrolled loop over struct fields
            @inbounds if isinitialized(x1, i)
                check_allinitialized(xtail, i)
                fields[i] = recursive_map_item(i, seen, f, maybe_y, xs, config)
            else
                return new_structv(T, fields, i - 1)
            end
        end
        @assert !isinitialized(x1, $nf)
        return new_structv(T, fields, $(nf - 1))
    end
end

@generated function recursive_map_new(
        seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    # direct construction of fully initialized non-cyclic structs
    nf = fieldcount(T)
    return quote
        @inline
        fields = Base.@ntuple $nf i -> @inbounds begin
            recursive_map_item(i, seen, f, maybe_y, xs, config)
        end
        newy = $(Expr(:splatnew, :T, :fields))
        return newy::T
    end
end

Base.@propagate_inbounds function recursive_map_item!(
        i, seen, f::F, newy::T, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    if isinitialized(first(xs), i)
        check_allinitialized(Base.tail(xs), i)
        setitem!(newy, i, recursive_map_item(i, seen, f, maybe_y, xs, config))
    elseif !isnothing(maybe_y)
        check_initialized(something(maybe_y), i, false)
    end
    return nothing
end

Base.@propagate_inbounds function recursive_map_item(
        i, seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}, config
    ) where {F, N, T}
    # recurse into the xs and apply recursive_map to items with index i
    maybe_y_i = if isnothing(maybe_y) || !isinitialized(something(maybe_y), i)
        nothing
    else
        Some(getitem(something(maybe_y), i))
    end
    return recursive_map_barrier(seen, f, maybe_y_i, config, getitems(xs, i)...)
end

# function barrier such that abstractly typed items trigger minimal runtime dispatch
# the idea is that SROA can eliminate the xs_i tuple in the above function, since it's
# splatted directly into a call; thus, instead of a dynamic dispatch to the Tuple
# constructor followed by a dynamic dispatch to recursive_map, we only incur a single
# dynamic dispatch to recursive_map_barrier
function recursive_map_barrier(
        seen, f::F, maybe_y_i::Maybe{ST}, config::InactiveConfig, xs_i::Vararg{ST, N}
    ) where {F, N, ST}
    return recursive_map_main(seen, f, maybe_y_i, xs_i, config)::ST
end

## recursion base case handlers
@inline function recursive_map_leaf(
        seen, f::F, maybe_y::Maybe{T}, xs::NTuple{N, T}
    ) where {F, N, T}
    # apply the mapped function to leaf values
    if isnothing(maybe_y) || isbitstype(T) || isscalartype(T)
        newy = f(xs...)::T
    else  # !isbitstype(T)
        y = something(maybe_y)
        newy = f(y, xs...)::T
        if ismutabletype(T)
            @assert newy === y
        end
    end
    maybecache!(seen, newy, xs)
    return newy::T
end

@inline function recursive_map_inactive(
        seen, maybe_y::Maybe{T}, (x1,)::NTuple{N, T}, ::InactiveConfig{copy_if_inactive}
    ) where {N, T, copy_if_inactive}
    newy = if !isnothing(maybe_y)
        something(maybe_y)
    elseif copy_if_inactive && !isbitstype(T)
        if isnothing(seen)
            deepcopy(x1)
        else
            Base.deepcopy_internal(x1, seen)
        end
    else
        x1
    end
    return newy::T
end

### recursive_map!: fully in-place wrapper around recursive_map
"""
    recursive_map!(
        [seen::Union{Nothing, IdDict},]
        f!!,
        y::T,
        xs::NTuple{N, T},
        isinactivetype::InactiveConfig = InactiveConfig(),
    )::Nothing

Recurse through `N` objects `xs = (x1::T, x2::T, ..., xN::T)` of the same type, mapping the
function `f!!` over every differentiable value encountered and updating `y::T` in-place with
the resulting values.

This is a simple wrapper that verifies that `T` is a type where all differentiable values
can be updated in-place, calls `recursive_map`, and verifies that the returned value is
indeed identically the same object `y`. See [`recursive_map`](@ref) for details.
"""
function recursive_map! end

function recursive_map!(
        f!!::F, y::T, xs::NTuple{N, T}, config::InactiveConfig = InactiveConfig()
    ) where {F, N, T}
    check_nonactive(T, config)
    newy = recursive_map(f!!, y, xs, config)
    @assert newy === y
    return nothing
end

function recursive_map!(
        seen::Union{Nothing, IdDict},
        f!!::F,
        y::T,
        xs::NTuple{N, T},
        config::InactiveConfig = InactiveConfig(),
    ) where {F, N, T}
    check_nonactive(T, config)
    newy = recursive_map(seen, f!!, y, xs, config)
    @assert newy === y
    return nothing
end

### recursive_map helpers
@generated function new_structv(::Type{T}, fields::Vector{Any}, nfields_) where {T}
    return quote
        @inline
        ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields, nfields_)::T
    end
end

@inline _similar(::T) where {T} = ccall(:jl_new_struct_uninit, Any, (Any,), T)::T
@inline _similar(x::T) where {T <: DenseArray} = similar(x)::T
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

Base.@propagate_inbounds function getitems(
        (x1, xtail...)::Tuple{T, T, Vararg{T, N}}, i
    ) where {T, N}
    return (getitem(x1, i), getitems(xtail, i)...)
end

Base.@propagate_inbounds getitems((x1,)::Tuple{T}, i) where {T} = (getitem(x1, i),)

## cache (seen) helpers
@inline function iscachedtype(::Type{T}) where {T}
    # cache all mutable types and any non-isbits types that are also leaf types
    return ismutabletype(T) || ((!isbitstype(T)) && isvectortype(T))
end

@inline shouldcache(::IdDict, ::Type{T}) where {T} = iscachedtype(T)
@inline shouldcache(::Nothing, ::Type{T}) where {T} = false

@inline function maybecache!(seen, newy::T, (x1, xtail...)::NTuple{N, T}) where {N, T}
    if shouldcache(seen, T)
        seen[x1] = if (N == 1)
            newy
        else
            (newy, xtail...)
        end
    end
    return nothing
end

@inline function hascache(seen, (x1,)::NTuple{N, T}) where {N, T}
    return shouldcache(seen, T) ? haskey(seen, x1) : false
end

@inline function getcached(seen::IdDict, (x1, xtail...)::NTuple{N, T}) where {N, T}
    newy = if (N == 1)
        seen[x1]::T
    else   # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
        cache = seen[x1]::NTuple{N, T}
        cachedtail = cache[2:end]
        check_identical(cachedtail, xtail)  # check compatible structure
        cache[1]
    end
    return newy::T
end

## argument validation
Base.@propagate_inbounds function check_initialized(x, i, initialized = true)
    if isinitialized(x, i) != initialized
        throw_initialized()  # TODO: hit this when VectorSpace implemented
    end
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(  # TODO: hit this when VectorSpace implemented
        (x1, xtail...)::Tuple{T, T, Vararg{T, N}}, i, initialized = true
    ) where {T, N}
    check_initialized(x1, i, initialized)
    check_allinitialized(xtail, i, initialized)
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(
        (x1,)::Tuple{T}, i, initialized = true
    ) where {T}
    check_initialized(x1, i, initialized)
    return nothing
end

Base.@propagate_inbounds check_allinitialized(::Tuple{}, i, initialized = true) = nothing

@inline function check_identical(u, v)  # TODO: hit this when VectorSpace implemented
    if u !== v
        throw_identical()
    end
    return nothing
end

@inline function check_nonactive(::Type{T}, config) where {T}
    if !isnonactivetype(T, config)
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
    msg = "recursive_map(!) called on objects whose structure don't match"
    throw(ArgumentError(msg))
end

@noinline function throw_nonactive()
    msg = "recursive_map! called on objects containing immutable differentiable values"
    throw(ArgumentError(msg))
end

### EnzymeCore.make_zero(!) implementation
@inline function EnzymeCore.make_zero(prev::T, args::Vararg{Any, M}; kws...) where {T, M}
    config = make_zero_config(args...; kws...)
    new = if iszero(M) && isempty(kws) && !isinactivetype(T, config) && isvectortype(T)  # fallback
        # isinactivetype has precedence over isvectortype for consistency with recursive_map
        convert(T, zero(prev))  # convert because zero(prev)::T may not hold when eltype(T) is abstract
    else
        recursive_map(_make_zero!!, (prev,), config)::T
    end
    return new::T
end

@inline function EnzymeCore.make_zero!(val::T, allargs::Vararg{Any, M}; kws...) where {T, M}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    seen, args = if (M > 0) && (first(allargs) isa IdDict)
        first(allargs), Base.tail(allargs)
    else
        nothing, allargs
    end
    config = make_zero_config!(args...; kws...)
    if iszero(M) && isempty(kws) && !isinactivetype(T, config) && isvectortype(T)  # fallback
        # isinactivetype has precedence over isvectortype for consistency with recursive_map
        fill!(val, false)
    elseif isnothing(seen)
        recursive_map!(_make_zero!!, val, (val,), config)
    else
        recursive_map!(seen, _make_zero!!, val, (val,), config)
    end
    return nothing
end

# map make_zero(!) args/kws to config
@inline make_zero_config(C) = InactiveConfig(; copy_if_inactive = C)
@inline make_zero_config(C, R) = InactiveConfig(; copy_if_inactive = C, runtime_inactive = R)
@inline make_zero_config(; kws...) = InactiveConfig(; kws...)

@inline make_zero_config!(R) = InactiveConfig(; runtime_inactive = R)
@inline function make_zero_config!(; runtime_inactive = nothing)
    if isnothing(runtime_inactive)
        return InactiveConfig()
    else
        return InactiveConfig(; runtime_inactive)
    end
end

# the mapped function: assert leaf type and call back into single-arg make_zero(!)
function _make_zero!!(prev::T) where {T}
    @assert isvectortype(T)  # otherwise infinite loop
    return EnzymeCore.make_zero(prev)::T
end

function _make_zero!!(val::T, _val::T) where {T}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    @assert isvectortype(T)   # otherwise infinite loop
    @assert val === _val
    EnzymeCore.make_zero!(val)
    return val::T
end

# alternative entry point for passing custom IdDict
@inline function EnzymeCore.make_zero(
        ::Type{T}, seen::IdDict, prev::T, args::Vararg{Any, M}; kws...
    ) where {T, M}
    new = recursive_map(seen, _make_zero!!, (prev,), make_zero_config(args...; kws...))
    return new::T
end

end  # module RecursiveMaps
