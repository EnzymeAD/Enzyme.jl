using .RecursiveMaps: RecursiveMaps, recursive_map, recursive_map!

"""
    recursive_add(x::T, y::T, f=identity, forcelhs=guaranteed_const)

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recursively construct `z::T` such that `zi = xi + f(yi)` where `zi`, `xi`, and `yi` are
corresponding values from `z`, `x`, and `y`. In other words, this is a recursive
generalization of `x .+ f.(y)`.

The function `f` must return values of the same type as its argument.

The optional argument `forcelhs` takes a callable such that if `forcelhs(S) == true`, values
`zi::S` will be set to `zi = xi`. The default returns true for non-differentiable types,
such that `zi = xi + f(yi)` applies to differentiable values, while `zi = xi` applies to
non-differentiable values.
"""
function recursive_add(x::T, y::T, f::F=identity, forcelhs::L=guaranteed_const) where {T,F,L}
    function addf(xi::S, yi::S) where {S}
        @assert EnzymeCore.isvectortype(S)
        return ((xi + f(yi))::S,)
    end
    return only(recursive_map(addf, Val(1), (x, y), Val(false), forcelhs))::T
end

"""
    accumulate_seen!(f, seen::IdDict, ::Val{runtime_inactive}=Val(false))
    accumulate_seen!(f, seen::IdDict, isinactivetype::RecursiveMaps.IsInactive)

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recursively accumulate from values into keys, generalizing key .+= f.(value), for each
key-value pair in `seen::IdDict` where each key must be a mutable object or non-isbits
vector type instance mappping to another object of the same type and structure. Typically
`seen` is populated by `make_zero` (or some other single-argument invocation of
`recursive_map`), mapping components of its argument to the corresponding component of the
returned value.

The recursion stops at instances of types that are themselves cached by `make_zero`
(`recursive_map`), as these objects should have their own entries in `seen`. The recursion
also stops at inactive objects that would not be zeroed by `make_zero`.

If the optional `::Val{runtime_inactive}` argument was passed to `make_zero`, the same value
should be passed to `accumulate_seen` for consistency. If needed, a custom
`RecursiveMaps.IsInactive` instance can be provided instead.
"""
function accumulate_seen!(
    f::F, seen::IdDict, ::Val{runtime_inactive}=Val(false)
) where {F,runtime_inactive}
    accumulate_seen!(f, seen, RecursiveMaps.IsInactive{runtime_inactive}())
    return nothing
end

function accumulate_seen!(
    f::F, seen::IdDict, isinactivetype::RecursiveMaps.IsInactive
) where {F}
    isinactive_or_cachedtype = RecursiveMaps.IsInactive(
        isinactivetype, RecursiveMaps.iscachedtype
    )
    for (k, v) in seen
        _accumulate_seen_item!(f, k, v, isinactivetype, isinactive_or_cachedtype)
    end
    return nothing
end

function _accumulate_seen_item!(
    f::F, k::T, v::T, isinactivetype, isinactive_or_cachedtype
) where {F,T}
    function addf!!(ki::S, vi::S) where {S}
        @assert EnzymeCore.isvectortype(S)
        return ((ki .+ f.(vi))::S,)
    end
    function addf!!(ki::S, _ki::S, vi::S) where {S}
        @assert !EnzymeCore.isscalartype(S)
        @assert EnzymeCore.isvectortype(S)
        @assert ki === _ki
        ki .+= f.(vi)
        return (ki::S,)
    end
    RecursiveMaps.check_nonactive(T, isinactivetype)
    if !isinactivetype(T)
        newks = RecursiveMaps.recursive_map_inner(
            nothing, addf!!, (k,), (k, v), Val(false), isinactive_or_cachedtype
        )
        @assert only(newks) === k
    end
    return nothing
end

"""
    accumulate_into!(into::T, from::T)

!!! warning
    Internal function, documented for developer convenience but not covered by semver API
    stability guarantees

Recursively accumulate from `from` into `into` and zero `from`, such that `into_i += from_i`
and `from_i = 0`, where `into_i` and `from_i` are corresponding values within `into` and
`from`. In other words, this is a recursive generalization of

```julia
into .+= from
from .= 0
```

The accumulation and zeroing is only applied to differentiable values; non-differentiable
values within both `into` and `from` are left untouched.
"""
function accumulate_into!(into::T, from::T) where {T}
    # may not show in coverage but both base cases are covered via deepcopy custom rule tests
    function accumulate_into!!(into_i::S, from_i::S) where {S}
        @assert EnzymeCore.isvectortype(S)
        return ((into_i + from_i)::S, convert(S, zero(from_i))::S)
    end
    function accumulate_into!!(into_i::S, from_i::S, _into_i::S, _from_i::S) where {S}
        @assert !EnzymeCore.isscalartype(S)
        @assert EnzymeCore.isvectortype(S)
        @assert (into_i === _into_i) && (from_i === _from_i)
        into_i .+= from_i
        fill!(from_i, false)
        return (into_i::S, from_i::S)
    end
    recursive_map!(accumulate_into!!, (into, from), (into, from))
    return nothing
end
