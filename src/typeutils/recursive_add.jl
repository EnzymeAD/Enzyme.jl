using .RecursiveMaps: RecursiveMaps, recursive_map, recursive_map!, recursive_map_inner

"""
    recursive_add(x::T, y::T, f = identity, forcelhs = guaranteed_const)

Recursively construct `z::T` such that `zi = xi + f(yi)` where `zi`, `xi`, and `yi` are
corresponding values from `z`, `x`, and `y`. In other words, this is a recursive
generalization of `x .+ f.(y)`.

The function `f` must return values of the same type as its argument.

The optional argument `forcelhs` takes a function such that if `forcelhs(S) == true`, values
`zi::S` will be set to `zi = xi`. The default returns true for non-differentiable (inactive)
types, such that `zi = xi + f(yi)` applies to differentiable values, while `zi = xi` applies
to non-differentiable values. If a custom callable is passed, it is combined with the
default, as `recursive_add` is not generally capable of traversing inactive objects.
"""
function recursive_add(
        x::T, y::T, f::F = identity, forcelhs::L = guaranteed_const
    ) where {T, F, L}
    function addf(xi::S, yi::S) where {S}
        @assert EnzymeCore.isvectortype(S)
        return (xi + f(yi))::S
    end
    config = RecursiveMaps.InactiveConfig(forcelhs)
    return recursive_map(addf, (x, y), config)::T
end

"""
    accumulate_seen!(f, seen::IdDict; runtime_inactive = Val(false))
    accumulate_seen!(f, seen::IdDict, ::Val{runtime_inactive})
    accumulate_seen!(
        f, seen::IdDict, config::RecursiveMaps.InactiveConfig = RecursiveMaps.InactiveConfig()
    )

Recursively accumulate from values into keys, generalizing `key .+= f.(value)` to arbitrary
types. This accumulation is applied to each key-value pair in `seen::IdDict` where each key
is of a mutable or non-isbits vector type and the corresponding value is of the same type
and structure. Typically `seen` is populated by `make_zero`/`recursive_map`, mapping parts
of its input to the corresponding parts of the returned value.

The recursion stops at objects of types that are themselves cached by
`make_zero`/`recursive_map`, as these objects should have their own entries in `seen`. The
recursion also stops at inactive objects that would be skipped by
`make_zero`/`recursive_map`.

If the optional argument `::Val{runtime_inactive}` was passed to `make_zero`, or
`config::RecursiveMaps.InactiveConfig` was passed to `recursive_map`, the same value should
be passed to `accumulate_seen` to ensure consistency.
"""
function accumulate_seen! end

function accumulate_seen!(f::F, seen::IdDict, args::Vararg{Any, M}; kws...) where {F, M}
    accumulate_seen!(f, seen, RecursiveMaps.make_zero_config!(args...; kws...))
    return nothing
end

function accumulate_seen!(f::F, seen::IdDict, config::RecursiveMaps.InactiveConfig) where {F}
    cachedconfig = RecursiveMaps.InactiveConfig(config, RecursiveMaps.iscachedtype)
    for (k, v) in seen
        _accumulate_seen_item!(f, k, v, config, cachedconfig)
    end
    return nothing
end

function _accumulate_seen_item!(f::F, k::T, v::T, config, cachedconfig) where {F, T}
    function addf!!(ki::S, vi::S) where {S}
        @assert EnzymeCore.isvectortype(S)
        return (ki .+ f.(vi))::S
    end
    function addf!!(ki::S, _ki::S, vi::S) where {S}
        @assert !EnzymeCore.isscalartype(S)
        @assert EnzymeCore.isvectortype(S)
        @assert ki === _ki
        ki .+= f.(vi)
        return ki::S
    end
    RecursiveMaps.check_nonactive(T, config)
    if !RecursiveMaps.isinactivetype(T, config)
        newk = recursive_map_inner(nothing, addf!!, Some(k), (k, v), cachedconfig)
        @assert newk === k
    end
    return nothing
end

"""
    accumulate_into!(into::T, from::T)

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
        return (into_i + from_i)::S
    end
    function accumulate_into!!(into_i::S, _into_i::S, from_i::S) where {S}
        @assert !EnzymeCore.isscalartype(S)
        @assert EnzymeCore.isvectortype(S)
        @assert into_i === _into_i
        into_i .+= from_i
        return into_i::S
    end
    recursive_map!(accumulate_into!!, into, (into, from))
    make_zero!(from)
    return nothing
end
