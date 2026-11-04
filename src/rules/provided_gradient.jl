"""
    WithGradient(primal, value_and_gradient)

Wrap a real scalar-valued function with an explicitly supplied gradient.
`f = WithGradient(primal, value_and_gradient)` calls `primal(x)` normally.
When differentiating `Const(f)`, Enzyme calls `value_and_gradient(x)`, which
must return `(primal(x), gradient)`. The input is a floating-point scalar or
an array of floating-point values; its gradient is respectively a real scalar
or a real array with the same axes.

Both callables must leave `x` unchanged and agree on the value. Enzyme treats
their captured data as constant. Forward mode contracts the gradient with
each input tangent; Reverse mode accumulates the output seed times the
gradient. Reverse mode saves a copy of an array gradient, so a provider may
reuse its output buffer between calls. Work and storage are linear in the
input size (and work is linear in the batch width).

This can supply derivatives for a function implemented by an opaque native
library. Higher derivatives additionally require differentiable provider
code or derivative rules for that provider.

```julia
square(x) = x * x
square_with_gradient(x) = (square(x), 2x)
f = WithGradient(square, square_with_gradient)
autodiff(Reverse, Const(f), Active, Active(3.0)) # ((6.0,),)
```
"""
struct WithGradient{F, G}
    primal::F
    value_and_gradient::G
end

const ProvidedGradientInput = Union{AbstractFloat, AbstractArray{<:AbstractFloat}}

@inline (f::WithGradient)(x::ProvidedGradientInput) = f.primal(x)

_provided_gradient_check(x::AbstractFloat, value::AbstractFloat, gradient::Real) = nothing
function _provided_gradient_check(
        x::AbstractArray, value::AbstractFloat, gradient::AbstractArray{<:Real}
    )
    axes(x) == axes(gradient) || throw(DimensionMismatch("WithGradient: gradient axes must match the input"))
    return nothing
end
function _provided_gradient_check(x, value, gradient)
    throw(ArgumentError("WithGradient: expected a floating-point value and a real gradient matching the input"))
end

_provided_gradient_copy(gradient::Real) = gradient
_provided_gradient_copy(gradient::AbstractArray) = copy(gradient)

@inline function _provided_gradient_product(gradient, seed, ::Val{StrongZero}) where {StrongZero}
    return StrongZero && iszero(seed) ? zero(gradient) * seed : gradient * seed
end

_provided_gradient_inactive(x, dx, runtime_activity) = false
_provided_gradient_inactive(x::AbstractArray, dx::AbstractArray, runtime_activity) = runtime_activity && x === dx

struct ProvidedGradientTangent{G, T, S, X}
    gradient::G
    strong_zero::S
    x::X
    runtime_activity::Bool
end
function (t::ProvidedGradientTangent{G, T})(dx::Real) where {G, T}
    return T(_provided_gradient_product(t.gradient, dx, t.strong_zero))
end
function (t::ProvidedGradientTangent{G, T})(dx::AbstractArray) where {G, T}
    _provided_gradient_inactive(t.x, dx, t.runtime_activity) && return zero(T)
    result = zero(T)
    for i in eachindex(t.gradient, dx)
        result += _provided_gradient_product(t.gradient[i], dx[i], t.strong_zero)
    end
    return T(result)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, f::Const{<:WithGradient}, ::Type{RT},
        x::Annotation{<:ProvidedGradientInput}
    ) where {RT}
    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? f.val(x.val) : nothing
    end
    value, gradient = f.val.value_and_gradient(x.val)
    _provided_gradient_check(x.val, value, gradient)
    sz = Val(EnzymeRules.strong_zero(config))
    tangent = ProvidedGradientTangent{typeof(gradient), typeof(value), typeof(sz), typeof(x.val)}(
        gradient, sz, x.val, EnzymeRules.runtime_activity(config)
    )
    if EnzymeRules.width(config) == 1
        shadow = x isa Const ? zero(value) : tangent(x.dval)
        return EnzymeRules.needs_primal(config) ? Duplicated(value, shadow) : shadow
    else
        shadow = x isa Const ? ntuple(Returns(zero(value)), Val(EnzymeRules.width(config))) : map(tangent, x.dval)
        return EnzymeRules.needs_primal(config) ? BatchDuplicated(value, shadow) : shadow
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, f::Const{<:WithGradient}, ::Type{RT},
        x::Annotation{<:ProvidedGradientInput}
    ) where {RT}
    if RT <: Const || x isa Const
        value = EnzymeRules.needs_primal(config) ? f.val(x.val) : nothing
        return EnzymeRules.AugmentedReturn(value, nothing, nothing)
    end
    value, gradient = f.val.value_and_gradient(x.val)
    _provided_gradient_check(x.val, value, gradient)
    tape = _provided_gradient_copy(gradient)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? value : nothing, nothing, tape)
end

function _provided_gradient_accumulate!(dx, gradient, seed, sz)
    for i in eachindex(dx, gradient)
        dx[i] += _provided_gradient_product(gradient[i], seed, sz)
    end
    return nothing
end

struct ProvidedGradientCotangent{G, T, S}
    gradient::G
    strong_zero::S
end
function (c::ProvidedGradientCotangent{G, T})(seed::Active) where {G, T}
    return T(_provided_gradient_product(c.gradient, seed.val, c.strong_zero))
end

function _provided_gradient_reverse(config, x::Active{T}, gradient, seed) where {T}
    sz = Val(EnzymeRules.strong_zero(config))
    cotangent = ProvidedGradientCotangent{typeof(gradient), T, typeof(sz)}(gradient, sz)
    return EnzymeRules.width(config) == 1 ? cotangent(seed) : map(cotangent, seed)
end
function _provided_gradient_reverse(config, x::Duplicated, gradient, seed)
    if !_provided_gradient_inactive(x.val, x.dval, EnzymeRules.runtime_activity(config))
        _provided_gradient_accumulate!(x.dval, gradient, seed.val, Val(EnzymeRules.strong_zero(config)))
    end
    return nothing
end
function _provided_gradient_reverse(config, x::BatchDuplicated, gradient, seed)
    for lane in 1:EnzymeRules.width(config)
        if !_provided_gradient_inactive(x.val, x.dval[lane], EnzymeRules.runtime_activity(config))
            _provided_gradient_accumulate!(x.dval[lane], gradient, seed[lane].val, Val(EnzymeRules.strong_zero(config)))
        end
    end
    return nothing
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, f::Const{<:WithGradient}, seed, tape,
        x::Annotation{<:ProvidedGradientInput}
    )
    if seed isa Type{<:Const} || x isa Const
        return (nothing,)
    end
    return (_provided_gradient_reverse(config, x, tape, seed),)
end
