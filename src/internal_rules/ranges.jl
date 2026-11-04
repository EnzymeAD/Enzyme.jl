# Ranges
# Float64 ranges in Julia use bitwise `&` with higher precision
# to correct for numerical error, thus we put rules over the
# operations as this is not directly differentiable.
# Exactly-rational floats additionally build the range from integers
# (`round(Int, ...)`), which loses the derivative entirely.
#
# Element `i` of a `StepRangeLen` is `ref + (i - offset) * step`, where Base
# places `offset` at the element of smallest magnitude. The derivative of a
# range is a range of the same type whose `ref` and `step` hold the
# derivatives of those fields for the same `offset`. The rules below
# differentiate the first element and the step of the range, and convert
# from (or to) `ref` using the `offset` of the primal range.

# The first element and step of the range of length `len` built by `f`,
# as a linear function of the floating-point arguments of `f`
_range_first_step(::Colon, len, start, step, stop) = (start, step)
_range_first_step(::typeof(Base.range_start_step_length), len, start, step) = (start, step)
function _range_first_step(::typeof(Base.range_start_stop_length), len, start, stop)
    return (start, len > 1 ? (stop - start) / (len - 1) : zero(stop - start))
end

# The adjoint of `_range_first_step`
_range_first_step_adjoint(::Colon, len, dfirst, dstep) = (dfirst, dstep, zero(dstep))
_range_first_step_adjoint(::typeof(Base.range_start_step_length), len, dfirst, dstep) = (dfirst, dstep)
function _range_first_step_adjoint(::typeof(Base.range_start_stop_length), len, dfirst, dstep)
    dstop = len > 1 ? dstep / (len - 1) : zero(dstep)
    return (dfirst - dstop, dstop)
end

# A `ref` or `step` field of type `R` holding the derivative `x`
_range_field(::Type{Base.TwicePrecision{T}}, x) where {T} = Base.TwicePrecision{T}(T(x), zero(T))
_range_field(::Type{T}, x) where {T <: Number} = T(x)

# A `TwicePrecision` field holds the value `hi + lo`, so the derivative with
# respect to the value is the derivative with respect to either part
_range_field_derivative(x::Base.TwicePrecision) = x.hi
_range_field_derivative(x::Number) = x

# The derivative of `r` for derivatives `dfirst` of its first element and `dstep` of its step
function _range_tangent(r::StepRangeLen{T, R, S, L}, dfirst, dstep) where {T, R, S, L}
    dref = dfirst + (r.offset - 1) * dstep
    return StepRangeLen{T, R, S, L}(_range_field(R, dref), _range_field(S, dstep), r.len, r.offset)
end

# The derivatives of the first element and the step of `r` from the derivative `dr` of `r`
function _range_first_step_cotangent(r::StepRangeLen, dr::StepRangeLen)
    dref = _range_field_derivative(dr.ref)
    return (dref, _range_field_derivative(dr.step) + (r.offset - 1) * dref)
end

# Maps the derivatives of the floating-point arguments of `f` to the derivative of `ret = f(...)`
struct RangeTangent{F, R}
    f::F
    ret::R
end
function (t::RangeTangent)(dargs::Vararg{Any, N}) where {N}
    return _range_tangent(t.ret, _range_first_step(t.f, length(t.ret), dargs...)...)
end

# Maps the derivative of `ret = f(...)` to the derivatives of the floating-point arguments of `f`
struct RangeCotangent{F, R}
    f::F
    ret::R
end
function (t::RangeCotangent)(dret::Active{<:StepRangeLen})
    return _range_first_step_adjoint(t.f, length(t.ret), _range_first_step_cotangent(t.ret, dret.val)...)
end

_range_dval(x::Const) = zero(x.val)
_range_dval(x) = x.dval
_range_dvals(x::Const, w::Val{W}) where {W} = ntuple(Returns(zero(x.val)), w)
_range_dvals(x, ::Val{W}) where {W} = x.dval

function _range_forward(config::EnzymeRules.FwdConfig, f, ret, args::Vararg{Annotation, N}) where {N}
    if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = RangeTangent(f, ret)(map(_range_dval, args)...)
            return EnzymeRules.needs_primal(config) ? Duplicated(ret, shadow) : shadow
        else
            w = Val(EnzymeRules.width(config))
            shadow = map(RangeTangent(f, ret), map(Base.Fix2(_range_dvals, w), args)...)
            return EnzymeRules.needs_primal(config) ? BatchDuplicated(ret, shadow) : shadow
        end
    elseif EnzymeRules.needs_primal(config)
        return ret
    else
        return nothing
    end
end

_range_derivative(::Const, d) = nothing
_range_derivative(::Active{T}, d) where {T} = T(d)
_range_derivative(::Active{T}, d::Tuple) where {T} = map(T, d)

_range_zero_derivative(::Const, ::Val{W}) where {W} = nothing
_range_zero_derivative(::Active{T}, ::Val{1}) where {T} = zero(T)
_range_zero_derivative(::Active{T}, w::Val{W}) where {T, W} = ntuple(Returns(zero(T)), w)

function _range_reverse(config::EnzymeRules.RevConfig, f, ret, dret::Active, args::Vararg{Annotation, N}) where {N}
    return map(_range_derivative, args, RangeCotangent(f, ret)(dret))
end

function _range_reverse(config::EnzymeRules.RevConfig, f, ret, dret::NTuple{W, Active}, args::Vararg{Annotation, N}) where {W, N}
    return map(_range_derivative, args, map(tuple, map(RangeCotangent(f, ret), dret)...))
end

function _range_reverse(config::EnzymeRules.RevConfig, f, ret, ::Type, args::Vararg{Annotation, N}) where {N}
    return map(Base.Fix2(_range_zero_derivative, Val(EnzymeRules.width(config))), args)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{Colon},
        RT::Type{
            <:Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed},
        },
        start::Annotation{<:AbstractFloat},
        step::Annotation{<:AbstractFloat},
        stop::Annotation{<:AbstractFloat},
    )
    ret = func.val(start.val, step.val, stop.val)
    return _range_forward(config, func.val, ret, start, step, stop)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{Colon},
        ::Type{RT},
        start::Annotation{<:AbstractFloat},
        step::Annotation{<:AbstractFloat},
        stop::Annotation{<:AbstractFloat},
    ) where {RT <: Union{Active, Const}}

    if EnzymeRules.needs_primal(config)
        primal = func.val(start.val, step.val, stop.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        Nothing,
        Nothing,
    }(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{Colon},
        dret,
        tape::Nothing,
        start::Annotation{<:AbstractFloat},
        step::Annotation{<:AbstractFloat},
        stop::Annotation{<:AbstractFloat},
    )
    ret = func.val(start.val, step.val, stop.val)
    return _range_reverse(config, func.val, ret, dret, start, step, stop)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.range_start_step_length)},
        RT,
        start::Annotation{T},
        step::Annotation{T},
        len::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    ret = func.val(start.val, step.val, len.val)
    return _range_forward(config, func.val, ret, start, step)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.range_start_step_length)},
        ::Type{RT},
        start::Annotation{T},
        step::Annotation{T},
        len::Annotation{<:Integer},
    ) where {RT <: Union{Active, Const}, T <: Base.IEEEFloat}
    if EnzymeRules.needs_primal(config)
        primal = func.val(start.val, step.val, len.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.range_start_step_length)},
        dret,
        tape,
        start::Annotation{T},
        step::Annotation{T},
        len::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    ret = func.val(start.val, step.val, len.val)
    return (_range_reverse(config, func.val, ret, dret, start, step)..., nothing)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        RT,
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    ret = func.val(start.val, stop.val, len.val)
    return _range_forward(config, func.val, ret, start, stop)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        ::Type{RT},
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{<:Base.Integer},
    ) where {RT <: Union{Active, Const}, T <: Base.IEEEFloat}
    if EnzymeRules.needs_primal(config)
        primal = func.val(start.val, stop.val, len.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.range_start_stop_length)},
        dret,
        tape,
        start::Annotation{T},
        stop::Annotation{T},
        len::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    ret = func.val(start.val, stop.val, len.val)
    return (_range_reverse(config, func.val, ret, dret, start, stop)..., nothing)
end

# `Base.truncbits` clears the low bits of a float so that twice-precision
# arithmetic (e.g. scaling or summing a `Float64` range) can split a value
# into exactly representable parts whose sum is the value itself. Its
# derivative is therefore the identity.
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.truncbits)},
        RT,
        x::Annotation{T},
        nb::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    ret = func.val(x.val, nb.val)
    if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            shadow = _range_dval(x)
            return EnzymeRules.needs_primal(config) ? Duplicated(ret, shadow) : shadow
        else
            shadow = _range_dvals(x, Val(EnzymeRules.width(config)))
            return EnzymeRules.needs_primal(config) ? BatchDuplicated(ret, shadow) : shadow
        end
    elseif EnzymeRules.needs_primal(config)
        return ret
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.truncbits)},
        ::Type{RT},
        x::Annotation{T},
        nb::Annotation{<:Integer},
    ) where {RT <: Union{Active, Const}, T <: Base.IEEEFloat}
    if EnzymeRules.needs_primal(config)
        primal = func.val(x.val, nb.val)
    else
        primal = nothing
    end
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

_active_val(x::Active) = x.val

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.truncbits)},
        dret::Active,
        tape,
        x::Annotation{T},
        nb::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    return (_range_derivative(x, dret.val), nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.truncbits)},
        dret::NTuple{W, Active},
        tape,
        x::Annotation{T},
        nb::Annotation{<:Integer},
    ) where {W, T <: Base.IEEEFloat}
    return (_range_derivative(x, map(_active_val, dret)), nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.truncbits)},
        dret::Type,
        tape,
        x::Annotation{T},
        nb::Annotation{<:Integer},
    ) where {T <: Base.IEEEFloat}
    return (_range_zero_derivative(x, Val(EnzymeRules.width(config))), nothing)
end
