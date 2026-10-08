function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    Ty::Const{Type{BigFloat}},
    RT::Type{<:Union{DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed}};
    kwargs...,
)

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return remove_innerty(RT)(Ty.val(; kwargs...), Ty.val(; kwargs...))
        else
            tup = ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Ty.val(; kwargs...)
            end
            return remove_innerty(RT)(Ty.val(; kwargs...), tup)
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Ty.val(; kwargs...)
        else
            return ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Ty.val(; kwargs...)
            end
        end
    elseif EnzymeRules.needs_primal(config)
        return Ty.val(; kwargs...)
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    Ty::Const{Type{BigFloat}},
    RT::Type{<:Union{DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed}};
    kwargs...,
)
    primal = if EnzymeRules.needs_primal(config)
        Ty.val(; kwargs...)
    else
        nothing
    end
    shadow = if RT <: Const
        shadow = nothing
    else
        if EnzymeRules.width(config) == 1
            Ty.val(; kwargs...)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                Ty.val(; kwargs...)
            end
        end
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    Ty::Const{Type{BigFloat}},
    RT::Type{<:Union{DuplicatedNoNeed,Duplicated,BatchDuplicated,BatchDuplicatedNoNeed}},
    tape;
    kwargs...,
)
    return ()
end

# `BigFloat(x::Integer)` lowers to `mpfr_set_si` / `mpfr_set_ui` (for `Clong` / `Culong`)
# or `mpfr_set_z` (via `BigInt`), for which there is no derivative, so any active use
# of an integer-constructed BigFloat otherwise fails. Note `Clong === Int32` on Windows,
# so `BigFloat(::Int64)` takes the `BigInt` path there.
_bigfloat_rounding_val(r::Const{Base.MPFR.MPFRRoundingMode}) = r.val

# Callable struct constructing `BigFloat(x, rs...; kwargs...)`; each call allocates a
# fresh BigFloat, so batched shadows built with `ntuple` don't alias one another.
struct BigFloatIntCtor{T <: Integer, R <: Tuple, K}
    x::T
    rs::R
    kwargs::K
end
@inline (f::BigFloatIntCtor)(_...) = BigFloat(f.x, f.rs...; f.kwargs...)

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        Ty::Const{Type{BigFloat}},
        RT::Type{<:Union{DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed}},
        x::Const{Ti},
        rs::Const{Base.MPFR.MPFRRoundingMode}...;
        kwargs...,
    ) where {Ti <: Integer}
    rvals = map(_bigfloat_rounding_val, rs)
    primal = BigFloatIntCtor(x.val, rvals, kwargs)
    shadow = BigFloatIntCtor(zero(Ti), rvals, kwargs)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return remove_innerty(RT)(primal(), shadow())
        else
            return BatchDuplicated(primal(), ntuple(shadow, Val(EnzymeRules.width(config))))
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return shadow()
        else
            return ntuple(shadow, Val(EnzymeRules.width(config)))
        end
    elseif EnzymeRules.needs_primal(config)
        return primal()
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        Ty::Const{Type{BigFloat}},
        RT::Type{<:Union{DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed}},
        x::Const{Ti},
        rs::Const{Base.MPFR.MPFRRoundingMode}...;
        kwargs...,
    ) where {Ti <: Integer}
    rvals = map(_bigfloat_rounding_val, rs)
    primal = if EnzymeRules.needs_primal(config)
        BigFloatIntCtor(x.val, rvals, kwargs)()
    else
        nothing
    end
    shadow = if RT <: Const
        nothing
    elseif EnzymeRules.width(config) == 1
        BigFloatIntCtor(zero(Ti), rvals, kwargs)()
    else
        ntuple(BigFloatIntCtor(zero(Ti), rvals, kwargs), Val(EnzymeRules.width(config)))
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        Ty::Const{Type{BigFloat}},
        RT::Type{<:Union{DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed}},
        tape,
        x::Const{<:Integer},
        rs::Const{Base.MPFR.MPFRRoundingMode}...;
        kwargs...,
    )
    # the integer argument carries no derivative
    return ntuple(Returns(nothing), Val(1 + length(rs)))
end

EnzymeRules.@easy_rule(+(a::BigFloat, b::Number), (1,1))
EnzymeRules.@easy_rule(+(a::Number, b::BigFloat), (1,1))
EnzymeRules.@easy_rule(+(a::BigFloat, b::BigFloat), (1,1))
EnzymeRules.@easy_rule(-(a::BigFloat, b::Number), (1,-1))
EnzymeRules.@easy_rule(-(a::Number, b::BigFloat), (1,-1))
EnzymeRules.@easy_rule(-(a::BigFloat, b::BigFloat), (1,-1))
EnzymeRules.@easy_rule(-(a::BigFloat), (-1,))
EnzymeRules.@easy_rule(*(a::BigFloat, b::BigFloat), (b, a))
EnzymeRules.@easy_rule(*(a::BigFloat, b::Number), (b, a))
EnzymeRules.@easy_rule(*(a::Number, b::BigFloat), (b, a))
EnzymeRules.@easy_rule(/(a::BigFloat, b::Number), (one(a)/b, -(a/b^2)))
EnzymeRules.@easy_rule(/(a::Number, b::BigFloat), (one(a)/b, -(a/b^2)))
EnzymeRules.@easy_rule(/(a::BigFloat, b::BigFloat), (one(a)/b, -(a/b^2)))
EnzymeRules.@easy_rule(Base.inv(a::BigFloat), (-(one(a)/a^2),))
EnzymeRules.@easy_rule(Base.sin(a::BigFloat), (cos(a),))
EnzymeRules.@easy_rule(Base.cos(a::BigFloat), (-sin(a),))
EnzymeRules.@easy_rule(Base.tan(a::BigFloat), (one(a) + Ω^2,))
EnzymeRules.@easy_rule(Base.zero(a::Type{BigFloat}), (0,))
