function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        Ty::Const{Type{BigFloat}},
        RT::Type{<:Union{DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed}};
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
        RT::Type{<:Union{DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed}},
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
        RT::Type{<:Union{DuplicatedNoNeed, Duplicated, BatchDuplicated, BatchDuplicatedNoNeed}},
        tape,
        kwargs...,
    )
    return ()
end

EnzymeRules.@easy_rule(+(a::BigFloat, b::Number), (1, 1))
EnzymeRules.@easy_rule(+(a::Number, b::BigFloat), (1, 1))
EnzymeRules.@easy_rule(+(a::BigFloat, b::BigFloat), (1, 1))
EnzymeRules.@easy_rule(-(a::BigFloat, b::Number), (1, -1))
EnzymeRules.@easy_rule(-(a::Number, b::BigFloat), (1, -1))
EnzymeRules.@easy_rule(-(a::BigFloat, b::BigFloat), (1, -1))
EnzymeRules.@easy_rule(*(a::BigFloat, b::BigFloat), (b, a))
EnzymeRules.@easy_rule(*(a::BigFloat, b::Number), (b, a))
EnzymeRules.@easy_rule(*(a::Number, b::BigFloat), (b, a))
EnzymeRules.@easy_rule(/(a::BigFloat, b::Number), (one(a) / b, -(a / b^2)))
EnzymeRules.@easy_rule(/(a::Number, b::BigFloat), (one(a) / b, -(a / b^2)))
EnzymeRules.@easy_rule(/(a::BigFloat, b::BigFloat), (one(a) / b, -(a / b^2)))
EnzymeRules.@easy_rule(Base.inv(a::BigFloat), (-(one(a) / a^2),))
EnzymeRules.@easy_rule(Base.sin(a::BigFloat), (cos(a),))
EnzymeRules.@easy_rule(Base.cos(a::BigFloat), (-sin(a),))
EnzymeRules.@easy_rule(Base.tan(a::BigFloat), (one(a) + Ω^2,))
