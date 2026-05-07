using Random

function EnzymeRules.inactive(
    ::typeof(Random.rand!),
    ::Random.AbstractRNG,
    ::Random.Sampler,
    ::AbstractArray,
)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.randn!), ::Random.AbstractRNG, ::AbstractArray)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.default_rng), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.seed!), args...)
    return nothing
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    Ty::Const{typeof(Random.rand!)},
    RT::Type,
    rng::Annotation{rngty},
    dst::Annotation{<:Array{FT}},
    smpl::Annotation{<:Random.SamplerTrivial{Random.CloseOpen01{FT}}},
) where {rngty<:Union{TaskLocalRNG,Xoshiro},FT<:Union{Float32,Float64}}
    Ty.val(rng.val, dst.val, smpl.val)

    if !(dst isa Const)
        if EnzymeRules.width(config) == 1
            fill!(dst.dval, 0)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                fill!(dst.dval[i], 0)
                nothing
            end
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        dst
    elseif EnzymeRules.needs_shadow(config)
        dst.dval
    elseif EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.rand!)},
    RT::Type,
    rng::Annotation{rngty},
    dst::Annotation{<:Array{FT}},
    smpl::Annotation{<:Random.SamplerTrivial{Random.CloseOpen01{FT}}},
) where {rngty<:Union{TaskLocalRNG,Xoshiro},FT<:Union{Float32,Float64}}
    Ty.val(rng.val, dst.val, smpl.val)
    if RT <: Duplicated || RT <: DuplicatedNoNeed
        fill!(dst.dval, 0)
        dst.dval
    elseif RT <: BatchDuplicated || RT <: BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            fill!(dst.dval[i], 0)
            nothing
        end
    end
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? dst.val : nothing,
        EnzymeRules.needs_shadow(config) ? dst.dval : nothing,
        nothing,
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.rand!)},
    RT::Type,
    tape,
    rng::Annotation{rngty},
    dst::Annotation{<:Array{FT}},
    smpl::Annotation{<:Random.SamplerTrivial{Random.CloseOpen01{FT}}},
) where {rngty<:Union{TaskLocalRNG,Xoshiro},FT<:Union{Float32,Float64}}
    return (nothing, nothing, nothing)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    Ty::Const{typeof(Random.randn!)},
    RT::Type,
    rng::Annotation{<:Random.AbstractRNG},
    dst::Annotation{<:AbstractArray})

    Ty.val(rng.val, dst.val)

    if !(dst isa Const)
        if EnzymeRules.width(config) == 1
            make_zero!(dst.dval)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                make_zero!(dst.dval[i])
                nothing
            end
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        dst
    elseif EnzymeRules.needs_shadow(config)
        dst.dval
    elseif EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.randn!)},
    RT::Type,
    rng::Annotation{<:Random.AbstractRNG},
    dst::Annotation{<:AbstractArray}
)
    Ty.val(rng.val, dst.val)
    if RT <: Duplicated || RT <: DuplicatedNoNeed
        make_zero!(dst.dval)
        dst.dval
    elseif RT <: BatchDuplicated || RT <: BatchDuplicatedNoNeed
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            make_zero!(dst.dval[i])
            nothing
        end
    end
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? dst.val : nothing,
        EnzymeRules.needs_shadow(config) ? dst.dval : nothing,
        nothing,
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    Ty::Const{typeof(Random.randn!)},
    RT::Type,
    tape,
    rng::Annotation{<:Random.AbstractRNG},
    dst::Annotation{<:AbstractArray})
    return (nothing, nothing)
end


