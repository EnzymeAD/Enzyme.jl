function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return xs
    elseif EnzymeRules.needs_shadow(config)
        return xs.dval
    elseif EnzymeRules.needs_primal(config)
        return xs.val
    else
        return nothing
    end
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,BatchDuplicatedNoNeed,BatchDuplicated}},
    xs::BatchDuplicated{T,N};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat},N}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    for i = 1:N
        xs.dval[i] .= xs.dval[i][inds]
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return xs
    elseif EnzymeRules.needs_shadow(config)
        return xs.dval
    elseif EnzymeRules.needs_primal(config)
        return xs.val
    else
        return nothing
    end
end


function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if EnzymeRules.needs_primal(config)
        primal = xs.val
    else
        primal = nothing
    end
    if RT <: Const
        shadow = nothing
    else
        shadow = xs.dval
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, inds)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(sort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    tape,
    xs::Duplicated{T};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = tape
    back_inds = sortperm(inds)
    xs.dval .= xs.dval[back_inds]
    return (nothing,)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(partialsort!)},
    RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if kv isa Integer
            return Duplicated(xs.val[kv], xs.dval[kv])
        else
            return Duplicated(view(xs.val, kv), view(xs.dval, kv))
        end
    elseif EnzymeRules.needs_shadow(config)
        return kv isa Integer ? xs.dval[kv] : view(xs.dval, kv)
    elseif EnzymeRules.needs_primal(config)
        return kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    else
        return nothing
    end
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    ::Const{typeof(partialsort!)},
    RT::Type{<:Union{Const,BatchDuplicatedNoNeed,BatchDuplicated}},
    xs::BatchDuplicated{T,N},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat},N}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    for i = 1:N
        xs.dval[i] .= xs.dval[i][inds]
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if kv isa Integer
            return BatchDuplicated(xs.val[kv], ntuple(i -> xs.dval[i][kv], N))
        else
            return BatchDuplicated(view(xs.val, kv), ntuple(i -> view(xs.dval[i], kv), N))
        end
    elseif EnzymeRules.needs_shadow(config)
        if kv isa Integer
            return ntuple(i -> xs.dval[i][kv], N)
        else
            return ntuple(i -> view(xs.dval[i], kv), N)
        end
    elseif EnzymeRules.needs_primal(config)
        return kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(partialsort!)},
    RT::Type{<:Union{Const,Active,DuplicatedNoNeed,Duplicated}},
    xs::Duplicated{T},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    xs.val .= xs.val[inds]
    xs.dval .= xs.dval[inds]
    if EnzymeRules.needs_primal(config)
        primal = kv isa Integer ? xs.val[kv] : view(xs.val, kv)
    else
        primal = nothing
    end
    if RT <: Const || RT <: Active
        shadow = nothing
    else
        shadow = kv isa Integer ? xs.dval[kv] : view(xs.dval, kv)
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, inds)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(partialsort!)},
    dret::Union{Active,Type{<:Union{Const,Active,DuplicatedNoNeed,Duplicated}}},
    tape,
    xs::Duplicated{T},
    k::Const{<:Union{Integer,OrdinalRange}};
    kwargs...,
) where {T<:AbstractArray{<:AbstractFloat}}
    inds = tape
    kv = k.val
    if dret isa Active
        if kv isa Integer
            xs.dval[kv] += dret.val
        else
            xs.dval[kv] .+= dret.val
        end
    end
    back_inds = sortperm(inds)
    xs.dval .= xs.dval[back_inds]
    return (nothing, nothing)
end

