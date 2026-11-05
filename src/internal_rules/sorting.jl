function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(sort!)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    copyto!(xs.val, xs.val[inds])
    copyto!(xs.dval, xs.dval[inds])
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
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::BatchDuplicated{T, N};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    inds = sortperm(xs.val; kwargs...)
    copyto!(xs.val, xs.val[inds])
    for i in 1:N
        copyto!(xs.dval[i], xs.dval[i][inds])
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
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    copyto!(xs.val, xs.val[inds])
    copyto!(xs.dval, xs.dval[inds])
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
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        tape,
        xs::Duplicated{T};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = tape
    copyto!(view(xs.dval, inds), copy(xs.dval))
    return (nothing,)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{N},
        ::Const{typeof(sort!)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::BatchDuplicated{T, N};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    inds = sortperm(xs.val; kwargs...)
    copyto!(xs.val, xs.val[inds])
    for i in 1:N
        copyto!(xs.dval[i], xs.dval[i][inds])
    end
    primal = EnzymeRules.needs_primal(config) ? xs.val : nothing
    shadow = RT <: Const ? nothing : xs.dval
    return EnzymeRules.AugmentedReturn(primal, shadow, inds)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{N},
        ::Const{typeof(sort!)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        tape,
        xs::BatchDuplicated{T, N};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    inds = tape
    for i in 1:N
        copyto!(view(xs.dval[i], inds), copy(xs.dval[i]))
    end
    return (nothing,)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::Const{T};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    y = sort(xs.val; kwargs...)
    W = EnzymeRules.width(config)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return W == 1 ? Duplicated(y, zero(y)) : BatchDuplicated(y, ntuple(_ -> zero(y), Val(W)))
    elseif EnzymeRules.needs_shadow(config)
        return W == 1 ? zero(y) : ntuple(_ -> zero(y), Val(W))
    elseif EnzymeRules.needs_primal(config)
        return y
    else
        return nothing
    end
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(xs.val[inds], xs.dval[inds])
    elseif EnzymeRules.needs_shadow(config)
        return xs.dval[inds]
    elseif EnzymeRules.needs_primal(config)
        return xs.val[inds]
    else
        return nothing
    end
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::BatchDuplicated{T, N};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    inds = sortperm(xs.val; kwargs...)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return BatchDuplicated(xs.val[inds], ntuple(i -> xs.dval[i][inds], Val(N)))
    elseif EnzymeRules.needs_shadow(config)
        return ntuple(i -> xs.dval[i][inds], Val(N))
    elseif EnzymeRules.needs_primal(config)
        return xs.val[inds]
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Union{Const{T}, Duplicated{T}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = sortperm(xs.val; kwargs...)
    y = xs.val[inds]
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = RT <: Const ? nothing : zero(y)
    tape = (RT <: Const || xs isa Const) ? nothing : (inds, shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        tape,
        xs::Union{Const{T}, Duplicated{T}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    if !(RT <: Const) && xs isa Duplicated
        inds, shadow = tape
        view(xs.dval, inds) .+= shadow
        make_zero!(shadow)
    end
    return (nothing,)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{N},
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::Union{Const{T}, BatchDuplicated{T, N}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    inds = sortperm(xs.val; kwargs...)
    y = xs.val[inds]
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    shadow = RT <: Const ? nothing : ntuple(_ -> zero(y), Val(N))
    tape = (RT <: Const || xs isa Const) ? nothing : (inds, shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{N},
        ::Const{typeof(sort)},
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        tape,
        xs::Union{Const{T}, BatchDuplicated{T, N}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    if !(RT <: Const) && xs isa BatchDuplicated
        inds, shadow = tape
        for i in 1:N
            view(xs.dval[i], inds) .+= shadow[i]
            make_zero!(shadow[i])
        end
    end
    return (nothing,)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(partialsort!)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    copyto!(xs.val, xs.val[inds])
    copyto!(xs.dval, xs.dval[inds])

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
        RT::Type{<:Union{Const, BatchDuplicatedNoNeed, BatchDuplicated}},
        xs::BatchDuplicated{T, N},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}, N}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    copyto!(xs.val, xs.val[inds])
    for i in 1:N
        copyto!(xs.dval[i], xs.dval[i][inds])
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
        RT::Type{<:Union{Const, Active, DuplicatedNoNeed, Duplicated}},
        xs::Duplicated{T},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    kv = k.val
    inds = collect(eachindex(xs.val))
    partialsortperm!(inds, xs.val, kv; kwargs...)
    copyto!(xs.val, xs.val[inds])
    copyto!(xs.dval, xs.dval[inds])
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
        dret::Union{Active, Type{<:Union{Const, Active, DuplicatedNoNeed, Duplicated}}},
        tape,
        xs::Duplicated{T},
        k::Const{<:Union{Integer, OrdinalRange}};
        kwargs...,
    ) where {T <: AbstractArray{<:AbstractFloat}}
    inds = tape
    kv = k.val
    if dret isa Active
        if kv isa Integer
            xs.dval[kv] += dret.val
        else
            xs.dval[kv] .+= dret.val
        end
    end
    copyto!(view(xs.dval, inds), copy(xs.dval))
    return (nothing, nothing)
end
