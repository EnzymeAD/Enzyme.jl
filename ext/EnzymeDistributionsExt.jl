module EnzymeDistributionsExt

using Distributions: Distributions, MvNormal
using FillArrays: Fill
using PDMats: ScalMat
using Enzyme
using Enzyme: EnzymeRules
using Enzyme.EnzymeCore: Annotation

# An isotropic normal with a `Fill` mean, as `product_distribution(Fill(Normal(μ, σ), n))`
# produces. Its whole state is one mean and one variance, so it is plain bits data.
const FillIsoNormal{T} = MvNormal{T, ScalMat{T}, <:Fill{T, 1}}

# `sqmahal(d, x) = sum(abs2, x .- μ) / v` for `v = d.Σ.value`. The generic code allocates
# `x .- μ`, and differentiating it allocates its shadow too, so differentiate it directly:
#     ∂/∂xᵢ = 2(xᵢ - μ)/v,  ∂/∂μ = -2 Σᵢ(xᵢ - μ)/v,  ∂/∂v = -S/v²  with S = Σᵢ(xᵢ - μ)².

const SqmahalArg{T} = Union{
    Const{<:AbstractVector{T}},
    Duplicated{<:AbstractVector{T}},
    BatchDuplicated{<:AbstractVector{T}},
}

# Returns `S` and `Σᵢ(xᵢ - μ)`.
function _sqmahal_sums(x::AbstractVector{T}, μ::T) where {T}
    s = zero(T)
    sumdiff = zero(T)
    @inbounds @simd for i in eachindex(x)
        δ = x[i] - μ
        s += δ * δ
        sumdiff += δ
    end
    return s, sumdiff
end

# Whether `x`'s shadow `dx` should receive a derivative: with runtime activity, a shadow
# that is the primal itself marks `x` as inactive.
_sqmahal_active_shadow(config, x, dx) = !(EnzymeRules.runtime_activity(config) && dx === x)

_sqmahal_shadows(x::Const) = ()
_sqmahal_shadows(x::Duplicated) = (x.dval,)
_sqmahal_shadows(x::BatchDuplicated) = x.dval

_sqmahal_tangent(d::FillIsoNormal{T}, dμ::T, dv::T) where {T} =
    typeof(d)(Fill(dμ, axes(d.μ)), ScalMat(d.Σ.dim, dv))

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(Distributions.sqmahal)},
        ::Type{RT},
        d::Union{Const{<:FillIsoNormal{T}}, Active{<:FillIsoNormal{T}}},
        x::SqmahalArg{T},
    ) where {T <: Base.IEEEFloat, RT <: Union{Const, Active}}
    s, sumdiff = _sqmahal_sums(x.val, d.val.μ.value)
    # Only the derivative for `x` needs `x` itself in the reverse pass.
    xcopy = if !(x isa Const) && EnzymeRules.overwritten(config)[3]
        copy(x.val)
    else
        nothing
    end
    primal = EnzymeRules.needs_primal(config) ? s / d.val.Σ.value : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (xcopy, s, sumdiff))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        ::Const{typeof(Distributions.sqmahal)},
        dret::Union{Active, Type{<:Const}},
        tape,
        d::Union{Const{<:FillIsoNormal{T}}, Active{<:FillIsoNormal{T}}},
        x::SqmahalArg{T},
    ) where {T <: Base.IEEEFloat}
    W = EnzymeRules.width(config)
    if !(dret isa Active)
        return (d isa Active ? _sqmahal_zero_tangent(d.val, Val(W)) : nothing, nothing)
    end
    xcopy, s, sumdiff = tape
    xv = xcopy === nothing ? x.val : xcopy
    μ = d.val.μ.value
    v = d.val.Σ.value
    dys = W == 1 ? (dret.val,) : dret.val
    for (k, dx) in enumerate(_sqmahal_shadows(x))
        _sqmahal_active_shadow(config, x.val, dx) || continue
        c = 2 * dys[k] / v
        @inbounds @simd for i in eachindex(xv, dx)
            dx[i] += c * (xv[i] - μ)
        end
    end
    dd = if d isa Active
        tangents = ntuple(
            k -> _sqmahal_tangent(d.val, -2 * dys[k] * sumdiff / v, -dys[k] * s / v^2), Val(W)
        )
        W == 1 ? tangents[1] : tangents
    else
        nothing
    end
    return (dd, nothing)
end

function _sqmahal_zero_tangent(d::FillIsoNormal{T}, ::Val{W}) where {T, W}
    z = _sqmahal_tangent(d, zero(T), zero(T))
    return W == 1 ? z : ntuple(_ -> z, Val(W))
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        ::Const{typeof(Distributions.sqmahal)},
        ::Type{RT},
        d::Union{
            Const{<:FillIsoNormal{T}},
            Duplicated{<:FillIsoNormal{T}},
            BatchDuplicated{<:FillIsoNormal{T}},
        },
        x::SqmahalArg{T},
    ) where {T <: Base.IEEEFloat, RT}
    W = EnzymeRules.width(config)
    xv = x.val
    μ = d.val.μ.value
    v = d.val.Σ.value
    s, _ = _sqmahal_sums(xv, μ)
    y = s / v
    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? y : nothing
    end
    dds = d isa Const ? () : d isa Duplicated ? (d.dval,) : d.dval
    xshadows = _sqmahal_shadows(x)
    dys = ntuple(Val(W)) do k
        dμ = d isa Const ? zero(T) : dds[k].μ.value
        dv = d isa Const ? zero(T) : dds[k].Σ.value
        dx = x isa Const ? nothing : xshadows[k]
        acc = zero(T)
        if dx !== nothing && _sqmahal_active_shadow(config, xv, dx)
            @inbounds @simd for i in eachindex(xv, dx)
                acc += (xv[i] - μ) * (dx[i] - dμ)
            end
        else
            _, sumdiff = _sqmahal_sums(xv, μ)
            acc = -sumdiff * dμ
        end
        2 * acc / v - s * dv / v^2
    end
    if W == 1
        return EnzymeRules.needs_primal(config) ? Duplicated(y, dys[1]) : dys[1]
    else
        return EnzymeRules.needs_primal(config) ? BatchDuplicated(y, dys) : dys
    end
end

end
