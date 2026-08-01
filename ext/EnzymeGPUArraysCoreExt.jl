module EnzymeGPUArraysCoreExt

using GPUArraysCore
using GPUArraysCore: AnyGPUArray
using Enzyme
using Enzyme: EnzymeRules
using LinearAlgebra: mul!

function Enzyme.zerosetfn(x::AbstractGPUArray, i::Int)
    res = zero(x)
    @allowscalar @inbounds res[i] = 1
    return res
end

function Enzyme.zerosetfn!(x::AbstractGPUArray, i::Int, val)
    @allowscalar @inbounds x[i] += val
    return
end

@inline function Enzyme.onehot(x::AbstractGPUArray)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, 0, length(x))
    N = length(x)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i] = 1
        return res
    end
end

@inline function onehot(x::AbstractArray, start::Int, endl::Int)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, start-1, endl-start+1)
    ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i + start - 1] = 1
        return res
    end
end

# ---------------------------------------------------------------------------
# mul! on GPU arrays (backend BLAS gemv/gemm)
#
# Enzyme's built-in BLAS differentiation only recognizes CPU BLAS symbols, so
# on the CPU `mul!` is handled by the core. On GPUs the backend BLAS call (e.g.
# `cublasSgemv`) is a foreign ccall Enzyme cannot differentiate through
# (staged scalars via a device `Ref`, `retry_reclaim` atomics, gc-transition
# bundles), which leaves the matrix gradient zero or trips an internal error.
# So we provide the analytic derivative directly, dispatched on `AnyGPUArray`
# (all backends, incl. wrapped transpose/adjoint/view; CPU `Array` is excluded
# so the core CPU-BLAS path is untouched).
# See https://github.com/EnzymeAD/Enzyme.jl/issues/2837.
#
# For `C = α*A*B + β*C`:
#   forward:  Ċ = β*Ċ + α*(Ȧ*B + A*Ḃ)  (+ α̇*A*B + β̇*C for active scalars)
#   reverse:  Ā += α*C̄*Bᵀ ;  B̄ += α*Aᵀ*C̄ ;  C̄ ← β*C̄
#             ᾱ = Σ C̄ ⊙ (A*B) ;  β̄ = Σ C̄ ⊙ C_old
# ---------------------------------------------------------------------------

const GPUAnnotation = Annotation{<:AnyGPUArray}

# Accumulate `α*X*Y` into `dst`. A plain GPU-array destination goes through the
# in-place backend BLAS; a wrapped destination (e.g. `transpose(dW)`) is not a
# valid BLAS output, so fall back to a broadcast into the wrapper.
@inline _addmul!(dst::AbstractGPUArray, X, Y, α) = (mul!(dst, X, Y, α, true); nothing)
@inline _addmul!(dst, X, Y, α) = (dst .+= α .* (X * Y); nothing)

function EnzymeRules.forward(
        config, ofn::Const{typeof(mul!)}, ::Type{RT},
        C::GPUAnnotation, A::GPUAnnotation, B::GPUAnnotation,
        α::Annotation{<:Number}, β::Annotation{<:Number}
    ) where {RT}
    W = EnzymeRules.width(config)
    for i in 1:W
        dC = W == 1 ? C.dval : C.dval[i]
        dC .*= β.val
        A isa Const || mul!(dC, W == 1 ? A.dval : A.dval[i], B.val, α.val, true)
        B isa Const || mul!(dC, A.val, W == 1 ? B.dval : B.dval[i], α.val, true)
        # `C.val` still holds the old value here (primal below), needed by the β̇ term
        α isa Const || (dC .+= (W == 1 ? α.dval : α.dval[i]) .* (A.val * B.val))
        β isa Const || (dC .+= (W == 1 ? β.dval : β.dval[i]) .* C.val)
    end
    ofn.val(C.val, A.val, B.val, α.val, β.val)

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return W == 1 ? Duplicated(C.val, C.dval) : BatchDuplicated(C.val, C.dval)
    elseif EnzymeRules.needs_shadow(config)
        return C.dval
    elseif EnzymeRules.needs_primal(config)
        return C.val
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config, ofn::Const{typeof(mul!)}, ::Type{RT},
        C::GPUAnnotation, A::GPUAnnotation, B::GPUAnnotation,
        α::Annotation{<:Number}, β::Annotation{<:Number}
    ) where {RT}
    # snapshot inputs needed to form the reverse products. Ā = α*C̄*Bᵀ needs B's
    # value, and B̄ = α*Aᵀ*C̄ needs A's value — i.e. each is needed when the *other*
    # operand is active.
    Aval = (B isa Const) ? nothing : copy(A.val)
    Bval = (A isa Const) ? nothing : copy(B.val)
    # α/β adjoints need A*B and the pre-update C respectively
    ABval = (α isa Const) ? nothing : A.val * B.val
    Cold = (β isa Const) ? nothing : copy(C.val)

    ofn.val(C.val, A.val, B.val, α.val, β.val)

    primal = EnzymeRules.needs_primal(config) ? C.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? C.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (Aval, Bval, α.val, β.val, ABval, Cold))
end

function EnzymeRules.reverse(
        config, ofn::Const{typeof(mul!)}, ::Type{RT}, tape,
        C::GPUAnnotation, A::GPUAnnotation, B::GPUAnnotation,
        α::Annotation{<:Number}, β::Annotation{<:Number}
    ) where {RT}
    Aval, Bval, αval, βval, ABval, Cold = tape
    W = EnzymeRules.width(config)
    Cbar(i) = W == 1 ? C.dval : C.dval[i]

    # scalar adjoints (use C̄ before it is scaled by β below)
    dα = if α isa Const
        nothing
    elseif W == 1
        sum(Cbar(1) .* ABval)
    else
        ntuple(i -> sum(Cbar(i) .* ABval), W)
    end
    dβ = if β isa Const
        nothing
    elseif W == 1
        sum(Cbar(1) .* Cold)
    else
        ntuple(i -> sum(Cbar(i) .* Cold), W)
    end

    for i in 1:W
        A isa Const || _addmul!(W == 1 ? A.dval : A.dval[i], Cbar(i), Bval', αval)
        B isa Const || _addmul!(W == 1 ? B.dval : B.dval[i], Aval', Cbar(i), αval)
        C isa Const || (Cbar(i) .*= βval)
    end
    return (nothing, nothing, nothing, dα, dβ)
end

end # module
