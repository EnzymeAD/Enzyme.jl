module EnzymeCUDAExt

using CUDA
using Enzyme
using EnzymeCore
using EnzymeCore.EnzymeRules
using LinearAlgebra: LinearAlgebra, mul!

# Enzyme cannot differentiate through the raw `cuMemcpy*Async` ccalls that back
# `Base.unsafe_copyto!` on device pointers (they carry a `gc-transition` operand
# bundle and have no registered augmented-forward pass), so we provide custom
# rules. See https://github.com/EnzymeAD/Enzyme.jl/issues/2837.
#
# The three device-touching methods are
#   unsafe_copyto!(dst::CuPtr, src::Ptr,   N)  # host -> device
#   unsafe_copyto!(dst::CuPtr, src::CuPtr, N)  # device -> device
#   unsafe_copyto!(dst::Ptr,   src::CuPtr, N)  # device -> host
# all of which are element-wise copies `dst[1:N] = src[1:N]` returning `dst`.

const PtrOrCuPtr{T} = Union{Ptr{T}, CuPtr{T}}

# Wrap a raw pointer as an array in its own memory space, without taking ownership.
@inline _wrap(p::CuPtr{T}, n::Integer) where {T} = unsafe_wrap(CuArray, p, (Int(n),))
@inline _wrap(p::Ptr{T}, n::Integer) where {T} = unsafe_wrap(Array, p, (Int(n),))

# Bring `x` into the memory space of `ref` so the two can be combined elementwise.
@inline _match(::CuArray, x::CuArray) = x
@inline _match(::CuArray, x::Array) = CuArray(x)
@inline _match(::Array, x::Array) = x
@inline _match(::Array, x::CuArray) = Array(x)

@inline function _copytangent!(ofn, dstdval, srcdval, n; kwargs...)
    ofn.val(dstdval, srcdval, n; kwargs...)
    return nothing
end

@inline function _zerotangent!(dstdval, n)
    fill!(_wrap(dstdval, n), false)
    return nothing
end

# reverse of a copy `dst = src`: accumulate the destination adjoint into the
# source adjoint (when the source is differentiated) and clear the destination
# adjoint. `dst` is always overwritten by the copy, so its adjoint must be
# cleared regardless of whether `src` is active.
@inline function _revpair!(dstdval, srcdval, n)
    ddst = _wrap(dstdval, n)
    if srcdval !== nothing
        dsrc = _wrap(srcdval, n)
        dsrc .+= _match(dsrc, ddst)
    end
    fill!(ddst, zero(eltype(ddst)))
    return nothing
end

## Forward mode

function _fwd(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        dst::Annotation, src::Annotation, N::Const; kwargs...
    ) where {RT}
    n = N.val
    # primal copy
    ofn.val(dst.val, src.val, n; kwargs...)

    if !(dst isa Const)
        if EnzymeRules.width(config) == 1
            if src isa Const
                _zerotangent!(dst.dval, n)
            else
                _copytangent!(ofn, dst.dval, src.dval, n; kwargs...)
            end
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                if src isa Const
                    _zerotangent!(dst.dval[i], n)
                else
                    _copytangent!(ofn, dst.dval[i], src.dval[i], n; kwargs...)
                end
                nothing
            end
        end
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Duplicated(dst.val, dst.dval)
        else
            return BatchDuplicated(dst.val, dst.dval)
        end
    elseif EnzymeRules.needs_shadow(config)
        return dst.dval
    elseif EnzymeRules.needs_primal(config)
        return dst.val
    else
        return nothing
    end
end

function EnzymeRules.forward(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        dst::Annotation{<:CuPtr}, src::Annotation{<:PtrOrCuPtr},
        N::Const; kwargs...
    ) where {RT}
    return _fwd(config, ofn, RT, dst, src, N; kwargs...)
end

function EnzymeRules.forward(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        dst::Annotation{<:Ptr}, src::Annotation{<:CuPtr},
        N::Const; kwargs...
    ) where {RT}
    return _fwd(config, ofn, RT, dst, src, N; kwargs...)
end

## Reverse mode

function _augmented(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        dst::Annotation, src::Annotation, N::Const; kwargs...
    ) where {RT}
    n = N.val
    ofn.val(dst.val, src.val, n; kwargs...)

    primal = EnzymeRules.needs_primal(config) ? dst.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dst.dval : nothing
    # remember the copy length so the reverse pass knows the region to touch
    return EnzymeRules.AugmentedReturn(primal, shadow, Int(n))
end

function _reverse(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        tape, dst::Annotation, src::Annotation, N::Const; kwargs...
    ) where {RT}
    n = tape
    if !(dst isa Const)
        if EnzymeRules.width(config) == 1
            _revpair!(dst.dval, src isa Const ? nothing : src.dval, n)
        else
            ntuple(Val(EnzymeRules.width(config))) do i
                Base.@_inline_meta
                _revpair!(dst.dval[i], src isa Const ? nothing : src.dval[i], n)
                nothing
            end
        end
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        dst::Annotation{<:CuPtr}, src::Annotation{<:PtrOrCuPtr},
        N::Const; kwargs...
    ) where {RT}
    return _augmented(config, ofn, RT, dst, src, N; kwargs...)
end

function EnzymeRules.augmented_primal(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        dst::Annotation{<:Ptr}, src::Annotation{<:CuPtr},
        N::Const; kwargs...
    ) where {RT}
    return _augmented(config, ofn, RT, dst, src, N; kwargs...)
end

function EnzymeRules.reverse(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        tape, dst::Annotation{<:CuPtr}, src::Annotation{<:PtrOrCuPtr},
        N::Const; kwargs...
    ) where {RT}
    return _reverse(config, ofn, RT, tape, dst, src, N; kwargs...)
end

function EnzymeRules.reverse(
        config, ofn::Const{typeof(Base.unsafe_copyto!)}, ::Type{RT},
        tape, dst::Annotation{<:Ptr}, src::Annotation{<:CuPtr},
        N::Const; kwargs...
    ) where {RT}
    return _reverse(config, ofn, RT, tape, dst, src, N; kwargs...)
end

# ---------------------------------------------------------------------------
# mul! on CuArrays (cuBLAS gemv/gemm)
#
# Enzyme's built-in BLAS differentiation only recognizes CPU BLAS symbols, not
# `cublasSgemv`/`cublasSgemm`, and neither CUDA.jl extension provides a rule for
# `mul!`. Differentiating through the wrapper therefore reaches the opaque cuBLAS
# ccall and drops the matrix gradient (the matrix adjoint comes back as zero).
# See https://github.com/EnzymeAD/Enzyme.jl/issues/2837.
#
# For `C = α*A*B + β*C`:
#   forward:  Ċ = β*Ċ + α*(Ȧ*B + A*Ḃ)  (+ α̇*A*B + β̇*C for active scalars)
#   reverse:  Ā += α*C̄*Bᵀ ;  B̄ += α*Aᵀ*C̄ ;  C̄ ← β*C̄
#             ᾱ = Σ C̄ ⊙ (A*B) ;  β̄ = Σ C̄ ⊙ C_old
# ---------------------------------------------------------------------------

const AnyCuAnnotation = EnzymeCore.Annotation{<:CuArray}

function EnzymeRules.forward(
        config, ofn::Const{typeof(mul!)}, ::Type{RT},
        C::AnyCuAnnotation, A::AnyCuAnnotation, B::AnyCuAnnotation,
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
        C::AnyCuAnnotation, A::AnyCuAnnotation, B::AnyCuAnnotation,
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
        C::AnyCuAnnotation, A::AnyCuAnnotation, B::AnyCuAnnotation,
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
        A isa Const || mul!(W == 1 ? A.dval : A.dval[i], Cbar(i), Bval', αval, true)
        B isa Const || mul!(W == 1 ? B.dval : B.dval[i], Aval', Cbar(i), αval, true)
        C isa Const || (Cbar(i) .*= βval)
    end
    return (nothing, nothing, nothing, dα, dβ)
end

end # module
