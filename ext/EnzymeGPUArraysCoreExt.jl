module EnzymeGPUArraysCoreExt

using GPUArraysCore
using Enzyme
using LinearAlgebra: LinearAlgebra, mul!, dot, Transpose, Adjoint, adjoint,
    UpperTriangular, LowerTriangular, Symmetric, Hermitian, triu, tril, diag, diagind
using Enzyme.EnzymeCore: EnzymeCore
using Enzyme.EnzymeCore.EnzymeRules:
    EnzymeRules,
    FwdConfig,
    RevConfig,
    Annotation,
    AugmentedReturn,
    needs_primal,
    needs_shadow,
    overwritten,
    width

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

@inline _bget(x, ::Val{1}, ::Int) = x
@inline _bget(x, ::Val{N}, i::Int) where {N} = x[i]

# Project an accumulated cotangent onto the (possibly real) parameter type.
_project(::Type{<:Real}, x) = real(x)
_project(::Type, x) = x

# A GPU array, or a structured/lazy wrapper of one; the operand types that show
# up in `A * B` matmuls (`transpose(X) * y`, `Symmetric(A) * x`, ...). CUBLAS /
# rocBLAS dispatch these to specialized kernels (trmm/symm/hemm), and the reverse
# pass projects the cotangent back onto each wrapper's stored entries (see
# `_accumulate_operand!`). UnitTriangular is intentionally excluded: its diagonal
# is structurally fixed, so it cannot represent a tangent/cotangent.
const MaybeWrappedGPU = Union{
    AbstractGPUArray,
    Transpose{<:Any, <:AbstractGPUArray},
    Adjoint{<:Any, <:AbstractGPUArray},
    UpperTriangular{<:Any, <:AbstractGPUArray},
    LowerTriangular{<:Any, <:AbstractGPUArray},
    Symmetric{<:Any, <:AbstractGPUArray},
    Hermitian{<:Any, <:AbstractGPUArray},
}

#=
Accumulate `factor .* G` into the shadow `s` of a matmul operand, where `G` is
the dense cotangent (`dY·B'` or `A'·dY`). For dense / transpose / adjoint shadows
the cotangent is added directly. For structured shadows only the stored entries
are free parameters, so `G` is projected onto that structure:

  UpperTriangular  : keep `triu(G)`
  LowerTriangular  : keep `tril(G)`
  Symmetric(uplo)  : off-diagonal (i,j) collects `G[i,j] + G[j,i]`, diagonal `G[i,i]`
  Hermitian(uplo)  : same with conjugation; the diagonal is real
=#
@inline function _accumulate_operand!(s, G, factor)
    s .+= factor .* G
    return nothing
end

function _accumulate_operand!(s::UpperTriangular, G, factor)
    parent(s) .+= factor .* triu(G)
    return nothing
end

function _accumulate_operand!(s::LowerTriangular, G, factor)
    parent(s) .+= factor .* tril(G)
    return nothing
end

function _accumulate_operand!(s::Symmetric, G, factor)
    p = parent(s)
    H = G .+ transpose(G)
    p .+= factor .* (s.uplo == 'U' ? triu(H, 1) : tril(H, -1))
    dg = view(p, diagind(p))
    dg .+= factor .* diag(G)
    return nothing
end

function _accumulate_operand!(s::Hermitian, G, factor)
    p = parent(s)
    H = G .+ adjoint(G)
    p .+= factor .* (s.uplo == 'U' ? triu(H, 1) : tril(H, -1))
    dg = view(p, diagind(p))
    dg .+= factor .* real.(diag(G))
    return nothing
end

#=
mul!(C, A, B, α, β):  C = α·A·B + β·C₀

JVP:      dC = α·(dA·B + A·dB) + β·dC₀   (+ dα·A·B + dβ·C₀)
Pullback: dA += conj(α)·dC·B'
          dB += conj(α)·A'·dC
          dC := conj(β)·dC                (cotangent w.r.t. C₀)
          dα  = conj(⟨dC, A·B⟩)
          dβ  = conj(⟨dC, C₀⟩)
=#

function EnzymeRules.forward(
        config::FwdConfig,
        func::Const{typeof(mul!)},
        RT::Type{<:Annotation},
        C::Annotation{<:AbstractGPUArray},
        A::Annotation{<:MaybeWrappedGPU},
        B::Annotation{<:MaybeWrappedGPU},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    )
    N = width(config)
    if !(C isa Const)
        # Update each output tangent from the old tangent values
        # before the in-place primal overwrites C.val.
        ntuple(Val(N)) do b
            Base.@_inline_meta
            dC = _bget(C.dval, Val(N), b)
            tmp = β.val .* dC
            if !(A isa Const)
                tmp = tmp .+ α.val .* (_bget(A.dval, Val(N), b) * B.val)
            end
            if !(B isa Const)
                tmp = tmp .+ α.val .* (A.val * _bget(B.dval, Val(N), b))
            end
            if !(α isa Const)
                tmp = tmp .+ _bget(α.dval, Val(N), b) .* (A.val * B.val)
            end
            if !(β isa Const)
                tmp = tmp .+ _bget(β.dval, Val(N), b) .* C.val
            end
            copyto!(dC, tmp)
            nothing
        end
    end

    func.val(C.val, A.val, B.val, α.val, β.val)

    if needs_primal(config) && needs_shadow(config)
        return N == 1 ? Duplicated(C.val, C.dval) : BatchDuplicated(C.val, C.dval)
    elseif needs_shadow(config)
        return C.dval
    elseif needs_primal(config)
        return C.val
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(mul!)},
        ::Type{RT},
        C::Annotation{<:AbstractGPUArray},
        A::Annotation{<:MaybeWrappedGPU},
        B::Annotation{<:MaybeWrappedGPU},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    # C₀ is needed for dβ; copy it before the primal overwrites C.val.
    cache_C = !(β isa Const) ? copy(C.val) : nothing

    func.val(C.val, A.val, B.val, α.val, β.val)

    primal = needs_primal(config) ? C.val : nothing
    shadow = needs_shadow(config) ? C.dval : nothing

    # A is needed for dB, B for dA; cache them only if overwritten before reverse.
    cache_A = (overwritten(config)[3] && !(B isa Const) && !(C isa Const)) ? copy(A.val) : nothing
    cache_B = (overwritten(config)[4] && !(A isa Const) && !(C isa Const)) ? copy(B.val) : nothing
    cache_α = !(α isa Const) ? A.val * B.val : nothing

    return AugmentedReturn(primal, shadow, (cache_C, cache_A, cache_B, cache_α))
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(mul!)},
        ::Type{RT},
        tape,
        C::Annotation{<:AbstractGPUArray},
        A::Annotation{<:MaybeWrappedGPU},
        B::Annotation{<:MaybeWrappedGPU},
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    cache_C, cache_A, cache_B, cache_α = tape
    Cval = cache_C !== nothing ? cache_C : C.val
    Aval = cache_A !== nothing ? cache_A : A.val
    Bval = cache_B !== nothing ? cache_B : B.val
    N = width(config)

    if !(C isa Const)
        dα = if !(α isa Const)
            if N == 1
                _project(typeof(α.val), conj(dot(C.dval, cache_α)))
            else
                ntuple(i -> _project(typeof(α.val), conj(dot(C.dval[i], cache_α))), Val(N))
            end
        else
            nothing
        end
        dβ = if !(β isa Const)
            if N == 1
                _project(typeof(β.val), conj(dot(C.dval, Cval)))
            else
                ntuple(i -> _project(typeof(β.val), conj(dot(C.dval[i], Cval))), Val(N))
            end
        else
            nothing
        end

        αc = conj(α.val)
        βc = conj(β.val)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dC = _bget(C.dval, Val(N), i)
            if !(A isa Const)
                _accumulate_operand!(_bget(A.dval, Val(N), i), dC * adjoint(Bval), αc)
            end
            if !(B isa Const)
                _accumulate_operand!(_bget(B.dval, Val(N), i), adjoint(Aval) * dC, αc)
            end
            dC .*= βc
            nothing
        end
    else
        dα = !(α isa Const) ? (N == 1 ? zero(α.val) : ntuple(Returns(zero(α.val)), Val(N))) : nothing
        dβ = !(β isa Const) ? (N == 1 ? zero(β.val) : ntuple(Returns(zero(β.val)), Val(N))) : nothing
    end

    return (nothing, nothing, nothing, dα, dβ)
end

#=
A * B:  allocating matmul (matrix or matrix-vector)

Unlike the in-place `mul!` rule above, `*` allocates its result, so the rule
owns the output shadow: `augmented_primal` returns a zeroed shadow and
`reverse` zeroes it again after reading. Without this, the freshly-allocated
result's shadow is uninitialized and downstream `+=` accumulates onto garbage.

JVP:      dY = dA·B + A·dB
Pullback: dA += dY·B'
          dB += A'·dY
=#

# dY_i = dA_i·B + A·dB_i  (factored out so inference sees a concrete array type)
@inline function _matmul_jvp(A::Annotation, B::Annotation, ::Val{N}, i::Int) where {N}
    if A isa Const
        return A.val * _bget(B.dval, Val(N), i)
    elseif B isa Const
        return _bget(A.dval, Val(N), i) * B.val
    else
        return _bget(A.dval, Val(N), i) * B.val .+ A.val * _bget(B.dval, Val(N), i)
    end
end

function EnzymeRules.forward(
        config::FwdConfig,
        func::Const{typeof(*)},
        RT::Type{<:Annotation},
        A::Annotation{<:MaybeWrappedGPU},
        B::Annotation{<:MaybeWrappedGPU},
    )
    if RT <: Const
        return needs_primal(config) ? A.val * B.val : nothing
    end

    N = width(config)
    if N == 1
        dY = _matmul_jvp(A, B, Val(1), 1)
        return RT <: DuplicatedNoNeed ? dY : Duplicated(A.val * B.val, dY)
    else
        dY = ntuple(i -> _matmul_jvp(A, B, Val(N), i), Val(N))
        return RT <: BatchDuplicatedNoNeed ? dY : BatchDuplicated(A.val * B.val, dY)
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(*)},
        ::Type{RT},
        A::Annotation{<:MaybeWrappedGPU},
        B::Annotation{<:MaybeWrappedGPU},
    ) where {RT}
    Y = A.val * B.val
    primal = needs_primal(config) ? Y : nothing

    N = width(config)
    dY = if RT <: Duplicated || RT <: DuplicatedNoNeed
        zero(Y)
    elseif RT <: BatchDuplicated || RT <: BatchDuplicatedNoNeed
        ntuple(_ -> zero(Y), Val(N))
    else
        nothing
    end

    cache_A = (overwritten(config)[2] && !(B isa Const)) ? copy(A.val) : A.val
    cache_B = (overwritten(config)[3] && !(A isa Const)) ? copy(B.val) : B.val

    return AugmentedReturn(primal, dY, (dY, cache_A, cache_B))
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(*)},
        ::Type{RT},
        tape,
        A::Annotation{<:MaybeWrappedGPU},
        B::Annotation{<:MaybeWrappedGPU},
    ) where {RT}
    dY, cache_A, cache_B = tape
    if dY !== nothing
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dYi = _bget(dY, Val(N), i)
            if !(A isa Const)
                _accumulate_operand!(_bget(A.dval, Val(N), i), dYi * adjoint(cache_B), true)
            end
            if !(B isa Const)
                _accumulate_operand!(_bget(B.dval, Val(N), i), adjoint(cache_A) * dYi, true)
            end
            fill!(dYi, zero(eltype(dYi)))
            nothing
        end
    end
    return (nothing, nothing)
end

#=
dot(a, b):  scalar inner product

JVP:      dr  = ⟨da, b⟩ + ⟨a, db⟩
Pullback: da += dr̄·b
          db += dr̄·a   (real convention; targets real GPU workloads)
=#

function EnzymeRules.forward(
        config::FwdConfig,
        func::Const{typeof(dot)},
        RT::Type{<:Annotation},
        a::Annotation{<:AbstractGPUArray},
        b::Annotation{<:AbstractGPUArray},
    )
    if needs_shadow(config)
        N = width(config)
        z = zero(promote_type(eltype(a.val), eltype(b.val)))
        if N == 1
            dr = (a isa Const ? z : dot(a.dval, b.val)) + (b isa Const ? z : dot(a.val, b.dval))
            return needs_primal(config) ? Duplicated(dot(a.val, b.val), dr) : dr
        else
            dr = ntuple(
                i -> (a isa Const ? z : dot(_bget(a.dval, Val(N), i), b.val)) +
                    (b isa Const ? z : dot(a.val, _bget(b.dval, Val(N), i))),
                Val(N),
            )
            return needs_primal(config) ? BatchDuplicated(dot(a.val, b.val), dr) : dr
        end
    elseif needs_primal(config)
        return dot(a.val, b.val)
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(dot)},
        ::Type,
        a::Annotation{<:AbstractGPUArray},
        b::Annotation{<:AbstractGPUArray},
    )
    primal = needs_primal(config) ? dot(a.val, b.val) : nothing
    cache_a = (overwritten(config)[2] && !(b isa Const)) ? copy(a.val) : nothing
    cache_b = (overwritten(config)[3] && !(a isa Const)) ? copy(b.val) : nothing
    return AugmentedReturn(primal, nothing, (cache_a, cache_b))
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(dot)},
        dret,
        tape,
        a::Annotation{<:AbstractGPUArray},
        b::Annotation{<:AbstractGPUArray},
    )
    if !(dret isa Const)
        cache_a, cache_b = tape
        av = cache_a !== nothing ? cache_a : a.val
        bv = cache_b !== nothing ? cache_b : b.val
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dr = _bget(dret.val, Val(N), i)
            if !(a isa Const)
                _bget(a.dval, Val(N), i) .+= dr .* bv
            end
            if !(b isa Const)
                _bget(b.dval, Val(N), i) .+= dr .* av
            end
            nothing
        end
    end
    return (nothing, nothing)
end

#=
sum(x):  scalar reduction

JVP:      dr  = sum(dx)
Pullback: dx += dr̄        (scalar broadcast over every element)
=#

function EnzymeRules.forward(
        config::FwdConfig,
        func::Const{typeof(sum)},
        RT::Type{<:Annotation},
        x::Annotation{<:AbstractGPUArray},
    )
    if needs_shadow(config)
        N = width(config)
        if N == 1
            dr = x isa Const ? zero(eltype(x.val)) : sum(x.dval)
            return needs_primal(config) ? Duplicated(sum(x.val), dr) : dr
        else
            dr = ntuple(i -> x isa Const ? zero(eltype(x.val)) : sum(_bget(x.dval, Val(N), i)), Val(N))
            return needs_primal(config) ? BatchDuplicated(sum(x.val), dr) : dr
        end
    elseif needs_primal(config)
        return sum(x.val)
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(sum)},
        ::Type,
        x::Annotation{<:AbstractGPUArray},
    )
    primal = needs_primal(config) ? sum(x.val) : nothing
    return AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(sum)},
        dret,
        tape,
        x::Annotation{<:AbstractGPUArray},
    )
    if !(dret isa Const) && !(x isa Const)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            _bget(x.dval, Val(N), i) .+= _bget(dret.val, Val(N), i)
            nothing
        end
    end
    return (nothing,)
end

end # module
