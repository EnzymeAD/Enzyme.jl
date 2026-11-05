# From LinearAlgebra ~/.julia/juliaup/julia-1.10.0-beta3+0.x64.apple.darwin14/share/julia/stdlib/v1.10/LinearAlgebra/src/generic.jl:1110
@inline function compute_lu_cache(cache_A::AT, b::BT) where {AT,BT}
    LinearAlgebra.require_one_based_indexing(cache_A, b)
    m, n = size(cache_A)

    if m == n
        if LinearAlgebra.istril(cache_A)
            if LinearAlgebra.istriu(cache_A)
                return LinearAlgebra.Diagonal(cache_A)
            else
                return LinearAlgebra.LowerTriangular(cache_A)
            end
        elseif LinearAlgebra.istriu(cache_A)
            return LinearAlgebra.UpperTriangular(cache_A)
        else
            return LinearAlgebra.lu(cache_A)
        end
    end
    return LinearAlgebra.qr(cache_A, ColumnNorm())
end

@inline onedimensionalize(::Type{T}) where {T<:Array} = Vector{eltype(T)}

# Triangular solves for derivatives. For a strided triangular BLAS matrix, `ldiv!` and `\`
# call LAPACK's trtrs, which besides solving checks the matrix for singularity, which the
# primal computation already did. OpenBLAS also runs trtrs multithreaded for any matrix with
# more than one right-hand side, so that solving a 2×2 system took microseconds. Solve with
# BLAS's trsv/trsm instead, which compute the same result.
const _EnzymeTriangular{T, S} = Union{
    LowerTriangular{T, S},
    UpperTriangular{T, S},
    UnitLowerTriangular{T, S},
    UnitUpperTriangular{T, S},
}

_tri_uplo(::Union{LowerTriangular, UnitLowerTriangular}) = 'L'
_tri_uplo(::Union{UpperTriangular, UnitUpperTriangular}) = 'U'
_tri_diag(::Union{LowerTriangular, UpperTriangular}) = 'N'
_tri_diag(::Union{UnitLowerTriangular, UnitUpperTriangular}) = 'U'
_flip_uplo(uplo::Char) = uplo == 'L' ? 'U' : 'L'

# The stored matrix, and the uplo, trans and diag arguments, with which BLAS solves with A, or
# nothing if it cannot.
_tri_blas_args(A) = nothing
_tri_blas_args(A::_EnzymeTriangular{T, <:StridedMatrix{T}}) where {T <: LinearAlgebra.BlasFloat} =
    (A.data, _tri_uplo(A), 'N', _tri_diag(A))
# The triangle used of the transpose of the stored matrix is the opposite one of the stored matrix.
_tri_blas_args(A::_EnzymeTriangular{T, <:Transpose{T, <:StridedMatrix{T}}}) where {T <: LinearAlgebra.BlasFloat} =
    (parent(A.data), _flip_uplo(_tri_uplo(A)), 'T', _tri_diag(A))
_tri_blas_args(A::_EnzymeTriangular{T, <:Adjoint{T, <:StridedMatrix{T}}}) where {T <: LinearAlgebra.BlasFloat} =
    (parent(A.data), _flip_uplo(_tri_uplo(A)), 'C', _tri_diag(A))
_tri_blas_args(A::Transpose{T, <:_EnzymeTriangular{T, <:StridedMatrix{T}}}) where {T <: LinearAlgebra.BlasFloat} =
    (parent(A).data, _tri_uplo(parent(A)), 'T', _tri_diag(parent(A)))
_tri_blas_args(A::Adjoint{T, <:_EnzymeTriangular{T, <:StridedMatrix{T}}}) where {T <: LinearAlgebra.BlasFloat} =
    (parent(A).data, _tri_uplo(parent(A)), 'C', _tri_diag(parent(A)))

# Like `ldiv!(A, B)`, for A triangular.
_trisolve!(A, B) = ldiv!(A, B)
function _trisolve!(A::AbstractMatrix{T}, B::StridedVecOrMat{T}) where {T <: LinearAlgebra.BlasFloat}
    args = _tri_blas_args(A)
    if args === nothing || stride(args[1], 1) != 1 || stride(B, 1) != 1
        return ldiv!(A, B)
    end
    data, uplo, trans, diag = args
    if B isa AbstractVector
        BLAS.trsv!(uplo, trans, diag, data, B)
    else
        BLAS.trsm!('L', uplo, trans, diag, one(T), data, B)
    end
    return B
end

# Like `A \ B`, for A triangular.
_trisolve(A, B) = A \ B
_trisolve(A::AbstractMatrix{T}, B::StridedVecOrMat{T}) where {T <: LinearAlgebra.BlasFloat} =
    _tri_blas_args(A) === nothing ? A \ B : _trisolve!(A, copy(B))

# y=inv(A) B
#   dA −= z y^T
#   dB += z, where  z = inv(A^T) dy
function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(\)},
    ::Type{RT},
    A::Annotation{AT},
    b::Annotation{BT},
) where {RT,AT<:Array,BT<:Array}

    cache_A = if EnzymeRules.overwritten(config)[2]
        copy(A.val)
    else
        A.val
    end

    cache_A = compute_lu_cache(cache_A, b.val)

    res = (cache_A \ b.val)::eltype(RT)

    dres = if EnzymeRules.width(config) == 1
        zero(res)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(res)
        end
    end

    retres = if EnzymeRules.needs_primal(config)
        res
    else
        nothing
    end

    cache_res = if EnzymeRules.needs_primal(config)
        copy(res)
    else
        res
    end

    cache_b = if EnzymeRules.overwritten(config)[3]
        copy(b.val)
    else
        nothing
    end

    UT = Union{
        LinearAlgebra.Diagonal{eltype(AT),onedimensionalize(BT)},
        LinearAlgebra.LowerTriangular{eltype(AT),AT},
        LinearAlgebra.UpperTriangular{eltype(AT),AT},
        LinearAlgebra.LU{eltype(AT),AT,Vector{Int}},
        LinearAlgebra.QRPivoted{eltype(AT),AT,onedimensionalize(BT),Vector{Int}},
    }

    cache = NamedTuple{
        (Symbol("1"), Symbol("2"), Symbol("3"), Symbol("4")),
        Tuple{
            eltype(RT),
            EnzymeRules.needs_shadow(config) ?
            (
                EnzymeRules.width(config) == 1 ? eltype(RT) :
                NTuple{EnzymeRules.width(config),eltype(RT)}
            ) : Nothing,
            UT,
            typeof(cache_b),
        },
    }((cache_res, dres, cache_A, cache_b))

    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        EnzymeRules.shadow_type(config, RT),
        typeof(cache),
    }(
        retres,
        dres,
        cache,
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(\)},
    ::Type{RT},
    cache,
    A::Annotation{<:Array},
    b::Annotation{<:Array},
) where {RT}

    y, dys, cache_A, cache_b = cache

    if !EnzymeRules.overwritten(config)[3]
        cache_b = b.val
    end

    if EnzymeRules.width(config) == 1
        dys = (dys,)
    end

    dAs = if EnzymeRules.width(config) == 1
        if typeof(A) <: Const
            (nothing,)
        else
            (A.dval,)
        end
    else
        if typeof(A) <: Const
            ntuple(Returns(nothing), Val(EnzymeRules.width(config)))
        else
            A.dval
        end
    end

    dbs = if EnzymeRules.width(config) == 1
        if typeof(b) <: Const
            (nothing,)
        else
            (b.dval,)
        end
    else
        if typeof(b) <: Const
            ntuple(Returns(nothing), Val(EnzymeRules.width(config)))
        else
            b.dval
        end
    end

    for (dA, db, dy) in zip(dAs, dbs, dys)
        z = _trisolve(transpose(cache_A), dy)
        if !(typeof(A) <: Const)
            dA .-= z * transpose(y)
        end
        if !(typeof(b) <: Const)
            db .+= z
        end
        dy .= eltype(dy)(0)
    end

    return (nothing, nothing)
end

const EnzymeTriangulars = Union{
    UpperTriangular{<:Complex},
    LowerTriangular{<:Complex},
    UnitUpperTriangular{<:Complex},
    UnitLowerTriangular{<:Complex},
}

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(ldiv!)},
    ::Type{RT},
    Y::Annotation{YT},
    A::Annotation{AT},
    B::Annotation{BT},
) where {RT,YT<:Array,AT<:EnzymeTriangulars,BT<:Array}
    cache_Y = EnzymeRules.overwritten(config)[1] ? copy(Y.val) : Y.val
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    cache_A = compute_lu_cache(cache_A, B.val)
    cache_B = EnzymeRules.overwritten(config)[3] ? copy(B.val) : nothing
    primal = EnzymeRules.needs_primal(config) ? Y.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Y.dval : nothing
    func.val(Y.val, A.val, B.val)
    return EnzymeRules.AugmentedReturn{
        EnzymeRules.primal_type(config, RT),
        EnzymeRules.shadow_type(config, RT),
        Tuple{typeof(cache_Y),typeof(cache_A),typeof(cache_B)},
    }(
        primal,
        shadow,
        (cache_Y, cache_A, cache_B),
    )
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(ldiv!)},
    ::Type{RT},
    cache,
    Y::Annotation{YT},
    A::Annotation{AT},
    B::Annotation{BT},
) where {YT<:Array,RT,AT<:EnzymeTriangulars,BT<:Array}
    if !isa(Y, Const)
        (cache_Yout, cache_A, cache_B) = cache
        for b = 1:EnzymeRules.width(config)
            dY = EnzymeRules.width(config) == 1 ? Y.dval : Y.dval[b]
            z = _trisolve(adjoint(cache_A), dY)
            if !isa(B, Const)
                dB = EnzymeRules.width(config) == 1 ? B.dval : B.dval[b]
                dB .+= z
            end
            if !isa(A, Const)
                dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[b]
                dA.data .-= _zero_unused_elements!(z * adjoint(cache_Yout), A.val)
            end
            dY .= zero(eltype(dY))
        end
    end
    return (nothing, nothing, nothing)
end

# `dB .-= A * B` for A triangular, using `tmp`, which is like B, as scratch. Unlike `mul!`, this
# uses BLAS also when A wraps the transpose or adjoint of a matrix, except for tiny products,
# for which the generic `mul!` is faster than a BLAS call (n^2 m <= 128, as measured with
# OpenBLAS).
_trimul_sub!(dB, A, B, tmp) = mul!(dB, A, B, -1, 1)
function _trimul_sub!(
        dB::StridedVecOrMat{T}, A::AbstractMatrix{T}, B::StridedVecOrMat{T}, tmp::StridedVecOrMat{T}
    ) where {T <: LinearAlgebra.BlasFloat}
    args = _tri_blas_args(A)
    if args === nothing || stride(args[1], 1) != 1 || stride(tmp, 1) != 1 ||
            size(A, 1)^2 * size(B, 2) <= 128
        return mul!(dB, A, B, -1, 1)
    end
    data, uplo, trans, diag = args
    copyto!(tmp, B)
    if tmp isa AbstractVector
        BLAS.trmv!(uplo, trans, diag, data, tmp)
    else
        BLAS.trmm!('L', uplo, trans, diag, one(T), data, tmp)
    end
    dB .-= tmp
    return dB
end

# Solve with, or multiply by, the lower (`Val(:L)`) or upper (`Val(:U)`) triangular factor of a
# Cholesky factorization C. Unlike `C.L` and `C.U`, these do not copy the factor that is not
# stored, but use its adjoint. They branch on `C.uplo` so that each branch has concrete types.
function _chol_trisolve!(C::Cholesky, ::Val{:L}, B)
    return C.uplo == 'L' ? _trisolve!(LowerTriangular(C.factors), B) :
        _trisolve!(LowerTriangular(C.factors'), B)
end
function _chol_trisolve!(C::Cholesky, ::Val{:U}, B)
    return C.uplo == 'U' ? _trisolve!(UpperTriangular(C.factors), B) :
        _trisolve!(UpperTriangular(C.factors'), B)
end
function _chol_trimul_sub!(dB, C::Cholesky, ::Val{:L}, B, tmp)
    return C.uplo == 'L' ? _trimul_sub!(dB, LowerTriangular(C.factors), B, tmp) :
        _trimul_sub!(dB, LowerTriangular(C.factors'), B, tmp)
end
function _chol_trimul_sub!(dB, C::Cholesky, ::Val{:U}, B, tmp)
    return C.uplo == 'U' ? _trimul_sub!(dB, UpperTriangular(C.factors), B, tmp) :
        _trimul_sub!(dB, UpperTriangular(C.factors'), B, tmp)
end

# y = inv(A) B
# dY = inv(A) [ dB - dA y ]
# ->
# B(out) = inv(A) B(in)
# dB(out) = inv(A) [ dB(in) - dA B(out) ]
function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(ldiv!)},
    RT::Type{<:Union{Const,Duplicated,BatchDuplicated}},
    fact::Annotation{<:Cholesky},
    B::Annotation{<:AbstractVecOrMat};
    kwargs...,
)
    if B isa Const
        retval = func.val(fact.val, B.val; kwargs...)
        if EnzymeRules.needs_primal(config)
            retval
        else
            return nothing
        end
    else
        N = EnzymeRules.width(config)
        retval = B.val

        # Scratch for the products with the factors' shadows, shared by all lanes
        tmp = fact isa Const ? nothing : similar(B.val)

        _chol_trisolve!(fact.val, Val(:L), B.val)
        ntuple(Val(N)) do b
            Base.@_inline_meta
            dB = N == 1 ? B.dval : B.dval[b]
            if !(fact isa Const)
                dfact = N == 1 ? fact.dval : fact.dval[b]
                _chol_trimul_sub!(dB, dfact, Val(:L), B.val, tmp)
            end
            _chol_trisolve!(fact.val, Val(:L), dB)
        end

        _chol_trisolve!(fact.val, Val(:U), B.val)
        dretvals = ntuple(Val(N)) do b
            Base.@_inline_meta
            dB = N == 1 ? B.dval : B.dval[b]
            if !(fact isa Const)
                dfact = N == 1 ? fact.dval : fact.dval[b]
                _chol_trimul_sub!(dB, dfact, Val(:U), B.val, tmp)
            end
            _chol_trisolve!(fact.val, Val(:U), dB)
            return dB
        end


        if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
            if EnzymeRules.width(config) == 1
                return Duplicated(retval, dretvals[1])
            else
                return BatchDuplicated(retval, dretvals)
            end
        elseif EnzymeRules.needs_shadow(config)
            if EnzymeRules.width(config) == 1
                return dretvals[1]
            else
                return dretvals
            end
        elseif EnzymeRules.needs_primal(config)
            return retval
        else
            return nothing
        end
    end
end


_zero_unused_elements!(X, ::UpperTriangular) = triu!(X)
_zero_unused_elements!(X, ::LowerTriangular) = tril!(X)
_zero_unused_elements!(X, ::UnitUpperTriangular) = triu!(X, 1)
_zero_unused_elements!(X, ::UnitLowerTriangular) = tril!(X, -1)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(LinearAlgebra.mul!)},
        ::Type{RT},
        C::Annotation{<:StridedVecOrMat},
        A::Annotation{<:SparseArrays.SparseMatrixCSCUnion},
        B::Annotation{<:StridedVecOrMat},
        α::Annotation{<:Number},
        β::Annotation{<:Number}
    ) where {RT}

    cache_C = !(isa(β, Const)) ? copy(C.val) : nothing
    # Always need to do forward pass otherwise primal may not be correct
    func.val(C.val, A.val, B.val, α.val, β.val)

    primal = if EnzymeRules.needs_primal(config)
        C.val
    else
        nothing
    end

    shadow = if EnzymeRules.needs_shadow(config)
        C.dval
    else
        nothing
    end


    # Check if A is overwritten and B is active (and thus required)
    cache_A = (
            EnzymeRules.overwritten(config)[3]
            && !(typeof(B) <: Const)
            && !(typeof(C) <: Const)
        ) ? copy(A.val) : nothing

    cache_B = (
            EnzymeRules.overwritten(config)[4]
            && !(typeof(A) <: Const)
            && !(typeof(C) <: Const)
        ) ? copy(B.val) : nothing

    if !isa(α, Const)
        cache_α = A.val * B.val
    else
        cache_α = nothing
    end

    cache = (cache_C, cache_A, cache_B, cache_α)

    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

# This is required to handle arguments that mix real and complex numbers
_project(::Type{<:Real}, x) = x
_project(::Type{<:Real}, x::Complex) = real(x)
_project(::Type{<:Complex}, x) = x

function _muladdproject!(::Type{<:Number}, dB::AbstractArray, A::AbstractArray, C::AbstractArray, α)
    return LinearAlgebra.mul!(dB, A, C, α, true)
end

function _muladdproject!(::Type{<:Complex}, dB::AbstractArray{<:Real}, A::AbstractArray, C::AbstractArray, α::Number)
    tmp = A * C
    return dB .+= real.(α .* tmp)
end


function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(LinearAlgebra.mul!)},
        ::Type{RT}, cache,
        C::Annotation{<:StridedVecOrMat},
        A::Annotation{<:SparseArrays.SparseMatrixCSCUnion},
        B::Annotation{<:StridedVecOrMat},
        α::Annotation{<:Number},
        β::Annotation{<:Number}
    ) where {RT}

    cache_C, cache_A, cache_B, cache_α = cache
    Cval = !isnothing(cache_C) ? cache_C : C.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    Bval = !isnothing(cache_B) ? cache_B : B.val

    rta = EnzymeRules.runtime_activity(config)
    A_is_const = isa(A, Const) || (rta && A.dval === A.val)
    B_is_const = isa(B, Const) || (rta && B.dval === B.val)
    C_is_const = isa(C, Const) || (rta && C.dval === C.val)

    N = EnzymeRules.width(config)
    if !C_is_const
        dCs = C.dval
        dBs = B_is_const ? dCs : B.dval
        dα = if !isa(α, Const)
            if N == 1
                _project(typeof(α.val), conj(LinearAlgebra.dot(C.dval, cache_α)))
            else
                ntuple(Val(N)) do i
                    Base.@_inline_meta
                    _project(typeof(α.val), conj(LinearAlgebra.dot(C.dval[i], cache_α)))
                end
            end
        else
            nothing
        end

        dβ = if !isa(β, Const)
            if N == 1
                _project(typeof(β.val), conj(LinearAlgebra.dot(C.dval, Cval)))
            else
                ntuple(Val(N)) do i
                    Base.@_inline_meta
                    _project(typeof(β.val), conj(LinearAlgebra.dot(C.dval[i], Cval)))
                end
            end
        else
            nothing
        end

        for i in 1:N
            if !A_is_const
                # dA .+= α'dC*B'
                # You need to be careful so that dA sparsity pattern does not change. Otherwise
                # you will get incorrect gradients. So for now we do the slow and bad way of accumulating
                dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[i]
                dC = EnzymeRules.width(config) == 1 ? C.dval : C.dval[i]
                # Now accumulate to preserve the correct sparsity pattern
                I, J, _ = SparseArrays.findnz(dA)
                for k in eachindex(I, J)
                    Ik, Jk = I[k], J[k]
                    # May need to widen if the eltype differ
                    tmp = zero(promote_type(eltype(dA), eltype(dC)))
                    for ti in axes(dC, 2)
                        tmp += dC[Ik, ti] * conj(Bval[Jk, ti])
                    end
                    dA[Ik, Jk] += _project(eltype(dA), conj(α.val) * tmp)
                end
                # mul!(dA, dCs, Bval', α.val, true)
            end

            if !B_is_const
                #dB .+= α*A'*dC
                # Get the type of all arguments since we may need to
                # project down to a smaller type during accumulation
                if N == 1
                    Targs = promote_type(eltype(Aval), eltype(dCs), typeof(α.val))
                    _muladdproject!(Targs, dBs, Aval', dCs, conj(α.val))
                else
                    Targs = promote_type(eltype(Aval[i]), eltype(dCs[i]), typeof(α.val))
                    _muladdproject!(Targs, dBs[i], Aval', dCs[i], conj(α.val))
                end
            end
            #dC = dC*conj(β.val)
            if N == 1
                dCs .*= _project(eltype(dCs), conj(β.val))
            else
                dCs[i] .*= _project(eltype(dCs[i]), conj(β.val))
            end
        end
    else
        # C is constant so there is no gradient information to compute

        dα = if !isa(α, Const)
            if N == 1
                zero(α.val)
            else
                ntuple(Returns(zero(α.val)), Val(N))
            end
        else
            nothing
        end


        dβ = if !isa(β, Const)
            if N == 1
                zero(β.val)
            else
                ntuple(Returns(zero(β.val)), Val(N))
            end
        else
            nothing
        end
    end
    
    return (nothing, nothing, nothing, dα, dβ)
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(det)},
    ::Type{RT},
    A::Annotation{AT},
) where {RT,AT<:StridedMatrix}
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    res = det(A.val)
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    dres = if EnzymeRules.width(config) == 1 && EnzymeRules.needs_shadow(config)
        zero(res)
    elseif EnzymeRules.width(config) > 1 && EnzymeRules.needs_shadow(config)
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(res)
        end
    else
        nothing
    end
    cache = (res, cache_A)
    return EnzymeRules.AugmentedReturn(retres, dres, cache)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(det)},
    dret::Active,
    cache,
    A::Annotation{<:StridedMatrix},
)
    ret, cache_A = cache
    if !isa(A, Const)
        A.dval .+= inv(cache_A)' * dot(ret, dret.val)
    end
    return (nothing,)
end
function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(det)},
    ::Type{<:Const},
    cache,
    A::Annotation{<:Array},
)
    return (nothing,)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(det)},
    ::Type{RT},
    A::Annotation{<:StridedMatrix},
) where {RT}
    res = det(A.val)
    dres = !isa(A, Const) ? res * sum(diag(A.val \ A.dval)) : zero(res)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(res, dres)
    elseif EnzymeRules.needs_primal(config)
        return res
    elseif EnzymeRules.needs_shadow(config)
        return dres
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(logdet)},
    ::Type{RT},
    A::Annotation{AT},
) where {RT,AT<:StridedMatrix}
    cache_A = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    res = logdet(A.val)
    retres = EnzymeRules.needs_primal(config) ? res : nothing
    dres = if EnzymeRules.width(config) == 1 && EnzymeRules.needs_shadow(config)
        zero(res)
    elseif EnzymeRules.width(config) > 1 && EnzymeRules.needs_shadow(config)
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero(res)
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(retres, dres, cache_A)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(logdet)},
    dret::Active,
    cache_A,
    A::Annotation{<:StridedMatrix},
)
    !isa(A, Const) && (A.dval .+= dret.val * inv(cache_A)')
    return (nothing,)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfig,
    func::Const{typeof(logdet)},
    ::Type{<:Const},
    cache,
    A::Annotation{<:StridedMatrix},
)
    return (nothing,)
end

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(logdet)},
    ::Type{RT},
    A::Annotation{<:StridedMatrix},
) where {RT}
    res = logdet(A.val)
    dres = !isa(A, Const) ? sum(diag(A.val \ A.dval)) : zero(res)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(res, dres)
    elseif EnzymeRules.needs_primal(config)
        return res
    elseif EnzymeRules.needs_shadow(config)
        return dres
    else
        return nothing
    end
end

# Whether the array x is constant. With runtime activity, x is also constant if its shadow is
# itself.
_isconst(config::EnzymeRules.FwdConfig, x::Annotation{<:AbstractArray}) =
    x isa Const || (EnzymeRules.runtime_activity(config) && x.dval === x.val)

function _fwd_return(config::EnzymeRules.FwdConfig, primal, shadow)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            return Duplicated(primal, shadow)
        else
            return BatchDuplicated(primal, shadow)
        end
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return primal
    else
        return nothing
    end
end

# X = op(A) X (side 'L') or X = X op(A) (side 'R'). For a vector x, side must be 'L'.
function _tri_mul!(
        side::Char, uplo::Char, trans::Char, diag::Char,
        A::StridedMatrix{T}, X::StridedMatrix{T}
    ) where {T <: LinearAlgebra.BlasFloat}
    return BLAS.trmm!(side, uplo, trans, diag, one(T), A, X)
end
function _tri_mul!(
        side::Char, uplo::Char, trans::Char, diag::Char,
        A::StridedMatrix{T}, x::StridedVector{T}
    ) where {T <: LinearAlgebra.BlasFloat}
    side == 'L' || throw(ArgumentError("side must be 'L' for a vector, got '$side'"))
    return BLAS.trmv!(uplo, trans, diag, A, x)
end

# tmp = op(tri(dA)) X0 (side 'L') or X0 op(tri(dA)) (side 'R'), where tri(dA) is the triangle
# of dA that trmm and trmv read, without the diagonal if diag == 'U'. The strict triangle of dA
# is the triangle of a submatrix of dA shifted by one row or column, so it is multiplied with
# that submatrix instead of subtracting the unit diagonal afterwards, which would cancel.
function _tri_mul_tangent!(
        tmp::StridedVecOrMat{T}, side::Char, uplo::Char, trans::Char, diag::Char,
        dA::StridedMatrix{T}, X0::StridedVecOrMat{T}
    ) where {T <: LinearAlgebra.BlasFloat}
    if diag == 'N'
        copyto!(tmp, X0)
        return _tri_mul!(side, uplo, trans, 'N', dA, tmp)
    end
    fill!(tmp, zero(T))
    n = size(dA, 1)
    n == 1 && return tmp
    lo, hi = 1:(n - 1), 2:n
    block = uplo == 'U' ? view(dA, lo, hi) : view(dA, hi, lo)
    # Whether op(tri(dA)) is upper triangular
    upper = (uplo == 'U') == (trans == 'N')
    dim = side == 'L' ? 1 : 2
    out, in = upper == (side == 'L') ? (lo, hi) : (hi, lo)
    tmp_out = selectdim(tmp, dim, out)
    copyto!(tmp_out, selectdim(X0, dim, in))
    _tri_mul!(side, uplo, trans, 'N', block, tmp_out)
    return tmp
end

# B = α op(A) B (side 'L') or α B op(A) (side 'R'):
# dB = op(A) (α dB + dα B) + α op(tri(dA)) B (side 'L', analogously for side 'R'), with B the
# input
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(BLAS.trmm!)},
        RT::Type,
        side::Const{Char},
        uplo::Const{Char},
        transa::Const{Char},
        diag::Const{Char},
        alpha::Annotation{<:Number},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
    ) where {T <: LinearAlgebra.BlasFloat}
    s, ul, tA, dg = side.val, uplo.val, transa.val, diag.val
    α = alpha.val
    if _isconst(config, B)
        func.val(s, ul, tA, dg, α, A.val, B.val)
        return _fwd_return(config, B.val, B isa Const ? nothing : B.dval)
    end

    A_const = _isconst(config, A)
    α_const = alpha isa Const
    B0 = A_const && α_const ? nothing : copy(B.val)
    tmp = A_const ? nothing : similar(B.val)
    func.val(s, ul, tA, dg, α, A.val, B.val)

    N = EnzymeRules.width(config)
    for i in 1:N
        dB = N == 1 ? B.dval : B.dval[i]
        if α_const
            func.val(s, ul, tA, dg, α, A.val, dB)
        else
            dα = N == 1 ? alpha.dval : alpha.dval[i]
            dB .= α .* dB .+ dα .* B0
            func.val(s, ul, tA, dg, one(T), A.val, dB)
        end
        if !A_const
            dA = N == 1 ? A.dval : A.dval[i]
            dB .+= α .* _tri_mul_tangent!(tmp, s, ul, tA, dg, dA, B0)
        end
    end
    return _fwd_return(config, B.val, B.dval)
end

# x = op(A) x: dx = op(A) dx + op(tri(dA)) x, with x the input
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(BLAS.trmv!)},
        RT::Type,
        uplo::Const{Char},
        trans::Const{Char},
        diag::Const{Char},
        A::Annotation{<:StridedMatrix{T}},
        x::Annotation{<:StridedVector{T}},
    ) where {T <: LinearAlgebra.BlasFloat}
    ul, tA, dg = uplo.val, trans.val, diag.val
    if _isconst(config, x)
        func.val(ul, tA, dg, A.val, x.val)
        return _fwd_return(config, x.val, x isa Const ? nothing : x.dval)
    end

    A_const = _isconst(config, A)
    x0 = A_const ? nothing : copy(x.val)
    tmp = A_const ? nothing : similar(x.val)
    func.val(ul, tA, dg, A.val, x.val)

    N = EnzymeRules.width(config)
    for i in 1:N
        dx = N == 1 ? x.dval : x.dval[i]
        func.val(ul, tA, dg, A.val, dx)
        if !A_const
            dA = N == 1 ? A.dval : A.dval[i]
            dx .+= _tri_mul_tangent!(tmp, 'L', ul, tA, dg, dA, x0)
        end
    end
    return _fwd_return(config, x.val, x.dval)
end

# B = op(A) \ B: dB = op(A) \ (dB - op(tri(dA)) B), with B the output.
# Enzyme's native trtrs rule does not support forward mode, and unlike for BLAS functions there
# is no fallback implementation of LAPACK functions. The tangents are solved with trsv/trsm
# instead of trtrs, see `_trisolve!`.
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(LAPACK.trtrs!)},
        RT::Type,
        uplo::Const{Char},
        trans::Const{Char},
        diag::Const{Char},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedVecOrMat{T}},
    ) where {T <: LinearAlgebra.BlasFloat}
    ul, tA, dg = uplo.val, trans.val, diag.val
    # Throws for singular A, as without differentiation
    func.val(ul, tA, dg, A.val, B.val)
    if _isconst(config, B)
        return _fwd_return(config, B.val, B isa Const ? nothing : B.dval)
    end

    A_const = _isconst(config, A)
    tmp = A_const ? nothing : similar(B.val)
    N = EnzymeRules.width(config)
    for i in 1:N
        dB = N == 1 ? B.dval : B.dval[i]
        if !A_const
            dA = N == 1 ? A.dval : A.dval[i]
            dB .-= _tri_mul_tangent!(tmp, 'L', ul, tA, dg, dA, B.val)
        end
        if dB isa AbstractVector
            BLAS.trsv!(ul, tA, dg, A.val, dB)
        else
            BLAS.trsm!('L', ul, tA, dg, one(T), A.val, dB)
        end
    end
    return _fwd_return(config, B.val, B.dval)
end

EnzymeRules.has_readonly_rule(::typeof(logdet), ::StridedMatrix) = true
EnzymeRules.has_attribute_preserving_rule(::typeof(logdet), ::StridedMatrix) = true
EnzymeRules.has_readonly_rule(::typeof(det), ::StridedMatrix) = true
EnzymeRules.has_attribute_preserving_rule(::typeof(det), ::StridedMatrix) = true
