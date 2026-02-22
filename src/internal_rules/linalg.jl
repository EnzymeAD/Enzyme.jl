# From LinearAlgebra ~/.julia/juliaup/julia-1.10.0-beta3+0.x64.apple.darwin14/share/julia/stdlib/v1.10/LinearAlgebra/src/generic.jl:1110
@inline function compute_lu_cache(cache_A::AT, b::BT) where {AT, BT}
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

@inline onedimensionalize(::Type{T}) where {T <: Array} = Vector{eltype(T)}

# y=inv(A) B
#   dA −= z y^T
#   dB += z, where  z = inv(A^T) dy
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(\)},
        ::Type{RT},
        A::Annotation{AT},
        b::Annotation{BT},
    ) where {RT, AT <: Array, BT <: Array}

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
        LinearAlgebra.Diagonal{eltype(AT), onedimensionalize(BT)},
        LinearAlgebra.LowerTriangular{eltype(AT), AT},
        LinearAlgebra.UpperTriangular{eltype(AT), AT},
        LinearAlgebra.LU{eltype(AT), AT, Vector{Int}},
        LinearAlgebra.QRPivoted{eltype(AT), AT, onedimensionalize(BT), Vector{Int}},
    }

    cache = NamedTuple{
        (Symbol("1"), Symbol("2"), Symbol("3"), Symbol("4")),
        Tuple{
            eltype(RT),
            EnzymeRules.needs_shadow(config) ?
                (
                    EnzymeRules.width(config) == 1 ? eltype(RT) :
                    NTuple{EnzymeRules.width(config), eltype(RT)}
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
        z = transpose(cache_A) \ dy
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
    ) where {RT, YT <: Array, AT <: EnzymeTriangulars, BT <: Array}
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
        Tuple{typeof(cache_Y), typeof(cache_A), typeof(cache_B)},
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
    ) where {YT <: Array, RT, AT <: EnzymeTriangulars, BT <: Array}
    if !isa(Y, Const)
        (cache_Yout, cache_A, cache_B) = cache
        for b in 1:EnzymeRules.width(config)
            dY = EnzymeRules.width(config) == 1 ? Y.dval : Y.dval[b]
            z = adjoint(cache_A) \ dY
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

# y = inv(A) B
# dY = inv(A) [ dB - dA y ]
# ->
# B(out) = inv(A) B(in)
# dB(out) = inv(A) [ dB(in) - dA B(out) ]
function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(ldiv!)},
        RT::Type{<:Union{Const, Duplicated, BatchDuplicated}},
        fact::Annotation{<:Cholesky},
        B::Annotation{<:AbstractVecOrMat};
        kwargs...,
    )
    return if B isa Const
        retval = func.val(fact.val, B.val; kwargs...)
        if EnzymeRules.needs_primal(config)
            retval
        else
            return nothing
        end
    else
        N = EnzymeRules.width(config)
        retval = B.val

        L = fact.val.L
        U = fact.val.U

        ldiv!(L, B.val)
        ntuple(Val(N)) do b
            Base.@_inline_meta
            dB = N == 1 ? B.dval : B.dval[b]
            if !(fact isa Const)
                dL = N == 1 ? fact.dval.L : fact.dval[b].L
                mul!(dB, dL, B.val, -1, 1)
            end
            ldiv!(L, dB)
        end

        ldiv!(U, B.val)
        dretvals = ntuple(Val(N)) do b
            Base.@_inline_meta
            dB = N == 1 ? B.dval : B.dval[b]
            if !(fact isa Const)
                dU = N == 1 ? fact.dval.U : fact.dval[b].U
                mul!(dB, dU, B.val, -1, 1)
            end
            ldiv!(U, dB)
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
            EnzymeRules.overwritten(config)[5]
            && !(typeof(B) <: Const)
            && !(typeof(C) <: Const)
        ) ? copy(A.val) : nothing

    cache_B = (
            EnzymeRules.overwritten(config)[6]
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

    N = EnzymeRules.width(config)
    if !isa(C, Const)
        dCs = C.dval
        dBs = isa(B, Const) ? dCs : B.dval
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
            if !isa(A, Const)
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

            if !isa(B, Const)
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

function cofactor(A)
    cofA = similar(A)
    minorAij = similar(A, size(A, 1) - 1, size(A, 2) - 1)
    for i in 1:size(A, 1), j in 1:size(A, 2)
        fill!(minorAij, zero(eltype(A)))

        # build minor matrix
        for k in 1:size(A, 1), l in 1:size(A, 2)
            if !(k == i || l == j)
                ki = k < i ? k : k - 1
                li = l < j ? l : l - 1
                @inbounds minorAij[ki, li] = A[k, l]
            end
        end
        @inbounds cofA[i, j] = (-1)^(i - 1 + j - 1) * det(minorAij)
    end
    return cofA
end

# partial derivative of the determinant is the matrix of cofactors
EnzymeRules.@easy_rule(LinearAlgebra.det(A::AbstractMatrix), (cofactor(A),))
