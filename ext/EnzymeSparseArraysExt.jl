module EnzymeSparseArraysExt

using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays
using Enzyme
using EnzymeCore: EnzymeRules

@inline Enzyme.Compiler.ptreltype(::Type{SparseArrays.CHOLMOD.Dense{T}}) where {T} = T
@inline Enzyme.Compiler.is_arrayorvararg_ty(::Type{SparseArrays.CHOLMOD.Dense{T}}) where {T} = true

Enzyme.Compiler.isa_cholmod_struct(::Core.Type{<:SparseArrays.LibSuiteSparse.cholmod_dense_struct}) = true
Enzyme.Compiler.isa_cholmod_struct(::Core.Type{<:SparseArrays.LibSuiteSparse.cholmod_sparse_struct}) = true
Enzyme.Compiler.isa_cholmod_struct(::Core.Type{<:SparseArrays.LibSuiteSparse.cholmod_factor_struct}) = true

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
                Enzyme._project(typeof(α.val), conj(LinearAlgebra.dot(C.dval, cache_α)))
            else
                ntuple(Val(N)) do i
                    Base.@_inline_meta
                    Enzyme._project(typeof(α.val), conj(LinearAlgebra.dot(C.dval[i], cache_α)))
                end
            end
        else
            nothing
        end

        dβ = if !isa(β, Const)
            if N == 1
                Enzyme._project(typeof(β.val), conj(LinearAlgebra.dot(C.dval, Cval)))
            else
                ntuple(Val(N)) do i
                    Base.@_inline_meta
                    Enzyme._project(typeof(β.val), conj(LinearAlgebra.dot(C.dval[i], Cval)))
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
                    dA[Ik, Jk] += Enzyme._project(eltype(dA), conj(α.val) * tmp)
                end
                # mul!(dA, dCs, Bval', α.val, true)
            end

            if !isa(B, Const)
                #dB .+= α*A'*dC
                # Get the type of all arguments since we may need to
                # project down to a smaller type during accumulation
                if N == 1
                    Targs = promote_type(eltype(Aval), eltype(dCs), typeof(α.val))
                    Enzyme._muladdproject!(Targs, dBs, Aval', dCs, conj(α.val))
                else
                    Targs = promote_type(eltype(Aval[i]), eltype(dCs[i]), typeof(α.val))
                    Enzyme._muladdproject!(Targs, dBs[i], Aval', dCs[i], conj(α.val))
                end
            end
            #dC = dC*conj(β.val)
            if N == 1
                dCs .*= Enzyme._project(eltype(dCs), conj(β.val))
            else
                dCs[i] .*= Enzyme._project(eltype(dCs[i]), conj(β.val))
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

end
