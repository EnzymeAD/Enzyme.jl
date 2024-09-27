module EnzymeSparseArraysExt

using Enzyme
using EnzymeCore
using EnzymeCore: EnzymeRules

using SparseArrays
using SparseArrays: LinearAlgebra, SparseMatrixCSCUnion


@inline Enzyme.Compiler.ptreltype(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = T
@inline Enzyme.Compiler.is_arrayorvararg_ty(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = true

# TODO don't limit A to be Const. Currently I'd have to implement a new matmul for this
# or bootstrap ChainRules ProjectTo mechanism to enforce the structural zeros
# Currently we put the rule on the 5-arg mul!, since spdensemul! isn't a public API and I am unsure
# how stable it is.
function EnzymeRules.augmented_primal(config::EnzymeRules.ConfigWidth, 
                                      func::Const{typeof(LinearAlgebra.mul!)},
                                      ::Type{RT}, 
                                      C::Annotation{<:StridedVecOrMat},
                                      A::Const{<:SparseMatrixCSCUnion},
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
    cache_A = ( EnzymeRules.overwritten(config)[5]
                && !(typeof(B) <: Const)
                && !(typeof(C) <: Const)
                ) ? copy(A.val) : nothing
    
    # cache_B = ( EnzymeRules.overwritten(config)[6]) ? copy(B.val) : nothing

    if !isa(α, Const)
        cache_α = A.val*B.val
    else
        cache_α = nothing
    end
    
    cache = (cache_C, cache_A, cache_α)

    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config,
                             func::Const{typeof(LinearAlgebra.mul!)},
                             ::Type{RT}, cache,
                             C::Annotation{<:StridedVecOrMat},
                             A::Const{<:SparseMatrixCSCUnion},
                             B::Annotation{<:StridedVecOrMat},
                             α::Annotation{<:Number},
                             β::Annotation{<:Number}
                             ) where {RT}

    cache_C, cache_A, cache_α = cache
    Cval = !isnothing(cache_C) ? cache_C : C.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    # Bval = !isnothing(cache_B) ? cache_B : B.val

    N = EnzymeRules.width(config)
    if !isa(C, Const)
        dCs = C.dval
        dBs  = isa(B, Const) ? dCs : B.dval

        dα = if !isa(α, Const)
                if N == 1
                    LinearAlgebra.dot(C.dval, cache_α)
                else
                    ntuple(Val(N)) do i
                        Base.@_inline_meta
                        LinearAlgebra.dot(C.dval[i], cache_α)
                    end
                end
        else
            nothing
        end

        dβ = if !isa(β, Const)
                if N == 1
                    LinearAlgebra.dot(C.dval, Cval)
                else
                    ntuple(Val(N)) do i
                        Base.@_inline_meta
                        LinearAlgebra.dot(C.dval[i], Cval)
                    end
                end
        else
            nothing
        end

        for i in 1:N

            # This rule is incorrect since I need to project dA to have the same
            # sparsity pattern as A.
            # if !isa(A, Const)
            #     dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[b]
            #     #dA .+= α*dC*B'
            #     mul!(dA, dC, Bval', α.val, true)
            # end

            if !isa(B, Const)
                #dB .+= α*A'*dC
                if N ==1
                    func.val(dBs, Aval', dCs, α.val, true)
                else
                    func.val(dBs[i], Aval', dCs[i], α.val, true)
                end
            end

            if N==1
                dCs .*= β.val
            else
                dCs[i] .*= β.val
            end
        end
    end
   
    return (nothing, nothing, nothing, dα, dβ)
end


end