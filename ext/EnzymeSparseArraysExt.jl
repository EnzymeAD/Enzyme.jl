module EnzymeSparseArraysExt

using Enzyme
using EnzymeCore
using EnzymeCore: EnzymeRules

using SparseArrays
using SparseArrays: LinearAlgebra, SparseMatrixCSCUnion
const SparseMatAdj = Union{SparseMatrixCSC, LinearAlgebra.Adjoint{T, SparseMatrixCSC} where T}

# TODO don't limit A to be Const. Currently I'd have to implement a new matmul for this
# or bootstrap ChainRules ProjectTo mechanism to enforce the structural zeros
# Currently we put the rule on the 5-arg mul!, since spdensemul! isn't a public API and I am unsure
# how stable it is.
function EnzymeRules.augmented_primal(config::EnzymeRules.ConfigWidth{1}, 
                                      func::Const{typeof(LinearAlgebra.mul!)},
                                      ::Type{RT}, 
                                      C::Annotation{<:StridedVecOrMat},
                                      A::Const{<:SparseMatAdj},
                                      B::Annotation{<:StridedVecOrMat},
                                      α::Annotation{<:Number},
                                      β::Annotation{<:Number}
                                    ) where {RT}

    # Q? Why doesn't EnzymeRules.overwritten(config) detect that C is overwritten?
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
    
    cache_B = ( EnzymeRules.overwritten(config)[6]) ? copy(B.val) : nothing

    if !isa(α, Const)
        cache_α = A.val*B.val
    else
        cache_α = nothing
    end
    
    cache = (cache_C, cache_A, cache_B, cache_α)

    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1}, 
                             func::Const{typeof(LinearAlgebra.mul!)},
                             ::Type{RT}, cache,
                             C::Annotation{<:StridedVecOrMat},
                             A::Const{<:SparseMatAdj},
                             B::Annotation{<:StridedVecOrMat},
                             α::Annotation{<:Number},
                             β::Annotation{<:Number}
                             ) where {RT}

    cache_C, cache_A, cache_B, cache_AB = cache
    Cval = !isnothing(cache_C) ? cache_C : C.val
    Aval = !isnothing(cache_A) ? cache_A : A.val
    Bval = !isnothing(cache_B) ? cache_B : B.val

    if !isa(α, Const)
        dα = zero(α.val)
        ABval = cache_AB
    else
        dα = nothing
    end

    if !isa(β, Const)
        dβ = zero(β.val)
    else
        dβ = nothing
    end

    if !isa(C, Const)

        for b in 1:EnzymeRules.width(config)
            dC = EnzymeRules.width(config) == 1 ? C.dval : C.dval[b]
            # TODO This rule is incorrect since I need to project dA to have the same 
            # sparsity pattern as A. 
            # if !isa(A, Const)
            #     dA = EnzymeRules.width(config) == 1 ? A.dval : A.dval[b]
            #     #dA .+= α*dC*B'
            #     mul!(dA, dC, Bval', α.val, true)
            # end

            if !isa(B, Const)
                dB = EnzymeRules.width(config) == 1 ? B.dval : B.dval[b]
                #dB .+= α*A'*dC
                func.val(dB, Aval', dC, α.val, true)
            end

            if !isa(α, Const)
                dα += LinearAlgebra.dot(dC, ABval)
            end

            if !isa(β, Const)
                dβ += LinearAlgebra.dot(dC, Cval)
            end

            # This is needed because dC may be done in place and accumulate into itself
            dC .= dC.*β.val
        end
    end
    return (nothing, nothing, nothing, dα, dβ)
end


end