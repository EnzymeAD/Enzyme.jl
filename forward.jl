function EnzymeRules.forward(
    ::Const{typeof(cholesky)},
    RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
    A::Union{Const, Duplicated};
    kwargs...
)
    @assert issymmetric(A.val)
    fact = cholesky(A.val; kwargs...)
    if RT <: Const
        return fact
    end
    # TODO: This will be a problem for sparse matrices as invL and dL are dense
    invL = inv(fact.L)
    # TODO: dL is dense even when L was sparse
    dL = Matrix(fact.L * LowerTriangular(invL * A.dval * invL' * 0.5 * I))
    # TODO: Stored as Cholesky, although it isn't a Cholesky factorization
    dfact = Cholesky(dL, 'L', 0)
    if RT <: DuplicatedNoNeed
        return fact
    else
        return Duplicated(fact, dfact)
    end
end

function EnzymeRules.forward(
        ::Const{typeof(\)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        fact::Union{Const, Duplicated}, B::Union{Const, Duplicated};
        kwargs...
)
    retval = copy(B.val)
    ldiv!(fact.val, retval)
    retdval = []
    if isa(fact, Duplicated) && isa(B, Duplicated)
        retdval = similar(retval)
        mul!(retdval, fact.dval.U, retval)
        mul!(retdval, fact.dval.L, retdval)
        retdval .= B.dval .- retdval
        ldiv!(fact.val, retdval)
    elseif isa(fact, Duplicated) && isa(B, Const)
        retdval = similar(retval)
        mul!(retdval, A, B.val)
        mul!(retdval, -1, retdval)
        ldiv!(fact.val, retdval)
    elseif isa(fact, Const) && isa(B, Duplicated)
        retdval = copy(B.dval)
        ldiv!(fact.val, retdval)
    elseif isa(fact, Const) && isa(B, Const)
        nothing
    else
        error("Error in forward \\ Enzyme rule $(typeof(fact)) $(typeof(x)).")
    end
    if RT <: Const
        return retval
    elseif RT <: DuplicatedNoNeed
        return retdval
    else
        return Duplicated(retval, retdval)
    end
end
