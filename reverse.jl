function EnzymeRules.augmented_primal(
    config,
    func::Const{typeof(cholesky)},
    RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
    A::Union{Const, Duplicated};
    kwargs...
)
    fact = cholesky(A.val; kwargs...)
    dA = similar(fact.factors)
    println("Custom Cholesky augmented forward rule")
    # dfact would be a dense matrix
    if needs_primal(config)
        return AugmentedReturn(fact, dfact, (fact, dfact, dA))
    else
        return AugmentedReturn(nothing, (bx, bstats), (fact,))
    end
end

function EnzymeRules.augmented_primal(
        config,
        func::Const{typeof(\)},
        RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        fact::Union{Const, Duplicated}, B::Union{Const, Duplicated};
        kwargs...
)
    x = copy(B.val)
    ldiv!(fact.val, x)
    dx = similar(x)
    if needs_primal(config)
        return AugmentedReturn(x, dx, (x, dx))
    else
        return AugmentedReturn(nothing, dx, (fact, B, x, dx))
    end
end

function EnzymeRules.reverse(
    config,
    ::Const{typeof(Cholesky)},
    dret,
    cache,
    A;
    kwargs...
)
    println("Custom Cholesky reverse rule")
    (fact, dfact, dA) = cache
    mul!(dA, fact.L', dfact.L)
    ldiv!(dA, fact.L)
    rdiv!(fact.L', dA)
    idx = diagind(dA)
    @views dA[idx] .= 0.5 .* dA[idx]
    dA = LowerTriangular(dA)
    dret = Cholesky(dA, 'L', 0)
    return (nothing, nothing)
end

function EnzymeRules.reverse(
    config,
    ::Const{typeof(\)},
    dret,
    cache,
    fact::Union{Const, Duplicated}, B::Union{Const, Duplicated};
    kwargs...
)

    (fact, dfact, dA, x) = cache
    ldiv!(B.dval, fact.dval, dret)
    dfact .= -x .* B.dval'
end