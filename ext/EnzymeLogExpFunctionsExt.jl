module EnzymeLogExpFunctionsExt

using LogExpFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(LogExpFunctions.xlogy)] = (:xlogy_jl, 2, nothing)
end

# xlog1py(x, y) = x * log1p(y); the iszero(x) branch is constant, so the primal
# differentiates to 0 there. Partials from LogExpFunctions' ChainRulesCore ext.
# x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/3579
EnzymeRules.@easy_rule(
    LogExpFunctions.xlog1py(x::Real, y::Real),
    (log1p(y), x / (1 + y)),
)

# xexpy(x, y) = x * exp(y), same zero branch, hence ∂/∂y is the primal Ω.
EnzymeRules.@easy_rule(
    LogExpFunctions.xexpy(x::Real, y::Real),
    (exp(y), Ω),
)

end
