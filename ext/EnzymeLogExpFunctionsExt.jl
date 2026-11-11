module EnzymeLogExpFunctionsExt

using LogExpFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(LogExpFunctions.xlogy)] = (:xlogy_jl, 2, nothing)
end

Enzyme.EnzymeRules.@easy_rule(
    LogExpFunctions.logistic(x::Real),
    (Ω * LogExpFunctions.logistic(-x),),
)

Enzyme.EnzymeRules.@easy_rule(
    LogExpFunctions.xlogx(x::Real),
    (one(Ω) + log(x),),
)

Enzyme.EnzymeRules.@easy_rule(
    LogExpFunctions.log1pexp(x::Real),
    (LogExpFunctions.logistic(x),),
)

Enzyme.EnzymeRules.@easy_rule(
    LogExpFunctions.logaddexp(x::Real, y::Real),
    (exp(x - Ω), exp(y - Ω)),
)

Enzyme.EnzymeRules.@easy_rule(
    LogExpFunctions.logsubexp(x::Real, y::Real),
    (x > y ? exp(x - Ω) : -exp(x - Ω), x > y ? -exp(y - Ω) : exp(y - Ω)),
)

end
