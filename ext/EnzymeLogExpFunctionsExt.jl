module EnzymeLogExpFunctionsExt

using LogExpFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(LogExpFunctions.xlogy)] = (:xlogy_jl, 2, nothing)
end

# Reverse mode formed 1/exp(-x), which overflows to Inf and leaves the derivative
# NaN. LogExpFunctions' ChainRulesCore ext writes this as Ω * (1 - Ω); logistic(-x)
# replaces 1 - Ω, which overflows nowhere but cancels to 0 for x > 37.
# x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/3583
EnzymeRules.@easy_rule(
    LogExpFunctions.logistic(x::Real),
    (Ω * LogExpFunctions.logistic(-x),),
)

end
