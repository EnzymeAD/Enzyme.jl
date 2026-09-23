module EnzymeLogExpFunctionsExt

using LogExpFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(LogExpFunctions.xlogy)] = (:xlogy_jl, 2, nothing)
end

# xlogx(x) = x * log(x); the explicit partial log(x) + 1 avoids the 0 * (1/x) NaN
# the product rule forms at x = 0. Partial as in LogExpFunctions' ChainRulesCore ext.
# x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/3582
EnzymeRules.@easy_rule(LogExpFunctions.xlogx(x::Real), (log(x) + 1,))

end
