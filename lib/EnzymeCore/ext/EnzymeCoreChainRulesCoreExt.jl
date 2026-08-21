module EnzymeCoreChainRulesCoreExt

using ChainRulesCore
using EnzymeCore

ChainRulesCore.@non_differentiable EnzymeCore.ignore_derivatives(x)

end #module
