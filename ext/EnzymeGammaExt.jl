module EnzymeGammaExt

using Gamma
using Enzyme

Enzyme.EnzymeRules.@easy_rule(
    Gamma.gamma(x::AbstractFloat),
    @setup(),
    (Ω * Gamma.digamma(x),),
)

end
