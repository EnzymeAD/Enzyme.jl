module EnzymeSpecialFunctionsExt

using SpecialFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(SpecialFunctions._logabsgamma)] = (:logabsgamma, 1, (:digamma, typeof(SpecialFunctions.digamma)))
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besselj)] = (:cmplx_jn, 2, nothing)
    Enzyme.Compiler.cmplx_known_ops[typeof(SpecialFunctions.besselk)] = (:cmplx_kn, 2, nothing)
end

end
