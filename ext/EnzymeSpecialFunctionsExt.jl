module EnzymeSpecialFunctionsExt

using SpecialFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(SpecialFunctions.logabsgamma)] = (:logabsgamma, 1, (:digamma, typeof(SpecialFunctions.digamma)))
end

end
