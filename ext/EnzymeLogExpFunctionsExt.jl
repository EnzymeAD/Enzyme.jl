module EnzymeLogExpFunctionsExt

using LogExpFunctions
using Enzyme

function __init__()
    Enzyme.Compiler.known_ops[typeof(LogExpFunctions.xlogy)] = (:xlogy_jl, 2, nothing)
end

end
