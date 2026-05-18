
function __init__()
    if VERSION >= v"1.11.0"
        @warn """You are using Julia v1.11 or later!"
                 Julia 1.11 changes the default Array type to contain a triply-nested pointer, rather than a doubly nested pointer."
                 This may cause programs (primal but especially derivatives) to be slower, or fail to differentiate with default settings when they previously worked on 1.10."
                 If you find issues, please report at https://github.com/EnzymeAD/Enzyme.jl/issues/new and try Julia 1.10 in the interim."""
    end
end