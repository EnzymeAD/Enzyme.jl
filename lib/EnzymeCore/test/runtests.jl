using Test
using EnzymeCore

import EnzymeCore.EnzymeRules: forward, has_frule_from_sig

g(x) = x ^ 2
function forward(::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@test has_frule_from_sig(Base.signature_type(g, Tuple{Float64}))

f(;kwargs) = 1.0

function forward(::Const{typeof(f)}, ::Type{<:Const}; kwargs...)
    return Const(f(; kwargs...))
end

@test has_frule_from_sig(Base.signature_type(f, Tuple{}))


