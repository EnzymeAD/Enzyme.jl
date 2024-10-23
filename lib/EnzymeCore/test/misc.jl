using Test
using EnzymeCore
import EnzymeCore.EnzymeRules: forward, has_frule_from_sig

g(x) = x ^ 2
function forward(config, ::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@test has_frule_from_sig(Base.signature_type(g, Tuple{Float64}))

f(;kwargs) = 1.0

function forward(config, ::Const{typeof(f)}, ::Type{<:Const}; kwargs...)
    return Const(f(; kwargs...))
end

@test has_frule_from_sig(Base.signature_type(f, Tuple{}))

data = [1.0, 2.0, 3.0, 4.0]

d = @view data[2:end]
y = @view data[3:end]
@test_skip @test_throws AssertionError Duplicated(d, y)

@test_throws ErrorException Active(data)
@test_skip @test_throws ErrorException Active(d)
