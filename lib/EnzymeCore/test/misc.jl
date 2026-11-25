using Test
using EnzymeCore
import EnzymeCore.EnzymeRules: forward, has_frule_from_sig

g(x) = x^2
function forward(config, ::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@test has_frule_from_sig(Base.signature_type(g, Tuple{Float64}))

f(; kwargs) = 1.0

function forward(config, ::Const{typeof(f)}, ::Type{<:Const}; kwargs...)
    return Const(f(; kwargs...))
end

@test has_frule_from_sig(Base.signature_type(f, Tuple{}))

# Signatures with `where` variables.
struct Scale{T}
    factor::T
end
(s::Scale)(x) = s.factor .* x

function forward(config, ::Const{F}, ::Type{<:Const}, x::Const{Float64}) where {F <: Scale}
    return Const(x.val)
end
function forward(config, ::Const{Scale{T}}, ::Type{<:Const}, x::Const{Vector{T}}) where {T <: Integer}
    return Const(x.val)
end

@test has_frule_from_sig(Tuple{F, Float64} where {F <: Scale})
@test has_frule_from_sig(Tuple{Scale{T}, Vector{T}} where {T <: Integer})
@test !has_frule_from_sig(Tuple{F, Int} where {F <: Scale})

data = [1.0, 2.0, 3.0, 4.0]

d = @view data[2:end]
y = @view data[3:end]
@test_skip @test_throws AssertionError Duplicated(d, y)

@test_throws ErrorException Active(data)
@test_skip @test_throws ErrorException Active(d)
