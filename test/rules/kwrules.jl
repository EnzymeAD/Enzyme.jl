module KWForwardRules

using Enzyme
using Enzyme.EnzymeRules
using Test

import .EnzymeRules: forward

function f_kw(x; kwargs...)
    return x^2
end

function forward(config, ::Const{typeof(f_kw)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated; kwargs...)
    return 10 + 2 * x.val * x.dval
end

@test autodiff(Forward, f_kw, Duplicated(2.0, 1.0))[1] ≈ 14.0

# TODO: autodiff wrapper with kwargs support

g(x, y) = f_kw(x; val = y)
@test autodiff(Forward, g, Duplicated(2.0, 1.0), Const(42.0))[1] ≈ 14.0

function f_kw2(x; kwargs...)
    return x^2
end

function forward(config, ::Const{typeof(f_kw2)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return 10 + 2 * x.val * x.dval
end

# Test that this errors due to missing kwargs in rule definition
g2(x, y) = f_kw2(x; val = y)
@test_throws MethodError autodiff(Forward, g2, Duplicated(2.0, 1.0), Const(42.0))[1] ≈ 14.0

function f_kw3(x; val = nothing)
    return x^2
end

function forward(config, ::Const{typeof(f_kw3)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated; dval = nothing)
    return 10 + 2 * x.val * x.dval
end

# Test that this errors due to missing kwargs in rule definition
g3(x, y) = f_kw3(x; val = y)
@test_throws MethodError autodiff(Forward, g3, Duplicated(2.0, 1.0), Const(42.0))[1] ≈ 14.0

function f_kw4(x; y = 2.0)
    return x * y
end

function forward(config, ::Const{typeof(f_kw4)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated; y)
    return 1000 * y + 2 * x.val * x.dval
end

# Test that this errors due to missing kwargs in rule definition
g4(x, y) = f_kw4(x; y)
@test autodiff(Forward, g4, Duplicated(2.0, 1.0), Const(42.0))[1] ≈ 42004.0
@test_throws Enzyme.Compiler.NonConstantKeywordArgException autodiff(Forward, g4, Duplicated(2.0, 1.0), Duplicated(42.0, 1.0))[1]

end # KWForwardRules
