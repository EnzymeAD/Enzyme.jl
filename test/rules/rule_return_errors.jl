module RuleReturnErrors

using Enzyme
using Enzyme.EnzymeRules
using Test

import .EnzymeRules: forward, augmented_primal, reverse

function f_kw(out)
    out[1] *= 2
    return nothing
end

function forward(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, x::Duplicated)
    f_kw(x.val)
    return 2
end

function augmented_primal(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, x::Duplicated)
    f_kw(x.val)
    return EnzymeRules.AugmentedReturn(2, nothing, nothing)
end

function reverse(config, ::Const{typeof(f_kw)}, ::Type{<:Const}, tape, x::Duplicated)
    f_kw(x.dval)
    return (nothing,)
end

function g_kw(out)
    out[1] *= 2
    return nothing
end

function augmented_primal(config, ::Const{typeof(g_kw)}, ::Type{<:Const}, x::Duplicated)
    f_kw(x.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function reverse(config, ::Const{typeof(g_kw)}, ::Type{<:Const}, tape, x::Duplicated)
    f_kw(x.dval)
    return ()
end

@testset "Forward Return Error" begin
    x = [2.7]
    dx = [3.1]
    @test_throws Enzyme.Compiler.ForwardRuleReturnError autodiff(Forward, f_kw, Duplicated(x, dx))
end

@testset "Augmented Return Error" begin
    x = [2.7]
    dx = [3.1]
    @test_throws Enzyme.Compiler.AugmentedRuleReturnError autodiff(Reverse, f_kw, Duplicated(x, dx))
end

@testset "Reverse Return Error" begin
    x = [2.7]
    dx = [3.1]
    @test_throws Enzyme.Compiler.ReverseRuleReturnError autodiff(Reverse, g_kw, Duplicated(x, dx))
end

end # RuleReturnErrors
