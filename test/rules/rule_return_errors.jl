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

# The data of the struct return is active, but its GC pointers are not. Enzyme
# does not support this mismatch and must report it as an error.
@noinline mixed_roots(x, n) = (x, n)
function forward(config, ::Const{typeof(mixed_roots)}, ::Type, x::Duplicated, n::Const)
    primal = mixed_roots(x.val, n.val)
    shadow = (x.dval, n.val)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(primal, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return primal
    else
        return nothing
    end
end
mixed_roots_caller(x, n) = (r = mixed_roots(x, n); r[1] * r[2][1])

@static if VERSION >= v"1.12"
    @testset "Return Roots Activity Mismatch Error" begin
        @test_throws Enzyme.Compiler.CallingConventionMismatchError autodiff(
            Forward, mixed_roots_caller, Duplicated(3.0, 1.0), Const([5])
        )
    end
end

end # RuleReturnErrors
