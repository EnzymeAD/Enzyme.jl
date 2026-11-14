using Enzyme
using Test
using Enzyme.EnzymeRules

function inner(f::F) where {F}
    s = f()
    return (s, 2.3)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(inner)}, ::Type, f
    )
    true_primal = inner(f.val)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(true_primal, ([2.7], 123.0))
    elseif EnzymeRules.needs_shadow(config)
        return ([2.7], 123.0)
    elseif EnzymeRules.needs_primal(config)
        return true_primal
    else
        return nothing
    end

    primal = EnzymeRules.needs_primal(config) ? true_primal : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        make_zero(true_primal)
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(inner)}, ::Type, f
    )
    true_primal = inner(f.val)
    primal = EnzymeRules.needs_primal(config) ? true_primal : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        make_zero(true_primal)
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(inner)}, shadow, tape, f
    )
    mz = Enzyme.Compiler.splatnew(typeof(f.val), (2.7 + 100 * tape[1][1],))
    return (mz,)
end

F_good(x) = inner(() -> [x])[1][1]


@testset "Simple Mixed Return" begin
    @test autodiff(Forward, F_good, Duplicated(0.3, 3.1))[1] ≈ 2.7

    if VERSION < v"1.12"
        @test_throws Enzyme.Compiler.MixedReturnException autodiff(Reverse, F_good, Active(0.3))
    else
        @test autodiff(Reverse, F_good, Active(0.3))[1][1] ≈ 102.7
    end
end
