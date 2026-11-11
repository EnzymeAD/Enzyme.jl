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

function inner_mixed(f::F) where {F}
    s = f()
    return (s, 2.3 * s[1])
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(inner_mixed)}, ::Type, f
    )
    true_primal = inner_mixed(f.val)
    primal = EnzymeRules.needs_primal(config) ? true_primal : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        make_zero(true_primal)
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(inner_mixed)}, shadow, tape, f
    )
    mz = Enzyme.Compiler.splatnew(typeof(f.val), (2.7 + 100 * tape[1][1],))
    return (mz,)
end

F_mixed_good(x) = inner_mixed(() -> [x])[1][1]
F_mixed_bad_inline(x) = inner_mixed(() -> [x])[2]
F_mixed_bad_both(x) = (r = inner_mixed(() -> [x]); r[1][1] + r[2])

@noinline inner_mixed_vec(v::Vector{Float64}) = (2 .* v, 2.3 * v[1])

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(inner_mixed_vec)}, ::Type, v
    )
    true_primal = inner_mixed_vec(v.val)
    primal = EnzymeRules.needs_primal(config) ? true_primal : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            make_zero(true_primal)
        else
            ntuple(_ -> make_zero(true_primal), Val(EnzymeRules.width(config)))
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, ::Const{typeof(inner_mixed_vec)}, ::Type, tape, v
    )
    if !(v isa Const) && tape !== nothing
        if EnzymeRules.width(config) == 1
            v.dval .+= 2 .* tape[1]
        else
            for b in 1:EnzymeRules.width(config)
                v.dval[b] .+= 2 .* tape[b][1]
            end
        end
    end
    return (nothing,)
end

F_mixed_vec_good(v) = inner_mixed_vec(v)[1][1]
F_mixed_vec_bad(v) = inner_mixed_vec(v)[2]

F_mixed_dyn_good(v) = Base.inferencebarrier(w -> inner_mixed_vec(w))(v)[1][1]
F_mixed_dyn_bad(v) = Base.inferencebarrier(w -> inner_mixed_vec(w))(v)[2]


@testset "Simple Mixed Return" begin
    @test autodiff(Forward, F_good, Duplicated(0.3, 3.1))[1] ≈ 2.7

    @test autodiff(Reverse, F_good, Active(0.3))[1][1] ≈ 102.7
    @test autodiff(Reverse, F_mixed_good, Active(0.3))[1][1] ≈ 102.7

    @test_throws Enzyme.Compiler.MixedReturnException autodiff(Reverse, F_mixed_bad_inline, Active(0.3))
    @test_throws Enzyme.Compiler.MixedReturnException autodiff(Reverse, F_mixed_bad_both, Active(0.3))

    v = [0.3]
    dv1, dv2 = [0.0], [0.0]
    autodiff(Reverse, F_mixed_vec_good, Active, BatchDuplicated(v, (dv1, dv2)))
    @test dv1 ≈ [2.0]
    @test dv2 ≈ [2.0]

    dv1, dv2 = [0.0], [0.0]
    @test_throws Enzyme.Compiler.MixedReturnException autodiff(Reverse, F_mixed_vec_bad, Active, BatchDuplicated(v, (dv1, dv2)))

    @static if VERSION < v"1.12"
        @test Enzyme.gradient(Reverse, F_mixed_dyn_good, [0.3])[1] ≈ [2.0]
        @test_throws Enzyme.Compiler.MixedReturnException Enzyme.gradient(Reverse, F_mixed_dyn_bad, [0.3])
    end
end
