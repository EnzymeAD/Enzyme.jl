module BijectorsIntegrationTests

using Bijectors: Bijectors
using Enzyme: Enzyme
using FiniteDifferences: FiniteDifferences
using LinearAlgebra: I, cholesky, Hermitian
using Random: randn
using StableRNGs: StableRNG
using Test: @test, @test_broken, @testset

rng = StableRNG(23)

"""
Enum type for choosing Enzyme autodiff modes.
"""
@enum ModeSelector Neither Forward Reverse Both

"""
Type for specifying a test case for `Enzyme.gradient`.

The test will check the accuracy of the gradient of `func` at `value` against `finitediff`,
with both forward and reverse mode autodiff. `name` is for diagnostic printing.
`runtime_activity`, `broken`, `skip` are for specifying whether to use
`Enzyme.set_runtime_activity` or not, whether the test is broken, and whether the test is so
broken we can't even run `@test_broken` on it (because it crashes Julia). All of them take
values `Neither`, `Forward`, `Reverse` or `Both`, to specify which mode to apply the setting
to. `splat` is for specifying whether to call the function as `func(value)` or as
`func(value...)`.

Default values are `name=nothing`, `runtime_activity=Neither`, `broken=Neither`,
`skip=Neither`, and `splat=false`.
"""
struct TestCase
    func::Function
    value
    name::Union{String, Nothing}
    runtime_activity::ModeSelector
    broken::ModeSelector
    skip::ModeSelector
    splat::Bool
end

# Default values for most arguments.
function TestCase(
        f, value;
        name = nothing, runtime_activity = Neither, broken = Neither, skip = Neither, splat = false
    )
    return TestCase(f, value, name, runtime_activity, broken, skip, splat)
end

"""
Test Enzyme.gradient, both Forward and Reverse mode, against FiniteDifferences.grad.
"""
function test_grad(case::TestCase; rtol = 1.0e-6, atol = 1.0e-6)
    @nospecialize
    f = case.func
    # We'll call the function as f(x...), so wrap in a singleton tuple if need be.
    x = case.splat ? case.value : (case.value,)
    finitediff = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), f, x...)[1]

    f_mode = if (case.runtime_activity === Both || case.runtime_activity === Forward)
        Enzyme.set_runtime_activity(Enzyme.Forward)
    else
        Enzyme.Forward
    end
    r_mode = if (case.runtime_activity === Both || case.runtime_activity === Reverse)
        Enzyme.set_runtime_activity(Enzyme.Reverse)
    else
        Enzyme.Reverse
    end

    if !(case.skip === Forward) && !(case.skip === Both)
        if case.broken === Both || case.broken === Forward
            @test_broken(
                Enzyme.gradient(f_mode, Enzyme.Const(f), x...)[1] ≈ finitediff,
                rtol = rtol,
                atol = atol,
            )
        else
            @test(
                Enzyme.gradient(f_mode, Enzyme.Const(f), x...)[1] ≈ finitediff,
                rtol = rtol,
                atol = atol,
            )
        end
    end

    if !(case.skip === Reverse) && !(case.skip === Both)
        if case.broken === Both || case.broken === Reverse
            @test_broken(
                Enzyme.gradient(r_mode, Enzyme.Const(f), x...)[1] ≈ finitediff,
                rtol = rtol,
                atol = atol,
            )
        else
            @test(
                Enzyme.gradient(r_mode, Enzyme.Const(f), x...)[1] ≈ finitediff,
                rtol = rtol,
                atol = atol,
            )
        end
    end
    return nothing
end

"""
A helper function that returns a TestCase that evaluates sum(bijector(inverse(bijector)(x)))
"""
function sum_b_binv_test_case(
        bijector, dim; runtime_activity = Neither, name = nothing, broken = Neither, skip = Neither
    )
    if name === nothing
        name = string(bijector)
    end
    b_inv = Bijectors.inverse(bijector)
    return TestCase(
        x -> sum(bijector(b_inv(x))),
        randn(rng, dim);
        runtime_activity = runtime_activity, name = name, broken = broken, skip = skip
    )
end

@testset "Bijectors integration tests" begin
    test_cases = TestCase[
        sum_b_binv_test_case(Bijectors.VecCorrBijector(), 3),
        sum_b_binv_test_case(Bijectors.VecCorrBijector(), (1, 1)),
        sum_b_binv_test_case(Bijectors.VecCorrBijector(), 0),
        sum_b_binv_test_case(Bijectors.CorrBijector(), (3, 3)),
        sum_b_binv_test_case(Bijectors.CorrBijector(), (0, 0)),
        sum_b_binv_test_case(Bijectors.VecCholeskyBijector(:L), 3),
        sum_b_binv_test_case(Bijectors.VecCholeskyBijector(:L), 0),
        sum_b_binv_test_case(Bijectors.VecCholeskyBijector(:U), 3),
        sum_b_binv_test_case(Bijectors.VecCholeskyBijector(:U), 0),
        sum_b_binv_test_case(Bijectors.Coupling(Bijectors.Shift, Bijectors.PartitionMask(3, [1], [2])), 3),
        sum_b_binv_test_case(
            Bijectors.InvertibleBatchNorm(3),
            (3, 3);
            runtime_activity = (VERSION >= v"1.11" ? Both : Neither)
        ),
        sum_b_binv_test_case(Bijectors.LeakyReLU(0.2), 3),
        sum_b_binv_test_case(Bijectors.Logit(0.1, 0.3), 3),
        sum_b_binv_test_case(Bijectors.PDBijector(), (3, 3)),
        sum_b_binv_test_case(Bijectors.PDVecBijector(), 3),
        sum_b_binv_test_case(
            Bijectors.Permute(
                [
                    0 1 0;
                    1 0 0;
                    0 0 1
                ]
            ),
            (3, 3),
        ),
        # NOTE(penelopeysm) This requires runtime activity, but forward-mode fails as this
        # calls gemm! and runtime activity is not yet supported for BLAS calls.
        sum_b_binv_test_case(Bijectors.PlanarLayer(3), (3, 3); runtime_activity = Reverse, broken = Forward),
        sum_b_binv_test_case(Bijectors.RadialLayer(3), 3),
        sum_b_binv_test_case(Bijectors.Reshape((2, 3), (3, 2)), (2, 3)),
        sum_b_binv_test_case(Bijectors.Scale(0.2), 3),
        sum_b_binv_test_case(Bijectors.Shift(-0.4), 3),
        sum_b_binv_test_case(Bijectors.SignFlip(), 3),
        sum_b_binv_test_case(Bijectors.SimplexBijector(), 3),
        sum_b_binv_test_case(Bijectors.TruncatedBijector(-0.2, 0.5), 3),

        # Below, some test cases that don't fit the sum_b_binv_test_case mold.
        TestCase(
            function (x)
                return sum(Bijectors.PDVecBijector()(x * x' + I))
            end,
            randn(rng, 4, 4),
            name = "PDVecBijector forward only",
        ),
        TestCase(
            function (x)
                binv = Bijectors.inverse(Bijectors.PDVecBijector())
                return sum(cholesky(Hermitian(binv(x), :L)).L)
            end,
            Bijectors.PDVecBijector()((x -> x * x' + I)(randn(rng, 4, 4))),
            name = "PDVecBijector inverse only + lower Cholesky",
        ),
        TestCase(
            function (x)
                binv = Bijectors.inverse(Bijectors.PDVecBijector())
                return sum(cholesky(Hermitian(binv(x), :U)).U)
            end,
            Bijectors.PDVecBijector()((x -> x * x' + I)(randn(rng, 4, 4))),
            name = "PDVecBijector inverse only + upper Cholesky",
        ),

        TestCase(
            function (x)
                b = Bijectors.RationalQuadraticSpline([-0.2, 0.1, 0.5], [-0.3, 0.3, 0.9], [1.0, 0.2, 1.0])
                binv = Bijectors.inverse(b)
                return sum(binv(b(x)))
            end,
            randn(rng);
            name = "RationalQuadraticSpline on scalar",
        ),

        TestCase(
            function (x)
                b = Bijectors.OrderedBijector()
                binv = Bijectors.inverse(b)
                return sum(binv(b(x)))
            end,
            randn(rng, 7);
            name = "OrderedBijector",
        ),

        TestCase(
            function (x)
                layer = Bijectors.PlanarLayer(x[1:2], x[3:4], x[5:5])
                flow = Bijectors.transformed(Bijectors.MvNormal(zeros(2), I), layer)
                x = x[6:7]
                return Bijectors.logpdf(flow.dist, x) - Bijectors.logabsdetjac(flow.transform, x)
            end,
            randn(rng, 7);
            name = "PlanarLayer7 forward"
        ),

        TestCase(
            function (x)
                layer = Bijectors.PlanarLayer(x[1:2], x[3:4], x[5:5])
                flow = Bijectors.transformed(Bijectors.MvNormal(zeros(2), I), layer)
                x = reshape(x[6:end], 2, :)
                return sum(Bijectors.logpdf(flow.dist, x) - Bijectors.logabsdetjac(flow.transform, x))
            end,
            randn(rng, 11);
            name = "PlanarLayer11 forward"
        ),

        TestCase(
            function (x)
                layer = Bijectors.PlanarLayer(x[1:2], x[3:4], x[5:5])
                flow = Bijectors.transformed(Bijectors.MvNormal(zeros(2), I), Bijectors.inverse(layer))
                x = x[6:7]
                return Bijectors.logpdf(flow.dist, x) - Bijectors.logabsdetjac(flow.transform, x)
            end,
            randn(rng, 7);
            name = "PlanarLayer7 inverse"
        ),

        TestCase(
            function (x)
                layer = Bijectors.PlanarLayer(x[1:2], x[3:4], x[5:5])
                flow = Bijectors.transformed(Bijectors.MvNormal(zeros(2), I), Bijectors.inverse(layer))
                x = reshape(x[6:end], 2, :)
                return sum(Bijectors.logpdf(flow.dist, x) - Bijectors.logabsdetjac(flow.transform, x))
            end,
            randn(rng, 11);
            name = "PlanarLayer11 inverse"
        ),
    ]

    @testset "$(case.name)" for case in test_cases
        test_grad(case)
    end
end

end
