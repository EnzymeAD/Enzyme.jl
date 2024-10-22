module DistributionsIntegrationTests

using Distributions
using Enzyme: Enzyme
using FillArrays: Fill
using FiniteDifferences: FiniteDifferences
using LinearAlgebra: Diagonal, Hermitian, I, Symmetric, diag, cholesky
using PDMats: PDMat
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
`runtime_activity` and `broken` are for specifying whether to use
`Enzyme.set_runtime_activity` or not and whether the test is broken. Both of them taken
values `Neither`, `Forward`, `Reverse` or `Both`, to specify which mode to apply the setting
to. `splat` is for specifying whether to splat the value into the function or not. If yes,
then value should be an iterable of arguments rather than a single argument.

A constructor is also provided for giving a `Distribution` instead of a function, in which
case the function is `x -> logpdf(distribution, x)`.

Default values are `name=nothing` or `name=string(nameof(typeof(distribution)))`,
`runtime_activity=Neither`, `broken=Neither` and `splat=false`.
"""
struct TestCase
    func::Function
    value
    name::Union{String, Nothing}
    runtime_activity::ModeSelector
    broken::ModeSelector
    splat::Bool
end

# Turn a distribution into a call to logpdf.
function TestCase(d::Distribution, value, name, runtime_activity, broken, splat)
    TestCase(x -> logpdf(d, x), value, name, runtime_activity, broken, splat)
end

# Defaults for name, runtime_activity and broken.
function TestCase(
    f, value;
    name=nothing, runtime_activity=Neither, broken=Neither, splat=false
)
    return TestCase(f, value, name, runtime_activity, broken, splat)
end

# Default name for a Distribution.
function TestCase(
    d::Distribution, value;
    name=string(nameof(typeof(d))), runtime_activity=Neither, broken=Neither, splat=false
)
    return TestCase(d, value, name, runtime_activity, broken, splat)
end

"""
Test Enzyme.gradient, both Forward and Reverse mode, against FiniteDifferences.grad.
"""
function test_grad(case::TestCase; rtol=1e-6, atol=1e-6)
    @nospecialize
    f = case.func
    # We'll call the function as f(x...), so wrap in a singleton tuple if need be.
    x = case.splat ? case.value : (case.value,)
    finitediff = collect(
        FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), f, x...)[1]
    )

    """
    A function for reshaping the output of Enzyme.gradient to match the shape of the
    output of FiniteDifferences.grad.
    """
    shape_grad(x) = reshape(collect(x), size(finitediff))

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

    if case.broken === Both || case.broken === Forward
        @test_broken(
            shape_grad(Enzyme.gradient(f_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    else
        @test(
            shape_grad(Enzyme.gradient(f_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    end

    if case.broken === Both || case.broken === Reverse
        @test_broken(
            shape_grad(Enzyme.gradient(r_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    else
        @test(
            shape_grad(Enzyme.gradient(r_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    end
    return nothing
end

_sym(A) = A'A
_pdmat(A) = PDMat(_sym(A) + 5I)

@testset "Distributions integration tests" begin
    test_cases = TestCase[

        #
        # Univariate
        #

        TestCase(Arcsine(), 0.5),
        TestCase(Arcsine(-0.3, 0.9), 0.5),
        TestCase(Arcsine(0.5, 1.1), 1.0),
        TestCase(Beta(1.1, 1.1), 0.5),
        TestCase(Beta(1.1, 1.5), 0.9),
        TestCase(Beta(1.6, 1.5), 0.5),
        TestCase(BetaPrime(1.1, 1.1), 0.5),
        TestCase(BetaPrime(1.1, 1.6), 0.5),
        TestCase(BetaPrime(1.6, 1.3), 0.9),
        TestCase(Biweight(1.0, 2.0), 0.5),
        TestCase(Biweight(-0.5, 2.5), -0.45),
        TestCase(Biweight(0.0, 1.0), 0.3),
        TestCase(Cauchy(), -0.5),
        TestCase(Cauchy(1.0), 0.99),
        TestCase(Cauchy(1.0, 0.1), 1.01),
        TestCase(Chi(2.5), 0.5),
        TestCase(Chi(5.5), 1.1),
        TestCase(Chi(0.1), 0.7),
        TestCase(Chisq(2.5), 0.5),
        TestCase(Chisq(5.5), 1.1),
        TestCase(Chisq(0.1), 0.7),
        TestCase(Cosine(0.0, 1.0), 0.5),
        TestCase(Cosine(-0.5, 2.0), -0.1),
        TestCase(Cosine(0.4, 0.5), 0.0),
        TestCase(Epanechnikov(0.0, 1.0), 0.5),
        TestCase(Epanechnikov(-0.5, 1.2), -0.9),
        TestCase(Epanechnikov(-0.4, 1.6), 0.1),
        TestCase(Erlang(), 0.5),
        TestCase(Erlang(), 0.1),
        TestCase(Erlang(), 0.9),
        TestCase(Exponential(), 0.1),
        TestCase(Exponential(0.5), 0.9),
        TestCase(Exponential(1.4), 0.05),
        TestCase(FDist(2.1, 3.5), 0.7),
        TestCase(FDist(1.4, 5.4), 3.5),
        TestCase(FDist(5.5, 3.3), 7.2),
        TestCase(Frechet(), 0.1),
        TestCase(Frechet(), 1.1),
        TestCase(Frechet(1.5, 2.4), 0.1),
        TestCase(Gamma(0.9, 1.2), 4.5),
        TestCase(Gamma(0.5, 1.9), 1.5),
        TestCase(Gamma(1.8, 3.2), 1.0),
        TestCase(GeneralizedExtremeValue(0.3, 1.3, 0.1), 2.4),
        TestCase(GeneralizedExtremeValue(-0.7, 2.2, 0.4), 1.1),
        TestCase(GeneralizedExtremeValue(0.5, 0.9, -0.5), -7.0),
        TestCase(GeneralizedPareto(0.3, 1.1, 1.1), 5.0),
        TestCase(GeneralizedPareto(-0.25, 0.9, 0.1), 0.8),
        TestCase(GeneralizedPareto(0.3, 1.1, -5.1), 0.31),
        TestCase(Gumbel(0.1, 0.5), 0.1),
        TestCase(Gumbel(-0.5, 1.1), -0.1),
        TestCase(Gumbel(0.3, 0.1), 0.3),
        TestCase(InverseGaussian(0.1, 0.5), 1.1),
        TestCase(InverseGaussian(0.2, 1.1), 3.2),
        TestCase(InverseGaussian(0.1, 1.2), 0.5),
        TestCase(JohnsonSU(0.1, 0.95, 0.1, 1.1), 0.1),
        TestCase(JohnsonSU(0.15, 0.9, 0.12, 0.94), 0.5),
        TestCase(JohnsonSU(0.1, 0.95, 0.1, 1.1), -0.3),
        TestCase(Kolmogorov(), 1.1),
        TestCase(Kolmogorov(), 0.9),
        TestCase(Kolmogorov(), 1.5),
        TestCase(Kumaraswamy(2.0, 5.0), 0.71),
        TestCase(Kumaraswamy(0.1, 5.0), 0.2),
        TestCase(Kumaraswamy(0.5, 4.5), 0.1),
        TestCase(Laplace(0.1, 1.0), 0.2),
        TestCase(Laplace(-0.5, 2.1), 0.5),
        TestCase(Laplace(-0.35, 0.4), -0.3),
        TestCase(Levy(0.1, 0.9), 4.1),
        TestCase(Levy(0.5, 0.9), 0.6),
        TestCase(Levy(1.1, 0.5), 2.2),
        TestCase(Lindley(0.5), 2.1),
        TestCase(Lindley(1.1), 3.1),
        TestCase(Lindley(1.9), 3.5),
        TestCase(Logistic(0.1, 1.2), 1.1),
        TestCase(Logistic(0.5, 0.7), 0.6),
        TestCase(Logistic(-0.5, 0.1), -0.4),
        TestCase(LogitNormal(0.1, 1.1), 0.5),
        TestCase(LogitNormal(0.5, 0.7), 0.6),
        TestCase(LogitNormal(-0.12, 1.1), 0.1),
        TestCase(LogNormal(0.0, 1.0), 0.5),
        TestCase(LogNormal(0.5, 1.0), 0.5),
        TestCase(LogNormal(-0.1, 1.3), 0.75),
        TestCase(LogUniform(0.1, 0.9), 0.75),
        TestCase(LogUniform(0.15, 7.8), 7.1),
        TestCase(LogUniform(2.0, 3.0), 2.1),
        TestCase(NoncentralBeta(1.1, 1.1, 1.2), 0.8; broken=Both), # foreigncall (Rmath.dnbeta).
        TestCase(NoncentralChisq(2, 3.0), 10.0; broken=Both), # foreigncall (Rmath.dnchisq).
        TestCase(NoncentralF(2, 3, 1.1), 4.1; broken=Both), # foreigncall (Rmath.dnf).
        TestCase(NoncentralT(1.3, 1.1), 0.1; broken=Both), # foreigncall (Rmath.dnt).
        TestCase(Normal(), 0.1),
        TestCase(Normal(0.0, 1.0), 1.0),
        TestCase(Normal(0.5, 1.0), 0.05),
        TestCase(Normal(0.0, 1.5), -0.1),
        TestCase(Normal(-0.1, 0.9), -0.3),
        # foreigncall -- https://github.com/JuliaMath/SpecialFunctions.jl/blob/be1fa06fee58ec019a28fb0cd2b847ca83a5af9a/src/bessel.jl#L265
        TestCase(NormalInverseGaussian(0.0, 1.0, 0.2, 0.1), 0.1; broken=Both),
        TestCase(Pareto(1.0, 1.0), 3.5),
        TestCase(Pareto(1.1, 0.9), 3.1),
        TestCase(Pareto(1.0, 1.0), 1.4),
        TestCase(PGeneralizedGaussian(0.2), 5.0),
        TestCase(PGeneralizedGaussian(0.5, 1.0, 0.3), 5.0),
        TestCase(PGeneralizedGaussian(-0.1, 11.1, 6.5), -0.3),
        TestCase(Rayleigh(0.5), 0.6),
        TestCase(Rayleigh(0.9), 1.1),
        TestCase(Rayleigh(0.55), 0.63),
        TestCase(Rician(0.5, 1.0), 2.1; broken=Both),  # foreigncall (Rmath.dnchisq).
        TestCase(Semicircle(1.0), 0.9),
        TestCase(Semicircle(5.1), 5.05),
        TestCase(Semicircle(0.5), -0.1),
        TestCase(SkewedExponentialPower(0.1, 1.0, 0.97, 0.7), -2.0),
        TestCase(SkewedExponentialPower(0.15, 1.0, 0.97, 0.7), -2.0),
        TestCase(SkewedExponentialPower(0.1, 1.1, 0.99, 0.7), 0.5),
        TestCase(SkewNormal(0.0, 1.0, -1.0), 0.1),
        TestCase(SkewNormal(0.5, 2.0, 1.1), 0.1),
        TestCase(SkewNormal(-0.5, 1.0, 0.0), 0.1),
        TestCase(SymTriangularDist(0.0, 1.0), 0.5),
        TestCase(SymTriangularDist(-0.5, 2.1), -2.0),
        TestCase(SymTriangularDist(1.7, 0.3), 1.75),
        TestCase(TDist(1.1), 99.1),
        TestCase(TDist(10.1), 25.0),
        TestCase(TDist(2.1), -89.5),
        TestCase(TriangularDist(0.0, 1.5, 0.5), 0.45),
        TestCase(TriangularDist(0.1, 1.4, 0.45), 0.12),
        TestCase(TriangularDist(0.0, 1.5, 0.5), 0.2),
        TestCase(Triweight(1.0, 1.0), 1.0),
        TestCase(Triweight(1.1, 2.1), 1.0),
        TestCase(Triweight(1.9, 10.0), -0.1),
        TestCase(Uniform(0.0, 1.0), 0.2),
        TestCase(Uniform(-0.1, 1.1), 1.0),
        TestCase(Uniform(99.5, 100.5), 100.0),
        TestCase(VonMises(0.5), 0.1),
        TestCase(VonMises(0.3), -0.1),
        TestCase(VonMises(0.2), -0.5),
        TestCase(Weibull(0.5, 1.0), 0.45),
        TestCase(Weibull(0.3, 1.1), 0.66),
        TestCase(Weibull(0.75, 1.3), 0.99),

        #
        # Multivariate
        #

        TestCase(MvNormal(1, 1.5), [-0.3]),
        TestCase(MvNormal(2, 0.5), [0.2, -0.3]),
        TestCase(MvNormal([1.0]), [-0.1]),
        TestCase(MvNormal([1.0, 0.9]), [-0.1, -0.7]),
        TestCase(MvNormal([0.0], 0.9), [0.1]),
        TestCase(MvNormal([0.0, 0.1], 0.9), [0.1, -0.05]),
        TestCase(MvNormal(Diagonal([0.1])), [0.1]),
        TestCase(MvNormal(Diagonal([0.1, 0.2])), [0.1, 0.15]),
        TestCase(MvNormal([0.1, -0.3], Diagonal(Fill(0.9, 2))), [0.1, -0.1]),
        TestCase(MvNormal([0.1, -0.1], 0.4I), [-0.1, 0.15]),
        TestCase(MvNormal([0.2, 0.3], Hermitian(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        TestCase(MvNormal([0.2, 0.3], Symmetric(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        TestCase(MvNormal([0.2, 0.3], Diagonal([0.5, 0.4])), [-0.1, 0.05]),
        # TODO Broken tests, see https://github.com/EnzymeAD/Enzyme.jl/issues/1991
        TestCase(MvNormal([-0.15], _pdmat([1.1]')), [-0.05]; broken=Forward),
        TestCase(
            MvNormal([0.2, -0.15], _pdmat([1.0 0.9; 0.7 1.1])), [0.05, -0.05];
            broken=Forward
        ),
        TestCase(MvNormal([0.2, -0.3], [0.5, 0.6]), [0.4, -0.3]),
        TestCase(MvNormalCanon([0.1, -0.1], _pdmat([0.5 0.4; 0.45 1.0])), [0.2, -0.25]),
        # TODO Broken tests, see https://github.com/EnzymeAD/Enzyme.jl/issues/1991
        TestCase(
            MvLogNormal(MvNormal([0.2, -0.1], _pdmat([1.0 0.9; 0.7 1.1]))), [0.5, 0.1];
            broken=Forward
        ),
        TestCase(product_distribution([Normal()]), [0.3]),
        TestCase(
            product_distribution([Normal(), Uniform()]), [-0.4, 0.3];
            runtime_activity=Both),

        #
        # Matrix-variate
        #

        TestCase(
            MatrixNormal(
                randn(rng, 2, 3), _pdmat(randn(rng, 2, 2)), _pdmat(randn(rng, 3, 3))
            ),
            randn(rng, 2, 3),
        ),
        # TODO Broken tests, see https://github.com/EnzymeAD/Enzyme.jl/issues/1821
        TestCase(
            Wishart(5, _pdmat(randn(rng, 3, 3))),
            Symmetric(collect(_pdmat(randn(rng, 3, 3))));
            broken=Forward
        ),
        TestCase(
            InverseWishart(5, _pdmat(randn(rng, 3, 3))),
            Symmetric(collect(_pdmat(randn(rng, 3, 3))));
            broken=Forward
        ),
        # TODO Broken tests, see https://github.com/EnzymeAD/Enzyme.jl/issues/1820
        TestCase(
            MatrixTDist(
                3.1,
                randn(rng, 2, 3),
                _pdmat(randn(rng, 2, 2)),
                _pdmat(randn(rng, 3, 3)),
            ),
            randn(rng, 2, 3);
            broken=Both
        ),
        TestCase(MatrixBeta(5, 6.0, 7.0), rand(rng, MatrixBeta(5, 6.0, 6.0)); broken=Both),
        TestCase(
            MatrixFDist(6.0, 7.0, _pdmat(randn(rng, 5, 5))),
            rand(rng, MatrixFDist(6.0, 7.0, _pdmat(randn(rng, 5, 5))));
            broken=Both
        ),
        TestCase(LKJ(5, 1.1), rand(rng, LKJ(5, 1.1)); broken=Both),

        #
        # Miscellaneous others
        #

        TestCase(
            (a, b, x) -> logpdf(InverseGamma(a, b), x), (1.5, 1.4, 0.4);
            name="InverseGamma", splat=true
        ),
        TestCase(
            (m, s, x) -> logpdf(NormalCanon(m, s), x), (0.1, 1.0, -0.5);
            name="NormalCanon", splat=true
        ),
        TestCase(x -> logpdf(Categorical(x, 1 - x), 1), 0.3; name="Categorical"),
        # TODO Broken test, see https://github.com/EnzymeAD/Enzyme.jl/issues/1995
        TestCase(
            (m, S, x) -> logpdf(MvLogitNormal(m, S), vcat(x, 1 - sum(x))),
            ([0.4, 0.6], _pdmat([0.9 0.4; 0.5 1.1]), [0.27, 0.24]);
            name="MvLogitNormal", runtime_activity=Forward, broken=Forward, splat=true,
        ),
        # TODO Broken test, see https://github.com/EnzymeAD/Enzyme.jl/issues/1996
        TestCase(
            (a, b, α, β, x) -> logpdf(truncated(Beta(α, β), a, b), x),
            (0.1, 0.9, 1.1, 1.3, 0.4);
            name="truncated Beta", splat=true, broken=Reverse
        ),
        # TODO Broken test, see https://github.com/EnzymeAD/Enzyme.jl/issues/1998
        TestCase(
            (a, b, x) -> logpdf(truncated(Normal(), a, b), x),
            (-0.3, 0.3, 0.1);
            name="allocs Normal", splat=true, broken=Forward
        ),
        TestCase(
            (a, b, α, β, x) -> logpdf(truncated(Uniform(α, β), a, b), x),
            (0.1, 0.9, -0.1, 1.1, 0.4);
            name="allocs Uniform", splat=true
        ),
        TestCase(
            (a, x) -> logpdf(Dirichlet(a), [x, 1 - x]), ([1.5, 1.1], 0.6);
            name="Dirichlet", splat=true, runtime_activity=Forward
        ),
        # TODO Broken test, see https://github.com/EnzymeAD/Enzyme.jl/issues/1997
        TestCase(
            x -> logpdf(reshape(product_distribution([Normal(), Uniform()]), 1, 2), x),
            [2.1 0.7];
            name="reshape", broken=Forward
        ),
        # TODO Broken test, see https://github.com/EnzymeAD/Enzyme.jl/issues/1820
        TestCase(
            x -> logpdf(vec(LKJ(2, 1.1)), x), [1.0, 0.489, 0.489, 1.0];
            name="vec", broken=Both
        ),
        # TODO Broken test, see https://github.com/EnzymeAD/Enzyme.jl/issues/1994
        TestCase(
            function (t)
                X, v = t
                # LKJCholesky distributes over the Cholesky factorisation of correlation
                # matrices, so the argument to `logpdf` must be such a matrix.
                S = X'X
                Λ = Diagonal(map(inv ∘ sqrt, diag(S)))
                C = cholesky(Symmetric(Λ * S * Λ))
                return logpdf(LKJCholesky(2, v), C)
            end,
            (randn(rng, 2, 2), 1.1);
            name="LKJCholesky", broken=Forward
        ),
    ]

    @testset "$(case.name)" for case in test_cases
        test_grad(case)
    end
end

end
