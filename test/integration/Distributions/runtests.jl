module DistributionsIntegrationTests

using Distributions
using Enzyme: Enzyme
using FillArrays: Fill
using FiniteDifferences: FiniteDifferences
using LinearAlgebra: Diagonal, Hermitian, I, Symmetric, diag, cholesky
using PDMats: PDMat
using Random: randn
using Test: @test, @testset

# TODO(mhauru) Could we at some point make do without this?
Enzyme.API.runtimeActivity!(true)

"""
Test Enzyme.gradient, both Forward and Reverse mode, against FiniteDifferences.grad, for a
given function f and argument x.
"""
function test_grad(f, x; rtol=1e-6, atol=1e-6)
    @nospecialize
    finitediff = FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), f, x)[1]
    # TODO(mhauru) The Val(1) works around https://github.com/EnzymeAD/Enzyme.jl/issues/1807
    @test(
        collect(Enzyme.gradient(Enzyme.Forward, Enzyme.Const(f), x, Val(1))) ≈ collect(finitediff),
        rtol = rtol,
        atol = atol
    )
    @test(
        Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), x) ≈ finitediff,
        rtol = rtol, atol = atol
    )
    return nothing
end

_sym(A) = A'A
_pdmat(A) = PDMat(_sym(A) + 5I)

@testset "Distributions integration tests" begin
    # Distributions for which to test differentiating `logpdf(d, x)`. The first value of the
    # tuple is the `d`, the second is `x`.
    logpdf_test_cases = Any[

        #
        # Univariate
        #

        (Arcsine(), 0.5),
        (Arcsine(-0.3, 0.9), 0.5),
        (Arcsine(0.5, 1.1), 1.0),
        (Beta(1.1, 1.1), 0.5),
        (Beta(1.1, 1.5), 0.9),
        (Beta(1.6, 1.5), 0.5),
        (BetaPrime(1.1, 1.1), 0.5),
        (BetaPrime(1.1, 1.6), 0.5),
        (BetaPrime(1.6, 1.3), 0.9),
        (Biweight(1.0, 2.0), 0.5),
        (Biweight(-0.5, 2.5), -0.45),
        (Biweight(0.0, 1.0), 0.3),
        (Cauchy(), -0.5),
        (Cauchy(1.0), 0.99),
        (Cauchy(1.0, 0.1), 1.01),
        (Chi(2.5), 0.5),
        (Chi(5.5), 1.1),
        (Chi(0.1), 0.7),
        (Chisq(2.5), 0.5),
        (Chisq(5.5), 1.1),
        (Chisq(0.1), 0.7),
        (Cosine(0.0, 1.0), 0.5),
        (Cosine(-0.5, 2.0), -0.1),
        (Cosine(0.4, 0.5), 0.0),
        (Epanechnikov(0.0, 1.0), 0.5),
        (Epanechnikov(-0.5, 1.2), -0.9),
        (Epanechnikov(-0.4, 1.6), 0.1),
        (Erlang(), 0.5),
        (Erlang(), 0.1),
        (Erlang(), 0.9),
        (Exponential(), 0.1),
        (Exponential(0.5), 0.9),
        (Exponential(1.4), 0.05),
        (FDist(2.1, 3.5), 0.7),
        (FDist(1.4, 5.4), 3.5),
        (FDist(5.5, 3.3), 7.2),
        (Frechet(), 0.1),
        (Frechet(), 1.1),
        (Frechet(1.5, 2.4), 0.1),
        (Gamma(0.9, 1.2), 4.5),
        (Gamma(0.5, 1.9), 1.5),
        (Gamma(1.8, 3.2), 1.0),
        (GeneralizedExtremeValue(0.3, 1.3, 0.1), 2.4),
        (GeneralizedExtremeValue(-0.7, 2.2, 0.4), 1.1),
        (GeneralizedExtremeValue(0.5, 0.9, -0.5), -7.0),
        (GeneralizedPareto(0.3, 1.1, 1.1), 5.0),
        (GeneralizedPareto(-0.25, 0.9, 0.1), 0.8),
        (GeneralizedPareto(0.3, 1.1, -5.1), 0.31),
        (Gumbel(0.1, 0.5), 0.1),
        (Gumbel(-0.5, 1.1), -0.1),
        (Gumbel(0.3, 0.1), 0.3),
        (InverseGaussian(0.1, 0.5), 1.1),
        (InverseGaussian(0.2, 1.1), 3.2),
        (InverseGaussian(0.1, 1.2), 0.5),
        (JohnsonSU(0.1, 0.95, 0.1, 1.1), 0.1),
        (JohnsonSU(0.15, 0.9, 0.12, 0.94), 0.5),
        (JohnsonSU(0.1, 0.95, 0.1, 1.1), -0.3),
        (Kolmogorov(), 1.1),
        (Kolmogorov(), 0.9),
        (Kolmogorov(), 1.5),
        (Kumaraswamy(2.0, 5.0), 0.71),
        (Kumaraswamy(0.1, 5.0), 0.2),
        (Kumaraswamy(0.5, 4.5), 0.1),
        (Laplace(0.1, 1.0), 0.2),
        (Laplace(-0.5, 2.1), 0.5),
        (Laplace(-0.35, 0.4), -0.3),
        (Levy(0.1, 0.9), 4.1),
        (Levy(0.5, 0.9), 0.6),
        (Levy(1.1, 0.5), 2.2),
        (Lindley(0.5), 2.1),
        (Lindley(1.1), 3.1),
        (Lindley(1.9), 3.5),
        (Logistic(0.1, 1.2), 1.1),
        (Logistic(0.5, 0.7), 0.6),
        (Logistic(-0.5, 0.1), -0.4),
        (LogitNormal(0.1, 1.1), 0.5),
        (LogitNormal(0.5, 0.7), 0.6),
        (LogitNormal(-0.12, 1.1), 0.1),
        (LogNormal(0.0, 1.0), 0.5),
        (LogNormal(0.5, 1.0), 0.5),
        (LogNormal(-0.1, 1.3), 0.75),
        (LogUniform(0.1, 0.9), 0.75),
        (LogUniform(0.15, 7.8), 7.1),
        (LogUniform(2.0, 3.0), 2.1),
        # (NoncentralBeta(1.1, 1.1, 1.2), 0.8), # foreigncall (Rmath.dnbeta).
        # (NoncentralChisq(2, 3.0), 10.0), # foreigncall (Rmath.dnchisq).
        # (NoncentralF(2, 3, 1.1), 4.1), # foreigncall (Rmath.dnf).
        # (NoncentralT(1.3, 1.1), 0.1), # foreigncall (Rmath.dnt).
        (Normal(), 0.1),
        (Normal(0.0, 1.0), 1.0),
        (Normal(0.5, 1.0), 0.05),
        (Normal(0.0, 1.5), -0.1),
        (Normal(-0.1, 0.9), -0.3),
        # (NormalInverseGaussian(0.0, 1.0, 0.2, 0.1), 0.1), # foreigncall -- https://github.com/JuliaMath/SpecialFunctions.jl/blob/be1fa06fee58ec019a28fb0cd2b847ca83a5af9a/src/bessel.jl#L265
        (Pareto(1.0, 1.0), 3.5),
        (Pareto(1.1, 0.9), 3.1),
        (Pareto(1.0, 1.0), 1.4),
        (PGeneralizedGaussian(0.2), 5.0),
        (PGeneralizedGaussian(0.5, 1.0, 0.3), 5.0),
        (PGeneralizedGaussian(-0.1, 11.1, 6.5), -0.3),
        (Rayleigh(0.5), 0.6),
        (Rayleigh(0.9), 1.1),
        (Rayleigh(0.55), 0.63),
        # (Rician(0.5, 1.0), 2.1), # foreigncall (Rmath.dnchisq). Not implemented anywhere.
        (Semicircle(1.0), 0.9),
        (Semicircle(5.1), 5.05),
        (Semicircle(0.5), -0.1),
        (SkewedExponentialPower(0.1, 1.0, 0.97, 0.7), -2.0),
        (SkewedExponentialPower(0.15, 1.0, 0.97, 0.7), -2.0),
        (SkewedExponentialPower(0.1, 1.1, 0.99, 0.7), 0.5),
        (SkewNormal(0.0, 1.0, -1.0), 0.1),
        (SkewNormal(0.5, 2.0, 1.1), 0.1),
        (SkewNormal(-0.5, 1.0, 0.0), 0.1),
        (SymTriangularDist(0.0, 1.0), 0.5),
        (SymTriangularDist(-0.5, 2.1), -2.0),
        (SymTriangularDist(1.7, 0.3), 1.75),
        (TDist(1.1), 99.1),
        (TDist(10.1), 25.0),
        (TDist(2.1), -89.5),
        (TriangularDist(0.0, 1.5, 0.5), 0.45),
        (TriangularDist(0.1, 1.4, 0.45), 0.12),
        (TriangularDist(0.0, 1.5, 0.5), 0.2),
        (Triweight(1.0, 1.0), 1.0),
        (Triweight(1.1, 2.1), 1.0),
        (Triweight(1.9, 10.0), -0.1),
        (Uniform(0.0, 1.0), 0.2),
        (Uniform(-0.1, 1.1), 1.0),
        (Uniform(99.5, 100.5), 100.0),
        (VonMises(0.5), 0.1),
        (VonMises(0.3), -0.1),
        (VonMises(0.2), -0.5),
        (Weibull(0.5, 1.0), 0.45),
        (Weibull(0.3, 1.1), 0.66),
        (Weibull(0.75, 1.3), 0.99),

        #
        # Multivariate
        #

        (MvNormal(1, 1.5), [-0.3]),
        (MvNormal(2, 0.5), [0.2, -0.3]),
        (MvNormal([1.0]), [-0.1]),
        (MvNormal([1.0, 0.9]), [-0.1, -0.7]),
        (MvNormal([0.0], 0.9), [0.1]),
        (MvNormal([0.0, 0.1], 0.9), [0.1, -0.05]),
        (MvNormal(Diagonal([0.1])), [0.1]),
        (MvNormal(Diagonal([0.1, 0.2])), [0.1, 0.15]),
        (MvNormal([0.1, -0.3], Diagonal(Fill(0.9, 2))), [0.1, -0.1]),
        (MvNormal([0.1, -0.1], 0.4I), [-0.1, 0.15]),
        (MvNormal([0.2, 0.3], Hermitian(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (MvNormal([0.2, 0.3], Symmetric(Diagonal([0.5, 0.4]))), [-0.1, 0.05]),
        (MvNormal([0.2, 0.3], Diagonal([0.5, 0.4])), [-0.1, 0.05]),
        (MvNormal([-0.15], _pdmat([1.1]')), [-0.05]),
        (MvNormal([0.2, -0.15], _pdmat([1.0 0.9; 0.7 1.1])), [0.05, -0.05]),
        (MvNormal([0.2, -0.3], [0.5, 0.6]), [0.4, -0.3]),
        (MvNormalCanon([0.1, -0.1], _pdmat([0.5 0.4; 0.45 1.0])), [0.2, -0.25]),
        (MvLogNormal(MvNormal([0.2, -0.1], _pdmat([1.0 0.9; 0.7 1.1]))), [0.5, 0.1]),
        (product_distribution([Normal()]), [0.3]),
        (product_distribution([Normal(), Uniform()]), [-0.4, 0.3]),

        #
        # Matrix-variate
        #

        (
            MatrixNormal(
                randn(2, 3), _pdmat(randn(2, 2)), _pdmat(randn(3, 3))
            ),
            randn(2, 3),
        ),
        (
            Wishart(5, _pdmat(randn(3, 3))),
            Symmetric(collect(_pdmat(randn(3, 3)))),
        ),
        (
            InverseWishart(5, _pdmat(randn(3, 3))),
            Symmetric(collect(_pdmat(randn(3, 3)))),
        ),
        (
            MatrixTDist(
                3.1,
                randn(2, 3),
                _pdmat(randn(2, 2)),
                _pdmat(randn(3, 3)),
            ),
            randn(2, 3),
        ),
        (MatrixBeta(5, 6.0, 7.0), rand(MatrixBeta(5, 6.0, 6.0))),
        (
            MatrixFDist(6.0, 7.0, _pdmat(randn(5, 5))),
            rand(MatrixFDist(6.0, 7.0, _pdmat(randn(5, 5)))),
        ),
        (LKJ(5, 1.1), rand(LKJ(5, 1.1))),
    ]

    # Cases where the function to differentiate isn't just straight up `logpdf(d, x)`.
    # Values in the tuple are (name, function to differentiate, value to differentiate at).
    work_around_test_cases = Any[
        ("InverseGamma", (a, b, x) -> logpdf(InverseGamma(a, b), x), (1.5, 1.4, 0.4)),
        ("NormalCanon", (m, s, x) -> logpdf(NormalCanon(m, s), x), (0.1, 1.0, -0.5)),
        ("Categorical", x -> logpdf(Categorical(x, 1 - x), 1), 0.3),
        (
            "MvLogitNormal",
            (m, S, x) -> logpdf(MvLogitNormal(m, S), vcat(x, 1 - sum(x))),
            ([0.4, 0.6], Symmetric(_pdmat([0.9 0.4; 0.5 1.1])), [0.27, 0.24]),
        ),
        (
            "truncated Beta",
            (a, b, α, β, x) -> logpdf(truncated(Beta(α, β), a, b), x),
            (0.1, 0.9, 1.1, 1.3, 0.4),
        ),
        (
            "allocs Normal",
            (a, b, x) -> logpdf(truncated(Normal(), a, b), x),
            (-0.3, 0.3, 0.1),
        ),
        (
            "allocs Uniform",
            (a, b, α, β, x) -> logpdf(truncated(Uniform(α, β), a, b), x),
            (0.1, 0.9, -0.1, 1.1, 0.4),
        ),
        ("Dirichlet", (a, x) -> logpdf(Dirichlet(a), [x, 1-x]), ([1.5, 1.1], 0.6)),
        (
            "reshape",
            x -> logpdf(reshape(product_distribution([Normal(), Uniform()]), 1, 2), x),
            ([2.1 0.7],),
        ),
        ("vec", x -> logpdf(vec(LKJ(2, 1.1)), x), ([1.0, 0.489, 0.489, 1.0],)),
        (
            "LKJCholesky",
            function(X, v)
                # LKJCholesky distributes over the Cholesky factorisation of correlation
                # matrices, so the argument to `logpdf` must be such a matrix.
                S = X'X
                Λ = Diagonal(map(inv ∘ sqrt, diag(S)))
                C = cholesky(Symmetric(Λ * S * Λ))
                return logpdf(LKJCholesky(2, v), C)
            end,
            (randn(2, 2), 1.1),
        ),
    ]

    @testset "$(nameof(typeof(d)))" for (d, x) in logpdf_test_cases
        test_grad(x -> logpdf(d, x), x)
    end

    @testset "$name" for (name, f, x) in work_around_test_cases
        test_grad(y -> f(y...), x)
    end
end

end
