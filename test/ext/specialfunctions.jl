using SpecialFunctions

include("../common.jl")

Enzyme.Compiler.VERBOSE_ERRORS[] = true

@testset "SpecialFunctions ext" begin
    lgabsg(x) = SpecialFunctions.logabsgamma(x)[1]
    test_scalar(lgabsg, 1.0; rtol = 1.0e-5, atol = 1.0e-5)
    test_scalar(lgabsg, 1.0f0; rtol = 1.0e-5, atol = 1.0e-5)
end

# From https://github.com/JuliaDiff/ChainRules.jl/blob/02e7857e34b5c01067a288262f69cfcb9fce069b/test/rulesets/packages/SpecialFunctions.jl#L1

@testset "SpecialFunctions" for x in (1, -1, 0, 0.5, 10, -17.1, 1.5 + 0.7im)
    # 32-bit erf currently broken
    if sizeof(Int) != sizeof(Int32)
        test_scalar(SpecialFunctions.erf, x)
        test_scalar(SpecialFunctions.erfc, x)
    end

    # Handled by openspec non defaultly done
    # test_scalar(SpecialFunctions.erfi, x)
    # test_scalar(SpecialFunctions.erfcx, x)
    # test_scalar(SpecialFunctions.airyai, x)
    # test_scalar(SpecialFunctions.airyaiprime, x)
    # test_scalar(SpecialFunctions.airybi, x)
    # test_scalar(SpecialFunctions.airybiprime, x)
    test_scalar(SpecialFunctions.besselj0, x)
    test_scalar(SpecialFunctions.besselj1, x)
    test_scalar((y) -> SpecialFunctions.besseli(2, y), x)
    test_scalar((y) -> SpecialFunctions.besselj(2, y), x)

    # Exponentially scaled Bessel functions (rules defined for real arguments)
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2880
    if x isa Real
        test_scalar((y) -> SpecialFunctions.besselix(2, y), x)
        test_scalar((y) -> SpecialFunctions.besseljx(2, y), x)
    end

    # test_scalar((y) -> SpecialFunctions.sphericalbessely(y, 0.5), 0.3)
    # test_scalar(SpecialFunctions.dawson, x)

    # Requires derivative of digamma/trigamma
    # if x isa Real
    #     test_scalar(SpecialFunctions.invdigamma, x)
    # end

    if x isa Real && 0 < x < 1
        # Requires GC -- avx functions appear
        # test_scalar(SpecialFunctions.erfinv, x)
        # test_scalar(SpecialFunctions.erfcinv, x)
    end

    if !(x isa Real) || x > 0
        test_scalar(SpecialFunctions.bessely0, x)
        test_scalar(SpecialFunctions.bessely1, x)
        test_scalar((y) -> SpecialFunctions.bessely(2, y), x)

        if x isa Real
            test_scalar((y) -> SpecialFunctions.besselyx(2, y), x)
            test_scalar((y) -> SpecialFunctions.besselkx(2, y), x)
        end

        # Complex gamma-family functions and trigamma are not tested here.
        # test_scalar(SpecialFunctions.trigamma, x)
    end
end

@testset "gamma and digamma derivatives" begin
    test_scalar(gamma, 2.5)
    test_scalar(gamma, -2.5)
    test_scalar(digamma, 2.5)
    test_scalar(digamma, -2.5)
    for T in (Float16, Float32, Float64), value in (-2.5, -0.5, 0.5, 1.0, 2.5, 10.0)
        x = T(value)
        expected = T(gamma(Float64(x)) * digamma(Float64(x)))
        rtol = 5 * eps(T)
        @test isapprox(Enzyme.gradient(Reverse, Const(gamma), x)[1], expected; rtol)
        @test isapprox(Enzyme.gradient(Forward, Const(gamma), x)[1], expected; rtol)
        expected_digamma = T(trigamma(Float64(x)))
        @test isapprox(Enzyme.gradient(Reverse, Const(digamma), x)[1], expected_digamma; rtol)
        @test isapprox(Enzyme.gradient(Forward, Const(digamma), x)[1], expected_digamma; rtol)

        forward = autodiff(ForwardWithPrimal, Const(gamma), Duplicated(x, T(2)))
        @test forward[2] == gamma(x)
        @test isapprox(forward[1], 2 * expected; rtol)
        reverse = autodiff(ReverseWithPrimal, Const(gamma), Active, Active(x))
        @test reverse[2] == gamma(x)
        @test isapprox(reverse[1][1], expected; rtol)

        batch = autodiff(Forward, Const(gamma), BatchDuplicated(x, (T(2), T(-3))))[1]
        @test isapprox(batch[1], 2 * expected; rtol)
        @test isapprox(batch[2], -3 * expected; rtol)
    end

    # loggamma makes libEnzyme's external digamma available, which used to hide
    # the missing-symbol error and expose the tgamma rule's missing gamma factor.
    gamma_and_loggamma(x) = gamma(x) + loggamma(x)
    for mode in (Forward, Reverse)
        @test Enzyme.gradient(mode, Const(gamma_and_loggamma), 2.5)[1] ≈ (gamma(2.5) + 1) * digamma(2.5)
    end

    dgamma(x) = autodiff(Reverse, Const(gamma), Active, Active(x))[1][1]
    dgamma_forward(x) = autodiff(Forward, Const(gamma), Duplicated(x, one(x)))[1]
    for x in (-2.5, -0.5, 0.5, 2.5, 10.0, 2.5f0)
        second = gamma(x) * (digamma(x)^2 + trigamma(x))
        @test autodiff(Forward, Const(dgamma), Duplicated(x, one(x)))[1] ≈ second
        @test autodiff(Forward, Const(dgamma_forward), Duplicated(x, one(x)))[1] ≈ second
    end

    # Higher gamma derivatives must also follow the analytic polygamma chain.
    ddgamma_forward(x) = autodiff(Forward, Const(dgamma_forward), Duplicated(x, one(x)))[1]
    dddgamma_forward(x) = autodiff(Forward, Const(ddgamma_forward), Duplicated(x, one(x)))[1]
    for x in (-2.5, -0.5, 2.5)
        ψ, ψ1, ψ2, ψ3 = digamma(x), trigamma(x), polygamma(2, x), polygamma(3, x)
        third = gamma(x) * (ψ^3 + 3 * ψ * ψ1 + ψ2)
        fourth = gamma(x) * (ψ^4 + 6 * ψ^2 * ψ1 + 3 * ψ1^2 + 4 * ψ * ψ2 + ψ3)
        @test dddgamma_forward(x) ≈ third
        @test autodiff(Forward, Const(dddgamma_forward), Duplicated(x, one(x)))[1] ≈ fourth
    end

    # Split reverse must retain the argument's contribution until the pullback,
    # and add it to an existing shadow with the caller's output seed.
    gamma_vector(x) = [gamma(x[1])]
    fwd, rev = autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(gamma_vector)}, Duplicated, Duplicated{Vector{Float64}})
    x, dx = [2.5], [7.0]
    tape, _, shadow = fwd(Const(gamma_vector), Duplicated(x, dx))
    shadow[1] = 3.0
    rev(Const(gamma_vector), Duplicated(x, dx), tape)
    @test dx[1] ≈ 7 + 3 * gamma(2.5) * digamma(2.5)
end

# SpecialFunctions 0.7->0.8 changes:
@testset "log gamma and co" begin
    #It is important that we have negative numbers with both odd and even integer parts
    for x in (1.5, 2.5, 10.5, -0.6, -2.6, -3.3, 1.6+1.6im, 1.6-1.6im, -4.6+1.6im)
        if isdefined(SpecialFunctions, :lgamma)
            # test_scalar(SpecialFunctions.lgamma, x)
        end
        if isdefined(SpecialFunctions, :loggamma)
            isreal(x) && x < 0 && continue
            # test_scalar(SpecialFunctions.loggamma, x)
        end
    end
end

# x/ref: https://github.com/JuliaMath/SpecialFunctions.jl/pull/506
@testset "incomplete beta: basic test_frule/test_rrule" begin
    # Use an expanded set of interior points (avoid endpoints for FD) to exercise many branches:
    # Rationale for x values:
    # - Include values around 0.1, 0.3, 0.5, 0.7, 0.9 to trigger different code paths.
    # - Include 0.14 and 0.28 to straddle the bx ≤ 0.7 power-series threshold for b ≈ 5 and 2.5.
    # - Include values near 0.5 (0.49, 0.51) to probe near-symmetry and tail swaps.
    # - Include additional midpoints to increase chance that x ≈ a/(a+b) for some (a,b), which makes λ ≈ 0
    #   in the large-parameter regime (key for choosing symmetric asymptotics when min(a,b) > 100).
    # - Add a few more around 0.6–0.8 to exercise continued fraction vs. asymptotics for large (a,b).
    test_points = (
        0.05, 0.08, 0.10, 0.12, 0.14, 0.18, 0.20, 0.22, 0.26,
        0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45,
        0.49, 0.50, 0.51, 0.55, 0.58, 0.60, 0.62, 0.65,
        0.68, 0.70, 0.72, 0.76, 0.80, 0.85, 0.90
    )
    # Rationale for a,b values:
    # - <1: 0.4, 0.6 to stress small-parameter power series branches.
    # - Near 1: 0.9, 1.0, 1.1 to test branch boundaries and continuity across a≈1, b≈1.
    #   Exactly 1 is a removable singularity in the CF coefficients: #3581.
    # - Moderate: 2.5, 5.0 where multiple algorithm choices engage based on x and bx.
    # - Large (≥15, ≥40) to drive large-parameter regimes: 16.0, 45.0.
    # - Very large (≫100): 100.5, 150.0 to ensure symmetric vs asymmetric asymptotics are exercised when λ
    #   is small/large, and continued fractions are robust for large shapes.
    ab = (0.4, 0.6, 0.9, 1.0, 1.1, 2.5, 5.0, 16.0, 45.0, 100.5, 150.0)

    # 3-argument beta_inc(a,b,x)
    for a in ab, b in ab, x in test_points
        0.0 < x < 1.0 || continue

        test_scalar(a -> first(SpecialFunctions.beta_inc(a, b, x)), a)
        test_scalar(b -> first(SpecialFunctions.beta_inc(a, b, x)), b)
        test_scalar(x -> first(SpecialFunctions.beta_inc(a, b, x)), x)
    end

    # Inverse beta: beta_inc_inv(a,b,p)
    for a in ab, b in ab, p in test_points
        0.0 < p < 1.0 || continue
        test_scalar(a -> first(SpecialFunctions.beta_inc_inv(a, b, p)), a)
        test_scalar(b -> first(SpecialFunctions.beta_inc_inv(a, b, p)), b)
        test_scalar(p -> first(SpecialFunctions.beta_inc_inv(a, b, p)), p)
    end
end
