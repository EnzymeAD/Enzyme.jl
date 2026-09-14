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

        # No derivative defined in Enzyme for libc atm
        # test_scalar(SpecialFunctions.gamma, x)
        # test_scalar(SpecialFunctions.digamma, x)
        # test_scalar(SpecialFunctions.trigamma, x)
    end
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
    # - Near 1: 0.9, 1.1 to test branch boundaries and continuity across a≈1, b≈1.
    # - Moderate: 2.5, 5.0 where multiple algorithm choices engage based on x and bx.
    # - Large (≥15, ≥40) to drive large-parameter regimes: 16.0, 45.0.
    # - Very large (≫100): 100.5, 150.0 to ensure symmetric vs asymmetric asymptotics are exercised when λ
    #   is small/large, and continued fractions are robust for large shapes.
    ab = (0.4, 0.6, 0.9, 1.1, 2.5, 5.0, 16.0, 45.0, 100.5, 150.0)

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

# x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/3580
@testset "incomplete gamma: shape and rate partials" begin
    gamma_inc_ext = Base.get_extension(Enzyme, :EnzymeSpecialFunctionsExt)

    # `_gamma_inc` branches to `gamma_inc_fsum` when 2a is an integer and a <= x, and
    # that routine uses a only as a loop count, so the shape partial differentiated to
    # exactly zero at these points before the analytic rule.
    for a in (0.5, 1.0, 1.5, 2.0)
        test_scalar(a -> first(SpecialFunctions.gamma_inc(a, 2.0)), a)
        test_scalar(a -> last(SpecialFunctions.gamma_inc(a, 2.0)), a)
    end

    # Neighbouring non-branch shapes, correct before the rule, as a regression guard.
    for a in (0.25, 1.25)
        test_scalar(a -> first(SpecialFunctions.gamma_inc(a, 2.0)), a)
    end

    # The x partial was correct throughout; check it still is on both outputs.
    for x in (0.5, 2.0, 5.0)
        test_scalar(x -> first(SpecialFunctions.gamma_inc(1.0, x)), x)
        test_scalar(x -> last(SpecialFunctions.gamma_inc(2.5, x)), x)
    end

    # Float32 tolerances follow the `logabsgamma` case above. They are set by the
    # Float32 finite difference `test_scalar` compares against, not by the partial:
    # Float16 and Float32 widen to Float64 and narrow back, so the partial itself
    # lands within an ulp rather than carrying the rounding of O(√x) accumulated
    # terms. The reference assertion further down pins that where an FD cannot.
    test_scalar(a -> first(SpecialFunctions.gamma_inc(a, 2.0f0)), 1.0f0; rtol = 1.0e-5, atol = 1.0e-5)
    test_scalar(x -> first(SpecialFunctions.gamma_inc(1.0f0, x)), 2.0f0; rtol = 1.0e-5, atol = 1.0e-5)

    # The 3-argument form carries the same partials; `ind` only sets the primal's
    # accuracy target, so a finite difference of it is too noisy for `test_scalar`.
    da2 = Enzyme.autodiff(Reverse, a -> first(SpecialFunctions.gamma_inc(a, 2.0)), Active, Active(1.0))[1][1]
    da3 = Enzyme.autodiff(Reverse, a -> first(SpecialFunctions.gamma_inc(a, 2.0, 1)), Active, Active(1.0))[1][1]
    @test da3 == da2

    # Shape derivative of log(Q) in the right tail, the quantity the regime split
    # exists for. Differentiating P everywhere and dividing by Q loses a digit per
    # decade Q falls and has the wrong sign by Q ≈ 1e-16; the continued fraction
    # leaves Q as a factor of ∂Q/∂a, so the quotient stays conditioned. References
    # are ∂log Q/∂a at 1024-bit precision. `test_scalar` cannot express these because
    # a central difference of log(Q) cannot resolve a derivative whose own function
    # underflows a short step away.
    dlogQ_da(k, x) = autodiff(
        Reverse, t -> log(last(SpecialFunctions.gamma_inc(t, x))), Active, Active(k)
    )[1][1]

    # Down the tail from Q ≈ 1e-6 to Q ≈ 1e-23, across shapes.
    @test isapprox(dlogQ_da(10.0, 30.0), 1.1935552300070054; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(10.0, 33.0), 1.283815721530926; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(10.0, 40.0), 1.4679114736958756; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(10.0, 60.0), 1.8617089624951197; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(10.0, 80.0), 2.1441192460521554; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(1.0, 40.0), 4.290499234095098; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(2.0, 40.0), 3.2910805852369234; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(5.0, 50.0), 2.42711839097637; rtol = 1.0e-14)
    @test isapprox(dlogQ_da(100.0, 170.0), 0.5490554382301781; rtol = 1.0e-14)

    # ∂P/∂a is -∂Q/∂a on the upper branch, so it inherits the same relative accuracy
    # where it used to come back with the wrong sign entirely (2.66e-15 at a=10, x=60).
    dP_da(k, x) = autodiff(
        Reverse, t -> first(SpecialFunctions.gamma_inc(t, x)), Active, Active(k)
    )[1][1]
    @test isapprox(dP_da(10.0, 60.0), -5.308677545135539e-16; rtol = 1.0e-14)
    @test isapprox(dP_da(5.0, 50.0), -1.3227071908086808e-16; rtol = 1.0e-14)

    # Widen/narrow at reduced precision: within an ulp of the Float64 reference.
    @test isapprox(dlogQ_da(10.0f0, 40.0f0), 1.4679115f0; rtol = 2 * eps(Float32))

    # Once the primal Q has underflowed to zero, ∂Q/∂a is rebuilt in log space rather
    # than returned as the product 0 * g. Only a couple of bits of a subnormal survive
    # at this depth, so the assertion is on magnitude and sign, not on value.
    @test last(SpecialFunctions.gamma_inc(1.0, 745.5)) == 0.0
    dQa_sub = gamma_inc_ext._gamma_inc_grad(1.0, 745.5, SpecialFunctions.gamma_inc(1.0, 745.5)...)[2]
    @test 0.0 < dQa_sub < 1.0e-322

    # Either side of Gautschi's regime boundary at x = a + 1: the lower series owns
    # the first of each pair and the continued fraction the second. They must agree.
    for a in (0.5, 2.0, 10.0, 100.0)
        lo, hi = dP_da(a, prevfloat(a + 1.0)), dP_da(a, a + 1.0)
        @test isapprox(lo, hi; rtol = 1.0e-12)
    end

    # P(a, 0) = 0 and Q(a, 0) = 1 for every a, so both shape partials are zero there;
    # `test_scalar` cannot express it because a central difference straddles the
    # x < 0 domain error.
    @test gamma_inc_ext._gamma_inc_grad(2.0, 0.0, 0.0, 1.0) === (0.0, 0.0, 0.0)
    @test gamma_inc_ext._gamma_inc_grad(1.0, 0.0, 0.0, 1.0) === (0.0, 0.0, 1.0)
    @test gamma_inc_ext._gamma_inc_grad(0.5, 0.0, 0.0, 1.0) === (0.0, 0.0, Inf)
end
