using SpecialFunctions

include("../common.jl")

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
    test_scalar((y) -> SpecialFunctions.besselj(2, y), x)

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
