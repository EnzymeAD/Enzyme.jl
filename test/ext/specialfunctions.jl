using SpecialFunctions

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
