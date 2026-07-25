using Gamma
using FiniteDifferences

include("../common.jl")

Enzyme.Compiler.VERBOSE_ERRORS[] = true

@testset "Gamma.gamma scalar derivative" begin
    # Points chosen to exercise every branch of the Cephes implementation:
    # the x<2 upshift loop, the plain 2≤x<3 region, the x≥3 downshift loop,
    # the x>11.5 asymptotic branch, and the x<0 reflection formula.
    for x in (0.5, 1.5, 2.5, 4.5, 8.5, 12.5, -0.6, -2.6)
        test_scalar(Gamma.gamma, x; rtol = 1.0e-5, atol = 1.0e-5)
    end
    test_scalar(Gamma.gamma, 0.5f0; rtol = 1.0e-4, atol = 1.0e-4)
    test_scalar(Gamma.gamma, 4.5f0; rtol = 1.0e-4, atol = 1.0e-4)
end

@testset "Gamma.gamma in-context (beta function)" begin
    # The rule must also fire for `gamma` calls inside a larger differentiated
    # function, with partials accumulating across several call sites.
    B(a, b) = Gamma.gamma(a) * Gamma.gamma(b) / Gamma.gamma(a + b)
    a, b = 1.1, 2.3
    fd = collect(FiniteDifferences.grad(central_fdm(5, 1), B, a, b))

    rev = collect(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(B), a, b))
    @test isapprox(rev, fd; rtol = 1.0e-6, atol = 1.0e-6)

    fwd = collect(Enzyme.gradient(Enzyme.Forward, Enzyme.Const(B), a, b))
    @test isapprox(fwd, fd; rtol = 1.0e-6, atol = 1.0e-6)
end
