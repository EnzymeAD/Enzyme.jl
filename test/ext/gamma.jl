using Gamma
using HypergeometricFunctions
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

@testset "Gamma.gamma in-context (HypergeometricFunctions ₂F₁)" begin
    # Reverse mode differentiating ₂F₁ w.r.t. all of (a, b, c) routes through the
    # parameter-derivative of `gamma` and reproduced the original failure.
    F(a, b, c, z) = HypergeometricFunctions._₂F₁(a, b, c, z)
    a, b, c, z = 1.1, 1.3, 2.3, 0.4
    fd = collect(FiniteDifferences.grad(central_fdm(5, 1), F, a, b, c, z))

    rev = collect(Float64.(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(F), a, b, c, z)))
    @test isapprox(rev, fd; rtol = 1.0e-5, atol = 1.0e-5)

    fwd = collect(Float64.(Enzyme.gradient(Enzyme.Forward, Enzyme.Const(F), a, b, c, z)))
    @test isapprox(fwd, fd; rtol = 1.0e-5, atol = 1.0e-5)
end
