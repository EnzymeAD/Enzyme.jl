using Enzyme, Test, LogExpFunctions

include("../common.jl")

xlogydiff(x) = xlogy(x[1], 23.0)
@testset "LogExpFunctions" begin

    x = [0.0]

    grad_forward = Enzyme.gradient(Enzyme.Forward, xlogydiff, x)
    grad_reverse = Enzyme.gradient(Enzyme.Reverse, xlogydiff, x)
    
    @test grad_forward[1] ≈ [log(23.0)] 
    @test grad_reverse[1] ≈ [log(23.0)] 
end

# Reverse mode returned NaN once 1/exp(-x) overflowed
# x/ref: https://github.com/EnzymeAD/Enzyme.jl/issues/3583
@testset "logistic" begin
    test_scalar(logistic, 0.3)
    test_scalar(logistic, 800.0)
    test_scalar(logistic, -700.0)
    # Finite differences are pure noise once logistic saturates, so compare the
    # saturated tail against exp(-x) directly. Guards against 1 - Ω, which
    # cancels to 0 here. Float32 saturates at x ≈ 88.7.
    dx, = autodiff(Reverse, logistic, Active, Active(40.0))[1]
    @test dx ≈ exp(-40.0) rtol = 1.0e-15
    dx32, = autodiff(Reverse, logistic, Active, Active(100.0f0))[1]
    @test dx32 == exp(-100.0f0)
end
