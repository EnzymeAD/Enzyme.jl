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

# https://github.com/EnzymeAD/Enzyme.jl/issues/3583
@testset "logistic" begin
    test_scalar(logistic, 0.3)
    test_scalar(logistic, 800.0)
    test_scalar(logistic, -700.0)
    # saturated tail, where a finite difference is noise and 1 - Ω cancels to 0
    dx, = autodiff(Reverse, logistic, Active, Active(40.0))[1]
    @test dx ≈ exp(-40.0) rtol = 1.0e-14
    dx32, = autodiff(Reverse, logistic, Active, Active(100.0f0))[1]
    @test dx32 ≈ exp(-100.0f0) rtol = 1.0e-5
end
