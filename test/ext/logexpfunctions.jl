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

@testset "xlogx" begin
    # test_scalar needs x - h > 0 and cannot express an infinite derivative
    @test Enzyme.autodiff(Reverse, xlogx, Active, Active(0.0))[1][1] == -Inf
    @test Enzyme.autodiff(Forward, xlogx, Duplicated(0.0, 1.0))[1] == -Inf
    @test Enzyme.autodiff(Reverse, xlogx, Active, Active(0.0f0))[1][1] === -Inf32
    @test Enzyme.autodiff(Reverse, xlogx, Active, Active(1.0e-8))[1][1] ≈ log(1.0e-8) + 1

    test_scalar(xlogx, 1.0)
    test_scalar(xlogx, 2.0)
    test_scalar(xlogx, 2.0f0; rtol = 1.0e-5, atol = 1.0e-5)
end
