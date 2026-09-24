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

@testset "LogExpFunctions xlog1py and xexpy" begin
    test_scalar(x -> xlog1py(x, -0.5), 0.0)
    test_scalar(x -> xlog1py(x, -0.5), 1.0)
    test_scalar(y -> xlog1py(2.0, y), -0.5)
    test_scalar(x -> xexpy(x, 2.0), 0.0)
    test_scalar(x -> xexpy(x, 2.0), 1.0)
    test_scalar(y -> xexpy(2.0, y), 2.0)
end
