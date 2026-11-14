using Enzyme, Test, LogExpFunctions

xlogydiff(x) = xlogy(x[1], 23.0)
@testset "LogExpFunctions" begin

    x = [0.0]

    grad_forward = Enzyme.gradient(Enzyme.Forward, xlogydiff, x)
    grad_reverse = Enzyme.gradient(Enzyme.Reverse, xlogydiff, x)

    @test grad_forward[1] ≈ [log(23.0)]
    @test grad_reverse[1] ≈ [log(23.0)]
end
