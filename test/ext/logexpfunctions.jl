using Enzyme, Test, LogExpFunctions

xlogydiff(x) = xlogy(x[1], 23.0)
@testset "LogExpFunctions" begin

    x = [0.0]

    grad_forward = Enzyme.gradient(Enzyme.Forward, xlogydiff, x)
    grad_reverse = Enzyme.gradient(Enzyme.Reverse, xlogydiff, x)

    @test grad_forward[1] ≈ [log(23.0)]
    @test grad_reverse[1] ≈ [log(23.0)]

    @testset "logistic (#3583)" begin
        for v in (-800.0, -700.0, 0.3, 40.0, 709.0, 709.78, 709.79, 720.0, 800.0)
            ref = logistic(v) * logistic(-v)
            @test autodiff(Reverse, logistic, Active, Active(v))[1][1] ≈ ref
            @test autodiff(Forward, logistic, Duplicated(v, 1.0))[1] ≈ ref
        end
        for v in (88.0f0, 88.8f0, 100.0f0)
            ref = logistic(v) * logistic(-v)
            @test autodiff(Reverse, logistic, Active, Active(v))[1][1] ≈ ref
            @test autodiff(Forward, logistic, Duplicated(v, 1.0f0))[1] ≈ ref
        end
    end

    @testset "xlogx (#3582)" begin
        @test autodiff(Reverse, xlogx, Active, Active(0.0))[1][1] == -Inf
        @test autodiff(Forward, xlogx, Duplicated(0.0, 1.0))[1] == -Inf
        for v in (1.0e-8, 1.0, 2.0)
            @test autodiff(Reverse, xlogx, Active, Active(v))[1][1] ≈ 1 + log(v)
            @test autodiff(Forward, xlogx, Duplicated(v, 1.0))[1] ≈ 1 + log(v)
        end
    end

    @testset "logaddexp and logsubexp (#2496)" begin
        @test collect(autodiff(Reverse, logaddexp, Active, Active(1.0), Active(1.0))[1]) ≈ [0.5, 0.5]
        @test autodiff(Forward, logaddexp, Duplicated(1.0, 1.0), Const(1.0))[1] ≈ 0.5
        @test autodiff(Forward, logaddexp, Const(1.0), Duplicated(1.0, 1.0))[1] ≈ 0.5
        @test collect(autodiff(Reverse, logaddexp, Active, Active(2.0), Active(0.5))[1]) ≈ [logistic(1.5), logistic(-1.5)]

        @test collect(autodiff(Reverse, logsubexp, Active, Active(2.0), Active(0.5))[1]) ≈ [exp(2.0) / (exp(2.0) - exp(0.5)), -exp(0.5) / (exp(2.0) - exp(0.5))]
        @test collect(autodiff(Reverse, logsubexp, Active, Active(0.5), Active(2.0))[1]) ≈ [-exp(0.5) / (exp(2.0) - exp(0.5)), exp(2.0) / (exp(2.0) - exp(0.5))]
    end
end
