using Enzyme, Test
using FiniteDifferences

@isdefined(UTILS_INCLUDE) || include("utils.jl")


@testset "Compare against" begin
    x = 3.0
    fd = central_fdm(5, 1)(sin, x)

    @test fd ≈ cos(x)
    @test fd ≈ first(autodiff(Reverse, sin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, sin, Duplicated(x, 1.0)))

    x = 0.2 + sin(3.0)
    fd = central_fdm(5, 1)(asin, x)

    @test fd ≈ 1/sqrt(1-x*x)
    @test fd ≈ first(autodiff(Reverse, asin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, asin, Duplicated(x, 1.0)))
    test_scalar(asin, x)

    function foo(x)
        a = sin(x)
        b = 0.2 + a
        c = asin(b)
        return c
    end

    x = 3.0
    fd = central_fdm(5, 1)(foo, x)

    @test fd ≈ cos(x)/sqrt(1-(0.2+sin(x))*(0.2+sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1.0)))
    test_scalar(foo, x)

    # Input type shouldn't matter
    x = 3
    @test fd ≈ cos(x)/sqrt(1-(0.2+sin(x))*(0.2+sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    # They do matter for duplicated, which can't be auto promoted
    # @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1)))

    f74(a, c) = a * √c
    @test √3 ≈ first(autodiff(Reverse, f74, Active, Active(2), Const(3)))[1]
    @test √3 ≈ first(autodiff(Forward, f74, Duplicated(2.0, 1.0), Const(3)))
end

