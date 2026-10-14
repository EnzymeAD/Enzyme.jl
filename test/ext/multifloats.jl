using Enzyme
using Test
using MultiFloats

# Derivatives in double-double arithmetic, checked against BigFloat.
mf_poly(x) = x * x * x + 2 * x
mf_rational(x) = (x * 0.1 + 1.0) / (x - 2.0)
mf_special(x) = sqrt(x) + exp(x) * log(x) + inv(x)
mf_sum_of_squares(v) = sum(abs2, v)

@testset "MultiFloats" begin
    T = Float64x2
    x = T(1) / 3
    xb = BigFloat(x; precision = 256)
    relerr(a, b) = Float64(abs(BigFloat(a; precision = 256) - b) / abs(b))

    for (f, df) in (
            (mf_poly, 3 * xb^2 + 2),
            (mf_rational, (BigFloat(0.1) * (xb - 2) - (BigFloat(0.1) * xb + 1)) / (xb - 2)^2),
            (mf_special, 1 / (2 * sqrt(xb)) + exp(xb) * log(xb) + exp(xb) / xb - 1 / xb^2),
        )
        # Forward mode is exact to double-double precision
        @test relerr(autodiff(Forward, f, Duplicated(x, one(T)))[1], df) < 1.0e-30
        # Reverse mode accumulates the cotangents of a reused value limb by limb, which is
        # exact only to the precision of a limb
        @test relerr(autodiff(Reverse, f, Active(x))[1][1], df) < 1.0e-15
    end
    # Without rules, reverse mode returned twice the gradient
    @test autodiff(Reverse, y -> y + y, Active(x))[1][1] == 2
    @test relerr(autodiff(Reverse, y -> y * y, Active(x))[1][1], 2 * xb) < 1.0e-30

    v = [T(1) / 3, T(2) / 7]
    dv = zero.(v)
    autodiff(Reverse, mf_sum_of_squares, Active, Duplicated(v, dv))
    @test all(relerr.(dv, 2 .* BigFloat.(v; precision = 256)) .< 1.0e-30)
end
