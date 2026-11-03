using Enzyme, Test, Distributions, FillArrays, PDMats, FiniteDifferences

const sqmahal = Distributions.sqmahal

iso(m, v, n) = MvNormal(Fill(m, n), ScalMat(n, v))
sqm(m, v, x) = sqmahal(iso(m, v, length(x)), x)
lpdf(m, s, x) = logpdf(product_distribution(Fill(Normal(m, s), length(x))), x)

fd(f, args...) = FiniteDifferences.grad(central_fdm(5, 1), f, args...)

@testset "sqmahal of an isotropic normal with a Fill mean" begin
    m, v = 0.3, 1.7
    x = [1.0, -2.0, 0.5, 3.0]

    @testset "reverse" begin
        dm, dv, _ = Enzyme.autodiff(Reverse, sqm, Active, Active(m), Active(v), Const(x))[1]
        dx = zero(x)
        Enzyme.autodiff(Reverse, sqm, Active, Const(m), Const(v), Duplicated(x, dx))
        rm, rv, rx = fd(sqm, m, v, x)
        @test dm ≈ rm
        @test dv ≈ rv
        @test dx ≈ rx

        dx = zero(x)
        ds = Enzyme.autodiff(Reverse, lpdf, Active, Active(m), Active(sqrt(v)), Duplicated(x, dx))[1]
        rm, rs, rx = fd(lpdf, m, sqrt(v), x)
        @test ds[1] ≈ rm
        @test ds[2] ≈ rs
        @test dx ≈ rx
    end

    @testset "runtime activity leaves an inactive x alone" begin
        xx = copy(x)
        dm = Enzyme.autodiff(
            set_runtime_activity(Reverse), sqm, Active, Active(m), Const(v), Duplicated(xx, xx)
        )[1][1]
        @test xx == x
        @test dm ≈ fd(sqm, m, v, x)[1]
    end

    @testset "forward" begin
        dm = Enzyme.autodiff(Forward, sqm, Duplicated(m, 1.0), Const(v), Const(x))[1]
        dv = Enzyme.autodiff(Forward, sqm, Const(m), Duplicated(v, 1.0), Const(x))[1]
        rm, rv, rx = fd(sqm, m, v, x)
        @test dm ≈ rm
        @test dv ≈ rv
        e1 = [1.0, 0, 0, 0]
        e3 = [0, 0, 1.0, 0]
        dxs = Enzyme.autodiff(Forward, sqm, Const(m), Const(v), BatchDuplicated(x, (e1, e3)))[1]
        @test dxs[1] ≈ rx[1]
        @test dxs[2] ≈ rx[3]
    end
end
