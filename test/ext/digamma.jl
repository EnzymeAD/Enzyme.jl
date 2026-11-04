module DigammaTests

using Enzyme
using SpecialFunctions
using Test

# Keep each nested call specialized on the function and mode.
struct GammaDerivative{M, F}
    mode::M
    f::F
end
(d::GammaDerivative)(x) = only(Enzyme.gradient(d.mode, d.f, x))

function gamma_to_ref!(f, y, x)
    y[] = f(x)
    return nothing
end

@testset "digamma family analytic derivatives" begin
    for T in (Float32, Float64), x0 in (-2.5, -1.5, -0.5, -0.7, -0.25, 0.25, 2.5, 10.0)
        x = T(x0)
        for (f, expected) in (
                (digamma, trigamma(x)), (trigamma, polygamma(2, x)),
                (Base.Fix1(polygamma, 2), polygamma(3, x)),
            )
            @test only(gradient(Reverse, Const(f), x)) ≈ expected
            @test only(gradient(Forward, Const(f), x)) ≈ expected
            dy, y = autodiff(ForwardWithPrimal, f, Duplicated(x, T(2)))
            @test y == f(x)
            @test dy ≈ 2expected
            dys, y = autodiff(ForwardWithPrimal, f, BatchDuplicated(x, (T(2), T(-3))))
            @test y == f(x)
            @test dys[1] ≈ 2expected
            @test dys[2] ≈ -3expected
            out = Ref(zero(x))
            ds = autodiff(
                Reverse, gamma_to_ref!, Const(f),
                BatchDuplicated(out, (Ref(T(2)), Ref(T(-3)))), Active(x)
            )[1][3]
            @test out[] == f(x)
            @test ds[1] ≈ 2expected
            @test ds[2] ≈ -3expected
        end
    end
    for mode in (Forward, Reverse)
        for x in (prevfloat(-0.5), nextfloat(-0.5))
            @test only(gradient(mode, digamma, x)) ≈ trigamma(x)
        end
        for n in 0:3
            @test only(gradient(mode, Base.Fix1(polygamma, n), -0.5)) ≈ polygamma(n + 1, -0.5)
        end
        @test_throws DomainError gradient(mode, Base.Fix1(polygamma, -1), 0.5)
    end
end

@testset "nested digamma derivatives at negative half-integers" begin
    for modes in (
            (Forward, Forward, Forward), (Forward, Forward, Reverse),
            (Forward, Reverse, Forward), (Reverse, Forward, Forward),
        )
        d1 = GammaDerivative(modes[3], digamma)
        d2 = GammaDerivative(modes[2], d1)
        d3 = GammaDerivative(modes[1], d2)
        d4 = GammaDerivative(Forward, d3)
        for x in (-2.5, -1.5, -0.5, -0.7, 0.25, 2.5)
            @test d1(x) ≈ trigamma(x)
            @test d2(x) ≈ polygamma(2, x)
            @test d3(x) ≈ polygamma(3, x)
            @test d4(x) ≈ polygamma(4, x)
        end
    end
end

end
