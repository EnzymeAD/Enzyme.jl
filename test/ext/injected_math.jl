module InjectedMathTests

using Enzyme
using Enzyme.EnzymeRules: AugmentedReturn, RevConfig, needs_primal
using SpecialFunctions
using Test
import Enzyme.EnzymeRules

struct BetaRule{Inline} end
(::BetaRule)(a, b) = logbeta(a, b)

@inline function beta_cut(a, b)
    y = logbeta(a, b)
    s = digamma(a + b)
    return (y, digamma(a) - s, digamma(b) - s)
end

@inline beta_cut(::Val{true}, a, b) = beta_cut(a, b)
@noinline beta_cut(::Val{false}, a, b) = beta_cut(a, b)

function EnzymeRules.augmented_primal(
        config::RevConfig, ::Const{BetaRule{Inline}}, ::Type{<:Active},
        a::Active, b::Active,
    ) where {Inline}
    y, da, db = beta_cut(Val(Inline), a.val, b.val)
    return AugmentedReturn(needs_primal(config) ? y : nothing, nothing, (da, db))
end

function EnzymeRules.reverse(
        ::RevConfig, ::Const{<:BetaRule}, dy::Active, tape, a::Active, b::Active,
    )
    return (dy.val * tape[1], dy.val * tape[2])
end

struct BetaObjective{Inline} end
(::BetaObjective{Inline})(x) where {Inline} = BetaRule{Inline}()(x[1], x[2])

struct BetaGradient{Inline} end
(::BetaGradient{Inline})(x) where {Inline} = only(gradient(Reverse, BetaObjective{Inline}(), x))

function beta_hvp(x, v)
    s = trigamma(x[1] + x[2]) * (v[1] + v[2])
    return [trigamma(x[1]) * v[1] - s, trigamma(x[2]) * v[2] - s]
end

@testset "Math implementations imported through a custom rule" begin
    # The outlined helper imports an already-lowered logabsgamma into the outer
    # differentiation. Its derivative still needs digamma (or digammaf), even
    # though the outer compilation did not lower that primal math function.
    for T in (Float32, Float64), inline in (false, true)
        f = BetaObjective{inline}()
        g = BetaGradient{inline}()
        for x in (T[2.3, 0.7], T[0.4, 1.2])
            before = copy(x)
            v = T[0.2, -0.1]
            w = T[-0.3, 0.4]
            expected = digamma.(x) .- digamma(sum(x))
            @test g(x) ≈ expected
            dv, value = autodiff(ForwardWithPrimal, g, Duplicated(x, v))
            @test dv ≈ beta_hvp(x, v)
            @test value ≈ expected
            dvs, value = autodiff(ForwardWithPrimal, g, BatchDuplicated(x, (v, w)))
            @test dvs[1] ≈ beta_hvp(x, v)
            @test dvs[2] ≈ beta_hvp(x, w)
            @test value ≈ expected
            @test only(gradient(Forward, f, x)) ≈ expected
            @test x == before
        end
    end
end

end
