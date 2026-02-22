module MixedRuleError

using Enzyme
using Enzyme.EnzymeRules
using Test

using Enzyme, LinearAlgebra

function handle_infinities(workfunc, f, s)
    s1, s2 = first(s), last(s)
    inf1, inf2 = isinf(s1), isinf(s2)
    if inf1 || inf2
        if inf1 && inf2 # x = t / (1 - t^2)
            return workfunc(
                function (t)
                    t2 = t * t
                    den = 1 / (1 - t2)
                    return f(oneunit(s1) * t * den) * (1 + t2) * den * den * oneunit(s1)
                end,
                map(s) do x
                    isinf(x) ? copysign(one(x), x) : 2x / (oneunit(x) + hypot(oneunit(x), 2x))
                end,
                t -> oneunit(s1) * t / (1 - t^2),
            )
        else
            (s0, si) = inf1 ? (s2, s1) : (s1, s2)
            if si < zero(si) # x = s0 - t / (1 - t)
                return workfunc(
                    function (t)
                        den = 1 / (1 - t)
                        return f(s0 - oneunit(s1) * t * den) * den * den * oneunit(s1)
                    end,
                    reverse(
                        map(s) do x
                            1 / (1 + oneunit(x) / (s0 - x))
                        end
                    ),
                    t -> s0 - oneunit(s1) * t / (1 - t),
                )
            else # x = s0 + t / (1 - t)
                return workfunc(
                    function (t)
                        den = 1 / (1 - t)
                        return f(s0 + oneunit(s1) * t * den) * den * den * oneunit(s1)
                    end,
                    map(s) do x
                        1 / (1 + oneunit(x) / (x - s0))
                    end,
                    t -> s0 + oneunit(s1) * t / (1 - t),
                )
            end
        end
    end
    return workfunc(f, s, identity)
end

outer(f, xs...) = handle_infinities((f_, xs_, _) -> inner(f_, xs_), f, xs)

function inner(f::F, xs) where {F}  # remove type annotation => problem solved
    s = sum(f, xs)
    return (s, norm(s))
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, ::Const{typeof(inner)}, ::Type, f, xs
    )
    true_primal = inner(f.val, xs.val)
    primal = EnzymeRules.needs_primal(config) ? true_primal : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            make_zero(true_primal)
        else
            ntuple(_ -> make_zero(true_primal), Val(EnzymeRules.width(config)))
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        ::EnzymeRules.RevConfig, ::Const{typeof(inner)}, shadow::Active, tape, f, xs
    )
    return ((f isa Active) ? f : nothing, (xs isa Active) ? xs : nothing)
end

F_good(x) = outer(y -> [cos(x * y)], 0.0, 1.0)[1][1]
F_bad(x) = outer(y -> [cos(y)], 0.0, x)[1][1]

@testset "Mixed Return Rule Error" begin
    @static if VERSION < v"1.12"
        @test_throws Enzyme.Compiler.MixedReturnException autodiff(Reverse, F_good, Active(0.3))
        @test_throws Enzyme.Compiler.MixedReturnException autodiff(Reverse, F_bad, Active(0.3))
    else
        @test_throws MethodError autodiff(Reverse, F_good, Active(0.3))
        @test_throws MethodError autodiff(Reverse, F_bad, Active(0.3))
    end
end

end # MixedRuleError
