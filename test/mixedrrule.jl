module ReverseMixedRules

using Enzyme
using Enzyme: EnzymeRules
using Test

function mixfnc(tup)
    return tup[1] * tup[2][1]
end

function mixouter(x,  y)
    res = mixfnc((x, y))
    fill!(y, 0.0)
    return res
end

function EnzymeRules.augmented_primal(config::ConfigWidth{1}, func::Const{typeof(mixfnc)},
    ::Type{<:Active}, tup)
    @show tup
    pval = func.val(tup.val)
    vec = copy(tup.val[2])
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

function EnzymeRules.reverse(config::ConfigWidth{1}, func::Const{Closure},
    dret::Active, tape, tup)
    dargs = 7 * tup[1].val * dret.val + tape[1] * 1000
    return (dres,)
end

@testset "Mixed activity rule" begin
    x = [3.14]
    dx = [0.0]
    res = autodiff(Reverse, mixouter, Active, Active(2.7), Duplicated(x, dx))[1][1]
    @test res ≈ 7 * 2.7 + 3.14 * 1000
    @test cl.v[1] ≈ 0.0
end

end # ReverseMixedRules
