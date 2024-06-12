module ReverseMixedRules

using Enzyme
using Enzyme: EnzymeRules
using Test

Enzyme.API.printall!(true)

import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

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

@inline function guaranteed_nonactive(::Type{T}) where T
    rt = Enzyme.Compiler.active_reg_inner(T, (), nothing)
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

function EnzymeRules.reverse(config::ConfigWidth{1}, func::Const{typeof(mixfnc)},
    dret::Active, tape, tup)
    prev = tup.dval[]
    dRT = typeof(prev)
    @show "rev", tup
    @show dRT, fieldcount(dRT)
    tup.dval[] = Enzyme.Compiler.splatnew(dRT, ntuple(Val(fieldcount(dRT))) do i
        Base.@_inline_meta
        pv = getfield(prev, i)
        if i == 1
            next = 7 * tape[1] * dret.val
            Enzyme.Compiler.recursive_add(pv, next, identity, guaranteed_nonactive)
        else
            pv
        end
    end)
    prev[2][1] = 1000 * dret.val * prev[1]
    return (nothing,)
end

@testset "Mixed activity rule" begin
    x = [3.14]
    dx = [0.0]
    res = autodiff(Reverse, mixouter, Active, Active(2.7), Duplicated(x, dx))[1][1]
    @test res ≈ 7 * 3.14
    @test dx[1] ≈ 1000 * 2.7
    @test x[1] ≈ 0.0
end

end # ReverseMixedRules
