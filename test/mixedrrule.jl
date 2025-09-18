module ReverseMixedRules

using Enzyme
using Enzyme: EnzymeRules
using Test

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

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(mixfnc)},
    ::Type{<:Active}, tup::MixedDuplicated{Tuple{Float64, Vector{Float64}}})
    pval = func.val(tup.val)
    vec = copy(tup.val[2])
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(mixfnc)},
    dret::Active, tape, tup::MixedDuplicated{Tuple{Float64, Vector{Float64}}})
    prev = tup.dval[]
    tup.dval[] = (7 * tape[1] * dret.val, prev[2])
    prev[2][1] = 1000 * dret.val * tup.val[1]
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


function recmixfnc(tup)
    return sum(tup[1]) * tup[2][1]
end

function recmixouter(x,  y, z)
    res = recmixfnc(((x, z), y))
    fill!(y, 0.0)
    return res
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(recmixfnc)},
    ::Type{<:Active}, tup)
    pval = func.val(tup.val)
    vec = copy(tup.val[2])
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

# check if a value is guaranteed to be not contain active[register] data
# (aka not either mixed or active)
@inline function guaranteed_nonactive(::Type{T}) where T
    rt = Enzyme.Compiler.active_reg_nothrow(T)
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(recmixfnc)},
    dret::Active, tape, tup)
    prev = tup.dval[]
    dRT = typeof(prev)

    tup.dval[] = Enzyme.Compiler.splatnew(dRT, ntuple(Val(fieldcount(dRT))) do i
        Base.@_inline_meta
        pv = getfield(prev, i)
        if i == 1
            next = (7 * tape[1] * dret.val, 31 * tape[1] * dret.val)
            Enzyme.Compiler.recursive_add(pv, next, identity, guaranteed_nonactive)
        else
            pv
        end
    end)
    prev[2][1] = 1000 * dret.val * tup.val[1][1] + .0001 * dret.val * tup.val[1][2]
    return (nothing,)
end

@testset "Recursive Mixed activity rule" begin
    x = [3.14]
    dx = [0.0]
    res = autodiff(Reverse, recmixouter, Active, Active(2.7), Duplicated(x, dx), Active(56.47))[1]
    @test res[1] ≈ 7 * 3.14
    @test res[3] ≈ 31 * 3.14
    @test dx[1] ≈ 1000 * 2.7 + .0001 * 56.47
    @test x[1] ≈ 0.0
end

end # ReverseMixedRules
