module ReverseMixedRules

using Enzyme
using Enzyme: EnzymeRules
using Test

import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

function mixfnc(tup)
    return tup[1] * tup[2][1]
end

function mixouter(x, y)
    res = mixfnc((x, y))
    fill!(y, 0.0)
    return res
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1}, func::Const{typeof(mixfnc)},
        ::Type{<:Active}, tup::MixedDuplicated{Tuple{Float64, Vector{Float64}}}
    )
    pval = func.val(tup.val)
    vec = copy(tup.val[2])
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1}, func::Const{typeof(mixfnc)},
        dret::Active, tape, tup::MixedDuplicated{Tuple{Float64, Vector{Float64}}}
    )
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

function recmixouter(x, y, z)
    res = recmixfnc(((x, z), y))
    fill!(y, 0.0)
    return res
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{1}, func::Const{typeof(recmixfnc)},
        ::Type{<:Active}, tup
    )
    pval = func.val(tup.val)
    vec = copy(tup.val[2])
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{1}, func::Const{typeof(recmixfnc)},
        dret::Active, tape, tup
    )
    prev = tup.dval[]
    dRT = typeof(prev)

    tup.dval[] = Enzyme.Compiler.splatnew(
        dRT, ntuple(Val(fieldcount(dRT))) do i
            Base.@_inline_meta
            pv = getfield(prev, i)
            if i == 1
                next = (7 * tape[1] * dret.val, 31 * tape[1] * dret.val)
                Enzyme.Compiler.recursive_add(pv, next, identity, Enzyme.Compiler.guaranteed_nonactive)
            else
                pv
            end
        end
    )
    prev[2][1] = 1000 * dret.val * tup.val[1][1] + 0.0001 * dret.val * tup.val[1][2]
    return (nothing,)
end

@testset "Recursive Mixed activity rule" begin
    x = [3.14]
    dx = [0.0]
    res = autodiff(Reverse, recmixouter, Active, Active(2.7), Duplicated(x, dx), Active(56.47))[1]
    @test res[1] ≈ 7 * 3.14
    @test res[3] ≈ 31 * 3.14
    @test dx[1] ≈ 1000 * 2.7 + 0.0001 * 56.47
    @test x[1] ≈ 0.0
end

@noinline function batch_mix_void!(tup::Tuple{Float64, Vector{Float64}})
    tup[2][1] *= tup[1]
    return nothing
end

function EnzymeRules.augmented_primal(
        config::RevConfigWidth{2}, func::Const{typeof(batch_mix_void!)},
        ::Type{Const{Nothing}}, tup::BatchMixedDuplicated{Tuple{Float64, Vector{Float64}}, 2}
    )
    orig = copy(tup.val[2])
    func.val(tup.val)
    return AugmentedReturn(nothing, nothing, orig)
end

function EnzymeRules.reverse(
        config::RevConfigWidth{2}, func::Const{typeof(batch_mix_void!)},
        ::Type{Const{Nothing}}, orig, tup::BatchMixedDuplicated{Tuple{Float64, Vector{Float64}}, 2}
    )
    for idx in 1:2
        prev = tup.dval[idx][]
        dy = prev[2][1]
        tup.dval[idx][] = (prev[1] + orig[1] * dy, prev[2])
        prev[2][1] = dy * tup.val[1]
    end
    return (nothing,)
end

@testset "BatchMixedDuplicated activity rule" begin
    t_prim = (3.0, [4.0])
    dt1 = Ref((1.0, [2.0]))
    dt2 = Ref((5.0, [7.0]))
    autodiff(
        Reverse,
        batch_mix_void!,
        Const,
        BatchMixedDuplicated(t_prim, (dt1, dt2)),
    )
    @test t_prim[2] ≈ [12.0]
    @test dt1[][1] ≈ 1.0 + 4.0 * 2.0
    @test dt1[][2] ≈ [6.0]
    @test dt2[][1] ≈ 5.0 + 4.0 * 7.0
    @test dt2[][2] ≈ [21.0]
end

end # ReverseMixedRules
