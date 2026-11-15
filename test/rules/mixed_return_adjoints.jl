using Enzyme
using Enzyme.EnzymeRules
using Test

mixed_pair(x, v; scale = 1.0) = (v, scale * x)

function EnzymeRules.augmented_primal(config::RevConfig, func::Const{typeof(mixed_pair)}, ::Type{RT}, x::Active, v::Annotation; scale = 1.0) where {RT}
    primal = func.val(x.val, v.val; scale)
    shadow = if needs_shadow(config)
        width(config) == 1 ? (v.dval, 0.0) : ntuple(i -> (v.dval[i], 0.0), Val(width(config)))
    else
        nothing
    end
    return AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, shadow)
end

function EnzymeRules.reverse(config::RevConfig, func::Const{typeof(mixed_pair)}, dret::Active, tape, x::Active, v::Duplicated; scale = 1.0)
    return (scale * dret.val[2], nothing)
end

function EnzymeRules.reverse(::RevConfig, ::Const{typeof(mixed_pair)}, dret::NTuple{N, Active{T}}, tape, x::Active, v::BatchDuplicated; scale = 1.0) where {N, T}
    return (map(d -> scale * d.val[2], dret), nothing)
end

mixed_pair_loss(x, v) = (c = mixed_pair(x, v); c[2] * sum(c[1]))
mixed_kw_loss(x, v) = (c = mixed_pair(x, v; scale = 3.0); c[2] * sum(c[1]))

@testset "Mixed return adjoints" begin
    if VERSION >= v"1.12"
        v = [1.0, 2.0, 3.0]
        @test Enzyme.gradient(Reverse, mixed_pair_loss, 2.0, v) == (6.0, fill(2.0, 3))
        @test Enzyme.gradient(Reverse, mixed_kw_loss, 2.0, v) == (18.0, fill(6.0, 3))

        fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(mixed_pair_loss)}, Active, Active{Float64}, BatchDuplicated{Vector{Float64}, 2})
        dv1, dv2 = zeros(3), fill(5.0, 3)
        args = (Const(mixed_pair_loss), Active(2.0), BatchDuplicated(v, (dv1, dv2)))
        tape, primal, shadow = fwd(args...)
        @test primal == 12.0
        @test rev(args..., (1.0, 3.0), tape)[1][1] == (6.0, 18.0)
        @test dv1 == fill(2.0, 3)
        @test dv2 == fill(11.0, 3)
    end
end
