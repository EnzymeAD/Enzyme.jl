using Enzyme
using Test
using Enzyme.EnzymeRules

# The augmented primal of a rule may read and write the shadow of its arguments. The
# shadow of an allocation local to the caller must therefore exist in the forward pass,
# also when the call to the rule sits behind a function that is not inlined,
# https://github.com/EnzymeAD/Enzyme.jl/issues/3562

function scale_ref!(r::Base.RefValue{Float64})
    r[] = 2 * r[]
    return nothing
end

const seen_dvals = Float64[]

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(scale_ref!)}, ::Type{RT}, r::Duplicated) where {RT}
    push!(seen_dvals, r.dval[])
    r.dval[] += 0.0
    scale_ref!(r.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(scale_ref!)}, ::Type{RT}, tape, r::Duplicated) where {RT}
    r.dval[] = 2 * r.dval[]
    return (nothing,)
end

function scale_direct(x)
    r = Ref(x)
    scale_ref!(r)
    return r[]
end

@noinline scale_helper!(r) = scale_ref!(r)

function scale_indirect(x)
    r = Ref(x)
    scale_helper!(r)
    return r[]
end

@testset "Shadow of a local allocation in augmented_primal: $f" for f in (scale_direct, scale_indirect)
    empty!(seen_dvals)
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Active{Float64})
    tape, res, _ = fwd(Const(f), Active(3.0))
    @test res == 6.0
    @test seen_dvals == [0.0]
    @test rev(Const(f), Active(3.0), 1.0, tape) == ((2.0,),)

    @test autodiff(Reverse, f, Active, Active(3.0)) == ((2.0,),)
end
