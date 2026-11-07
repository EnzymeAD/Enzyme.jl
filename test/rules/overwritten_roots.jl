# Custom rules whose by-reference argument (an immutable struct mixing GC-tracked
# pointer fields with inline fields) lives in a stack slot that a loop overwrites
# every iteration. The reverse pass has to see each iteration's own primal and
# shadow, not whatever the slot held last.
# https://github.com/EnzymeAD/Enzyme.jl/issues/3477
# https://github.com/EnzymeAD/Enzyme.jl/issues/3602
module OverwrittenRootsRules

using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using Test

struct PtrInt
    data::Vector{Float64}
    k::Int
end

struct PtrFloatInt
    data::Vector{Float64}
    w::Float64
    k::Int
end

const COEFFS = [1.0, 2.0, 3.0, 4.0]

# --- Duplicated argument: inline part is inactive (Int only) ---

red_int(x) = x.k * sum(x.data)

const seen_int = Tuple{Int, Int, Vector{Float64}}[]

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(red_int)}, ::Type{RT}, A::Annotation
    ) where {RT}
    ret = red_int(A.val)
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? ret : nothing,
        EnzymeRules.needs_shadow(config) ? zero(ret) : nothing, nothing
    )
end
function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(red_int)}, dret::Active, cache, A::Annotation
    )
    if !(A isa Const)
        push!(seen_int, (A.val.k, A.dval.k, A.val.data))
        A.dval.data .+= dret.val * A.val.k
    end
    return (nothing,)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{2},
        ::Const{typeof(red_int)}, ::Type{RT}, A::Annotation
    ) where {RT}
    ret = red_int(A.val)
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? ret : nothing,
        EnzymeRules.needs_shadow(config) ? (zero(ret), zero(ret)) : nothing, nothing
    )
end
function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{2},
        ::Const{typeof(red_int)}, dret::NTuple{2, <:Active}, cache, A::Annotation
    )
    if !(A isa Const)
        push!(seen_int, (A.val.k, A.dval[1].k, A.val.data))
        for (d, dr) in zip(A.dval, dret)
            d.data .+= dr.val * A.val.k
        end
    end
    return (nothing,)
end

@noinline mk_int(a, i) = PtrInt(COEFFS[i] .* copy(a.data), a.k + i)
loop_int(a) = (
    s = 0.0; for i in 1:4
        s += red_int(mk_int(a, i))
    end; s
)

# d loop_int / d a.data[j] = sum_i (a.k + i) * COEFFS[i]
expected_int(a) = sum((a.k + i) * COEFFS[i] for i in 1:4)

@testset "Duplicated mixed struct overwritten in loop" begin
    for mode in (:combined, :split)
        empty!(seen_int)
        a = PtrInt([1.0, 2.0, 3.0], 3)
        da = Enzyme.make_zero(a)
        if mode == :combined
            autodiff(Reverse, loop_int, Active, Duplicated(a, da))
        else
            fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(loop_int)}, Active, Duplicated{PtrInt})
            tape, _, _ = fwd(Const(loop_int), Duplicated(a, da))
            rev(Const(loop_int), Duplicated(a, da), 1.0, tape)
        end
        @test all(da.data .≈ expected_int(a))
        # reverse visits iterations 4,3,2,1 and each must see its own values
        @test [s[1] for s in seen_int] == [7, 6, 5, 4]
        @test [s[2] for s in seen_int] == [7, 6, 5, 4]
        @test [s[3] for s in seen_int] == [COEFFS[i] .* a.data for i in 4:-1:1]
    end
end

@testset "BatchDuplicated mixed struct overwritten in loop" begin
    empty!(seen_int)
    a = PtrInt([1.0, 2.0, 3.0], 3)
    da1 = Enzyme.make_zero(a)
    da2 = Enzyme.make_zero(a)
    autodiff(Reverse, loop_int, Active, BatchDuplicated(a, (da1, da2)))
    @test all(da1.data .≈ expected_int(a))
    @test all(da2.data .≈ expected_int(a))
    @test [s[1] for s in seen_int] == [7, 6, 5, 4]
end

# --- MixedDuplicated argument: inline part holds an active Float64 and an Int ---

red_mixed(x) = x.w * sum(x.data)

const seen_mixed = Tuple{Int, Int}[]

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(red_mixed)}, ::Type{RT}, A::Annotation
    ) where {RT}
    ret = red_mixed(A.val)
    return EnzymeRules.AugmentedReturn(
        EnzymeRules.needs_primal(config) ? ret : nothing,
        EnzymeRules.needs_shadow(config) ? zero(ret) : nothing, nothing
    )
end
function EnzymeRules.reverse(
        ::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(red_mixed)}, dret::Active, cache, A::MixedDuplicated
    )
    d = A.dval[]
    push!(seen_mixed, (A.val.k, d.k))
    d.data .+= dret.val * A.val.w
    A.dval[] = PtrFloatInt(d.data, d.w + dret.val * sum(A.val.data), d.k)
    return (nothing,)
end

@noinline mk_mixed(a, i) = PtrFloatInt(COEFFS[i] .* copy(a.data), a.w * i, a.k + i)
loop_mixed(a) = (
    s = 0.0; for i in 1:4
        s += red_mixed(mk_mixed(a, i))
    end; s
)

# loop_mixed(a) = sum_i (a.w * i) * COEFFS[i] * sum(a.data)
expected_mixed_data(a) = a.w * sum(i * COEFFS[i] for i in 1:4)
expected_mixed_w(a) = sum(i * COEFFS[i] for i in 1:4) * sum(a.data)

@testset "MixedDuplicated mixed struct overwritten in loop" begin
    for mode in (:combined, :split)
        empty!(seen_mixed)
        a = PtrFloatInt([1.0, 2.0, 3.0], 0.5, 3)
        r = Ref(Enzyme.make_zero(a))
        if mode == :combined
            autodiff(Reverse, loop_mixed, Active, MixedDuplicated(a, r))
        else
            fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(loop_mixed)}, Active, MixedDuplicated{PtrFloatInt})
            tape, _, _ = fwd(Const(loop_mixed), MixedDuplicated(a, r))
            rev(Const(loop_mixed), MixedDuplicated(a, r), 1.0, tape)
        end
        da = r[]
        @test all(da.data .≈ expected_mixed_data(a))
        @test da.w ≈ expected_mixed_w(a)
        @test [s[1] for s in seen_mixed] == [7, 6, 5, 4]
        @test [s[2] for s in seen_mixed] == [7, 6, 5, 4]
    end
end

end # module
