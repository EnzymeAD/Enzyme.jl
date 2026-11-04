using Enzyme
using Test
using ChainRules
using ChainRulesCore
using LinearAlgebra
using EnzymeTestUtils

module MockModule
    struct MockType
        x::Float32
    end

    mock_function(x::MockType) = 2 * x.x
end

function ChainRulesCore.frule((_, ẋ), ::typeof(MockModule.mock_function), x)
    y = MockModule.mock_function(x)
    ẏ = 3 * ẋ.x
    return y, ẏ
end

function ChainRulesCore.rrule(::typeof(MockModule.mock_function), x)
    y = MockModule.mock_function(x)
    return y, ȳ -> (NoTangent(), MockModule.MockType(2 * ȳ))
end

fdiff(f, x::Number) = autodiff(ForwardWithPrimal, f, Duplicated, Duplicated(x, one(x)))[1]
fdiff(f, x::MockModule.MockType) = autodiff(ForwardWithPrimal, f, Duplicated, Duplicated(x, MockModule.MockType(one(x.x))))[1]

fdiff2(f, x::Number) = autodiff(Forward, f, Duplicated, Duplicated(x, one(x)))[1]
fdiff2(f, x::MockModule.MockType) = autodiff(Forward, f, Duplicated, Duplicated(x, MockModule.MockType(one(x.x))))[1]

@testset "import_frule" begin
    f1(x) = 2 * x
    ChainRulesCore.@scalar_rule f1(x)  (5 * one(x),)
    Enzyme.@import_frule typeof(f1) Any
    @test fdiff(f1, 1.0f0) === 5.0f0
    @test fdiff(f1, 1.0) === 5.0
    @test fdiff2(f1, 1.0f0) === 5.0f0
    @test fdiff2(f1, 1.0) === 5.0

    # specific signature
    f2(x) = 2 * x
    ChainRulesCore.@scalar_rule f2(x)  (5 * one(x),)
    Enzyme.@import_frule typeof(f2) Float32
    @test fdiff(f2, 1.0f0) === 5.0f0
    @test fdiff(f2, 1.0) === 2.0
    @test fdiff2(f2, 1.0f0) === 5.0f0
    @test fdiff2(f2, 1.0) === 2.0

    # two arguments
    f3(x, y) = 2 * x + y
    ChainRulesCore.@scalar_rule f3(x, y)  (5 * one(x), y)
    Enzyme.@import_frule typeof(f3) Any Any
    @test fdiff(x -> f3(x, 1.0), 2.0) === 5.0
    @test fdiff(y -> f3(1.0, y), 2.0) === 2.0
    @test fdiff2(x -> f3(x, 1.0), 2.0) === 5.0
    @test fdiff2(y -> f3(1.0, y), 2.0) === 2.0

    # external module (checks correct type escaping, PR #1446)
    Enzyme.@import_frule typeof(MockModule.mock_function) MockModule.MockType
    @test fdiff(MockModule.mock_function, MockModule.MockType(1.0f0)) === 3.0f0

    @testset "batch duplicated" begin
        x = [1.0, 2.0, 0.0]
        Enzyme.@import_frule typeof(Base.sort) Any

        test_forward(Base.sort, Duplicated, (x, Duplicated))
        # Unsupported by EnzymeTestUtils
        # test_forward(Base.sort, Duplicated, (x, DuplicatedNoNeed))
        test_forward(Base.sort, DuplicatedNoNeed, (x, Duplicated))
        # Unsupported by EnzymeTestUtils
        # test_forward(Base.sort, DuplicatedNoNeed, (x, DuplicatedNoNeed))
        test_forward(Base.sort, Const, (x, Duplicated))
        # Unsupported by EnzymeTestUtils
        # test_forward(Base.sort, Const, (x, DuplicatedNoNeed))

        test_forward(Base.sort, Const, (x, Const))

        # ChainRules does not support this case (returning notangent)
        # test_forward(Base.sort, Duplicated, (x, Const))
        # test_forward(Base.sort, DuplicatedNoNeed, (x, Const))

        test_forward(Base.sort, BatchDuplicated, (x, BatchDuplicated))
        # Unsupported by EnzymeTestUtils
        # test_forward(Base.sort, BatchDuplicated, (x, BatchDuplicatedNoNeed))
        test_forward(Base.sort, BatchDuplicatedNoNeed, (x, BatchDuplicated))
        # Unsupported by EnzymeTestUtils
        # test_forward(Base.sort, BatchDuplicatedNoNeed, (x, BatchDuplicatedNoNeed))
        test_forward(Base.sort, Const, (x, BatchDuplicated))
        # Unsupported by EnzymeTestUtils
        # test_forward(Base.sort, Const, (x, BatchDuplicatedNoNeed))

        # ChainRules does not support this case (returning notangent)
        # test_forward(Base.sort, BatchDuplicated, (x, Const))
        # test_forward(Base.sort, BatchDuplicatedNoNeed, (x, Const))
    end
end

rdiff(f, x::Number) = autodiff(Reverse, f, Active, Active(x))[1][1]
rdiff(f, x::MockModule.MockType) = autodiff(Reverse, f, Active, Active(x))[1][1]

@testset "import_rrule" begin
    f1(x) = 2 * x
    ChainRulesCore.@scalar_rule f1(x)  (5 * one(x),)
    Enzyme.@import_rrule typeof(f1) Any
    @test rdiff(f1, 1.0f0) === 5.0f0
    @test rdiff(f1, 1.0) === 5.0

    # specific signature
    f2(x) = 2 * x
    ChainRulesCore.@scalar_rule f2(x)  (5 * one(x),)
    Enzyme.@import_rrule typeof(f2) Float32
    @test rdiff(f2, 1.0f0) === 5.0f0
    @test rdiff(f2, 1.0) === 2.0

    # two arguments
    f3(x, y) = 2 * x + y
    ChainRulesCore.@scalar_rule f3(x, y)  (5 * one(x), y)
    Enzyme.@import_rrule typeof(f3) Any Any
    @test rdiff(x -> f3(x, 1.0), 2.0) === 5.0
    @test rdiff(y -> f3(1.0, y), 2.0) === 2.0

    # external module (checks correct type escaping, PR #1446)
    Enzyme.@import_rrule typeof(MockModule.mock_function) MockModule.MockType
    @test rdiff(MockModule.mock_function, MockModule.MockType(1.0f0)) === MockModule.MockType(2.0f0)

    @testset "batch duplicated" begin
        x = [1.0, 2.0, 0.0]
        Enzyme.@import_rrule typeof(Base.sort) Any

        test_reverse(Base.sort, Duplicated, (x, Duplicated))
        # Unsupported by EnzymeTestUtils
        # test_reverse(Base.sort, Duplicated, (x, DuplicatedNoNeed))
        test_reverse(Base.sort, DuplicatedNoNeed, (x, Duplicated))
        # Unsupported by EnzymeTestUtils
        # test_reverse(Base.sort, DuplicatedNoNeed, (x, DuplicatedNoNeed))
        test_reverse(Base.sort, Const, (x, Duplicated))
        # Unsupported by EnzymeTestUtils
        # test_reverse(Base.sort, Const, (x, DuplicatedNoNeed))

        test_reverse(Base.sort, Const, (x, Const))

        # ChainRules does not support this case (returning notangent)
        # test_reverse(Base.sort, Duplicated, (x, Const))
        # test_reverse(Base.sort, DuplicatedNoNeed, (x, Const))

        test_reverse(Base.sort, BatchDuplicated, (x, BatchDuplicated))
        # Unsupported by EnzymeTestUtils
        # test_reverse(Base.sort, BatchDuplicated, (x, BatchDuplicatedNoNeed))
        test_reverse(Base.sort, BatchDuplicatedNoNeed, (x, BatchDuplicated))
        # Unsupported by EnzymeTestUtils
        # test_reverse(Base.sort, BatchDuplicatedNoNeed, (x, BatchDuplicatedNoNeed))
        test_reverse(Base.sort, Const, (x, BatchDuplicated))
        # Unsupported by EnzymeTestUtils
        # test_reverse(Base.sort, Const, (x, BatchDuplicatedNoNeed))

        # ChainRules does not support this case (returning notangent)
        # test_reverse(Base.sort, BatchDuplicated, (x, Const))
        # test_reverse(Base.sort, BatchDuplicatedNoNeed, (x, Const))
    end

    @testset "unread GC-tracked struct field read by rrule" begin
        @test Enzyme.EnzymeRules.has_readonly_rule(MockModule.mock_function, MockModule.MockType(1.0f0))
        @test !Enzyme.EnzymeRules.has_attribute_preserving_rule_from_sig(Tuple{typeof(MockModule.mock_function), MockModule.MockType})
    end
end

struct RRulePartialArg
    mode::Int
    weights::Vector{Float64}
end

const rrule_partial_calls = Ref(0)

@noinline rrule_partial(p::RRulePartialArg, u::Vector{Float64}) = p.mode .* u

function ChainRulesCore.rrule(::typeof(rrule_partial), p::RRulePartialArg, u::Vector{Float64})
    rrule_partial_calls[] += 1
    y = p.weights .* (p.mode .* u)
    function rrule_partial_pullback(dy)
        dy_unthunked = ChainRulesCore.unthunk(dy)
        return NoTangent(), NoTangent(), dy_unthunked .* (p.mode .* p.weights)
    end
    return y, rrule_partial_pullback
end

Enzyme.@import_rrule typeof(rrule_partial) RRulePartialArg Vector{Float64}

rrule_partial_loss(u) = sum(rrule_partial(RRulePartialArg(2, [3.0, 5.0]), u))
function rrule_partial_const_call(u)
    _ = rrule_partial(RRulePartialArg(2, [3.0, 5.0]), [1.0, 1.0])
    return sum(u)
end

@testset "import_rrule preserves unread GC-tracked struct fields" begin
    @test Enzyme.gradient(Reverse, rrule_partial_loss, [1.0, 2.0])[1] ≈ [6.0, 10.0]
    @test Enzyme.gradient(Enzyme.set_runtime_activity(Reverse), rrule_partial_loss, [1.0, 2.0])[1] ≈ [6.0, 10.0]

    rrule_partial_calls[] = 0
    @test Enzyme.gradient(Reverse, rrule_partial_const_call, [1.0, 2.0])[1] ≈ [1.0, 1.0]
    @test rrule_partial_calls[] == 0
end

rrule_typed_scale(z::Matrix{T}, s::Int) where {T <: AbstractFloat} = T(s) .* z
@noinline rrule_untyped_scale(z::Matrix{T}, s::Int) where {T <: AbstractFloat} = Base.inferencebarrier(T(s) .* z)

for fn in (:rrule_typed_scale, :rrule_untyped_scale)
    @eval function ChainRulesCore.rrule(::typeof($fn), z::Matrix{T}, s::Int) where {T <: AbstractFloat}
        y = $fn(z, s)
        pb(Δ) = (NoTangent(), T(s) .* T.(ChainRulesCore.unthunk(Δ)), NoTangent())
        return y, pb
    end
    @eval Enzyme.@import_rrule typeof($fn) Matrix{Float32} Int
end

const rrule_scale_weights = Float32[1 0.5; -1 2]
rrule_typed_loss(x) = sum(rrule_scale_weights .* rrule_typed_scale(x, 3))
rrule_untyped_loss(x) = sum(rrule_scale_weights .* (rrule_untyped_scale(x, 3)::Matrix{Float32}))

@testset "import_rrule non-concrete cache tuple (#2503)" begin
    z = Float32[1 2; 3 4]
    @test Enzyme.gradient(Reverse, rrule_typed_loss, z)[1] ≈ 3.0f0 .* rrule_scale_weights
    @test Enzyme.gradient(Reverse, rrule_untyped_loss, z)[1] ≈ 3.0f0 .* rrule_scale_weights
end
