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
end
