using Enzyme
using Test
using ChainRules
using ChainRulesCore
using LinearAlgebra
using EnzymeTestUtils

fdiff(f, x::Number) = autodiff(Forward, f, Duplicated, Duplicated(x, one(x)))[2]

@testset "import_frule" begin
    f1(x) = 2*x
    ChainRulesCore.@scalar_rule f1(x)  (5*one(x),)
    Enzyme.@import_frule typeof(f1) Any
    @test fdiff(f1, 1f0) === 5f0
    @test fdiff(f1, 1.0) === 5.0

    # specific signature    
    f2(x) = 2*x
    ChainRulesCore.@scalar_rule f2(x)  (5*one(x),)
    Enzyme.@import_frule typeof(f2) Float32
    @test fdiff(f2, 1f0) === 5f0
    @test fdiff(f2, 1.0) === 2.0

    # two arguments
    f3(x, y) = 2*x + y
    ChainRulesCore.@scalar_rule f3(x, y)  (5*one(x), y)
    Enzyme.@import_frule typeof(f3) Any Any    
    @test fdiff(x -> f3(x, 1.0), 2.) === 5.0
    @test fdiff(y -> f3(1.0, y), 2.) === 2.0

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

@testset "import_rrule" begin
    f1(x) = 2*x
    ChainRulesCore.@scalar_rule f1(x)  (5*one(x),)
    Enzyme.@import_frule typeof(f1) Any
    @test rdiff(f1, 1f0) === 5f0
    @test rdiff(f1, 1.0) === 5.0

    # specific signature    
    f2(x) = 2*x
    ChainRulesCore.@scalar_rule f2(x)  (5*one(x),)
    Enzyme.@import_frule typeof(f2) Float32
    @test rdiff(f2, 1f0) === 5f0
    @test rdiff(f2, 1.0) === 2.0

    # two arguments
    f3(x, y) = 2*x + y
    ChainRulesCore.@scalar_rule f3(x, y)  (5*one(x), y)
    Enzyme.@import_frule typeof(f3) Any Any    
    @test rdiff(x -> f3(x, 1.0), 2.) === 5.0
    @test rdiff(y -> f3(1.0, y), 2.) === 2.0

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







