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

    @testset "batch duplicated" begin 
        x = [1.0, 2.0, 0.0]        
        Enzyme.@import_frule typeof(Base.sort)  Any
        test_forward(Base.sort, Duplicated, (x, Duplicated))
        # test_forward(Base.sort, Duplicated, (x, BatchDuplicated))
        # test_forward(Base.sort, DuplicatedNoNeed, (x, Duplicated))
        # test_forward(Base.sort, DuplicatedNoNeed, (x, BatchDuplicated))
        # for Tret in (Duplicated, DuplicatedNoNeed)
        #     for Tx in (Duplicated, BatchDuplicated)
        #         test_forward(sort, Tret, (x, Tx))
        #     end
        # end
    end
end





