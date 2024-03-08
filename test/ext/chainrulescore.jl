using Enzyme
using Test
using ChainRules
using ChainRulesCore
using LinearAlgebra
using EnzymeTestUtils

# utility functions
fdiff(f, x::Number) = autodiff(Forward, f, Duplicated, Duplicated(x, one(x)))[2]
rdiff(f, x::Number) = autodiff(Reverse, f, Active, Active(x))[1][1]
rdiff(f, x::AbstractArray) = autodiff(Reverse, f, Active, Duplicated(x, zero(x)))[1][1]

@testset "import_frule" begin
    f1(x) = 2*x
    ChainRulesCore.@scalar_rule f1(x)  (5.0,)
    Enzyme.@import_frule typeof(f1) Any
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
        for Tret in (Duplicated, DuplicatedNoNeed)
            for Tx in (Duplicated, BatchDuplicated)
                test_forward(sort, Tret, (x, Tx))
            end
        end
    end
end





