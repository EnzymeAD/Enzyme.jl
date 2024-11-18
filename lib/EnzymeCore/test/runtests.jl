using Test
using EnzymeCore

@testset verbose = true "EnzymeCore" begin
    @testset "WithPrimal" begin
        @test EnzymeCore.WithPrimal(Reverse) === ReverseWithPrimal
        @test EnzymeCore.NoPrimal(Reverse) === Reverse
        @test EnzymeCore.WithPrimal(ReverseWithPrimal) === ReverseWithPrimal
        @test EnzymeCore.NoPrimal(ReverseWithPrimal) === Reverse
    
        @test EnzymeCore.WithPrimal(EnzymeCore.set_runtime_activity(Reverse)) === EnzymeCore.set_runtime_activity(ReverseWithPrimal)
    
        @test EnzymeCore.WithPrimal(Forward) === ForwardWithPrimal
        @test EnzymeCore.NoPrimal(Forward) === Forward
        @test EnzymeCore.WithPrimal(ForwardWithPrimal) === ForwardWithPrimal
        @test EnzymeCore.NoPrimal(ForwardWithPrimal) === Forward
    
        @test EnzymeCore.WithPrimal(ReverseSplitNoPrimal) === ReverseSplitWithPrimal
        @test EnzymeCore.NoPrimal(ReverseSplitNoPrimal) === ReverseSplitNoPrimal
        @test EnzymeCore.WithPrimal(ReverseSplitWithPrimal) === ReverseSplitWithPrimal
        @test EnzymeCore.NoPrimal(ReverseSplitWithPrimal) === ReverseSplitNoPrimal
    end

    @testset "needs_primal" begin
        @test EnzymeCore.needs_primal(Reverse) === false
        @test EnzymeCore.needs_primal(ReverseWithPrimal) === true
        @test EnzymeCore.needs_primal(Forward) === false
        @test EnzymeCore.needs_primal(ForwardWithPrimal) === true
        @test EnzymeCore.needs_primal(ReverseSplitNoPrimal) === false
        @test EnzymeCore.needs_primal(ReverseSplitWithPrimal) === true
    end

    @testset "Miscellaneous" begin
        include("misc.jl")
    end
end
