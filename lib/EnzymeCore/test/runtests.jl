using Test
using EnzymeCore

@testset verbose = true "EnzymeCore" begin
    @testset "WithPrimal" begin
        @test WithPrimal(Reverse) === ReverseWithPrimal
        @test NoPrimal(Reverse) === Reverse
        @test WithPrimal(ReverseWithPrimal) === ReverseWithPrimal
        @test NoPrimal(ReverseWithPrimal) === Reverse
    
        @test WithPrimal(set_runtime_activity(Reverse)) === set_runtime_activity(ReverseWithPrimal)
    
        @test WithPrimal(Forward) === ForwardWithPrimal
        @test NoPrimal(Forward) === Forward
        @test WithPrimal(ForwardWithPrimal) === ForwardWithPrimal
        @test NoPrimal(ForwardWithPrimal) === Forward
    
        @test WithPrimal(ReverseSplitNoPrimal) === ReverseSplitWithPrimal
        @test NoPrimal(ReverseSplitNoPrimal) === ReverseSplitNoPrimal
        @test WithPrimal(ReverseSplitWithPrimal) === ReverseSplitWithPrimal
        @test NoPrimal(ReverseSplitWithPrimal) === ReverseSplitNoPrimal
    end

    @testset "needs_primal" begin
        @test needs_primal(Reverse) === false
        @test needs_primal(ReverseWithPrimal) === true
        @test needs_primal(Forward) === false
        @test needs_primal(ForwardWithPrimal) === true
        @test needs_primal(ReverseSplitNoPrimal) === false
        @test needs_primal(ReverseSplitWithPrimal) === true
    end

    @testset "Miscellaneous" begin
        include("misc.jl")
    end
end
