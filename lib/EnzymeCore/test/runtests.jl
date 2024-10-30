using Test
using EnzymeCore

@testset verbose = true "EnzymeCore" begin
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
