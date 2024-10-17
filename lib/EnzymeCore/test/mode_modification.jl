using EnzymeCore
using EnzymeCore: split_mode, unsplit_mode
using Test

@testset "Split / unsplit mode" begin
    @test split_mode(Reverse) == ReverseSplitNoPrimal
    @test split_mode(ReverseWithPrimal) == ReverseSplitWithPrimal
    @test split_mode(ReverseSplitNoPrimal) == ReverseSplitNoPrimal
    @test split_mode(ReverseSplitWithPrimal) == ReverseSplitWithPrimal

    @test unsplit_mode(Reverse) == Reverse
    @test unsplit_mode(ReverseWithPrimal) == ReverseWithPrimal
    @test unsplit_mode(ReverseSplitNoPrimal) == Reverse
    @test unsplit_mode(ReverseSplitWithPrimal) == ReverseWithPrimal
end
