using EnzymeCore
using EnzymeCore: InlineABI, ReverseModeSplit, Split, Combined, set_runtime_activity, set_err_if_func_written, set_abi
using Test

@testset "Split / unsplit mode" begin
    @test Split(Reverse) == ReverseSplitNoPrimal
    @test Split(ReverseWithPrimal) == ReverseSplitWithPrimal
    @test Split(ReverseSplitNoPrimal) == ReverseSplitNoPrimal
    @test Split(ReverseSplitWithPrimal) == ReverseSplitWithPrimal

    @test Split(set_runtime_activity(Reverse)) == set_runtime_activity(ReverseSplitNoPrimal)
    @test Split(set_err_if_func_written(Reverse)) == set_err_if_func_written(ReverseSplitNoPrimal)
    @test Split(set_abi(Reverse, InlineABI)) == set_abi(ReverseSplitNoPrimal, InlineABI)

    @test Split(Reverse, Val(:ReturnShadow), Val(:Width), Val(:ModifiedBetween), Val(:ShadowInit)) == ReverseModeSplit{false,:ReturnShadow,false,false,:Width,:ModifiedBetween,EnzymeCore.DefaultABI,false,false,:ShadowInit}()

    @test Combined(Reverse) == Reverse
    @test Combined(ReverseWithPrimal) == ReverseWithPrimal
    @test Combined(ReverseSplitNoPrimal) == Reverse
    @test Combined(ReverseSplitWithPrimal) == ReverseWithPrimal

    @test Combined(set_runtime_activity(ReverseSplitNoPrimal)) == set_runtime_activity(Reverse)
    @test Combined(set_err_if_func_written(ReverseSplitNoPrimal)) == set_err_if_func_written(Reverse)
    @test Combined(set_abi(ReverseSplitNoPrimal, InlineABI)) == set_abi(Reverse, InlineABI)
end
