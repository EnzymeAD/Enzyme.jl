using EnzymeCore
using EnzymeCore: InlineABI, ReverseModeSplit, split_mode, combined_mode, set_runtime_activity, set_err_if_func_written, set_abi
using Test

@testset "Split / unsplit mode" begin
    @test split_mode(Reverse) == ReverseSplitNoPrimal
    @test split_mode(ReverseWithPrimal) == ReverseSplitWithPrimal
    @test split_mode(ReverseSplitNoPrimal) == ReverseSplitNoPrimal
    @test split_mode(ReverseSplitWithPrimal) == ReverseSplitWithPrimal

    @test split_mode(set_runtime_activity(Reverse)) == set_runtime_activity(ReverseSplitNoPrimal)
    @test split_mode(set_err_if_func_written(Reverse)) == set_err_if_func_written(ReverseSplitNoPrimal)
    @test split_mode(set_abi(Reverse, InlineABI)) == set_abi(ReverseSplitNoPrimal, InlineABI)

    @test split_mode(Reverse, Val(:ReturnShadow), Val(:Width), Val(:ModifiedBetween), Val(:ShadowInit)) == ReverseModeSplit{false,:ReturnShadow,false,:Width,:ModifiedBetween,EnzymeCore.DefaultABI,false,false,:ShadowInit}()

    @test combined_mode(Reverse) == Reverse
    @test combined_mode(ReverseWithPrimal) == ReverseWithPrimal
    @test combined_mode(ReverseSplitNoPrimal) == Reverse
    @test combined_mode(ReverseSplitWithPrimal) == ReverseWithPrimal

    @test combined_mode(set_runtime_activity(ReverseSplitNoPrimal)) == set_runtime_activity(Reverse)
    @test combined_mode(set_err_if_func_written(ReverseSplitNoPrimal)) == set_err_if_func_written(Reverse)
    @test combined_mode(set_abi(ReverseSplitNoPrimal, InlineABI)) == set_abi(Reverse, InlineABI)
end
