using EnzymeCore
using EnzymeCore: InlineABI, ReverseModeSplit, Split, Combined, set_runtime_activity, set_err_if_func_written, set_abi, set_strong_zero, forward_counterpart, reverse_counterpart
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

@testset "Forward / reverse counterparts" begin
    @test forward_counterpart(Reverse) == Forward
    @test forward_counterpart(ReverseWithPrimal) == ForwardWithPrimal
    @test forward_counterpart(ReverseHolomorphic) == Forward
    @test forward_counterpart(ReverseSplitNoPrimal) == Forward
    @test forward_counterpart(ReverseSplitWithPrimal) == ForwardWithPrimal
    @test forward_counterpart(Forward) == Forward
    @test forward_counterpart(ForwardWithPrimal) == ForwardWithPrimal

    @test forward_counterpart(set_runtime_activity(Reverse)) == set_runtime_activity(Forward)
    @test forward_counterpart(set_err_if_func_written(Reverse)) == set_err_if_func_written(Forward)
    @test forward_counterpart(set_strong_zero(Reverse)) == set_strong_zero(Forward)
    @test forward_counterpart(set_abi(Reverse, InlineABI)) == set_abi(Forward, InlineABI)
    @test forward_counterpart(set_runtime_activity(ReverseSplitNoPrimal)) == set_runtime_activity(Forward)

    @test reverse_counterpart(Forward) == Reverse
    @test reverse_counterpart(ForwardWithPrimal) == ReverseWithPrimal
    @test reverse_counterpart(Reverse) == Reverse
    @test reverse_counterpart(ReverseWithPrimal) == ReverseWithPrimal
    @test reverse_counterpart(ReverseSplitNoPrimal) == ReverseSplitNoPrimal
    @test reverse_counterpart(ReverseSplitWithPrimal) == ReverseSplitWithPrimal

    @test reverse_counterpart(set_runtime_activity(Forward)) == set_runtime_activity(Reverse)
    @test reverse_counterpart(set_err_if_func_written(Forward)) == set_err_if_func_written(Reverse)
    @test reverse_counterpart(set_strong_zero(Forward)) == set_strong_zero(Reverse)
    @test reverse_counterpart(set_abi(Forward, InlineABI)) == set_abi(Reverse, InlineABI)
end
