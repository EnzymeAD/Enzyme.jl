using EnzymeCore
using EnzymeCore: InlineABI, ReverseModeSplit, Split, Combined, set_runtime_activity, set_err_if_func_written, set_abi, set_strong_zero, as_forward, as_reverse
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

@testset "as_forward / as_reverse" begin
    @test as_forward(Reverse) == Forward
    @test as_forward(ReverseWithPrimal) == ForwardWithPrimal
    @test as_forward(ReverseHolomorphic) == Forward
    @test as_forward(ReverseSplitNoPrimal) == Forward
    @test as_forward(ReverseSplitWithPrimal) == ForwardWithPrimal
    @test as_forward(Forward) == Forward
    @test as_forward(ForwardWithPrimal) == ForwardWithPrimal

    @test as_forward(set_runtime_activity(Reverse)) == set_runtime_activity(Forward)
    @test as_forward(set_err_if_func_written(Reverse)) == set_err_if_func_written(Forward)
    @test as_forward(set_strong_zero(Reverse)) == set_strong_zero(Forward)
    @test as_forward(set_abi(Reverse, InlineABI)) == set_abi(Forward, InlineABI)
    @test as_forward(set_runtime_activity(ReverseSplitNoPrimal)) == set_runtime_activity(Forward)

    @test as_reverse(Forward) == Reverse
    @test as_reverse(ForwardWithPrimal) == ReverseWithPrimal
    @test as_reverse(Reverse) == Reverse
    @test as_reverse(ReverseWithPrimal) == ReverseWithPrimal
    @test as_reverse(ReverseSplitNoPrimal) == ReverseSplitNoPrimal
    @test as_reverse(ReverseSplitWithPrimal) == ReverseSplitWithPrimal

    @test as_reverse(set_runtime_activity(Forward)) == set_runtime_activity(Reverse)
    @test as_reverse(set_err_if_func_written(Forward)) == set_err_if_func_written(Reverse)
    @test as_reverse(set_strong_zero(Forward)) == set_strong_zero(Reverse)
    @test as_reverse(set_abi(Forward, InlineABI)) == set_abi(Reverse, InlineABI)
end
