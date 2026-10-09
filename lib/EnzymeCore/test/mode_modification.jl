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

@testset "Strong zero" begin
    using EnzymeCore: set_strong_zero, clear_strong_zero, strong_zero
    using EnzymeCore.EnzymeRules: FwdConfig, RevConfig

    for mode in (Forward, Reverse)
        @test strong_zero(set_strong_zero(mode))
        @test !strong_zero(clear_strong_zero(set_strong_zero(mode)))
        @test set_strong_zero(mode, true) === set_strong_zero(mode)
        @test set_strong_zero(set_strong_zero(mode), false) === mode
        @test set_strong_zero(mode, set_strong_zero(mode)) === set_strong_zero(mode)
    end

    # Take the setting from a rule config, independently of runtime activity
    for (config_sz, config_rt) in ((FwdConfig{false, true, 1, false, true}(),
                                    FwdConfig{false, true, 1, true, false}()),
                                   (RevConfig{false, true, 1, (), false, true}(),
                                    RevConfig{false, true, 1, (), true, false}()))
        for mode in (Forward, Reverse)
            @test set_strong_zero(mode, config_sz) === set_strong_zero(mode)
            @test set_strong_zero(mode, config_rt) === mode
            @test set_runtime_activity(mode, config_rt) === set_runtime_activity(mode)
            @test set_runtime_activity(mode, config_sz) === mode
        end
    end
end
