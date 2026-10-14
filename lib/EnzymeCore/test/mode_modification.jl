using EnzymeCore
using EnzymeCore: InlineABI, ReverseMode, ReverseModeSplit, Split, Combined, set_runtime_activity, set_err_if_func_written, clear_err_if_func_written, set_strong_zero, set_abi, set_atomic_add, atomic_add, WithPrimal, NoPrimal
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

@testset "Atomic accumulation setting" begin
    @test atomic_add(Reverse) === nothing
    @test atomic_add(ReverseSplitNoPrimal) === nothing
    for setting in (true, false, nothing)
        mode = set_atomic_add(Reverse, setting)
        @test atomic_add(mode) === setting
        @test atomic_add(typeof(mode)) === setting
        @test atomic_add(set_runtime_activity(mode)) === setting
        @test atomic_add(set_strong_zero(mode)) === setting
        @test atomic_add(set_err_if_func_written(mode)) === setting
        @test atomic_add(set_abi(mode, InlineABI)) === setting
        @test atomic_add(WithPrimal(mode)) === setting
        @test atomic_add(Split(mode)) === setting
        @test Combined(Split(mode)) == mode

        split = set_atomic_add(ReverseSplitWithPrimal, setting)
        @test atomic_add(split) === setting
        @test atomic_add(NoPrimal(split)) === setting
        @test atomic_add(clear_err_if_func_written(set_err_if_func_written(split))) === setting
        @test atomic_add(Combined(split)) === setting
    end
    # Modes spelled without the setting, as code written before it existed does, leave it to Enzyme
    @test ReverseMode{false, false, false, EnzymeCore.DefaultABI, false, false}() === Reverse
    @test ReverseModeSplit{false, true, false, false, 0, true, EnzymeCore.DefaultABI, false, false, false}() === ReverseSplitNoPrimal
end
