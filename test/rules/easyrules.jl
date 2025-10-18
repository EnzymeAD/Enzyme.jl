module EasyRules

using Enzyme
using Enzyme: EnzymeRules
using Test

import .EnzymeRules: forward, Annotation, has_frule_from_sig, FwdConfig

function mysin(x)
    return sin(x)
end

const calls = Ref(0)

function myexp(x)
    global calls
    calls[] += 1
    return exp(x)
end

EnzymeRules.@easy_scalar_rule(
    mysin(x1::Float32),
    @setup(),
    (myexp(x1),)
)

@testset "Forward EasyRule mysin" begin
    calls[] = 0

    @test autodiff(Forward, mysin, Duplicated(2.0f0, 1.2f0))[1] ≈ 1.2f0 * exp(2.0f0)
    @test calls[] == 1

    @test autodiff(Forward, mysin, Duplicated(2.0, 1.2))[1] ≈ cos(2.0f0)
    @test calls[] == 1

    @test autodiff(Forward, mysin, BatchDuplicated(2.0f0, (1.2f0, 1.4f0)))[1] ≈ (1.2f0 * exp(2.0f0), 1.4f0 * exp(2.0f0))
    @test calls[] == 2

    @test autodiff(ForwardWithPrimal, mysin, Duplicated(2.0f0, 1.2f0))[1] ≈ (mysin(2.0f0), 1.2f0 * exp(2.0f0))
    @test calls[] == 3

    @test autodiff(ForwardWithPrimal, mysin, BatchDuplicated(2.0f0, (1.2f0, 1.4f0)))[1] ≈ (mysin(2.0f0), (1.2f0 * exp(2.0f0), 1.4f0 * exp(2.0f0)))
    @test calls[] == 4
end

function mymul(x, y)
    return sin(x)
end

EnzymeRules.@easy_scalar_rule(
    mymul(x::Float64, y::Float64),
    (myexp(x1), Enzyme.Const)
)

@testset "Forward EasyRule mymul" begin
    calls[] = 0

    @test autodiff(Forward, mymul, Duplicated(2.0, 1.2), Duplicated(3.1, 2.7))[1] ≈ 1.2 * exp(2.0)
    @test calls[] == 1

    @test autodiff(Forward, mymul, BatchDuplicated(2.0, (1.2, 1.4)), BatchDuplicated(2.0, (1.2, 1.4)), BatchDuplicated(3.1, (2.7, 2.9)))[1] ≈ (1.2 * exp(2.0), 1.4 * exp(2.0))
    @test calls[] == 2

end

function mytup(x, y)
    return (sin(x), cos(y))
end

EnzymeRules.@easy_scalar_rule(
    mytup(x::Float64, y::Float64),
    (myexp(x), Enzyme.Const),
    (Enzyme.Const, 0.123456),
)

@testset "Forward EasyRule mytup" begin
    calls[] = 0

    @test autodiff(Forward, mytup, Duplicated(2.0, 1.2), Duplicated(3.1, 2.7))[1] ≈ (1.2 * exp(2.0), 2.7 * 0.123456)
end

end # module EasyRules
