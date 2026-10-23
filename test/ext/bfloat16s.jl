using Enzyme
using Test
using BFloat16s

@testset "bfloat16s" begin
    @static if isdefined(Core, :BFloat16) && Core.BFloat16 === BFloat16
        @test Enzyme.gradient(Reverse, sum, ones(BFloat16, 10))[1] ≈ ones(BFloat16, 10)
    else
        @test_broken Enzyme.gradient(Reverse, sum, ones(BFloat16, 10))[1] ≈
            ones(BFloat16, 10)
    end
    @test_broken Enzyme.gradient(Forward, sum, ones(BFloat16, 10))[1] ≈ ones(BFloat16, 10)
end

@static if isdefined(Core, :BFloat16) && Core.BFloat16 === BFloat16
    # https://github.com/EnzymeAD/Enzyme.jl/issues/3430
    bf16_conv(x) = sum(abs2, BFloat16.(x))
    @testset "bfloat16 in otherwise Float32 IR (#3430)" begin
        x = Float32[1.0, 2.0, 3.0]
        dx = zero(x)
        Enzyme.autodiff(Reverse, bf16_conv, Active, Duplicated(x, dx))
        @test dx == 2 .* Float32.(BFloat16.(x))
    end

    # https://github.com/EnzymeAD/Enzyme.jl/issues/3762
    @noinline bf16_pair(x) = (x[1] * x[2], x[2] + x[1])
    bf16_pair_sum(x) = (t = bf16_pair(x); t[1] + t[2])
    @testset "bfloat16 aggregate return (#3762)" begin
        x = BFloat16[1, 2, 3, 4]
        dx = zeros(BFloat16, 4)
        Enzyme.autodiff(Reverse, bf16_pair_sum, Active, Duplicated(x, dx))
        @test dx == BFloat16[3, 2, 0, 0]

        dx = BFloat16[1, 0, 0, 0]
        @test Enzyme.autodiff(Forward, bf16_pair_sum, Duplicated(x, dx))[1] == BFloat16(3)
    end
end
