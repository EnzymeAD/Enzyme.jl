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
end
