using Enzyme
using Test
using BFloat16s

using Enzyme: gradient

@testset "BFloat16s ext" begin
    @test_broken gradient(Reverse, sum, ones(BFloat16, 10)) ≈ ones(BFloat16, 10)
    @test_broken gradient(Forward, sum, ones(BFloat16, 10)) ≈ ones(BFloat16, 10)
end
