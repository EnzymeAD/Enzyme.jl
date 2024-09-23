using Enzyme
using Test
using BFloat16s

@test_broken Enzyme.gradient(Reverse, sum, ones(BFloat16, 10))[1] ≈ ones(BFloat16, 10)

@test_broken Enzyme.gradient(Forward, sum, ones(BFloat16, 10))[1] ≈ ones(BFloat16, 10)
