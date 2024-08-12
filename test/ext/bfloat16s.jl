using Enzyme
using Test
using BFloat16s

msum(x) = sum(x)

@test Enzyme.gradient(Reverse, msum, ones(BFloat16, 10)) ≈ ones(BFloat16, 10)

@test Enzyme.gradient(Forward, msum, ones(BFloat16, 10)) ≈ ones(BFloat16, 10)

@test_broken Enzyme.gradient(Reverse, sum, ones(BFloat16, 10)) ≈ ones(BFloat16, 10)

@test_broken Enzyme.gradient(Forward, sum, ones(BFloat16, 10)) ≈ ones(BFloat16, 10)
