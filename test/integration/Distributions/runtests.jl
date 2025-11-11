module DistributionsIntegrationTests

using Distributions
using Enzyme: Enzyme
using FillArrays: Fill
using FiniteDifferences: FiniteDifferences
using LinearAlgebra: Diagonal, Hermitian, I, Symmetric, diag, cholesky
using PDMats: PDMat
using Random: randn
using StableRNGs: StableRNG
using Test: @test, @test_broken, @testset

Enzyme.API.printall!(true)
Enzyme.API.printactivity!(true)

rng = StableRNG(23)

"""
Enum type for choosing Enzyme autodiff modes.
"""
@enum ModeSelector Neither Forward Reverse Both

"""
Type for specifying a test case for `Enzyme.gradient`.
The test will check the accuracy of the gradient of `func` at `value` against `finitediff`,
with both forward and reverse mode autodiff. `name` is for diagnostic printing.
`runtime_activity` and `broken` are for specifying whether to use
`Enzyme.set_runtime_activity` or not and whether the test is broken. Both of them taken
values `Neither`, `Forward`, `Reverse` or `Both`, to specify which mode to apply the setting
to. `splat` is for specifying whether to call the function as `func(value)` or as
`func(value...)`.
A constructor is also provided for giving a `Distribution` instead of a function, in which
case the function is `x -> logpdf(distribution, x)`.
Default values are `name=nothing` or `name=string(nameof(typeof(distribution)))`,
`runtime_activity=Neither`, `broken=Neither` and `splat=false`.
"""
struct TestCase
    func::Function
    value
    name::Union{String, Nothing}
    runtime_activity::ModeSelector
    broken::ModeSelector
    splat::Bool
end

# Turn a distribution into a call to logpdf.
function TestCase(d::Distribution, value, name, runtime_activity, broken, splat)
    TestCase(x -> logpdf(d, x), value, name, runtime_activity, broken, splat)
end

# Defaults for name, runtime_activity and broken.
function TestCase(
    f, value;
    name=nothing, runtime_activity=Neither, broken=Neither, splat=false
)
    return TestCase(f, value, name, runtime_activity, broken, splat)
end

# Default name for a Distribution.
function TestCase(
    d::Distribution, value;
    name=string(nameof(typeof(d))), runtime_activity=Neither, broken=Neither, splat=false
)
    return TestCase(d, value, name, runtime_activity, broken, splat)
end

"""
Test Enzyme.gradient, both Forward and Reverse mode, against FiniteDifferences.grad.
"""
function test_grad(case::TestCase; rtol=1e-6, atol=1e-6)
    @nospecialize
    f = case.func
    # We'll call the function as f(x...), so wrap in a singleton tuple if need be.
    x = case.splat ? case.value : (case.value,)
    finitediff = collect(
        FiniteDifferences.grad(FiniteDifferences.central_fdm(4, 1), f, x...)[1]
    )

    f_mode = if (case.runtime_activity === Both || case.runtime_activity === Forward)
        Enzyme.set_runtime_activity(Enzyme.Forward)
    else
        Enzyme.Forward
    end
    r_mode = if (case.runtime_activity === Both || case.runtime_activity === Reverse)
        Enzyme.set_runtime_activity(Enzyme.Reverse)
    else
        Enzyme.Reverse
    end

    if case.broken === Both || case.broken === Forward
        @test_broken(
            collect(Enzyme.gradient(f_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    else
        @test(
            collect(Enzyme.gradient(f_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    end

    if case.broken === Both || case.broken === Reverse
        @test_broken(
            collect(Enzyme.gradient(r_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    else
        @test(
            collect(Enzyme.gradient(r_mode, Enzyme.Const(f), x...)[1]) ≈ finitediff,
            rtol = rtol,
            atol = atol,
        )
    end
    return nothing
end

_sym(A) = A'A
_pdmat(A) = PDMat(_sym(A) + 5I)

@testset "Distributions integration tests" begin
    test_cases = TestCase[
        TestCase(
            function (X, v)
                # LKJCholesky distributes over the Cholesky factorisation of correlation
                # matrices, so the argument to `logpdf` must be such a matrix.
                S = X'X
                Λ = Diagonal(map(inv ∘ sqrt, diag(S)))
                C = cholesky(Symmetric(Λ * S * Λ))
                return logpdf(LKJCholesky(2, v), C)
            end,
            (randn(rng, 2, 2), 1.1);
            name="LKJCholesky", splat=true
        ),
    ]

    @testset "$(case.name)" for case in test_cases
        test_grad(case)
    end
end

end