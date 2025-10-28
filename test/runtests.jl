import Enzyme
import Enzyme_jll
using ParallelTestRunner: addworker, filter_tests!, find_tests, parse_args, runtests

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)
# Add threads tests to be run with multiple Julia threads (will be configured in
# `test_worker`).
testsuite["threads/2"] = :(include($(joinpath(@__DIR__, "threads.jl"))))
# Exclude integration tests, they're handled differently (they each run in their
# own environment)
for (k, _) in testsuite
    startswith(k, "integration/") && delete!(testsuite, k)
end

# Parse arguments
args = parse_args(ARGS)

if filter_tests!(testsuite, args)
    # Skip GPU-specific tests by default.
    delete!(testsuite, "metal")
    delete!(testsuite, "cuda")
    delete!(testsuite, "amdgpu")

    # Skipped until https://github.com/EnzymeAD/Enzyme.jl/issues/2620 is fixed.
    if Sys.iswindows()
        delete!(testsuite, "ext/specialfunctions")
    end
end

function test_worker(name)
    if name == "threads/2"
        # Run the `threads/2` testset, with multiple threads.
        return addworker(; exeflags = ["--threads=2"])
    end
end

const init_code = quote
    using Test
    using FiniteDifferences
    using Enzyme

    function isapproxfn(fn, args...; kwargs...)
        return isapprox(args...; kwargs...)
    end

    # Test against FiniteDifferences
    function test_scalar(f, x; rtol = 1.0e-9, atol = 1.0e-9, fdm = central_fdm(5, 1), kwargs...)
        ∂x, = autodiff(ReverseHolomorphic, f, Active, Active(x))[1]

        finite_diff = if typeof(x) <: Complex
            RT = typeof(x).parameters[1]
            (fdm(dx -> f(x + dx), RT(0)) - im * fdm(dy -> f(x + im * dy), RT(0))) / 2
        else
            fdm(f, x)
        end

        @test isapproxfn((Enzyme.Reverse, f), ∂x, finite_diff; rtol = rtol, atol = atol, kwargs...)

        if typeof(x) <: Integer
            x = Float64(x)
        end

        if typeof(x) <: Complex
            ∂re, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
            ∂im, = autodiff(Forward, f, Duplicated(x, im * one(typeof(x))))
            ∂x = (∂re - im * ∂im) / 2
        else
            ∂x, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
        end

        return @test isapproxfn((Enzyme.Reverse, f), ∂x, finite_diff; rtol = rtol, atol = atol, kwargs...)
    end
end

@info "Testing against" Enzyme_jll.libEnzyme
runtests(Enzyme, args; testsuite, init_code, test_worker)
