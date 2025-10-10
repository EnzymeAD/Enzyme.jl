using ParallelTestRunner: runtests
using Enzyme_jll: Enzyme_jll

function testfilter(test)
    if test ∈ ("metal", "cuda", "amdgpu")
        return false
    end
    if Sys.iswindows() && test == "ext/specialfunctions"
        return false
    end
    return true
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
runtests(ARGS; testfilter, init_code)
