# HACK: work around Pkg.jl#2500
if VERSION < v"1.8-"
test_project = Base.active_project()
preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
if isfile(preferences_file) && !isfile(test_preferences_file)
    cp(preferences_file, test_preferences_file)
end
end

using GPUCompiler
using Enzyme
using Test
using FiniteDifferences
using ForwardDiff
using Statistics
using LinearAlgebra

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

# Test against FiniteDifferences
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    ∂x, = autodiff(Reverse, f, Active, Active(x))[1]
    if typeof(x) <: Complex
    else
      @test isapprox(∂x, fdm(f, x); rtol=rtol, atol=atol, kwargs...)
    end

    rm = ∂x
    if typeof(x) <: Integer
        x = Float64(x)
    end
    ∂x, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
    if typeof(x) <: Complex
      @test ∂x ≈ rm
    else
      @test isapprox(∂x, fdm(f, x); rtol=rtol, atol=atol, kwargs...)
    end
end

@testset "Simple tests" begin
    Enzyme.API.printall!(true)
    test_scalar(Base.sinc, 2.2)
end
