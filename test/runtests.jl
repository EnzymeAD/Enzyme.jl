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
using Aqua
using Statistics
using LinearAlgebra

import Enzyme: API

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

function test_matrix_to_number(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    dx_fd = map(eachindex(x)) do i
        fdm(x[i]) do xi
            x2 = copy(x)
            x2[i] = xi
            f(x2)
        end
    end

    dx = zero(x)
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    @test isapprox(reshape(dx, length(dx)), dx_fd; rtol=rtol, atol=atol, kwargs...)

    dx_fwd = map(eachindex(x)) do i
        dx = zero(x)
        dx[i] = 1
        ∂x = autodiff(Forward, f, Duplicated(x, dx))
        isempty(∂x) ? zero(eltype(dx)) : ∂x[1]
    end
    @test isapprox(dx_fwd, dx_fd; rtol=rtol, atol=atol, kwargs...)
end

# Aqua.test_all(Enzyme, unbound_args=false, piracy=false)
# 
# include("abi.jl")
# include("typetree.jl")

@static if Enzyme.EnzymeRules.issupported()
    include("rules.jl")
    include("rrules.jl")
    include("kwrules.jl")
    include("kwrrules.jl")
    @static if VERSION ≥ v"1.9-"
        # XXX invalidation does not work on Julia 1.8
        include("ruleinvalidation.jl")
    end
end


@testset "Null init union" begin
    @noinline function unionret(itr, cond)
        if cond
            return Base._InitialValue()
        else
            return itr[1]
        end
    end

    function fwdunion(data::Vector{Float64})::Real
        unionret(data, false)
    end

    data = ones(Float64, 500)
    ddata = zeros(Float64, 500)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((false, true))), Const{typeof(fwdunion)}, Active, Duplicated{Vector{Float64}})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(data, ddata))

	function firstimpl(itr)
		v = firstfold(itr)
		@assert !(v isa Base._InitialValue)
		return v
	end

	function firstfold(itr)
		op, itr = Base._xfadjoint(Base.BottomRF(Base.add_sum), Base.Generator(Base.identity, itr))
		y = iterate(itr)
		init = Base._InitialValue()
		y === nothing && return init
		v = op(init, y[1])
		return v
	end

	function smallrf(weights::Vector{Float64}, data::Vector{Float64})::Float64
		itr1 = (weight for (weight, mean) in zip(weights, weights))

		itr2 = (firstimpl(itr1) for x in data)

		firstimpl(itr2)
	end

	data = ones(Float64, 1)

	weights = [0.2]
	dweights = [0.0]
    # Technically this test doesn't need runtimeactivity since the closure combo of active itr1 and const data
    # doesn't use any of the const data values, but now that we error for activity confusion, we need to
    # mark runtimeActivity to let this pass
    Enzyme.API.runtimeActivity!(true)
	Enzyme.autodiff(Enzyme.Reverse, smallrf, Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    @test dweights[1] ≈ 1.

    function invokesum(weights::Vector{Float64}, data::Vector{Float64})::Float64
        sum(
            sum(
                weight
                for (weight, mean) in zip(weights, weights)
            )
            for x in data
        )
    end

    data = ones(Float64, 20)

    weights = [0.2, 0.8]
    dweights = [0.0, 0.0]

    Enzyme.autodiff(Enzyme.Reverse, invokesum, Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    Enzyme.API.runtimeActivity!(false)
    @test dweights[1] ≈ 20.
    @test dweights[2] ≈ 20.
end
