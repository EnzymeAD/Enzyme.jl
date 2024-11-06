# # work around https://github.com/JuliaLang/Pkg.jl/issues/1585
# using Pkg
# Pkg.develop(PackageSpec(; path=joinpath(dirname(@__DIR__), "lib", "EnzymeTestUtils")))

using GPUCompiler
using Enzyme
using Test
using FiniteDifferences
using Aqua
using Statistics
using LinearAlgebra
using InlineStrings

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

function isapproxfn(fn, args...; kwargs...)
    isapprox(args...; kwargs...)
end
# Test against FiniteDifferences
function test_scalar(f, x; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)
    ∂x, = autodiff(ReverseHolomorphic, f, Active, Active(x))[1]

    finite_diff = if typeof(x) <: Complex
      RT = typeof(x).parameters[1]
      (fdm(dx -> f(x+dx), RT(0)) - im * fdm(dy -> f(x+im*dy), RT(0)))/2
    else
      fdm(f, x)
    end

    @test isapproxfn((Enzyme.Reverse, f), ∂x, finite_diff; rtol=rtol, atol=atol, kwargs...)

    if typeof(x) <: Integer
        x = Float64(x)
    end

    if typeof(x) <: Complex
        ∂re, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
        ∂im, = autodiff(Forward, f, Duplicated(x, im*one(typeof(x))))
        ∂x = (∂re - im*∂im)/2
    else
        ∂x, = autodiff(Forward, f, Duplicated(x, one(typeof(x))))
    end

    @test isapproxfn((Enzyme.Reverse, f), ∂x, finite_diff; rtol=rtol, atol=atol, kwargs...)

end


genlatestsin(x)::Float64 = Base.invokelatest(sin, x)
function genlatestsinx(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    Base.invokelatest(sin, x)::Float64 + 1
end

function loadsin(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    sin(x)
end
function invsin(xp)
    xp = Base.invokelatest(convert, Vector{Float64}, xp)
    loadsin(xp)
end

@testset "generic" begin
    @test -0.4161468365471424 ≈ Enzyme.autodiff(Reverse, genlatestsin, Active, Active(2.0))[1][1]
    @test -0.4161468365471424 ≈ Enzyme.autodiff(Forward, genlatestsin, Duplicated(2.0, 1.0))[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(Reverse, genlatestsinx, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(Reverse, invsin, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

	function inactive_gen(x)
		n = 1
		for k in 1:2
			y = falses(n)
		end
		return x
	end
    @test 1.0 ≈ Enzyme.autodiff(Reverse, inactive_gen, Active, Active(1E4))[1][1]
	@test 1.0 ≈ Enzyme.autodiff(Forward, inactive_gen, Duplicated(1E4, 1.0))[1]

    function whocallsmorethan30args(R)
        temp = diag(R)     
         R_inv = [temp[1] 0. 0. 0. 0. 0.; 
             0. temp[2] 0. 0. 0. 0.; 
             0. 0. temp[3] 0. 0. 0.; 
             0. 0. 0. temp[4] 0. 0.; 
             0. 0. 0. 0. temp[5] 0.; 
         ]

        return sum(R_inv)
    end

    R = zeros(6,6)    
    dR = zeros(6, 6)

    @static if VERSION ≥ v"1.10-"
        @test_broken autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    else
        autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    	@test 1.0 ≈ dR[1, 1]
    	@test 1.0 ≈ dR[2, 2]
    	@test 1.0 ≈ dR[3, 3]
    	@test 1.0 ≈ dR[4, 4]
    	@test 1.0 ≈ dR[5, 5]
    	@test 0.0 ≈ dR[6, 6]
    end
end
