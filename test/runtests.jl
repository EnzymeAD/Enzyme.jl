# HACK: work around Pkg.jl#2500
if VERSION < v"1.8-"
test_project = Base.active_project()
preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
if isfile(preferences_file) && !isfile(test_preferences_file)
    cp(preferences_file, test_preferences_file)
end
end

using Enzyme
using Test
using LinearAlgebra

using Enzyme_jll


Enzyme.API.printall!(true)

    function whocallsmorethan30args(R)
        temp = diag(R)     
         R_inv = [temp[1] 0. 0. 0. 0. 0.; 
             0. temp[2] 0. 0. 0. 0.; 
         ]
    
        return sum(R_inv)
    end

@testset "generic" begin
    
    R = zeros(6,6)    
    dR = zeros(6, 6)

    @static if VERSION ≥ v"1.10-"
        @test_broken autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    else
        autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
	@show R
	@show dR
    	@test 1.0 ≈ dR[1, 1]
    	@test 1.0 ≈ dR[2, 2]
    	#@test 1.0 ≈ dR[3, 3]
    	#@test 1.0 ≈ dR[4, 4]
    	#@test 1.0 ≈ dR[5, 5]
    	#@test 0.0 ≈ dR[6, 6]
    end
end
