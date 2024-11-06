# # work around https://github.com/JuliaLang/Pkg.jl/issues/1585
# using Pkg
# Pkg.develop(PackageSpec(; path=joinpath(dirname(@__DIR__), "lib", "EnzymeTestUtils")))

using GPUCompiler
using Enzyme
using Test
using Statistics
using LinearAlgebra
using InlineStrings

using Enzyme_jll
@info "Testing against" Enzyme_jll.libEnzyme

@testset "generic" begin
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
