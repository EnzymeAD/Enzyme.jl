using Enzyme
using Test
using LinearAlgebra

using Enzyme_jll


Enzyme.API.printall!(true)

@noinline function mydiag(R)
   @inbounds R[1:4:9]
end

    function whocallsmorethan30args(R)
        return @inbounds mydiag(R)[1]    
    end

@testset "generic" begin
    
    R = zeros(3,3)    
    dR = zeros(3,3)

	autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
	@show R
	@show dR
	@test 1.0 ≈ dR[1, 1]
end
