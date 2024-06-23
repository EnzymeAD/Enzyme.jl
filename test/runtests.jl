using Enzyme
using Test
using LinearAlgebra

using Enzyme_jll


Enzyme.API.printall!(true)

@noinline function mygetindex(R, len)
  out = Vector{Float64}(undef, 3)
  oi = 1
  i = 1
  while i < unsafe_load(len)
    @inbounds out[oi] = @inbounds R[i]
    oi+=1
    i += 4
  end
  out
end

@noinline function mydiag(R)
   len = Base.reinterpret(Ptr{Int}, Libc.malloc(8))
   unsafe_store!(len, 9)
   mygetindex(R, len)
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
