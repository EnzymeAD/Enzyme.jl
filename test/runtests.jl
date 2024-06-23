using Enzyme
using Test
using LinearAlgebra

using Enzyme_jll


Enzyme.API.printall!(true)


@noinline function mygetindex(R, len)
  out = Vector{Float64}(undef, 3)
  oi = 1
  i = 1
  v = @inbounds Core.arrayref(false, R, 1)
  while true
    Core.arrayset(false, out, v, i)
    # unsafe_store!(out, v, oi)
    i += 1
    if i >= unsafe_load(len)
	break
    end
  end
  out
end

@inline function mydiag(R)
   len = Base.reinterpret(Ptr{Int}, Libc.malloc(8))
   unsafe_store!(len, 3)
   mygetindex(R, len)
end

function whocallsmorethan30args(R)
   # return unsafe_load(mydiag(R))
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
