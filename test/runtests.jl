using Enzyme
using Test
using LinearAlgebra

using Enzyme_jll


Enzyme.API.printall!(true)


@noinline function mygetindex(R, len)
  # out = Vector{Float64}(undef, 3)
  out = Base.llvmcall(("
declare i8* @malloc(i64)

define i64 @f() {
  %r = call i8* @malloc(i64 24)
  %p = ptrtoint i8* %r to i64
  ret i64 %p
}", "f"), Ptr{Float64}, Tuple{})
  oi = 1
  i = 1
  while i < unsafe_load(len)
    v = Core.arrayref(false, R, i)
    # Core.arrayset(false, out, v, oi)
    unsafe_store!(out, v, oi)
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
   return unsafe_load(mydiag(R))
   # return @inbounds mydiag(R)[1]    
end

@testset "generic" begin
    
    R = zeros(3,3)    
    dR = zeros(3,3)

	autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
	@show R
	@show dR
	@test 1.0 ≈ dR[1, 1]
end
