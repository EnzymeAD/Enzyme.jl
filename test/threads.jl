using Enzyme
using Test

# Enzyme.API.printall!(true)

# @testset "Threads $(Threads.nthreads())" begin
#     function tasktest(M, x)
#         xr = Ref(x)
#         task = Threads.@spawn begin
#             @inbounds M[1] = xr[]
#         end
#         @inbounds M[2] = x
#         wait(task)
#         nothing
#     end

#     R = Float64[0., 0.]
#     dR = Float64[2., 3.]

#     @test 5.0 ≈ Enzyme.autodiff(tasktest, Duplicated(R, dR), Active(2.0))[1]
#     @test Float64[2.0, 2.0] ≈ R
#     @test Float64[0.0, 0.0] ≈ dR
    
#     Enzyme.fwddiff(tasktest, Duplicated(R, dR), Duplicated(2.0, 1.0))
#     @test Float64[1.0, 1.0] ≈ dR

#     function tasktest2(M, x)
#         task = Threads.@spawn begin
#            return
#         end
#         Base.wait(task)
#         nothing
#     end
#     # The empty return previously resulted in an illegal instruction error
#     @test 0.0 ≈ Enzyme.autodiff(tasktest2, Duplicated(R, dR), Active(2.0))[1]
#     @test () === Enzyme.fwddiff(tasktest, Duplicated(R, dR), Duplicated(2.0, 1.0))
# end

@testset "Advanced Threads $(Threads.nthreads())" begin
    function foo(y)
        Threads.@threads for i in 1:3
            y[i] *= 2
        end
        nothing
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    Enzyme.autodiff(foo, Duplicated(x, dx))
    @test 2.0 ≈ x[1]
    @test 4.0 ≈ x[2]
    @test 6.0 ≈ x[3]
    @test 2.0 ≈ dx[1]
    @test 2.0 ≈ dx[2]
    @test 2.0 ≈ dx[3]
end
