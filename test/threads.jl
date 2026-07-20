using Enzyme
using Test

@testset "Threads $(Threads.nthreads())" begin
    function tasktest(M, x)
        xr = Ref(x)
        task = Threads.@spawn begin
            @inbounds M[1] = xr[]
        end
        @inbounds M[2] = x
        wait(task)
        nothing
    end

    R = Float64[0., 0.]
    dR = Float64[2., 3.]

    # We define the local variable `autodiff` as a `Ref` wrapper only to be able to capture
    # and test the warning emitted by the `@generated` function `Enzyme.autodiff`.
    autodiff = Ref{Any}(Enzyme.autodiff)
    @test 5.0 ≈ @test_warn r"active variables passed by value to jl_new_task are not yet supported" autodiff[](Reverse, tasktest, Duplicated(R, dR), Active(2.0))[1][2]
    @test Float64[2.0, 2.0] ≈ R
    @test Float64[0.0, 0.0] ≈ dR
    
    Enzyme.autodiff(Forward, tasktest, Duplicated(R, dR), Duplicated(2.0, 1.0))
    @test Float64[1.0, 1.0] ≈ dR
end
