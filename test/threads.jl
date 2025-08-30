using Enzyme
using Test

@testset "Closure-less threads $(Threads.nthreads())" begin
    function bf(i, x)
      x[i] *= x[i]
      nothing
    end

    function psquare0(x)
      Enzyme.pmap(bf, 10, x)
    end

    xs = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dxs = ones(10)

    Enzyme.autodiff(Reverse, psquare0, Duplicated(xs, dxs))
    @test Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 20] ≈ dxs 

    function psquare1(x)
      Enzyme.@parallel x for i = 1:10
        @inbounds x[i] *= x[i]
      end
    end
    
    xs = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dxs = ones(10)

    Enzyme.autodiff(Reverse, psquare1, Duplicated(xs, dxs))
    @test Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 20] ≈ dxs 

    function psquare2(x, y)
      Enzyme.@parallel x y for i = 1:10
        @inbounds x[i] *= y[i]
      end
    end
    
    xs = Float64[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dxs = ones(10)
    
    Enzyme.autodiff(Reverse, psquare2, Duplicated(xs, dxs), Duplicated(xs, dxs))
    @test Float64[2, 4, 6, 8, 10, 12, 14, 16, 18, 20] ≈ dxs 
end
