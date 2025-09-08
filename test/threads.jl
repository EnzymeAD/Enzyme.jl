using Enzyme
using Test

Enzyme.API.printall!(true)
Enzyme.Compiler.DumpPostOpt[] = true

@testset "Advanced, Active-var Threads $(Threads.nthreads())" begin
    function f_multi(out, in)
        Threads.@threads for idx in 1:length(out)
            out[idx] = in
        end
        return nothing
    end

    out = [1.0, 2.0]
    dout = [1.0, 1.0]
    res = autodiff(Reverse, f_multi, Const, Duplicated(out, dout), Active(2.0))
    @test res[1][2] ≈ 2.0
end

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

# TODO on 1.8 having `Inactive threads` after `UndefVar` in the main `runtest.jl` leads to a GC verification bug
@testset "Inactive threads" begin
    function thr_inactive(x, y)
        if x
            Threads.@threads for N in 1:5:20
                println("The number of this iteration is $N")
            end
        end
        y
    end
    @test 1.0 ≈ autodiff(Reverse, thr_inactive, Const(false), Active(2.14))[1][2]
    @test 1.0 ≈ autodiff(Forward, thr_inactive, Const(false), Duplicated(2.14, 1.0))[1]
    
    @test 1.0 ≈ autodiff(Reverse, thr_inactive, Const(true), Active(2.14))[1][2]
    @test 1.0 ≈ autodiff(Forward, thr_inactive, Const(true), Duplicated(2.14, 1.0))[1]
end

@testset "Batched Forward" begin
    function f2(du)
        Threads.@threads for i in eachindex(du)
            du[i] *= 2
        end
        return nothing
    end
    du = zeros(2)
    ty = ntuple(i -> (i + 1) * ones(2), Val(4))
    y_and_ty = BatchDuplicated(du, ty)
    autodiff(Forward, f2, Const, y_and_ty)
    @test ty[1] ≈ [4.0, 4.0]
    @test ty[2] ≈ [6.0, 6.0]
    @test ty[3] ≈ [8.0, 8.0]
    @test ty[4] ≈ [10.0, 10.0]
end
