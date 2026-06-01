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

    function tasktest2(M, x)
        task = Threads.@spawn begin
           return
        end
        Base.wait(task)
        nothing
    end
    # The empty return previously resulted in an illegal instruction error
    @test 0.0 ≈ @test_warn r"active variables passed by value to jl_new_task are not yet supported" autodiff[](Reverse, tasktest2, Duplicated(R, dR), Active(2.0))[1][2]
    @test () === Enzyme.autodiff(Forward, tasktest, Duplicated(R, dR), Duplicated(2.0, 1.0))
end

@testset "Advanced Threads $(Threads.nthreads())" begin
    function foo(y)
        Threads.@threads for i in 1:3
            y[i] *= 2
        end
        nothing
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    Enzyme.autodiff(Reverse, foo, Duplicated(x, dx))
    @test 2.0 ≈ x[1]
    @test 4.0 ≈ x[2]
    @test 6.0 ≈ x[3]
    @test 2.0 ≈ dx[1]
    @test 2.0 ≈ dx[2]
    @test 2.0 ≈ dx[3]

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    Enzyme.autodiff(Forward, foo, Duplicated(x, dx))
    @test 2.0 ≈ x[1]
    @test 4.0 ≈ x[2]
    @test 6.0 ≈ x[3]
    @test 2.0 ≈ dx[1]
    @test 2.0 ≈ dx[2]
    @test 2.0 ≈ dx[3]
end

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

@testset "Task Rules" begin
    function wait_f(t)
        Base.wait(t)
        nothing
    end
    function _wait_f(t)
        Base._wait(t)
        nothing
    end
    function schedule_f(t)
        Base.schedule(t)
        nothing
    end
    function enq_work_f(t)
        Base.enq_work(t)
        nothing
    end

    t1 = Task(()->nothing)
    t2 = Task(()->nothing)
    t3 = Task(()->nothing)
    t4 = Task(()->nothing)
    
    # Pre-schedule tasks for wait so we don't deadlock
    Base.schedule(t1)
    Base.schedule(t2)
    Base.schedule(t3)
    Base.schedule(t4)
    Base.wait(t1)
    Base.wait(t2)
    Base.wait(t3)
    Base.wait(t4)
    
    @test Enzyme.autodiff(Reverse, wait_f, Const(t1)) === ()
    @test Enzyme.autodiff(Reverse, wait_f, Duplicated(t1, t2)) === ()
    @test Enzyme.autodiff(Reverse, wait_f, BatchDuplicated(t1, (t2, t3))) === ()

    @test Enzyme.autodiff(Forward, wait_f, Const(t1)) === ()
    @test Enzyme.autodiff(Forward, wait_f, Duplicated(t1, t2)) === ()
    @test Enzyme.autodiff(Forward, wait_f, BatchDuplicated(t1, (t2, t3))) === ()

    @test Enzyme.autodiff(Reverse, _wait_f, Const(t1)) === ()
    @test Enzyme.autodiff(Reverse, _wait_f, Duplicated(t1, t2)) === ()
    @test Enzyme.autodiff(Reverse, _wait_f, BatchDuplicated(t1, (t2, t3))) === ()

    @test Enzyme.autodiff(Forward, _wait_f, Const(t1)) === ()
    @test Enzyme.autodiff(Forward, _wait_f, Duplicated(t1, t2)) === ()
    @test Enzyme.autodiff(Forward, _wait_f, BatchDuplicated(t1, (t2, t3))) === ()


    t5 = Task(()->nothing)
    t6 = Task(()->nothing)
    t7 = Task(()->nothing)
    t8 = Task(()->nothing)

    @test Enzyme.autodiff(Reverse, schedule_f, Const(t5)) === ()
    @test Enzyme.autodiff(Forward, schedule_f, Const(t6)) === ()

    t9 = Task(()->nothing); t10 = Task(()->nothing)
    @test Enzyme.autodiff(Reverse, enq_work_f, Const(t9)) === ()
    @test Enzyme.autodiff(Forward, enq_work_f, Const(t10)) === ()
end
