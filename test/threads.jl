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

@testset "GEP non-inline type analysis" begin
    function spread_no_broadcast!(bufs, q, ::Val{N}) where N
        Threads.@threads for t in 1:N
            for i in t:N:length(q)
                @inbounds bufs[t][i, i + 1, i + 2] += q[i]
            end
        end
        return sum(sum, bufs)
    end

    q = [1.0, 2.0]
    dq = [0.0, 0.0]
    bufs = [zeros(4, 4, 4) for _ in 1:2]
    dbufs = [zeros(4, 4, 4) for _ in 1:2]

    autodiff(ReverseWithPrimal, spread_no_broadcast!, Active, Duplicated(bufs, dbufs),
             Duplicated(q, dq), Const(Val(2)))
    @test dq ≈ [1.0, 1.0]
end


@testset "Atomic accumulation into shadows $(Threads.nthreads())" begin
    function sumsq(x)
        s = 0.0
        @inbounds for i in eachindex(x)
            s += x[i]^2
        end
        return s
    end
    function spawn_sumsq!(out, x)
        task = Threads.@spawn begin
            @inbounds out[1] = sumsq(x)
        end
        @inbounds out[2] = sumsq(x)
        wait(task)
        nothing
    end
    function threaded_sumsq(x)
        out = zeros(4)
        Threads.@threads for i in 1:4
            @inbounds out[i] = i * sumsq(x)
        end
        return sum(out)
    end
    function atomics(f, A, TT, setting)
        ir = sprint() do io
            Enzyme.Compiler.enzyme_code_llvm(io, f, A, TT; dump_module = true, AtomicAdd = setting)
        end
        return count("atomicrmw", ir)
    end

    @test atomic_add(Reverse) === nothing
    @test atomic_add(set_atomic_add(Reverse, false)) === false
    @test atomic_add(set_atomic_add(ReverseSplitWithPrimal, true)) === true

    multithreaded = Threads.nthreads() > 1
    VecDup = Tuple{Duplicated{Vector{Float64}}}
    # By default a CPU derivative accumulates atomically exactly when Julia runs several threads
    @test (atomics(sumsq, Active, VecDup, nothing) > 0) == multithreaded
    @test atomics(sumsq, Active, VecDup, true) > 0
    @test atomics(sumsq, Active, VecDup, false) == 0
    # A function that runs work on other threads keeps the default (its primal code has atomics
    # of its own, so compare against the other settings rather than against zero)
    for (f, A, TT) in ((threaded_sumsq, Active, VecDup), (spawn_sumsq!, Const, Tuple{Duplicated{Vector{Float64}}, Duplicated{Vector{Float64}}}))
        @test atomics(f, A, TT, false) == atomics(f, A, TT, nothing)
        @test (atomics(f, A, TT, nothing) == atomics(f, A, TT, true)) == multithreaded
    end

    x = collect(1.0:10.0)
    for mode in (Reverse, set_atomic_add(Reverse, false), set_atomic_add(Reverse, true))
        dx = zero(x)
        autodiff(mode, Const(sumsq), Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test gradient(mode, sumsq, x)[1] ≈ 2 .* x

        dx = zero(x)
        autodiff(mode, Const(threaded_sumsq), Active, Duplicated(x, dx))
        @test dx ≈ 20 .* x

        dx = zero(x)
        autodiff(mode, Const(spawn_sumsq!), Const, Duplicated(zeros(2), ones(2)), Duplicated(x, dx))
        @test dx ≈ 4 .* x
    end

    fwd, rev = autodiff_thunk(set_atomic_add(ReverseSplitNoPrimal, false), Const{typeof(sumsq)}, Active, Duplicated{Vector{Float64}})
    dx = zero(x)
    tape, = fwd(Const(sumsq), Duplicated(x, dx))
    rev(Const(sumsq), Duplicated(x, dx), 1.0, tape)
    @test dx ≈ 2 .* x

    # By default, `autodiff` calls on several threads may share one shadow
    dx = zero(x)
    Threads.@threads for i in 1:1000
        autodiff(Reverse, Const(sumsq), Active, Duplicated(x, dx))
    end
    @test dx ≈ 2000 .* x
end
