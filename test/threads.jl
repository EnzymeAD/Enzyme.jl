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

#= Julia's `@cfunction` thunks stay callable from a thread Julia does not know
about yet: the thunk null-checks the pgcstack and calls `jl_adopt_thread` before
touching it. Enzyme used to hoist a `julia.get_pgcstack` into the entry block of
those thunks, so the GC frame was pushed through the still-null pgcstack and the
foreign thread died with a segfault. CUDA.jl reaches this by lazily spawning its
nonblocking synchronization worker from inside differentiated code. =#
const foreign_thread_calls = Threads.Atomic{Int}(0)

function foreign_thread_worker(::Ptr{Cvoid})
    # allocate, so that the thunk needs a GC frame in the first place
    v = Base.inferencebarrier(Any[1, 2, 3])
    Threads.atomic_add!(foreign_thread_calls, length(v)::Int)
    return nothing
end

@noinline function run_on_foreign_thread(cb::Ptr{Cvoid})
    tid = Ref{NTuple{32, UInt8}}(ntuple(Returns(UInt8(0)), Val(32)))
    err = @ccall uv_thread_create(tid::Ptr{Cvoid}, cb::Ptr{Cvoid}, C_NULL::Ptr{Cvoid})::Cint
    err == 0 || Base.uv_error("uv_thread_create", err)
    # spin rather than join, so that this thread keeps hitting safepoints while
    # the foreign one allocates
    while foreign_thread_calls[] == 0
        ccall(:jl_cpu_pause, Cvoid, ())
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
    @ccall uv_thread_join(tid::Ptr{Cvoid})::Cint
    return nothing
end

# the `@cfunction` has to be created here, inside the differentiated code, so
# that Enzyme rather than Julia emits the thunk
function spawner(x)
    run_on_foreign_thread(@cfunction(foreign_thread_worker, Cvoid, (Ptr{Cvoid},)))
    return x * x
end

@testset "@cfunction thunk on a foreign thread" begin
    @test spawner(3.0) == 9.0
    @test foreign_thread_calls[] == 3

    foreign_thread_calls[] = 0
    @test autodiff(Forward, spawner, Duplicated(3.0, 1.0))[1] == 6.0
    @test foreign_thread_calls[] == 3

    foreign_thread_calls[] = 0
    @test autodiff(ReverseWithPrimal, spawner, Active, Active(3.0))[2] == 9.0
    @test foreign_thread_calls[] > 0
end
