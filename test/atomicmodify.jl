using Enzyme
using FileCheck
using InteractiveUtils
using Test

# Atomic modifications of integer fields (`@atomic x.n += v`) next to active
# data. Julia 1.13 emits these as the `julia.atomicmodify` pseudo-intrinsic,
# which Enzyme has to keep intact and replicate on the shadow object so that
# counters and flags of shadow objects stay consistent with the primal ones.

mutable struct AtomicCounter
    @atomic n::Int
    y::Float64
end

function counter_inc(c, v)
    @atomic c.n += 1
    return c.y * v
end

# a modify op that is not an addition
function counter_max(c, v)
    @atomic max(c.n, 7)
    return c.y * v
end

# non-default ordering, non-unit increment from an argument
function counter_add(c, k, v)
    @atomic :monotonic c.n += k
    return c.y * c.y * v
end

# Before Julia 1.13 these modifications are emitted as a cmpxchg loop, which
# Enzyme does not handle.
@static if VERSION >= v"1.13-"

    @testset "IR uses julia.atomicmodify" begin
        @test @filecheck begin
            @check_label "define"
            @check "call { i64, i64 }"
            @check_same "@julia.atomicmodify.i64.p11("
            @check_same "i64 1"
            code_llvm(counter_inc, Tuple{AtomicCounter, Float64}; raw = true, optimize = false)
        end
        @test @filecheck begin
            @check_label "define"
            @check "call { i64, i64 }"
            @check_same "@julia.atomicmodify.i64.p11("
            @check_same "i64 7"
            code_llvm(counter_max, Tuple{AtomicCounter, Float64}; raw = true, optimize = false)
        end
    end

    @testset "Forward counter increment" begin
        c = AtomicCounter(3, 2.5)
        dc = AtomicCounter(10, 0.0)
        r = autodiff(ForwardWithPrimal, counter_inc, Duplicated(c, dc), Duplicated(4.0, 1.0))
        @test r[1] ≈ 2.5
        @test r[2] ≈ 2.5 * 4.0
        # both the primal and the shadow counter are incremented exactly once
        @test c.n == 4
        @test dc.n == 11
    end

    @testset "Reverse counter increment" begin
        c = AtomicCounter(3, 2.5)
        dc = AtomicCounter(10, 0.0)
        r = autodiff(Reverse, counter_inc, Active, Duplicated(c, dc), Active(4.0))
        @test r[1][2] ≈ 2.5
        @test dc.y ≈ 4.0
        @test c.n == 4
        @test dc.n == 11
    end

    @testset "Forward counter max" begin
        c = AtomicCounter(3, 2.5)
        dc = AtomicCounter(10, 0.0)
        r = autodiff(ForwardWithPrimal, counter_max, Duplicated(c, dc), Duplicated(4.0, 1.0))
        @test r[1] ≈ 2.5
        @test r[2] ≈ 2.5 * 4.0
        @test c.n == 7
        @test dc.n == 10
    end

    @testset "Reverse counter max" begin
        c = AtomicCounter(3, 2.5)
        dc = AtomicCounter(10, 0.0)
        r = autodiff(Reverse, counter_max, Active, Duplicated(c, dc), Active(4.0))
        @test r[1][2] ≈ 2.5
        @test dc.y ≈ 4.0
        @test c.n == 7
        @test dc.n == 10
    end

    @testset "Forward monotonic counter add" begin
        c = AtomicCounter(3, 2.5)
        dc = AtomicCounter(10, 0.0)
        r = autodiff(ForwardWithPrimal, counter_add, Duplicated(c, dc), Const(5), Duplicated(4.0, 1.0))
        @test r[1] ≈ 2.5 * 2.5
        @test r[2] ≈ 2.5 * 2.5 * 4.0
        @test c.n == 8
        @test dc.n == 15
    end

    @testset "Reverse monotonic counter add" begin
        c = AtomicCounter(3, 2.5)
        dc = AtomicCounter(10, 0.0)
        r = autodiff(Reverse, counter_add, Active, Duplicated(c, dc), Const(5), Active(4.0))
        @test r[1][3] ≈ 2.5 * 2.5
        @test dc.y ≈ 2 * 2.5 * 4.0
        @test c.n == 8
        @test dc.n == 15
    end

end # VERSION >= v"1.13-"
