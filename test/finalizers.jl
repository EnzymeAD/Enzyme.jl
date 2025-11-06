using Enzyme
using Test

const FREE_LIST = Vector{Any}()

mutable struct Container
    value::Float64
    function Container(v::Float64)
        c = new(v)
        finalizer(c) do c
            # Necromance object
            push!(FREE_LIST, c)
        end
        return c
    end
end

@noinline function compute(c)
    return c.value^2
end

function compute(x::Float64)
    c = Container(x)
    return compute(c)
end

compute(1.0)
GC.gc()
@test length(FREE_LIST) == 1
empty!(FREE_LIST)

@testset "forward" begin
    dx, x = autodiff(ForwardWithPrimal, compute, Duplicated(1.0, 2.0))
    @test dx == 4.0
    GC.gc()
    @test length(FREE_LIST) == 2
    empty!(FREE_LIST)

    dx, = autodiff(Forward, compute, Duplicated(1.0, 2.0))
    @test dx == 4.0
    GC.gc()
    @test length(FREE_LIST) == 2
    empty!(FREE_LIST)
end

@testset "batched forward" begin
    dx, x = autodiff(ForwardWithPrimal, compute, BatchDuplicated(1.0, (1.0, 2.0)))
    @test dx == 4.0
    GC.gc()
    @test length(FREE_LIST) == 3
    empty!(FREE_LIST)

    dx, = autodiff(Forward, compute, BatchDuplicated(1.0, (1.0, 2.0)))
    @test dx == 4.0
    GC.gc()
    @test length(FREE_LIST) == 3
    empty!(FREE_LIST)
end

@testset "reverse" begin
    ((dx,), x) = autodiff(ReverseWithPrimal, compute, Active(1.0))
    @test dx == 2.0
    GC.gc()
    @test length(FREE_LIST) == 2
    empty!(FREE_LIST)

    ((dx,), x) = autodiff(Reverse, compute, Active(1.0))
    @test dx == 2.0
    GC.gc()
    @test length(FREE_LIST) == 2
    empty!(FREE_LIST)
end
