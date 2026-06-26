using Enzyme
using Test

const my_cache_lock = ReentrantLock()

function my_lock()
       lock(my_cache_lock);
       unlock(my_cache_lock);
       return nothing
end

@testset "Lock forward" begin
    Enzyme.autodiff(Forward, my_lock, Const)
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3086
mutable struct AtomicCounter
    @atomic count::Int
end

const my_counter = AtomicCounter(0)

function my_atomic_modify(x)
    @atomic my_counter.count += 1
    return x * x
end

@testset "Atomic modify" begin
    @test Enzyme.autodiff(Reverse, my_atomic_modify, Active, Active(2.0))[1][1] ≈ 4.0
    @test Enzyme.autodiff(Forward, my_atomic_modify, Duplicated(3.0, 1.0))[1] ≈ 6.0
    @test (@atomic my_counter.count) == 2
end

mutable struct DupAtomicCounter
    @atomic count::Int
    x::Float64
end

function my_dup_atomic_modify(c)
    @atomic c.count += 1
    return c.x * c.x
end

@testset "Atomic modify duplicated" begin
    c = DupAtomicCounter(0, 2.0)
    dc = DupAtomicCounter(0, 0.0)
    Enzyme.autodiff(Reverse, my_dup_atomic_modify, Active, Duplicated(c, dc))
    @test dc.x ≈ 4.0
    @test (@atomic c.count) == 1
    # the modification is replicated on the shadow
    @test (@atomic dc.count) == 1
end

