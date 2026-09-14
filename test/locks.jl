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
    @test !islocked(my_cache_lock)
end

@testset "Lock reverse" begin
    Enzyme.autodiff(Reverse, my_lock, Const)
    @test !islocked(my_cache_lock)
end

# A lock that records every acquire/release into a shared log, so the tests
# can check that the reverse sweep holds the lock over the same region as the
# forward sweep: the adjoint of `lock` is `unlock` and vice versa.
mutable struct RecordingLock <: Base.AbstractLock
    inner::ReentrantLock
    name::Symbol
    log::Vector{Tuple{Symbol, Symbol}}
end
RecordingLock(name, log) = RecordingLock(ReentrantLock(), name, log)
function Base.lock(l::RecordingLock)
    push!(l.log, (l.name, :lock))
    return lock(l.inner)
end
function Base.unlock(l::RecordingLock)
    push!(l.log, (l.name, :unlock))
    return unlock(l.inner)
end
function Base.trylock(l::RecordingLock)
    push!(l.log, (l.name, :trylock))
    return trylock(l.inner)
end
Base.islocked(l::RecordingLock) = islocked(l.inner)

mutable struct LockedBuf{L <: Base.AbstractLock}
    lock::L
    data::Vector{Float64}
end

function copy_locked!(d::LockedBuf, s::LockedBuf, n)
    lock(d.lock)
    try
        lock(s.lock)
        try
            @inbounds for i in 1:n
                d.data[i] = s.data[i]
            end
        finally
            unlock(s.lock)
        end
    finally
        unlock(d.lock)
    end
    return nothing
end

function trylock_copy!(d::LockedBuf, s::LockedBuf, n)
    if trylock(d.lock)
        @inbounds for i in 1:n
            d.data[i] = s.data[i]
        end
        unlock(d.lock)
    end
    return nothing
end

@testset "Lock inside Duplicated forward" begin
    for f in (copy_locked!, trylock_copy!)
        log = Tuple{Symbol, Symbol}[]
        d = LockedBuf(RecordingLock(:d, log), zeros(3))
        s = LockedBuf(RecordingLock(:s, log), [1.0, 2.0, 3.0])
        dd = LockedBuf(RecordingLock(:dd, log), zeros(3))
        ds = LockedBuf(RecordingLock(:ds, log), [1.0, 1.0, 1.0])
        Enzyme.autodiff(Forward, f, Const, Duplicated(d, dd), Duplicated(s, ds), Const(3))
        @test d.data == [1.0, 2.0, 3.0]
        @test dd.data == [1.0, 1.0, 1.0]
        @test !islocked(d.lock)
        @test !islocked(s.lock)
        @test !islocked(dd.lock)
        @test !islocked(ds.lock)
        # Only the primal locks are touched, and only once each.
        if f === copy_locked!
            @test log == [(:d, :lock), (:s, :lock), (:s, :unlock), (:d, :unlock)]
        else
            @test log == [(:d, :trylock), (:d, :unlock)]
        end
    end
end

@testset "Lock inside Duplicated reverse" begin
    for f in (copy_locked!, trylock_copy!)
        log = Tuple{Symbol, Symbol}[]
        d = LockedBuf(RecordingLock(:d, log), zeros(3))
        s = LockedBuf(RecordingLock(:s, log), [1.0, 2.0, 3.0])
        dd = LockedBuf(RecordingLock(:dd, log), ones(3))
        ds = LockedBuf(RecordingLock(:ds, log), zeros(3))
        Enzyme.autodiff(Reverse, f, Const, Duplicated(d, dd), Duplicated(s, ds), Const(3))
        @test d.data == [1.0, 2.0, 3.0]
        @test ds.data == [1.0, 1.0, 1.0]
        @test dd.data == [0.0, 0.0, 0.0]
        @test !islocked(d.lock)
        @test !islocked(s.lock)
        @test !islocked(dd.lock)
        @test !islocked(ds.lock)
        # The reverse sweep re-acquires the primal locks over the mirrored
        # region: the adjoint of `unlock` is `lock` and the adjoint of `lock`
        # (or a successful `trylock`) is `unlock`.  Shadow locks are untouched.
        if f === copy_locked!
            @test log == [
                (:d, :lock), (:s, :lock), (:s, :unlock), (:d, :unlock),   # forward
                (:d, :lock), (:s, :lock), (:s, :unlock), (:d, :unlock),   # reverse
            ]
        else
            @test log == [
                (:d, :trylock), (:d, :unlock),   # forward
                (:d, :lock), (:d, :unlock),      # reverse
            ]
        end
    end
end

# The reverse of a failed `trylock` must not release a lock it never held.
@testset "trylock failure reverse" begin
    log = Tuple{Symbol, Symbol}[]
    d = LockedBuf(RecordingLock(:d, log), zeros(3))
    s = LockedBuf(RecordingLock(:s, log), [1.0, 2.0, 3.0])
    dd = LockedBuf(RecordingLock(:dd, log), ones(3))
    ds = LockedBuf(RecordingLock(:ds, log), zeros(3))
    # Hold the lock from another task so the primal trylock fails (a
    # ReentrantLock held by this task would be re-acquired successfully).
    started = Base.Event()
    release = Base.Event()
    holder = Threads.@spawn begin
        lock(d.lock.inner)
        notify(started)
        wait(release)
        unlock(d.lock.inner)
    end
    wait(started)
    Enzyme.autodiff(Reverse, trylock_copy!, Const, Duplicated(d, dd), Duplicated(s, ds), Const(3))
    notify(release)
    wait(holder)
    @test d.data == [0.0, 0.0, 0.0]
    @test ds.data == [0.0, 0.0, 0.0]
    @test dd.data == [1.0, 1.0, 1.0]
    @test log == [(:d, :trylock)]
    @test !islocked(d.lock)
end
