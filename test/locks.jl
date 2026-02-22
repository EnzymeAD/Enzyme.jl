using Enzyme
using Test

const my_cache_lock = ReentrantLock()

function my_lock()
    lock(my_cache_lock)
    unlock(my_cache_lock)
    return nothing
end

@testset "Lock forward" begin
    Enzyme.autodiff(Forward, my_lock, Const)
end
