using Enzyme, Test

# A compiled thunk is kept by the job without its world, and reused in another world for as
# long as a compiled instance of the generated `thunk` call still holds there, which is
# when no edge of the compilation was invalidated: here, when another interpreter (the one
# of a differentiated function that itself differentiates) infers the same call after an
# unrelated definition moved the world.
thunk_cache_inner(x) = sin(x) * x
thunk_cache_outer(x) = Enzyme.autodiff(Reverse, thunk_cache_inner, Active(x))[1][1]
thunk_cache_filler_1() = nothing

@testset "compiled thunks are reused across worlds while their instance holds" begin
    C = Enzyme.Compiler
    @test Enzyme.autodiff(Reverse, thunk_cache_inner, Active(2.0))[1][1] ≈ cos(2.0) * 2.0 + sin(2.0)
    n = length(C.cache)
    @eval thunk_cache_filler_2() = nothing
    # The outer thunk is a new entry. The nested call is inferred by the `EnzymeInterpreter`
    # in the new world; its generator runs again and finds the compiled inner thunk, so
    # there is no third entry.
    @test Enzyme.autodiff(Forward, thunk_cache_outer, Duplicated(2.0, 1.0))[1] ≈ 2 * cos(2.0) - sin(2.0) * 2.0
    @test length(C.cache) == n + 1

    # Whereas a rule for the inner function invalidates the instance: the thunk is
    # compiled again, replacing the entry rather than adding one.
    @eval Enzyme.EnzymeRules.inactive(::typeof(thunk_cache_inner), args...) = nothing
    @test Enzyme.autodiff(Reverse, thunk_cache_inner, Active(2.0))[1][1] == 0.0
    @test length(C.cache) == n + 1
end

@testset "compiled_instance_covers" begin
    C = Enzyme.Compiler
    sig = Tuple{typeof(thunk_cache_filler_1)}
    thunk_cache_filler_1()
    @test C.compiled_instance_covers(sig, Base.get_world_counter())
    @test !C.compiled_instance_covers(Tuple{typeof(thunk_cache_filler_1), Int}, Base.get_world_counter())
end
