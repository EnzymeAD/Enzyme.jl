using Enzyme, Test

# An IdSet compares its keys by identity, so inserting, finding and deleting an
# active array carries no derivative.
function idset_sumsq(x)
    s = Base.IdSet{Any}()
    push!(s, x)
    x in s || return 0.0
    delete!(s, x)
    push!(s, x)
    return sum(abs2, x)
end

# On Julia 1.10, IdSet is built on IdDict, whose deletion (jl_eqtable_pop) has
# no derivative yet.
if VERSION >= v"1.11"
    @testset "IdSet of active arrays" begin
        x = [1.0, 2.0, 3.0]
        @test Enzyme.gradient(Reverse, idset_sumsq, x)[1] ≈ 2 .* x
        @test Enzyme.gradient(set_runtime_activity(Reverse), idset_sumsq, x)[1] ≈ 2 .* x
        @test Enzyme.gradient(Forward, idset_sumsq, x)[1] ≈ 2 .* x
    end
end
