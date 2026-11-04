using Enzyme, Test

# Inserting, finding and deleting an active array in an IdSet.
function idset_sumsq(x)
    s = Base.IdSet{Any}()
    push!(s, x)
    x in s || return 0.0
    delete!(s, x)
    push!(s, x)
    return sum(abs2, x)
end

# The derivative flows through the set: the array read back out of it is `x`.
function idset_through(x)
    s = Base.IdSet{Any}()
    push!(s, x)
    return sum(abs2, first(s)::Vector{Float64})
end

# Enough insertions and deletions that the keys are grown, compacted and rehashed.
function idset_churn(x)
    s = Base.IdSet{Any}()
    others = [[Float64(i)] for i in 1:20]
    for o in others
        push!(s, o)
    end
    push!(s, :sym)
    push!(s, x)
    for o in others[1:2:end]
        delete!(s, o)
    end
    delete!(s, x)
    push!(s, x)
    for o in others[1:2:end]
        push!(s, o)
    end
    t = 0.0
    for y in s
        if y === x
            t += sum(abs2, y::Vector{Float64})
        end
    end
    return t
end

function idset_push_sum(s, x)
    push!(s, x)
    t = 0.0
    for y in s
        t += sum(abs2, y::Vector{Float64})
    end
    return t
end

function idset_push_two(c, x)
    s = Base.IdSet{Any}()
    push!(s, c)
    push!(s, x)
    t = 0.0
    for y in s
        t += sum(abs2, y::Vector{Float64})
    end
    return t
end

# On Julia 1.10, IdSet is built on IdDict, whose deletion (jl_eqtable_pop) has
# no derivative yet.
if VERSION >= v"1.11"
    @testset "IdSet of active arrays" begin
        x = [1.0, 2.0, 3.0]
        for f in (idset_sumsq, idset_through, idset_churn)
            @test Enzyme.gradient(Reverse, f, x)[1] ≈ 2 .* x
            @test Enzyme.gradient(set_runtime_activity(Reverse), f, x)[1] ≈ 2 .* x
            @test Enzyme.gradient(Forward, f, x)[1] ≈ 2 .* x
        end
        dxs = Enzyme.autodiff(
            Forward, idset_churn, BatchDuplicated(x, ([1.0, 0, 0], [0, 0, 1.0]))
        )[1]
        @test dxs[1] ≈ 2x[1]
        @test dxs[2] ≈ 2x[3]

        # The shadow set receives the shadow of what the primal receives.
        s = Base.IdSet{Any}()
        ds = Base.IdSet{Any}()
        dx = zero(x)
        Enzyme.autodiff(Reverse, idset_push_sum, Duplicated(s, ds), Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test only(s) === x
        @test only(ds) === dx
        @test dx in ds

        # An inactive array alongside an active one, under runtime activity.
        c = [4.0, 5.0]
        dx = zero(x)
        Enzyme.autodiff(
            set_runtime_activity(Reverse), idset_push_sum,
            Duplicated(Base.IdSet{Any}(Any[c]), Base.IdSet{Any}(Any[c])), Duplicated(x, dx)
        )
        @test dx ≈ 2 .* x
        dx = zero(x)
        Enzyme.autodiff(
            set_runtime_activity(Reverse), idset_push_two, Const(c), Duplicated(x, dx)
        )
        @test dx ≈ 2 .* x

        # A set whose shadow is itself is inactive, and left as the primal makes it.
        s = Base.IdSet{Any}()
        dx = zero(x)
        Enzyme.autodiff(
            set_runtime_activity(Reverse), idset_push_sum, Duplicated(s, s), Duplicated(x, dx)
        )
        @test only(s) === x
    end
end
