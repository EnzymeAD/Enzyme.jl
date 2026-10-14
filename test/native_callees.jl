using Enzyme, Test

# Derivative-free callees called natively instead of being emitted (Julia 1.10 and later).

struct NCStructure
    idx::Vector{Int}
    n::Int
end
@noinline nc_structure(n::Int) = NCStructure(collect(1:n), n)

function nc_loss(x::Vector{Float64})
    s = nc_structure(length(x))
    acc = 0.0
    for i in s.idx
        acc += x[i]^2
    end
    return acc
end

# `_sortperm` on integers takes and returns only boxed values, so Julia compiles it with
# the boxed `jl_fptr_args` ABI and not a specialized entry: it must stay emitted.
nc_sortperm(x) = (p = sortperm([3, 1, 2]); x * p[1])

# A read-only function whose callee calls an effect-free native callee. On Julia 1.10 and
# 1.11 that call goes through `jl_invoke`; unless it is marked read-only-or-throw, Enzyme
# no longer finds `nc_shared` read-only-or-throw and clones it, and on Julia 1.10 (typed
# pointers) that clone failed LLVM verification.
struct NCView
    parent::Vector{Float64}
    size::NTuple{5, Int}
    strides::NTuple{5, Int}
    offset::Int
end
@noinline function nc_strides(dims::NTuple{5, Int})
    s2 = dims[1]
    s3 = s2 * dims[2]
    s4 = s3 * dims[3]
    return (1, s2, s3, s4, s4 * dims[4])
end
@noinline function nc_view(x::Vector{Float64})
    dims = (length(x) ÷ 72, 3, 4, 3, 2)
    return NCView(x, dims, nc_strides(dims), 0)
end
@noinline function nc_shared(x::Vector{Float64}, p::NTuple{3, Int})
    t = nc_view(x)
    return t.size[p[1]] * t.strides[p[2]] == t.size[p[3]]
end
nc_branch(x, p) = nc_shared(x, p) ? sum(x) : 2 * sum(x)

@static if VERSION < v"1.12-" || v"1.12-beta3" <= VERSION
    @testset "Native derivative-free callees" begin
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test autodiff(Forward, nc_loss, Duplicated(x, ones(3)))[1] ≈ 12.0

        @test autodiff(Reverse, nc_sortperm, Active, Active(2.0))[1][1] ≈ 2.0
        @test autodiff(Forward, nc_sortperm, Duplicated(2.0, 1.0))[1] ≈ 2.0

        # TODO: with code coverage `nc_shared` writes its counters, so Enzyme clones it
        # anyway, and on Julia 1.10 the clone fails verification until Enzyme_jll
        # carries EnzymeAD/Enzyme#3491.
        @static if VERSION >= v"1.11-" || Base.JLOptions().code_coverage == 0
            x = ones(144)
            for p in ((1, 3, 2), (2, 1, 2))
                dx = zero(x)
                autodiff(Reverse, nc_branch, Active, Duplicated(x, dx), Const(p))
                @test dx == fill(nc_shared(x, p) ? 1.0 : 2.0, 144)
            end
        end
    end
end
