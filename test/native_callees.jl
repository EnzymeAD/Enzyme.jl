using Enzyme, Test

# Derivative-free callees called natively instead of being emitted (Julia 1.10 to 1.13).

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

@static if VERSION < v"1.12-" || v"1.12-beta3" <= VERSION < v"1.14-"
    @testset "Native derivative-free callees" begin
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test autodiff(Forward, nc_loss, Duplicated(x, ones(3)))[1] ≈ 12.0

        @test autodiff(Reverse, nc_sortperm, Active, Active(2.0))[1][1] ≈ 2.0
        @test autodiff(Forward, nc_sortperm, Duplicated(2.0, 1.0))[1] ≈ 2.0
    end
end
