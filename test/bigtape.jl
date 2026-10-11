using Enzyme, Test
using LinearAlgebra
using Random
using FiniteDifferences

# A tape whose nested sub-tape is 32768 bytes or more cannot be stored inline by
# Julia (`jl_fielddesc16_t` only has a 15-bit size field, so `allocatedinline`
# gives up and the field is boxed), while Enzyme keeps writing it inline.  The
# augmented forward then stores a tape Julia cannot hold and the reverse pass
# cannot read it back, which used to surface as
#
#   AssertionError: Enzyme Internal Error: Illegal calling convention fixup
#     ctype = LLVM.PointerType(ptr addrspace(10))
#
# out of `calling_conv_fixup`.  Three ingredients are needed together to build
# such a tape:
#
#   1. `map` over a container with an abstract element type, whose closure
#      differentiates through the abstractly-typed element.  This puts Enzyme on
#      its runtime-generic path, where the tape is only partially known.
#   2. A `@noinline` callee picked by a runtime value, with several call sites,
#      so its tape is a merge of several sub-tapes.
#   3. Enough work in that callee for the merged sub-tape to cross the size
#      threshold.  Below it (a 5-matmul chain, or fewer than 4 call sites) the
#      same program compiles fine.

const BIGTAPE_N = 8

# Scaled so that a chain of seven products stays O(1) and the finite-difference
# check below is well conditioned.
bigtape_mat(rng) = randn(rng, BIGTAPE_N, BIGTAPE_N) ./ sqrt(BIGTAPE_N)

const BIGTAPE_A, BIGTAPE_B, BIGTAPE_C, BIGTAPE_D = let rng = MersenneTwister(1234)
    (bigtape_mat(rng), bigtape_mat(rng), bigtape_mat(rng), bigtape_mat(rng))
end

# Callee with a large reverse-mode tape.
@noinline function bigtape_chain(x::Matrix{Float64})
    t1 = BIGTAPE_A * x
    t2 = t1 * BIGTAPE_B
    t3 = x' * t2
    t4 = t3 * BIGTAPE_C
    t5 = t4 * t1
    t6 = t5 * BIGTAPE_D
    t7 = t6 * t3
    return tr(t7) / tr(t5)
end

# Several call sites of `bigtape_chain`, selected by a runtime value, so the
# tape Enzyme builds for `bigtape_dispatch` merges all of their sub-tapes.
@noinline function bigtape_dispatch(k::Vector{Int}, x)
    k[1] == 0 && return bigtape_chain(x)
    k[1] == -1 && return bigtape_chain(x)
    k[1] == -2 && return bigtape_chain(x)
    k[1] == -3 && return bigtape_chain(x)
    return bigtape_chain(x)
end

# `Any` value type: `op` in the closure below is only known as `Any`, which is
# what forces the runtime-generic path.
const BIGTAPE_TERMS = let rng = MersenneTwister(5678)
    Pair{Vector{Int}, Any}[[1, 2] => bigtape_mat(rng), [1, 3] => bigtape_mat(rng)]
end

bigtape_f(x) = sum(
    map(BIGTAPE_TERMS) do (k, op)
        tr(op * x) * bigtape_dispatch(k, x)
    end
)

@testset "boxed nested tape aggregate" begin
    x = bigtape_mat(MersenneTwister(91011))
    dx = zero(x)

    _, primal = Enzyme.autodiff(
        set_runtime_activity(ReverseWithPrimal), Const(bigtape_f), Active, Duplicated(x, dx)
    )

    @test primal ≈ bigtape_f(x)
    @test dx ≈ FiniteDifferences.grad(central_fdm(5, 1), bigtape_f, x)[1] rtol = 1.0e-5
end
