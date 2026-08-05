module NestedCache

using Enzyme
using Enzyme: EnzymeRules
using LinearAlgebra
using Test

import .EnzymeRules: augmented_primal, reverse
using .EnzymeRules

# A custom-ruled function whose rule bodies pull in enough code (mul!, norm, Dict) that
# `nested_codegen!` has real work to do for them. See EnzymeAD/Enzyme.jl#3392.
function heavy(x::Float64)
    A = fill(x, 8, 8)
    return norm(A * A) + sum(A)
end

const SCRATCH = Dict{Symbol, Matrix{Float64}}()

function heavy_deriv(x::Float64)
    A = get(SCRATCH, :A, nothing)
    if A === nothing
        A = zeros(8, 8)
        SCRATCH[:A] = A
    end
    fill!(A, x)
    B = similar(A)
    mul!(B, A, A)
    dB = B ./ norm(B)
    dA = similar(A)
    mul!(dA, dB, transpose(A))
    mul!(dA, transpose(A), dB, 1.0, 1.0)
    return sum(dA) + length(A)
end

function augmented_primal(config::RevConfig, func::Const{typeof(heavy)}, ::Type{<:Active}, x::Active{Float64})
    primal = needs_primal(config) ? heavy(x.val) : nothing
    return AugmentedReturn(primal, nothing, nothing)
end

function reverse(config::RevConfig, func::Const{typeof(heavy)}, dret::Active, tape, x::Active{Float64})
    return (heavy_deriv(x.val) * dret.val,)
end

# Distinct outer functions, so each `autodiff` is a separate thunk build, but all of them
# need the identical `augmented_primal`/`reverse` method instances.
outer1(x) = heavy(x) * 1.0
outer2(x) = heavy(x) + 0.0
outer3(x) = heavy(x) - 0.0

# Enzyme builds its thunk while the *calling* function is compiled, so each build has to be
# its own freshly-compiled expression for the cache to be observed in between.
function build_and_run(fname::Symbol)
    before = length(Enzyme.Compiler.NESTED_CODEGEN_CACHE)
    res = Core.eval(@__MODULE__, :(autodiff(Reverse, $fname, Active, Active(1.3))))
    return res[1][1], before, length(Enzyme.Compiler.NESTED_CODEGEN_CACHE)
end

@testset "nested codegen cache" begin
    Enzyme.Compiler.clear_nested_codegen_cache!()
    @test length(Enzyme.Compiler.NESTED_CODEGEN_CACHE) == 0

    expected = heavy_deriv(1.3)

    d1, before1, after1 = build_and_run(:outer1)
    # The first build compiles both rule bodies and populates the cache.
    @test after1 > before1
    @test d1 ≈ expected

    # Subsequent builds need the same rule method instances and must reuse them rather
    # than recompiling: the entry count stays put, and the derivative is unchanged.
    for fname in (:outer2, :outer3)
        d, before, after = build_and_run(fname)
        @test before == after
        @test d ≈ expected
    end

    @test Enzyme.Compiler.NESTED_CODEGEN_CACHE_BYTES[] > 0

    # Clearing forces the next build to regenerate the modules from scratch, which must
    # still produce the same answer.
    Enzyme.Compiler.clear_nested_codegen_cache!()
    @test Enzyme.Compiler.NESTED_CODEGEN_CACHE_BYTES[] == 0
    outer4(x) = heavy(x) / 1.0
    @test autodiff(Reverse, outer4, Active, Active(1.3))[1][1] ≈ expected
end

end # module NestedCache
