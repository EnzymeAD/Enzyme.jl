using Enzyme, Test, JLArrays
using LinearAlgebra: mul!, dot, transpose, adjoint, Symmetric

function jlres(x)
    2 * collect(x)
end

@testset "JLArrays" begin
    # TODO fix activity of jlarray
    # Enzyme.jacobian(Forward, jlres, JLArray([3.0, 5.0]))
    # Enzyme.jacobian(Reverse, jlres, JLArray([3.0, 5.0]))
end

#=
The AbstractGPUArray matmul/dot rules. JLArray is a CPU-backed AbstractGPUArray,
so it hits the same LinearAlgebra entry points a real backend does, no device
needed.

The rules sit on `generic_matmatmul!` / `generic_matvecmul!`, so these tests call
`mul!` into an explicit buffer and seed the output shadow with ones, which is what
`sum(C)` would hand back. A reduction can't be used instead: `mapreducedim!` over
a JLArray doesn't differentiate yet (LLVM verifier error out of
`EnzymeCreateAugmentedPrimal`), and `dot` on GPUArrays infers as `Any`, so nesting
it asks the rule for a scalar's shadow.

The allocating `A * B` isn't tested here either. The shadow of the array `similar`
returns is never zeroed for an AbstractGPUArray, so the reverse pass reads junk
instead of the incoming cotangent. That affects any rule with a freshly allocated
output and doesn't happen on CUDA.
=#
@testset "GPUArrays linalg rules" begin
    jl(x) = JLArray(x)

    function matmul!(C, A, B)
        mul!(C, A, B)
        return nothing
    end

    # Central differences over the entries of `X0`. For complex inputs Enzyme's
    # cotangent of a real loss is ∂L/∂Re + i·∂L/∂Im, so the imaginary direction
    # gets its own difference.
    function fdgrad(f, X0, ϵ = 1.0e-6)
        g = zero(X0)
        for idx in eachindex(X0)
            Xp = copy(X0); Xp[idx] += ϵ
            Xm = copy(X0); Xm[idx] -= ϵ
            g[idx] = (f(Xp) - f(Xm)) / (2ϵ)
            if eltype(X0) <: Complex
                Xip = copy(X0); Xip[idx] += im * ϵ
                Xim = copy(X0); Xim[idx] -= im * ϵ
                g[idx] += im * (f(Xip) - f(Xim)) / (2ϵ)
            end
        end
        return g
    end

    @testset "matmul reverse ($m×$k × $k×$n)" for (m, k, n) in ((3, 4, 2), (5, 5, 1))
        A0 = randn(m, k)
        B0 = randn(k, n)

        # dC = ones is the cotangent of sum(A·B); analytic grads dA = ones·B',
        # dB = A'·ones
        dA = jl(zero(A0))
        dB = jl(zero(B0))
        Enzyme.autodiff(
            Reverse, matmul!, Const,
            Duplicated(jl(zeros(m, n)), jl(ones(m, n))),
            Duplicated(jl(A0), dA), Duplicated(jl(B0), dB),
        )

        ones_mn = ones(m, n)
        @test collect(dA) ≈ ones_mn * B0'
        @test collect(dB) ≈ A0' * ones_mn
    end

    # Width 2, the only case that reaches `_bget` and the `ntuple(Val(N))` loops.
    # The shadows start at different values, so this also checks each batch
    # element accumulates into its own array.
    @testset "batched matmul reverse" begin
        m, k, n = 3, 4, 2
        A0 = randn(m, k)
        B0 = randn(k, n)
        dA1 = jl(fill(1.0, m, k))
        dA2 = jl(fill(-2.0, m, k))
        Enzyme.autodiff(
            Reverse, matmul!, Const,
            BatchDuplicated(jl(zeros(m, n)), (jl(ones(m, n)), jl(ones(m, n)))),
            BatchDuplicated(jl(A0), (dA1, dA2)), Const(jl(B0)),
        )
        expected = ones(m, n) * B0'
        @test collect(dA1) ≈ 1.0 .+ expected
        @test collect(dA2) ≈ -2.0 .+ expected
    end

    # Active α at width 2, where the cotangent comes back as a tuple of scalars:
    # dα[i] = ⟨dCᵢ, A·B⟩.
    @testset "batched matmul reverse, Active alpha" begin
        m, k, n = 3, 4, 2
        A0 = randn(m, k)
        B0 = randn(k, n)
        function lossα!(C, A, B, α)
            mul!(C, A, B, α, 0.5)
            return nothing
        end
        out = Enzyme.autodiff(
            Reverse, lossα!, Const,
            BatchDuplicated(jl(randn(m, n)), (jl(ones(m, n)), jl(fill(2.0, m, n)))),
            Const(jl(A0)), Const(jl(B0)), Active(2.0),
        )
        s = sum(A0 * B0)
        @test all(out[1][4] .≈ (s, 2.0 * s))
    end

    # A `Const` output has no shadow, so the rule returns early and the operands
    # get nothing.
    @testset "Const output buffer" begin
        m, k, n = 3, 4, 2
        dA = jl(fill(7.0, m, k))
        Enzyme.autodiff(
            Reverse, matmul!, Const,
            Const(jl(zeros(m, n))), Duplicated(jl(randn(m, k)), dA),
            Const(jl(randn(k, n))),
        )
        @test all(collect(dA) .≈ 7.0)
    end

    # The same operand plain and transposed, both products matrix-vector: two
    # passes through `generic_matvecmul!`, tA = 'N' then 'T'.
    @testset "matvec reverse with transpose" begin
        X0 = randn(6, 3)
        β0 = randn(3)
        function g!(y, z, X, β)
            mul!(z, X, β)
            mul!(y, transpose(X), z)
            return nothing
        end
        dX = jl(zero(X0))
        dβ = jl(zero(β0))
        Enzyme.autodiff(
            Reverse, g!, Const,
            Duplicated(jl(zeros(3)), jl(ones(3))), Duplicated(jl(zeros(6)), jl(zeros(6))),
            Duplicated(jl(X0), dX), Duplicated(jl(β0), dβ),
        )

        gcpu(X, β) = sum(transpose(X) * (X * β))
        @test collect(dβ) ≈ fdgrad(β -> gcpu(X0, β), β0) rtol = 1.0e-4
        @test collect(dX) ≈ fdgrad(X -> gcpu(X, β0), X0) rtol = 1.0e-4
    end

    #=
    Wrapped operands, where the char algebra in `_pullback!` actually matters: the
    chars differ per combination, 'C' is a real adjoint once the eltype is complex,
    and a transposed complex operand needs a materialized conjugate that no char
    can express. A real `dC` of ones seeds `real(sum(C))`, which is what `fdgrad`
    differentiates.
    =#
    @testset "matmul reverse, $name" for (name, T, wa, wb) in (
            ("transpose(A)·B", Float64, transpose, identity),
            ("A·transpose(B)", Float64, identity, transpose),
            ("adjoint(A)·B, complex", ComplexF64, adjoint, identity),
            ("transpose(A)·B, complex", ComplexF64, transpose, identity),
            ("A·adjoint(B), complex", ComplexF64, identity, adjoint),
        )
        m, k, n = 3, 4, 2
        A0 = randn(T, wa === identity ? (m, k) : (k, m))
        B0 = randn(T, wb === identity ? (k, n) : (n, k))
        cpu(A) = real(sum(wa(A) * wb(B0)))

        function wrapped_mul!(C, A, B)
            mul!(C, wa(A), wb(B))
            return nothing
        end
        dA = jl(zero(A0))
        Enzyme.autodiff(
            Reverse, wrapped_mul!, Const,
            Duplicated(jl(zeros(T, m, n)), jl(ones(T, m, n))),
            Duplicated(jl(copy(A0)), dA), Const(jl(B0)),
        )

        @test collect(dA) ≈ fdgrad(cpu, A0) rtol = 1.0e-4
    end

    # 5-arg `mul!`, which the tests above miss: β ≠ 0 scales dC by conj(β), and an
    # Active α/β has to come back as the scalar's own type (one `MulAddMul` before
    # Julia 1.12).
    @testset "mul! beta != 0 and Active alpha/beta" begin
        A0 = randn(3, 4)
        B0 = randn(4, 2)
        C0 = randn(3, 2)

        # loss = sum(2·A·B + 0.5·C₀)  ⇒  dA = 2·ones·B',  dC₀ = 0.5
        function loss!(C, A, B)
            mul!(C, A, B, 2.0, 0.5)
            return nothing
        end
        dA = jl(zero(A0))
        dC = jl(ones(3, 2))
        Enzyme.autodiff(
            Reverse, loss!, Const,
            Duplicated(jl(copy(C0)), dC), Duplicated(jl(A0), dA), Const(jl(B0)),
        )
        @test collect(dA) ≈ 2.0 .* (ones(3, 2) * B0')
        @test all(collect(dC) .≈ 0.5)

        # dα = sum(A·B) and dβ = sum(C₀) come back as Active argument cotangents
        function lossα!(C, A, B, α)
            mul!(C, A, B, α, 0.5)
            return nothing
        end
        outα = Enzyme.autodiff(
            Reverse, lossα!, Const,
            Duplicated(jl(copy(C0)), jl(ones(3, 2))), Duplicated(jl(A0), jl(zero(A0))),
            Const(jl(B0)), Active(2.0),
        )
        @test outα[1][4] ≈ sum(A0 * B0)

        function lossβ!(C, A, B, β)
            mul!(C, A, B, 2.0, β)
            return nothing
        end
        outβ = Enzyme.autodiff(
            Reverse, lossβ!, Const,
            Duplicated(jl(copy(C0)), jl(ones(3, 2))), Duplicated(jl(A0), jl(zero(A0))),
            Const(jl(B0)), Active(0.5),
        )
        @test outβ[1][4] ≈ sum(C0)
    end

    @testset "dot reverse" begin
        a0 = randn(8)
        b0 = randn(8)
        h(a, b) = dot(a, b)
        da = jl(zero(a0))
        db = jl(zero(b0))
        Enzyme.autodiff(Reverse, h, Active, Duplicated(jl(a0), da), Duplicated(jl(b0), db))
        @test collect(da) ≈ b0
        @test collect(db) ≈ a0
    end

    # Batched, so the return cotangent is a tuple of `Active`s. The shadows start
    # apart to check per-element accumulation.
    @testset "batched dot reverse" begin
        a0 = randn(6)
        b0 = randn(6)
        h(a, b) = dot(a, b)
        da1 = jl(fill(1.0, 6))
        da2 = jl(fill(-2.0, 6))
        Enzyme.autodiff(
            Reverse, h, Active,
            BatchDuplicated(jl(a0), (da1, da2)), Const(jl(b0)),
        )
        @test collect(da1) ≈ 1.0 .+ b0
        @test collect(da2) ≈ -2.0 .+ b0
    end

    # Only `a`'s cotangent picks up conj(dr), which a real cotangent can't tell
    # apart, so the loss is complex (dr = 2-3im). Reference values come from Enzyme
    # on plain `Array`s, which these rules don't touch.
    @testset "dot reverse, complex conjugation" begin
        a0 = ComplexF64[1 + 2im, -3 + 1im, 0.5 - 1.5im]
        b0 = ComplexF64[2 - 1im, 0.25 + 3im, -1 + 0.5im]
        closs(a, b) = real((2 + 3im) * dot(a, b))

        ref_da, ref_db = zero(a0), zero(b0)
        Enzyme.autodiff(
            Reverse, closs, Active,
            Duplicated(copy(a0), ref_da), Duplicated(copy(b0), ref_db),
        )

        da, db = jl(zero(a0)), jl(zero(b0))
        Enzyme.autodiff(
            Reverse, closs, Active,
            Duplicated(jl(copy(a0)), da), Duplicated(jl(copy(b0)), db),
        )
        @test collect(da) ≈ ref_da
        @test collect(db) ≈ ref_db
        # and explicitly: conj on `a`'s cotangent only
        @test collect(da) ≈ conj(2 - 3im) .* b0
        @test collect(db) ≈ (2 - 3im) .* a0
    end

    # Symmetric/Hermitian operands need their cotangent projected onto the stored
    # triangle, which isn't a matmul, so the rule rejects them for now. Triangular
    # operands go elsewhere in LinearAlgebra and have no rule at all.
    @testset "Symmetric operand is rejected" begin
        n = 4
        X0 = randn(n, n)
        dX = jl(zero(X0))
        @test_throws ArgumentError Enzyme.autodiff(
            Reverse, matmul!, Const,
            Duplicated(jl(zeros(n, 3)), jl(ones(n, 3))),
            Duplicated(Symmetric(jl(X0), :U), Symmetric(dX, :U)), Const(jl(randn(n, 3))),
        )
    end
end
