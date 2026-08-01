using Enzyme, Test, JLArrays
using LinearAlgebra: mul!, lmul!, rmul!, dot, transpose, adjoint,
    UpperTriangular, LowerTriangular, Symmetric, Hermitian

function jlres(x)
    2 * collect(x)
end

@testset "JLArrays" begin
    # TODO fix activity of jlarray
    # Enzyme.jacobian(Forward, jlres, JLArray([3.0, 5.0]))
    # Enzyme.jacobian(Reverse, jlres, JLArray([3.0, 5.0]))
end

#=
AbstractGPUArray linear-algebra rules (matmul / dot / sum). JLArray is a
CPU-backed `AbstractGPUArray`, so it reaches these rules through the same
LinearAlgebra entry points a real backend does, without needing a device.

The rules live on `generic_matmatmul!` / `generic_matvecmul!` /
`generic_trimatmul!` / `generic_mattrimul!`, so the tests drive `mul!` with an
explicit output buffer. The allocating `A * B` reaches the same rules, but on
JLArray the shadow of the array it allocates is never zeroed and the reverse pass
reads junk out of it -- see the `@test_broken` at the end.
=#
@testset "GPUArrays linalg rules" begin
    jl(x) = JLArray(x)

    function matmul_sum(C, A, B)
        mul!(C, A, B)
        return sum(C)
    end

    #=
    Central differences over the entries of `X0`. For a complex input Enzyme's
    cotangent of a real-valued loss is ∂L/∂Re + i·∂L/∂Im, so both directions are
    needed.
    =#
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

        # loss = sum(A·B); analytic grads dA = ones*B', dB = A'*ones
        dA = jl(zero(A0))
        dB = jl(zero(B0))
        Enzyme.autodiff(
            Reverse, matmul_sum, Active,
            Duplicated(jl(zeros(m, n)), jl(zeros(m, n))),
            Duplicated(jl(A0), dA), Duplicated(jl(B0), dB),
        )

        ones_mn = ones(m, n)
        @test collect(dA) ≈ ones_mn * B0'
        @test collect(dB) ≈ A0' * ones_mn
    end

    #=
    Batch width > 1, which `_bget`, `_dscalar` and every `ntuple(Val(N))` in the
    rules exist to serve and nothing else here reaches. The shadows start at
    different values so this also pins that each batch element accumulates into
    its own array rather than all of them landing in the first.
    =#
    @testset "batched matmul reverse" begin
        m, k, n = 3, 4, 2
        A0 = randn(m, k)
        B0 = randn(k, n)
        dA1 = jl(fill(1.0, m, k))
        dA2 = jl(fill(-2.0, m, k))
        Enzyme.autodiff(
            Reverse, matmul_sum, Active,
            BatchDuplicated(jl(zeros(m, n)), (jl(zeros(m, n)), jl(zeros(m, n)))),
            BatchDuplicated(jl(A0), (dA1, dA2)), Const(jl(B0)),
        )
        expected = ones(m, n) * B0'
        @test collect(dA1) ≈ 1.0 .+ expected
        @test collect(dA2) ≈ -2.0 .+ expected
    end

    # A `Const` output has no shadow, so nothing flows back through the matmul and
    # the rule's early return is what runs -- dA here comes only from `sum(A)`.
    @testset "Const output buffer" begin
        m, k, n = 3, 4, 2
        A0 = randn(m, k)
        function const_out(C, A, B)
            mul!(C, A, B)
            return sum(A)
        end
        dA = jl(zero(A0))
        Enzyme.autodiff(
            Reverse, const_out, Active,
            Const(jl(zeros(m, n))), Duplicated(jl(A0), dA), Const(jl(randn(k, n))),
        )
        @test all(collect(dA) .≈ 1)
    end

    #=
    The same operand plain and transposed, and both products matrix-vector: two
    passes through `generic_matvecmul!`, with tA = 'N' and then 'T'.
    =#
    @testset "matvec reverse with transpose" begin
        X0 = randn(6, 3)
        β0 = randn(3)
        function g(y, z, X, β)
            mul!(z, X, β)
            mul!(y, transpose(X), z)
            return sum(y)
        end
        dX = jl(zero(X0))
        dβ = jl(zero(β0))
        Enzyme.autodiff(
            Reverse, g, Active,
            Duplicated(jl(zeros(3)), jl(zeros(3))), Duplicated(jl(zeros(6)), jl(zeros(6))),
            Duplicated(jl(X0), dX), Duplicated(jl(β0), dβ),
        )

        gcpu(X, β) = sum(transpose(X) * (X * β))
        @test collect(dβ) ≈ fdgrad(β -> gcpu(X0, β), β0) rtol = 1.0e-4
        @test collect(dX) ≈ fdgrad(X -> gcpu(X, β0), X0) rtol = 1.0e-4
    end

    #=
    Wrapped operands, where the pullback's char algebra stops being trivial: it
    rewrites each cotangent as one more `generic_matmatmul!` with β = 1, and the
    chars it has to pick differ per combination -- 'C' is a genuine adjoint once
    the eltype is complex, and a transposed complex operand needs a materialized
    conjugate that no char can express. Enzyme's cotangent for a real-valued loss
    of complex inputs is ∂L/∂Re + i·∂L/∂Im, which central differences reproduce.
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

        function wrapped_sum(C, A, B)
            mul!(C, wa(A), wb(B))
            return real(sum(C))
        end
        dA = jl(zero(A0))
        Enzyme.autodiff(
            Reverse, wrapped_sum, Active,
            Duplicated(jl(zeros(T, m, n)), jl(zeros(T, m, n))),
            Duplicated(jl(copy(A0)), dA), Const(jl(B0)),
        )

        @test collect(dA) ≈ fdgrad(cpu, A0) rtol = 1.0e-4
    end

    #=
    5-arg `mul!`, which the tests above miss: β ≠ 0 scales dC by conj(β), and an
    Active α/β has to come back typed exactly as the scalar was (and before Julia
    1.12 the two of them arrive as one `MulAddMul`).
    =#
    @testset "mul! beta != 0 and Active alpha/beta" begin
        A0 = randn(3, 4)
        B0 = randn(4, 2)
        C0 = randn(3, 2)

        # loss = sum(2·A·B + 0.5·C₀)  ⇒  dA = 2·ones·B',  dC₀ = 0.5
        function loss(C, A, B)
            mul!(C, A, B, 2.0, 0.5)
            return sum(C)
        end
        dA = jl(zero(A0))
        dC = jl(zero(C0))
        Enzyme.autodiff(
            Reverse, loss, Active,
            Duplicated(jl(copy(C0)), dC), Duplicated(jl(A0), dA), Const(jl(B0)),
        )
        @test collect(dA) ≈ 2.0 .* (ones(3, 2) * B0')
        @test all(collect(dC) .≈ 0.5)

        # dα = sum(A·B) and dβ = sum(C₀) come back as Active scalar returns
        function lossα(C, A, B, α)
            mul!(C, A, B, α, 0.5)
            return sum(C)
        end
        outα = Enzyme.autodiff(
            Reverse, lossα, Active,
            Duplicated(jl(copy(C0)), jl(zero(C0))), Duplicated(jl(A0), jl(zero(A0))),
            Const(jl(B0)), Active(2.0),
        )
        @test outα[1][4] ≈ sum(A0 * B0)

        function lossβ(C, A, B, β)
            mul!(C, A, B, 2.0, β)
            return sum(C)
        end
        outβ = Enzyme.autodiff(
            Reverse, lossβ, Active,
            Duplicated(jl(copy(C0)), jl(zero(C0))), Duplicated(jl(A0), jl(zero(A0))),
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

    # Batched, where the return cotangent arrives as a tuple of `Active`s rather
    # than as one `Active`. The shadows start apart so this also pins that each
    # batch element accumulates into its own array.
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

    #=
    `a`'s cotangent picks up conj(dr) and `b`'s does not, which a real cotangent
    cannot tell apart -- hence the complex loss (dr = 2-3im). Ground truth is
    Enzyme on plain `Array`, which no rule here touches.
    =#
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

    @testset "sum reverse" begin
        x0 = randn(10)
        da = jl(zero(x0))
        Enzyme.autodiff(Reverse, sum, Active, Duplicated(jl(x0), da))
        @test all(collect(da) .≈ 1)
        #=
        `sum(A .* B)` leans on this rule too, but JLArrays can't differentiate the
        broadcast kernel, so that one lives in the CUDA tests.
        =#
    end

    #=
    Structured operands arrive unwrapped: Symmetric/Hermitian as an 'S'/'H' char
    to `generic_matmatmul!`, triangular ones as uploc/isunitc/tfun to
    `generic_trimatmul!`/`generic_mattrimul!`. Only stored entries are free
    parameters, so finite differences over the raw data are the check -- the
    non-stored ones have to come back 0.
    =#
    @testset "structured operand: $name" for (name, wrap) in (
            ("UpperTriangular", UpperTriangular),
            ("LowerTriangular", LowerTriangular),
            ("Symmetric(:U)", X -> Symmetric(X, :U)),
            ("Symmetric(:L)", X -> Symmetric(X, :L)),
            ("Hermitian(:U)", X -> Hermitian(X, :U)),
            ("transpose(UpperTriangular)", X -> transpose(UpperTriangular(X))),
        )
        n = 4
        X0 = randn(n, n)
        B0 = randn(n, 3)

        dX = jl(zero(X0))
        Enzyme.autodiff(
            Reverse, matmul_sum, Active,
            Duplicated(jl(zeros(n, 3)), jl(zeros(n, 3))),
            Duplicated(wrap(jl(X0)), wrap(dX)), Const(jl(B0)),
        )

        @test collect(dX) ≈ fdgrad(X -> sum(wrap(X) * B0), X0) rtol = 1.0e-5
    end

    #=
    The one combination where the projection stops being linear in α: a Hermitian
    operand sums a term and its conjugate and reads the diagonal as real, so α has
    to be folded into the cotangent before the projection, not after. It also needs
    a complex eltype to reach at all -- `wrapper_char` sends a real Hermitian as
    'S'.
    =#
    @testset "Hermitian operand, complex alpha" begin
        n, p = 3, 2
        A0 = randn(ComplexF64, n, n)
        B0 = randn(ComplexF64, n, p)
        α = 1.7 - 0.9im
        cpu(A) = real(sum(Hermitian(A, :U) * B0 * α))

        function herm_sum(C, A, B)
            mul!(C, Hermitian(A, :U), B, α, false)
            return real(sum(C))
        end
        dA = jl(zero(A0))
        Enzyme.autodiff(
            Reverse, herm_sum, Active,
            Duplicated(jl(zeros(ComplexF64, n, p)), jl(zeros(ComplexF64, n, p))),
            Duplicated(jl(copy(A0)), dA), Const(jl(B0)),
        )

        @test collect(dA) ≈ fdgrad(cpu, A0) rtol = 1.0e-4
    end

    #=
    A structured operand times a vector goes through `generic_matvecmul!` instead,
    where the projection lands on an outer-product cotangent. `:L` also checks that
    the rule reads the triangle out of the char's case ('s', not 'S').
    =#
    @testset "structured matvec: Symmetric(:L)" begin
        n = 4
        X0 = randn(n, n)
        v0 = randn(n)
        dX = jl(zero(X0))
        function matvec_sum(y, A, x)
            mul!(y, A, x)
            return sum(y)
        end
        Enzyme.autodiff(
            Reverse, matvec_sum, Active,
            Duplicated(jl(zeros(n)), jl(zeros(n))),
            Duplicated(Symmetric(jl(X0), :L), Symmetric(dX, :L)), Const(jl(v0)),
        )
        @test collect(dX) ≈ fdgrad(X -> sum(Symmetric(X, :L) * v0), X0) rtol = 1.0e-5
    end

    @testset "structured operand on the right: $name" for (name, wrap) in (
            ("UpperTriangular", UpperTriangular),
            ("transpose(UpperTriangular)", X -> transpose(UpperTriangular(X))),
            ("Symmetric(:L)", X -> Symmetric(X, :L)),
        )
        m, n = 3, 4
        A0 = randn(m, n)
        X0 = randn(n, n)
        dX = jl(zero(X0))
        Enzyme.autodiff(
            Reverse, matmul_sum, Active,
            Duplicated(jl(zeros(m, n)), jl(zeros(m, n))),
            Const(jl(A0)), Duplicated(wrap(jl(X0)), wrap(dX)),
        )
        @test collect(dX) ≈ fdgrad(X -> sum(A0 * wrap(X)), X0) rtol = 1.0e-5
    end

    #=
    `lmul!`/`rmul!` reach the triangular rules with the output aliasing the operand
    it overwrites, so the operand's cotangent has to replace the shared shadow
    rather than accumulate into it. Only cotangents are checked here, which is just
    as well -- the GPUArrays triangular kernels accumulate into their output, so
    the in-place primal is itself off by the operand on JLArray.

    Both cases wrap in `UpperTriangular` because Julia 1.11's `lmul!`/`rmul!` ask
    `istriu` first, which for a `LowerTriangular` has to read the data and so
    scalar-indexes a GPU array. The rule is oblivious to which triangle it is; the
    testsets above cover both.
    =#
    @testset "triangular in place: $name" for (name, f, cpu) in (
            (
                "lmul!", (B, X) -> (lmul!(UpperTriangular(X), B); sum(B)),
                (B, X) -> sum(UpperTriangular(X) * B),
            ),
            (
                "rmul!", (A, X) -> (rmul!(A, UpperTriangular(X)); sum(A)),
                (A, X) -> sum(A * UpperTriangular(X)),
            ),
        )
        n = 4
        M0 = randn(n, n)
        X0 = randn(n, n)
        dM = jl(zero(M0))
        dX = jl(zero(X0))
        Enzyme.autodiff(
            Reverse, f, Active, Duplicated(jl(copy(M0)), dM), Duplicated(jl(X0), dX),
        )

        @test collect(dM) ≈ fdgrad(M -> cpu(M, X0), M0) rtol = 1.0e-5
        @test collect(dX) ≈ fdgrad(X -> cpu(M0, X), X0) rtol = 1.0e-5
    end

    #=
    `A * B` is `mul!(similar(...), A, B)`, so it reaches the same rules -- but
    Enzyme never zeroes the shadow of what `similar` hands back for an
    AbstractGPUArray, and the reverse pass reads that memory instead of the
    incoming cotangent. An allocation problem, not a matmul one: it hits any rule
    whose output was freshly allocated. Doesn't reproduce on CUDA.
    =#
    @testset "allocating A * B (unzeroed shadow)" begin
        A0 = randn(3, 4)
        B0 = randn(4, 2)
        dA = jl(zero(A0))
        Enzyme.autodiff(
            Reverse, (A, B) -> sum(A * B), Active,
            Duplicated(jl(A0), dA), Const(jl(B0)),
        )
        @test_broken collect(dA) ≈ ones(3, 2) * B0'
    end
end
