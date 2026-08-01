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

        # finite-difference check against the CPU primal
        gcpu(X, β) = sum(transpose(X) * (X * β))
        ϵ = 1.0e-6
        fdβ = map(eachindex(β0)) do i
            βp = copy(β0); βp[i] += ϵ
            βm = copy(β0); βm[i] -= ϵ
            (gcpu(X0, βp) - gcpu(X0, βm)) / (2ϵ)
        end
        @test collect(dβ) ≈ fdβ rtol = 1.0e-4
        fdX = map(eachindex(X0)) do i
            Xp = copy(X0); Xp[i] += ϵ
            Xm = copy(X0); Xm[i] -= ϵ
            (gcpu(Xp, β0) - gcpu(Xm, β0)) / (2ϵ)
        end
        @test collect(dX) ≈ reshape(fdX, size(X0)) rtol = 1.0e-4
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

        ϵ = 1.0e-6
        fd = zero(A0)
        for idx in eachindex(A0)
            Ap = copy(A0); Ap[idx] += ϵ
            Am = copy(A0); Am[idx] -= ϵ
            g = (cpu(Ap) - cpu(Am)) / (2ϵ)
            if T <: Complex
                Aip = copy(A0); Aip[idx] += im * ϵ
                Aim = copy(A0); Aim[idx] -= im * ϵ
                g += im * (cpu(Aip) - cpu(Aim)) / (2ϵ)
            end
            fd[idx] = g
        end
        @test collect(dA) ≈ fd rtol = 1.0e-4
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

        ϵ = 1.0e-6
        fd = zero(X0)
        for idx in eachindex(X0)
            Xp = copy(X0); Xp[idx] += ϵ
            Xm = copy(X0); Xm[idx] -= ϵ
            fd[idx] = (sum(wrap(Xp) * B0) - sum(wrap(Xm) * B0)) / (2ϵ)
        end
        @test collect(dX) ≈ fd rtol = 1.0e-5
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
        ϵ = 1.0e-6
        fd = zero(X0)
        for idx in eachindex(X0)
            Xp = copy(X0); Xp[idx] += ϵ
            Xm = copy(X0); Xm[idx] -= ϵ
            fd[idx] = (sum(Symmetric(Xp, :L) * v0) - sum(Symmetric(Xm, :L) * v0)) / (2ϵ)
        end
        @test collect(dX) ≈ fd rtol = 1.0e-5
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
        ϵ = 1.0e-6
        fd = zero(X0)
        for idx in eachindex(X0)
            Xp = copy(X0); Xp[idx] += ϵ
            Xm = copy(X0); Xm[idx] -= ϵ
            fd[idx] = (sum(A0 * wrap(Xp)) - sum(A0 * wrap(Xm))) / (2ϵ)
        end
        @test collect(dX) ≈ fd rtol = 1.0e-5
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

        ϵ = 1.0e-6
        fdM = zero(M0)
        for idx in eachindex(M0)
            Mp = copy(M0); Mp[idx] += ϵ
            Mm = copy(M0); Mm[idx] -= ϵ
            fdM[idx] = (cpu(Mp, X0) - cpu(Mm, X0)) / (2ϵ)
        end
        fdX = zero(X0)
        for idx in eachindex(X0)
            Xp = copy(X0); Xp[idx] += ϵ
            Xm = copy(X0); Xm[idx] -= ϵ
            fdX[idx] = (cpu(M0, Xp) - cpu(M0, Xm)) / (2ϵ)
        end
        @test collect(dM) ≈ fdM rtol = 1.0e-5
        @test collect(dX) ≈ fdX rtol = 1.0e-5
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
