using Enzyme, Test, JLArrays
using LinearAlgebra: mul!, dot

function jlres(x)
    2 * collect(x)
end

@testset "JLArrays" begin
    # TODO fix activity of jlarray
    # Enzyme.jacobian(Forward, jlres, JLArray([3.0, 5.0]))
    # Enzyme.jacobian(Reverse, jlres, JLArray([3.0, 5.0]))
end

# AbstractGPUArray linear-algebra rules (matmul / dot / sum). JLArray is a
# CPU-backed `AbstractGPUArray`, so it exercises exactly the dispatch the
# CUDA/AMDGPU/Metal backends hit, without needing a device.
@testset "GPUArrays linalg rules" begin
    jl(x) = JLArray(x)

    @testset "matmul reverse ($m×$k × $k×$n)" for (m, k, n) in ((3, 4, 2), (5, 5, 1))
        A0 = randn(m, k)
        B0 = randn(k, n)

        # loss = sum(A*B); analytic grads dA = ones*B', dB = A'*ones
        f(A, B) = sum(A * B)
        dA = jl(zero(A0))
        dB = jl(zero(B0))
        Enzyme.autodiff(Reverse, f, Active, Duplicated(jl(A0), dA), Duplicated(jl(B0), dB))

        ones_mn = ones(m, n)
        @test collect(dA) ≈ ones_mn * B0'
        @test collect(dB) ≈ A0' * ones_mn
    end

    @testset "matmul reverse with transpose" begin
        # mirrors pmcmc_matmul(transpose(X), X*β): a composed/wrapped matmul
        X0 = randn(6, 3)
        β0 = randn(3)
        g(X, β) = sum(transpose(X) * (X * β))
        dX = jl(zero(X0))
        dβ = jl(zero(β0))
        Enzyme.autodiff(Reverse, g, Active, Duplicated(jl(X0), dX), Duplicated(jl(β0), dβ))

        # finite-difference check against the CPU primal
        gcpu(X, β) = sum(transpose(X) * (X * β))
        ϵ = 1.0e-6
        fdβ = map(eachindex(β0)) do i
            βp = copy(β0); βp[i] += ϵ
            βm = copy(β0); βm[i] -= ϵ
            (gcpu(X0, βp) - gcpu(X0, βm)) / (2ϵ)
        end
        @test collect(dβ) ≈ fdβ rtol = 1.0e-4
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

    @testset "sum reverse" begin
        x0 = randn(10)
        da = jl(zero(x0))
        Enzyme.autodiff(Reverse, sum, Active, Duplicated(jl(x0), da))
        @test all(collect(da) .≈ 1)
        # Note: `sum(A .* B)` (reduction over a broadcast) also relies on this
        # rule, but JLArrays cannot differentiate the broadcast kernel itself,
        # so that pattern is covered in the CUDA test instead.
    end

    @testset "matmul forward" begin
        A0 = randn(3, 4)
        B0 = randn(4, 2)
        dA = randn(3, 4)
        dB = randn(4, 2)
        # d(A*B) = dA*B + A*dB
        out = Enzyme.autodiff(
            Forward, (A, B) -> A * B, Duplicated,
            Duplicated(jl(A0), jl(dA)), Duplicated(jl(B0), jl(dB)),
        )
        @test collect(out[1]) ≈ dA * B0 + A0 * dB
    end
end
