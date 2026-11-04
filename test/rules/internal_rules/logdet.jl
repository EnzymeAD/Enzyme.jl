using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using Test

stored_logdet(A, W, uplo) = logdet(W(A, uplo))
gram_logdet(A) = logdet(I + Symmetric(A' * A))
scaled_gram_logdet(U, D) = logdet(I + Symmetric(U' * (U ./ D .^ 2)))
view_logdet(A, W, uplo) = logdet(W(view(A, 1:2:4, 1:2:4), uplo))
selected_logdet(A, B, choose) = logdet(choose ? A : B)
function overwrite_logdet!(A)
    y = logdet(A)
    fill!(parent(A), 0)
    return y
end

@testset "Dense structured logdet" begin
    @testset "$W $T $uplo" for (W, T) in ((Symmetric, Float64), (Hermitian, Float64), (Hermitian, ComplexF64)),
            uplo in (:U, :L)
        # Deliberately different triangles: only the one selected by uplo is read.
        M = T[4 1; 0.3 3]
        if T <: Complex
            M[1, 2] += 0.2im
            M[2, 1] += 0.1im
            M[1, 1] += 0.3im
        end
        A = W(M, uplo)
        for RT in (Const, Duplicated, DuplicatedNoNeed), AT in (Const, Duplicated)
            test_forward(logdet, RT, (A, AT))
        end
        for RT in (Const, BatchDuplicated, BatchDuplicatedNoNeed),
                AT in (Const, BatchDuplicated)
            test_forward(logdet, RT, (A, AT))
        end
        for RT in (Const, Active), AT in (Const, Duplicated, BatchDuplicated)
            test_reverse(logdet, RT, (A, AT))
        end

        # Check the backing storage too: the unused triangle and imaginary parts
        # of Hermitian diagonals must receive no contribution.
        for AT in (Duplicated, BatchDuplicated)
            test_reverse(stored_logdet, Active, (M, AT), Const(W), Const(uplo))
        end
    end
end

@testset "Logdet views and overwritten arguments" begin
    M = [4.0 0.2 1.0 0.4; 0.5 6.0 0.7 0.8; 0.3 1.0 3.0 1.2; 1.3 1.4 1.5 7.0]
    for W in (Symmetric, Hermitian), uplo in (:U, :L)
        test_forward(view_logdet, Duplicated, (M, Duplicated), Const(W), Const(uplo))
        test_reverse(view_logdet, Active, (M, Duplicated), Const(W), Const(uplo))
        test_reverse(overwrite_logdet!, Active, (W(copy(M), uplo), Duplicated))
    end
end

@testset "Low-rank Gaussian entropy logdet" begin
    for T in (Float32, Float64)
        A = T[0.3 -0.2; 0.1 0.4; -0.5 0.2]
        expected = 2A * inv(I + A' * A)
        @test Enzyme.gradient(Reverse, gram_logdet, A)[1] ≈ expected
        @test Enzyme.gradient(Forward, gram_logdet, A)[1] ≈ expected
        dA = T[0.1 0.2; 0.3 0.4; 0.5 0.6]
        @test only(autodiff(Forward, gram_logdet, Duplicated(A, dA))) ≈ dot(expected, dA)
    end
    U = [0.3 -0.2; 0.1 0.4; -0.5 0.2]
    D = [0.8, 1.1, 0.9]
    test_reverse(scaled_gram_logdet, Active, (U, Duplicated), (D, Duplicated))
    test_forward(scaled_gram_logdet, Duplicated, (U, Duplicated), (D, Duplicated))
end

@testset "Batched dense logdet" begin
    A = [4.0 0.3; 1.0 3.0]
    for RT in (BatchDuplicated, BatchDuplicatedNoNeed)
        test_forward(logdet, RT, (A, BatchDuplicated))
    end
    test_reverse(logdet, Active, (A, BatchDuplicated))
end

@testset "Runtime-inactive logdet argument" begin
    for W in (Symmetric, Hermitian), choose in (false, true)
        A = W([4.0 1.0; 1.0 3.0])
        B = W([5.0 0.5; 0.5 2.0])
        test_forward(selected_logdet, Duplicated, (A, Duplicated), Const(B), Const(choose); runtime_activity = true)
        test_reverse(selected_logdet, Active, (A, Duplicated), Const(B), Const(choose); runtime_activity = true)
    end
end
