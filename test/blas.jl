using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using Test

@testset "BLAS rules" begin
    RTs = (Float32, Float64)
    RCs = (ComplexF32, ComplexF64)
    n = 10

    @testset for fun in (BLAS.dot, BLAS.dotu, BLAS.dotc)

        # Wrap `test_reverse` and `test_forward` in `Ref` containers to be able to capture
        # and test the warnings issued by `@generated` functions.  Also, prepare the
        # expected warning message to test.
        testrev = Ref{Any}(test_reverse)
        testfwd = Ref{Any}(test_forward)
        warn_msg = fun === BLAS.dot ? "" : r"Using fallback BLAS replacements"

        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Const, Duplicated, BatchDuplicated),
                T in (fun == BLAS.dot ? RTs : RCs),
                (sz, inc) in ((10, 1), ((2, 20), -2))

                are_activities_compatible(Tret, Tx, Ty) || continue

                x = randn(T, sz)
                y = randn(T, sz)
                atol = rtol = sqrt(eps(real(T)))
                @test_warn warn_msg testfwd[](fun, Tret, n, (x, Tx), inc, (y, Ty), inc; atol, rtol)
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const, Active),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Const, Duplicated, BatchDuplicated),
                T in (fun == BLAS.dot ? RTs : RCs),
                (sz, inc) in ((10, 1), ((2, 20), -2))

                are_activities_compatible(Tret, Tx, Ty) || continue

                x = randn(T, sz)
                y = randn(T, sz)
                atol = rtol = sqrt(eps(real(T)))
                @test_warn warn_msg testrev[](fun, Tret, n, (x, Tx), inc, (y, Ty), inc; atol, rtol)
            end
        end
    end

    @testset "trmm! forward" begin
        @testset for T in (Float64, ComplexF64),
                side in ('L', 'R'), uplo in ('U', 'L'), transa in ('N', 'T', 'C'), diag in ('N', 'U'),
                Tα in (Const, Duplicated),
                TA in (Const, Duplicated, BatchDuplicated),
                TB in (Duplicated, BatchDuplicated)

            are_activities_compatible(TB, Tα, TA, TB) || continue
            m, k = 4, 3
            A = randn(T, side == 'L' ? m : k, side == 'L' ? m : k)
            B = randn(T, m, k)
            test_forward(
                BLAS.trmm!, TB, side, uplo, transa, diag, (T(0.7), Tα), (A, TA), (B, TB);
                atol = 0, rtol = sqrt(eps(real(T)))
            )
        end
    end

    @testset "trmv! forward" begin
        @testset for T in (Float64, ComplexF64),
                uplo in ('U', 'L'), trans in ('N', 'T', 'C'), diag in ('N', 'U'),
                TA in (Const, Duplicated, BatchDuplicated),
                Tx in (Duplicated, BatchDuplicated)

            are_activities_compatible(Tx, TA, Tx) || continue
            A = randn(T, 4, 4)
            x = randn(T, 4)
            test_forward(
                BLAS.trmv!, Tx, uplo, trans, diag, (A, TA), (x, Tx);
                atol = 0, rtol = sqrt(eps(real(T)))
            )
        end
    end

    @testset "trtrs! forward" begin
        @testset for T in (Float64, ComplexF64),
                uplo in ('U', 'L'), trans in ('N', 'T', 'C'), diag in ('N', 'U'),
                nrhs in (nothing, 3),
                TA in (Const, Duplicated, BatchDuplicated),
                TB in (Duplicated, BatchDuplicated)

            are_activities_compatible(TB, TA, TB) || continue
            # Well-conditioned also for diag == 'N'
            A = randn(T, 4, 4) + 4I
            B = nrhs === nothing ? randn(T, 4) : randn(T, 4, nrhs)
            test_forward(
                LAPACK.trtrs!, TB, uplo, trans, diag, (A, TA), (B, TB);
                atol = 0, rtol = sqrt(eps(real(T)))
            )
        end
    end
end

# Call BLAS directly, so that the forward rules for `BLAS.trmm!` and `BLAS.trmv!` do not apply
# and Enzyme uses the fallback BLAS implementation.
function _ccall_dtrmm!(side, uplo, transa, diag, alpha, A, B)
    m, n = size(B)
    ccall(
        (BLAS.@blasfunc(dtrmm_), BLAS.libblastrampoline), Cvoid,
        (
            Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
            Ref{Float64}, Ptr{Float64}, Ref{BLAS.BlasInt}, Ptr{Float64}, Ref{BLAS.BlasInt},
            Clong, Clong, Clong, Clong,
        ),
        side, uplo, transa, diag, m, n, alpha, A, stride(A, 2), B, stride(B, 2), 1, 1, 1, 1
    )
    return B
end
function _ccall_dtrmv!(uplo, trans, diag, A, x)
    ccall(
        (BLAS.@blasfunc(dtrmv_), BLAS.libblastrampoline), Cvoid,
        (
            Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ptr{Float64}, Ref{BLAS.BlasInt},
            Ptr{Float64}, Ref{BLAS.BlasInt}, Clong, Clong, Clong,
        ),
        uplo, trans, diag, size(A, 1), A, stride(A, 2), x, stride(x, 1), 1, 1, 1
    )
    return x
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3905, https://github.com/EnzymeAD/Enzyme.jl/issues/2618
@testset "Fallback BLAS in forward mode" begin
    A = randn(3, 3)
    B = randn(3, 2)
    x = randn(3)
    rtol = sqrt(eps(Float64))
    @test_warn r"Using fallback BLAS replacements" test_forward(
        _ccall_dtrmm!, Duplicated, 'L', 'U', 'T', 'U', (0.7, Duplicated), (A, Duplicated), (B, Duplicated);
        atol = 0, rtol
    )
    @test_warn r"Using fallback BLAS replacements" test_forward(
        _ccall_dtrmv!, Duplicated, 'L', 'N', 'N', (A, Duplicated), (x, Duplicated);
        atol = 0, rtol
    )
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3905
@testset "Matrix(::Cholesky)" begin
    U = [1.4 0.3 0.0; 0.0 1.2 0.2; 0.0 0.0 1.1]
    g(x) = sum(Matrix(Cholesky(reshape(x, 3, 3), 'U', 0)))
    # The forward rule for `BLAS.trmm!` is used, without a fallback BLAS warning
    dx = @test_nowarn Enzyme.gradient(Forward, g, vec(U))[1]
    @test dx ≈ Enzyme.gradient(Reverse, g, vec(U))[1]
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/1991
@testset "Triangular solve in forward mode" begin
    x, L, µ = [0.6, 0.98], [1.0 0.5; 0.5 1.0], [-0.79, 0.47]
    f(x, L, µ) = LowerTriangular(L) \ (x .- µ)
    # `\` calls `LAPACK.trtrs!`, for which the forward rule is used
    J = @test_nowarn Enzyme.gradient(Forward, f, x, Const(L), Const(µ))[1]
    @test J ≈ inv(LowerTriangular(L))
end
