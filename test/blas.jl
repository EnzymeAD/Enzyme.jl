using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using MetaTesting
using Test

discard(_) = nothing

@testset "BLAS rules" begin
    BLASReals = (Float32, Float64)
    BLASComplexes = (ComplexF32, ComplexF64)
    BLASFloats = (BLASReals..., BLASComplexes...)

    @testset "BLAS.scal!" begin
        n = 10
        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Ta in (Const, Duplicated, BatchDuplicated),
                Tx in (Duplicated, BatchDuplicated),
                T in BLASFloats

                are_activities_compatible(Tret, Ta, Tx) || continue
                atol = rtol = sqrt(eps(real(T)))

                @testset "BLAS.scal!(n, a, x, incx)" begin
                    @testset for (sz, inc) in ((n, 1), ((2, 2n), -2))
                        a = randn(T)
                        x = randn(T, sz)
                        @test !fails() do
                            test_forward(
                                BLAS.scal!, Tret, n, (a, Ta), (x, Tx), inc; atol, rtol
                            )
                        end broken = (T <: ComplexF32 && !(Ta <: Const) && !(Tx <: Const))
                    end
                end
                @testset "BLAS.scal!(a, x)" begin
                    a = randn(T)
                    x = randn(T, n)
                    @test !fails() do
                        test_forward(BLAS.scal!, Tret, (a, Ta), (x, Tx); atol, rtol)
                    end broken = (T <: ComplexF32 && !(Ta <: Const) && !(Tx <: Const))
                end
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const,),
                Ta in (Const, Active),
                Tx in (Const, Duplicated, BatchDuplicated),
                T in BLASFloats

                are_activities_compatible(Tret, Ta, Tx) || continue
                atol = rtol = sqrt(eps(real(T)))

                if T <: Complex && Ta <: Active && Tx <: BatchDuplicated
                    # avoid failure that crashes Julia
                    @test false skip = true
                    continue
                end

                @testset "BLAS.scal!(n, a, x, incx)" begin
                    @testset for (sz, inc) in ((n, 1), ((2, 2n), -2))
                        a = randn(T)
                        x = randn(T, sz)
                        @test !fails() do
                            test_reverse(
                                discard ∘ BLAS.scal!,
                                Tret,
                                n,
                                (a, Ta),
                                (x, Tx),
                                inc;
                                atol,
                                rtol,
                            )
                        end broken = (Tx <: BatchDuplicated && sz isa Int)
                    end
                end

                @testset "BLAS.scal!(a, x)" begin
                    a = randn(T)
                    x = randn(T, n)
                    @test !fails() do
                        test_reverse(
                            discard ∘ BLAS.scal!, Tret, (a, Ta), (x, Tx); atol, rtol
                        )
                    end broken = (Tx <: BatchDuplicated)
                end
            end
        end
    end

    @testset for fun in (BLAS.dot, BLAS.dotu, BLAS.dotc)
        n = 10
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
                T in (fun == BLAS.dot ? BLASReals : BLASComplexes)

                are_activities_compatible(Tret, Tx, Ty) || continue
                atol = rtol = sqrt(eps(real(T)))

                @testset "$fun(n, x, incx, y, incy)" begin
                    @testset for (sz, inc) in ((n, 1), ((2, 2n), -2))
                        x = randn(T, sz)
                        y = randn(T, sz)
                        test_forward(fun, Tret, n, (x, Tx), inc, (y, Ty), inc; atol, rtol)
                    end
                end

                @testset "$fun(x, y)" begin
                    x = randn(T, n)
                    y = randn(T, n)
                    test_forward(fun, Tret, (x, Tx), (y, Ty); atol, rtol)
                end
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const, Active),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Const, Duplicated, BatchDuplicated),
                T in (fun == BLAS.dot ? BLASReals : BLASComplexes)

                are_activities_compatible(Tret, Tx, Ty) || continue
                atol = rtol = sqrt(eps(real(T)))

                @testset "$fun(n, x, incx, y, incy)" begin
                    @testset for (sz, inc) in ((n, 1), ((2, 2n), -2))
                        x = randn(T, sz)
                        y = randn(T, sz)
                        test_reverse(fun, Tret, n, (x, Tx), inc, (y, Ty), inc; atol, rtol)
                    end
                end

                @testset "$fun(x, y)" begin
                    x = randn(T, n)
                    y = randn(T, n)
                    test_reverse(fun, Tret, (x, Tx), (y, Ty); atol, rtol)
                end
            end
        end
    end

    @testset "BLAS.axpy!" begin
        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Ta in (Const, Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Ta, Tx, Ty) || continue

                @testset for T in BLASFloats, sz in (10, (2, 5), (3, 4, 5))
                    a = randn(T)
                    x = randn(T, sz)
                    y = randn(T, sz)
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_forward(
                            BLAS.axpy!, Tret, (a, Ta), (x, Tx), (y, Ty); atol, rtol
                        )
                    end broken = (T <: ComplexF32 && !(Ta <: Const) && !(Ty <: Const))
                end
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const,),
                Ta in (Const, Active),
                Tx in (Const, Duplicated, BatchDuplicated),
                Ty in (Const, Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Ta, Tx, Ty) || continue

                @testset for T in BLASFloats, sz in (10, (2, 5), (3, 4, 5))
                    a = randn(T)
                    x = randn(T, sz)
                    y = randn(T, sz)
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_reverse(
                            discard ∘ BLAS.axpy!,
                            Tret,
                            (a, Ta),
                            (x, Tx),
                            (y, Ty);
                            atol,
                            rtol,
                        )
                    end
                end
            end
        end
    end

    @testset "BLAS.gemv!" begin
        sz = (2, 3)
        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Talpha in (Const, Duplicated, BatchDuplicated),
                TA in (Const, Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated),
                Tbeta in (Const, Duplicated, BatchDuplicated),
                Ty in (Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Talpha, TA, Tx, Tbeta, Ty) || continue

                @testset for T in BLASFloats, t in ('N', 'T', 'C')
                    alpha, beta = randn(T, 2)
                    A = t === 'N' ? randn(T, sz...) : randn(T, reverse(sz)...)
                    x = randn(T, sz[2])
                    y = randn(T, sz[1])
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_forward(
                            BLAS.gemv!,
                            Tret,
                            t,
                            (alpha, Talpha),
                            (A, TA),
                            (x, Tx),
                            (beta, Tbeta),
                            (y, Ty);
                            atol,
                            rtol,
                        )
                    end broken = (
                        T <: ComplexF32 &&
                        !(Ty <: Const) &&
                        !(Talpha <: Const && Tbeta <: Const)
                    )
                end
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const,),
                Talpha in (Const, Active),
                TA in (Const, Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated),
                Tbeta in (Const, Active),
                Ty in (Const, Duplicated, BatchDuplicated),
                T in BLASFloats

                are_activities_compatible(Tret, Talpha, TA, Tx, Tbeta, Ty) || continue

                if T <: Complex && any(Base.Fix2(<:, BatchDuplicated), (TA, Tx, Ty))
                    # avoid failure that crashes Julia
                    @test false skip = true
                    continue
                end

                @testset for t in ('N', 'T', 'C')
                    alpha, beta = randn(T, 2)
                    A = t === 'N' ? randn(T, sz...) : randn(T, reverse(sz)...)
                    x = randn(T, sz[2])
                    y = randn(T, sz[1])
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_reverse(
                            discard ∘ BLAS.gemv!,
                            Tret,
                            t,
                            (alpha, Talpha),
                            (A, TA),
                            (x, Tx),
                            (beta, Tbeta),
                            (y, Ty);
                            atol,
                            rtol,
                        )
                    end broken = any(Base.Fix2(<:, BatchDuplicated), (Tx, Ty))
                end
            end
        end
    end

    @testset "BLAS.spmv!" begin
        n = 5
        m = div(n * (n + 1), 2)
        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Talpha in (Const, Duplicated, BatchDuplicated),
                TAP in (Const, Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated),
                Tbeta in (Const, Duplicated, BatchDuplicated),
                Ty in (Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Talpha, TAP, Tx, Tbeta, Ty) || continue

                @testset for T in BLASReals, uplo in ('U', 'L')
                    alpha, beta = randn(T, 2)
                    AP = randn(T, m)
                    x = randn(T, n)
                    y = randn(T, n)
                    atol = rtol = sqrt(eps(real(T)))
                    test_forward(
                        BLAS.spmv!,
                        Tret,
                        uplo,
                        (alpha, Talpha),
                        (AP, TAP),
                        (x, Tx),
                        (beta, Tbeta),
                        (y, Ty);
                        atol,
                        rtol,
                    )
                end
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const,),
                Talpha in (Const, Active),
                TAP in (Const, Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated),
                Tbeta in (Const, Active),
                Ty in (Const, Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Talpha, TAP, Tx, Tbeta, Ty) || continue

                @testset for T in BLASReals, uplo in ('U', 'L')
                    alpha, beta = randn(T, 2)
                    AP = randn(T, m)
                    x = randn(T, n)
                    y = randn(T, n)
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_reverse(
                            discard ∘ BLAS.spmv!,
                            Tret,
                            uplo,
                            (alpha, Talpha),
                            (AP, TAP),
                            (x, Tx),
                            (beta, Tbeta),
                            (y, Ty);
                            atol,
                            rtol,
                        )
                    end broken = TAP <: BatchDuplicated
                end
            end
        end
    end

    @testset "BLAS.gemm!" begin
        szA = (2, 3)
        szB = (3, 4)
        @testset "forward" begin
            @testset for Tret in (
                    Const,
                    Duplicated,
                    DuplicatedNoNeed,
                    BatchDuplicated,
                    BatchDuplicatedNoNeed,
                ),
                Talpha in (Const, Duplicated, BatchDuplicated),
                TA in (Const, Duplicated, BatchDuplicated),
                TB in (Const, Duplicated, BatchDuplicated),
                Tbeta in (Const, Duplicated, BatchDuplicated),
                TC in (Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Talpha, TA, TB, Tbeta, TC) || continue

                @testset for T in BLASFloats, tA in ('N', 'T', 'C'), tB in ('N', 'T', 'C')
                    alpha, beta = randn(T, 2)
                    A = tA === 'N' ? randn(T, szA...) : randn(T, reverse(szA)...)
                    B = tB === 'N' ? randn(T, szB...) : randn(T, reverse(szB)...)
                    C = randn(T, (szA[1], szB[2]))
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_forward(
                            BLAS.gemm!,
                            Tret,
                            tA,
                            tB,
                            (alpha, Talpha),
                            (A, TA),
                            (B, TB),
                            (beta, Tbeta),
                            (C, TC);
                            atol,
                            rtol,
                        )
                    end broken = T <: ComplexF32 && !(Talpha <: Const && Tbeta <: Const)
                end
            end
        end

        @testset "reverse" begin
            @testset for Tret in (Const,),
                Talpha in (Const, Active),
                TA in (Const, Duplicated, BatchDuplicated),
                TB in (Const, Duplicated, BatchDuplicated),
                Tbeta in (Const, Active),
                TC in (Duplicated, BatchDuplicated),
                T in BLASFloats

                are_activities_compatible(Tret, Talpha, TA, TB, Tbeta, TC) || continue

                if T <: Complex && any(Base.Fix2(<:, BatchDuplicated), (TA, TB, TC))
                    # avoid failure that crashes Julia
                    @test false skip = true
                    continue
                end

                @testset for tA in ('N', 'T', 'C'), tB in ('N', 'T', 'C')
                    alpha, beta = randn(T, 2)
                    A = tA === 'N' ? randn(T, szA...) : randn(T, reverse(szA)...)
                    B = tB === 'N' ? randn(T, szB...) : randn(T, reverse(szB)...)
                    C = randn(T, (szA[1], szB[2]))
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_reverse(
                            discard ∘ BLAS.gemm!,
                            Tret,
                            tA,
                            tB,
                            (alpha, Talpha),
                            (A, TA),
                            (B, TB),
                            (beta, Tbeta),
                            (C, TC);
                            atol,
                            rtol,
                        )
                    end broken = (
                        T <: Complex && any(Base.Fix2(<:, BatchDuplicated), (TB, TC))
                    )
                end
            end
        end
    end
end
