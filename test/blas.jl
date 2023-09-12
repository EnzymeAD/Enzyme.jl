using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using MetaTesting
using Test

@testset "BLAS rules" begin
    BLASReals = (Float32, Float64)
    BLASComplexes = (ComplexF32, ComplexF64)
    BLASFloats = (BLASReals..., BLASComplexes...)
    n = 10

    @testset "BLAS.scal!" begin
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
                    @testset for (sz, inc) in ((10, 1), ((2, 20), -2))
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
                    @testset for (sz, inc) in ((10, 1), ((2, 20), -2))
                        a = randn(T)
                        x = randn(T, sz)
                        @test !fails() do
                            test_reverse(
                                Tret, n, (a, Ta), (x, Tx), inc; atol, rtol
                            ) do n, a, x, inc
                                BLAS.scal!(n, a, x, inc)
                                return nothing
                            end
                        end broken = (Tx <: BatchDuplicated && sz isa Int)
                    end
                end

                @testset "BLAS.scal!(a, x)" begin
                    a = randn(T)
                    x = randn(T, n)
                    @test !fails() do
                        test_reverse(Tret, (a, Ta), (x, Tx); atol, rtol) do a, x
                            BLAS.scal!(a, x)
                            return nothing
                        end
                    end broken = (Tx <: BatchDuplicated)
                end
            end
        end
    end

    @testset for fun in (BLAS.dot, BLAS.dotu, BLAS.dotc)
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
                    @testset for (sz, inc) in ((10, 1), ((2, 20), -2))
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
                    @testset for (sz, inc) in ((10, 1), ((2, 20), -2))
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
                        test_reverse(Tret, (a, Ta), (x, Tx), (y, Ty); atol, rtol) do a, x, y
                            BLAS.axpy!(a, x, y)
                            return nothing
                        end
                    end broken = (
                        T <: ComplexF32 &&
                        xor(Tx <: BatchDuplicated, Ty <: BatchDuplicated) &&
                        !(Ta <: Const) &&
                        sz isa Int
                    )
                end
            end
        end
    end

    @testset "BLAS.gemv!" begin
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
                    sz = (2, 3)
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
                    sz = (2, 3)
                    alpha, beta = randn(T, 2)
                    A = t === 'N' ? randn(T, sz...) : randn(T, reverse(sz)...)
                    x = randn(T, sz[2])
                    y = randn(T, sz[1])
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_reverse(
                            Tret,
                            t,
                            (alpha, Talpha),
                            (A, TA),
                            (x, Tx),
                            (beta, Tbeta),
                            (y, Ty);
                            atol,
                            rtol,
                        ) do args...
                            BLAS.gemv!(args...)
                            return nothing
                        end
                    end broken = any(Base.Fix2(<:, BatchDuplicated), (Tx, Ty))
                end
            end
        end
    end
end
