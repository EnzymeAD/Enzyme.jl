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
                Tx in (Duplicated, BatchDuplicated),
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
end
