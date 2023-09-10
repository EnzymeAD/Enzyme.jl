using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using Test

@testset "BLAS rules" begin
    BLASReals = (Float32, Float64)
    BLASFloats = (ComplexF32, ComplexF64)
    n = 10

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
                T in (fun == BLAS.dot ? BLASReals : BLASFloats)

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
                T in (fun == BLAS.dot ? BLASReals : BLASFloats)

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
