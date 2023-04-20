using Enzyme
using FiniteDifferences
using LinearAlgebra
using Test

@testset "BLAS rules" begin
    fdm = central_fdm(5, 1)
    RTs = (Float32, Float64)
    RCs = (ComplexF32, ComplexF64)
    n = 10

    @testset for fun in (BLAS.dot, BLAS.dotu, BLAS.dotc)
        @testset for T in (fun == BLAS.dot ? RTs : RCs),
            (sx, incx) in ((10, 1), ((2, 10), 2)), (sy, incy) in ((10, 1), ((2, 10), 2))

            @testset "forward" begin
                @testset for Tx in (Duplicated, Const),
                    Ty in (Duplicated, Const),
                    Tret in (Const, Duplicated, DuplicatedNoNeed),
                    pfun in (identity, pointer)

                    Tx <: Const && Ty <: Const && !(Tret <: Const) && continue

                    x, ∂x = ntuple(_ -> randn(T, sx), 2)
                    y, ∂y = ntuple(_ -> randn(T, sy), 2)

                    x_annot = Tx <: Const ? Const(pfun(x)) : Duplicated(pfun(x), pfun(∂x))
                    y_annot = Ty <: Const ? Const(pfun(y)) : Duplicated(pfun(y), pfun(∂y))

                    vexp = fun(n, x, incx, y, incy)
                    dexp = FiniteDifferences.jvp(
                        fdm,
                        (x, y) -> fun(n, x, incx, y, incy),
                        Tx <: Const ? (x, zero(x)) : (x, ∂x),
                        Ty <: Const ? (y, zero(y)) : (y, ∂y),
                    )[1]
                    ret = autodiff(
                        Forward,
                        fun,
                        Tret,
                        Const(n),
                        x_annot,
                        Const(incx),
                        y_annot,
                        Const(incy),
                    )

                    Tret <: Const && @test ret === ()
                    if Tret <: Duplicated
                        v, d = ret
                        @test v ≈ vexp
                        @test d ≈ dexp
                    elseif Tret <: DuplicatedNoNeed
                        @test only(ret) ≈ dexp
                    end
                end
            end

            @testset "reverse" begin
                @testset for Tx in (Duplicated, Const),
                    Ty in (Duplicated, Const),
                    Tret in (Const, Active),
                    pfun in (identity, pointer)

                    Tx <: Const && Ty <: Const && !(Tret <: Const) && continue

                    x, ∂x = ntuple(_ -> randn(T, sx), 2)
                    y, ∂y = ntuple(_ -> randn(T, sy), 2)
                    ∂z = randn(T)
                    ∂xcopy = copy(∂x)
                    ∂ycopy = copy(∂y)

                    x_annot =
                        Tx <: Const ? Const(pfun(x)) : Duplicated(pfun(x), pfun(∂xcopy))
                    y_annot =
                        Ty <: Const ? Const(pfun(y)) : Duplicated(pfun(y), pfun(∂ycopy))

                    dexp = FiniteDifferences.j′vp(
                        fdm, (x, y) -> fun(n, x, incx, y, incy), one(T), x, y
                    )
                    ret = autodiff(
                        ReverseWithPrimal,
                        fun,
                        Tret,
                        Const(n),
                        x_annot,
                        Const(incx),
                        y_annot,
                        Const(incy),
                    )
                    dval, val = ret

                    @test all(isnothing, dval)
                    @test val ≈ fun(n, x, incx, y, incy)
                    @test ∂xcopy ≈ dexp[1] * !(Tx <: Const || Tret <: Const) + ∂x
                    @test ∂ycopy ≈ dexp[2] * !(Ty <: Const || Tret <: Const) + ∂y
                end
            end
        end
    end
end
