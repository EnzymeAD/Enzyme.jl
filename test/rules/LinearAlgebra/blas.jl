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
        @testset "forward" begin
            @testset for T in (fun == BLAS.dot ? RTs : RCs),
                Tret in (Const, Duplicated, DuplicatedNoNeed),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                pfun in (identity, pointer),
                (sz, inc) in ((10, 1), ((2, 10), 2))

                Tx <: Const && Ty <: Const && !(Tret <: Const) && continue

                x, ∂x = ntuple(_ -> randn(T, sz), 2)
                y, ∂y = ntuple(_ -> randn(T, sz), 2)

                x_annot = Tx <: Const ? Const(pfun(x)) : Duplicated(pfun(x), pfun(∂x))
                y_annot = Ty <: Const ? Const(pfun(y)) : Duplicated(pfun(y), pfun(∂y))

                vexp = fun(n, x, inc, y, inc)
                dexp = FiniteDifferences.jvp(
                    fdm,
                    (x, y) -> fun(n, x, inc, y, inc),
                    Tx <: Const ? (x, zero(x)) : (x, ∂x),
                    Ty <: Const ? (y, zero(y)) : (y, ∂y),
                )[1]
                ret = autodiff(
                    Forward, fun, Tret, Const(n), x_annot, Const(inc), y_annot, Const(inc)
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
            fun_overwrite_x = (x, y) -> (s = dot(x, y); fill!(x, 0); s)
            fun_overwrite_y = (x, y) -> (s = dot(x, y); fill!(y, 0); s)

            @testset for T in (fun == BLAS.dot ? RTs : RCs),
                Tret in (Const, Active),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                pfun in (identity, pointer),
                (sz, inc) in ((10, 1), ((2, 10), 2)),
                f in (fun, fun_overwrite_x, fun_overwrite_y)

                Tx <: Const && Ty <: Const && !(Tret <: Const) && continue

                x, ∂x = ntuple(_ -> randn(T, sz), 2)
                y, ∂y = ntuple(_ -> randn(T, sz), 2)
                ∂z = randn(T)
                xcopy, ycopy, ∂xcopy, ∂ycopy = map(copy, (x, y, ∂x, ∂y))

                x_annot =
                    Tx <: Const ? Const(pfun(x)) : Duplicated(pfun(xcopy), pfun(∂xcopy))
                y_annot =
                    Ty <: Const ? Const(pfun(y)) : Duplicated(pfun(ycopy), pfun(∂ycopy))

                vexp = fun(n, x, inc, y, inc)
                dexp = FiniteDifferences.j′vp(
                    fdm, (x, y) -> fun(n, x, inc, y, inc), one(T), x, y
                )
                ret = autodiff(
                    ReverseWithPrimal,
                    fun,
                    Tret,
                    Const(n),
                    x_annot,
                    Const(inc),
                    y_annot,
                    Const(inc),
                )
                dval, val = ret

                @test all(isnothing, dval)
                @test val ≈ vexp
                @test ∂xcopy ≈ dexp[1] * !(Tx <: Const || Tret <: Const) + ∂x
                @test ∂ycopy ≈ dexp[2] * !(Ty <: Const || Tret <: Const) + ∂y
            end
        end
    end
end
