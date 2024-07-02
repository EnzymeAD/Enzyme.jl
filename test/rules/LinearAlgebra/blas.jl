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
            @testset for Tret in (Const, Duplicated, DuplicatedNoNeed),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                pfun in (identity, pointer),
                T in (fun == BLAS.dot ? RTs : RCs),
                (sz, inc) in ((10, 1), ((2, 20), -2))

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

                if pfun === identity && sz == n && inc == 1
                    @testset "consistency of 2-arg version" begin
                        ret2 = autodiff(Forward, fun, Tret, x_annot, y_annot)
                        @test ret2 == ret
                    end
                end
            end

            @testset for Tret in (BatchDuplicated, BatchDuplicatedNoNeed),
                T in (fun == BLAS.dot ? RTs : RCs)

                batch_size = 3
                inc = 1
                x = randn(T, n)
                y = randn(T, n)
                ∂xs = ntuple(_ -> randn(T, n), batch_size)
                ∂ys = ntuple(_ -> randn(T, n), batch_size)
                vexp = fun(n, x, inc, y, inc)
                dexp = map(∂xs, ∂ys) do ∂x, ∂y
                    FiniteDifferences.jvp(
                        fdm, (x, y) -> fun(n, x, inc, y, inc), (x, ∂x), (y, ∂y)
                    )[1]
                end
                ret = autodiff(
                    Forward,
                    fun,
                    Tret,
                    Const(n),
                    BatchDuplicated(x, ∂xs),
                    Const(inc),
                    BatchDuplicated(y, ∂ys),
                    Const(inc),
                )
                if Tret <: BatchDuplicated
                    v, ds = ret
                    @test v ≈ vexp
                else
                    ds = only(ret)
                end
                @test all(map(≈, values(ds), dexp))
            end
        end

        @testset "reverse" begin
            function fun_overwrite!(n, x, incx, y, incy)
                d = fun(n, x, incx, y, incy)
                x[1] = 0
                y[1] = 0
                return d
            end
            function fun_overwrite!(x, y)
                d = fun(x, y)
                x[1] = 0
                y[1] = 0
                return d
            end

            @testset for Tret in (Const, Active),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                pfun in (identity, pointer),
                T in (fun == BLAS.dot ? RTs : RCs),
                (sz, inc) in ((10, 1), ((2, 20), -2)),
                f in (pfun === identity ? (fun, fun_overwrite!) : (fun,))

                Tx <: Const && Ty <: Const && !(Tret <: Const) && continue

                x, ∂x = ntuple(_ -> randn(T, sz), 2)
                y, ∂y = ntuple(_ -> randn(T, sz), 2)
                ∂z = randn(T)
                xcopy, ycopy, ∂xcopy, ∂ycopy = map(copy, (x, y, ∂x, ∂y))

                x_annot =
                    Tx <: Const ? Const(pfun(xcopy)) : Duplicated(pfun(xcopy), pfun(∂xcopy))
                y_annot =
                    Ty <: Const ? Const(pfun(ycopy)) : Duplicated(pfun(ycopy), pfun(∂ycopy))
                activities = (Const(n), x_annot, Const(inc), y_annot, Const(inc))

                vexp = fun(n, x, inc, y, inc)
                dret = randn(typeof(vexp))

                dexp = FiniteDifferences.j′vp(
                    fdm, (x, y) -> fun(n, x, inc, y, inc), dret, x, y
                )
                fwd, rev = autodiff_thunk(
                    ReverseSplitWithPrimal,
                    Const{typeof(f)},
                    Tret,
                    map(typeof, activities)...,
                )
                tape, val, shadow_val = fwd(Const(f), activities...)
                if Tret <: Const
                    dval, = rev(Const(f), activities..., tape)
                else
                    dval, = rev(Const(f), activities..., dret, tape)
                end

                @test all(isnothing, dval)
                @test val ≈ vexp
                @test ∂xcopy ≈
                    dexp[1] .* !(Tx <: Const || Tret <: Const) .+
                      ∂x .* ((Tx <: Const) .| (x .== xcopy))
                @test ∂ycopy ≈
                    dexp[2] .* !(Ty <: Const || Tret <: Const) .+
                      ∂y .* ((Ty <: Const) .| (y .== ycopy))

                if pfun === identity && sz == n && inc == 1
                    @testset "consistency of 2-arg version" begin
                        xcopy2, ycopy2, ∂xcopy2, ∂ycopy2 = map(copy, (x, y, ∂x, ∂y))
                        x_annot = if Tx <: Const
                            Const(pfun(xcopy2))
                        else
                            Duplicated(pfun(xcopy2), pfun(∂xcopy2))
                        end
                        y_annot = if Ty <: Const
                            Const(pfun(ycopy2))
                        else
                            Duplicated(pfun(ycopy2), pfun(∂ycopy2))
                        end
                        activities = (x_annot, y_annot)
                        fwd, rev = autodiff_thunk(
                            ReverseSplitWithPrimal,
                            Const{typeof(f)},
                            Tret,
                            map(typeof, activities)...,
                        )
                        tape, val2, shadow_val = fwd(Const(f), activities...)
                        if Tret <: Const
                            dval2, = rev(Const(f), activities..., tape)
                        else
                            dval2, = rev(Const(f), activities..., dret, tape)
                        end
                        @test all(isnothing, dval2)
                        @test val2 == val
                        @test ∂xcopy2 == ∂xcopy
                        @test ∂ycopy2 == ∂ycopy
                    end
                end
            end
        end
    end
end
