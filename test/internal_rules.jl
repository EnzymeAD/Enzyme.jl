module InternalRules

using Enzyme
using Enzyme.EnzymeRules
using EnzymeTestUtils
using FiniteDifferences
using LinearAlgebra
using SparseArrays
using Test
import Random

struct TPair
    a::Float64
    b::Float64
end

function sorterrfn(t, x)
    function lt(a, b)
        return a.a < b.a
    end
    return first(sortperm(t; lt=lt)) * x
end

@testset "Sort rules" begin
    function f1(x)
        a = [1.0, 3.0, x]
        sort!(a)
        return a[2]
    end

    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1] == 1
    @test autodiff(Forward, f1, BatchDuplicated(2.0, (1.0, 2.0)))[1] ==
        (var"1"=1.0, var"2"=2.0)
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 1
    @test autodiff(Forward, f1, Duplicated(4.0, 1.0))[1] == 0
    @test autodiff(Forward, f1, BatchDuplicated(4.0, (1.0, 2.0)))[1] ==
        (var"1"=0.0, var"2"=0.0)
    @test autodiff(Reverse, f1, Active, Active(4.0))[1][1] == 0

    function f2(x)
        a = [1.0, -3.0, -x, -2x, x]
        sort!(a; rev=true, lt=(x, y) -> abs(x) < abs(y) || (abs(x) == abs(y) && x < y))
        return sum(a .* [1, 2, 3, 4, 5])
    end

    @test autodiff(Forward, f2, Duplicated(2.0, 1.0))[1] == -3
    @test autodiff(Forward, f2, BatchDuplicated(2.0, (1.0, 2.0)))[1] ==
        (var"1"=-3.0, var"2"=-6.0)
    @test autodiff(Reverse, f2, Active, Active(2.0))[1][1] == -3

    dd = Duplicated(
        [TPair(1, 2), TPair(2, 3), TPair(0, 1)], [TPair(0, 0), TPair(0, 0), TPair(0, 0)]
    )
    res = Enzyme.autodiff(Reverse, sorterrfn, dd, Active(1.0))

    @test res[1][2] ≈ 3
    @test dd.dval[1].a ≈ 0
    @test dd.dval[1].b ≈ 0
    @test dd.dval[2].a ≈ 0
    @test dd.dval[2].b ≈ 0
    @test dd.dval[3].a ≈ 0
    @test dd.dval[3].b ≈ 0
end

@testset "Linear Solve" begin
    A = Float64[2 3; 5 7]
    dA = zero(A)
    b = Float64[11, 13]
    db = zero(b)

    forward, pullback = Enzyme.autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(\)},
        Duplicated,
        Duplicated{typeof(A)},
        Duplicated{typeof(b)},
    )

    tape, primal, shadow = forward(Const(\), Duplicated(A, dA), Duplicated(b, db))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Duplicated(A, dA), Duplicated(b, db), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test dA ≈ (-z * transpose(y))
    @test db ≈ z

    db = zero(b)

    forward, pullback = Enzyme.autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(\)},
        Duplicated,
        Const{typeof(A)},
        Duplicated{typeof(b)},
    )

    tape, primal, shadow = forward(Const(\), Const(A), Duplicated(b, db))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Const(A), Duplicated(b, db), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test db ≈ z

    dA = zero(A)

    forward, pullback = Enzyme.autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{typeof(\)},
        Duplicated,
        Duplicated{typeof(A)},
        Const{typeof(b)},
    )

    tape, primal, shadow = forward(Const(\), Duplicated(A, dA), Const(b))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Duplicated(A, dA), Const(b), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test dA ≈ (-z * transpose(y))
end

@static if VERSION > v"1.8"
    @testset "Cholesky" begin
        function cholesky_testfunction_symmetric(A, b, x1, x2)
            C1 = cholesky(A * A') # test factorization without wrapper
            C2 = cholesky(Symmetric(A * A')) # test factorization with wrapper
            x1 .= C1 \ b # test linear solve with factorization object without wrapper
            x2 .= C2 \ b # test linear solve with factorization object with wrapper
            return sum(abs2, C1.L * C1.U) + sum(abs2, C2.L * C2.U) # test factorization itself
        end
        function cholesky_testfunction_hermitian(A, b, x1, x2)
            C1 = cholesky(A * adjoint(A)) # test factorization without wrapper
            C2 = cholesky(Hermitian(A * adjoint(A))) # test factorization with wrapper
            x1 .= C1 \ b # test linear solve with factorization object without wrapper
            x2 .= C2 \ b # test linear solve with factorization object with wrapper
            return sum(abs2, C1.L * C1.U) + sum(abs2, C2.L * C2.U) # test factorization itself
        end
        @testset for (TE, testfunction) in (
            Float64 => cholesky_testfunction_symmetric,
            Float64 => cholesky_testfunction_hermitian,
        )
            @testset for TA in (Const, Duplicated),
                Tb in (Const, Duplicated),
                Tx1 in (Const, Duplicated),
                Tx2 in (Const, Duplicated)

                A = rand(TE, 5, 5)
                b = rand(TE, 5)
                x1 = rand(TE, 5)
                x2 = rand(TE, 5)
                # ishermitian(A * adjoint(A)) || continue
                @testset for Tret in (Const, Duplicated)
                    are_activities_compatible(Tret, TA, Tb, Tx1, Tx2) || continue
                    test_forward(testfunction, Tret, (A, TA), (b, Tb), (x1, Tx1), (x2, Tx2))
                end
                @testset for Tret in (Const, Active)
                    are_activities_compatible(Tret, TA, Tb, Tx1, Tx2) || continue
                    test_reverse(testfunction, Tret, (A, TA), (b, Tb), (x1, Tx1), (x2, Tx2))
                end
            end
        end
    end

    @testset "Linear solve for triangular matrices" begin
        @testset for T in (
                UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular
            ),
            TE in (Float64, ComplexF64),
            sizeB in ((3,), (3, 3))

            n = sizeB[1]
            M = rand(TE, n, n)
            B = rand(TE, sizeB...)
            Y = zeros(TE, sizeB...)
            A = T(M)
            @testset "test through constructor" begin
                _A = T(A)
                function f!(Y, A, B, ::T) where {T}
                    ldiv!(Y, T(A), B)
                    return nothing
                end
                for TY in (Const, Duplicated, BatchDuplicated),
                    TM in (Const, Duplicated, BatchDuplicated),
                    TB in (Const, Duplicated, BatchDuplicated)

                    are_activities_compatible(Const, TY, TM, TB) || continue
                    test_reverse(f!, Const, (Y, TY), (M, TM), (B, TB), (_A, Const))
                end
            end
            @testset "test through `Adjoint` wrapper (regression test for #1306)" begin
                # Test that we get the same derivative for `M` as for the adjoint of its
                # (materialized) transpose. It's the same matrix, but represented differently
                function f!(Y, A, B)
                    ldiv!(Y, A, B)
                    return nothing
                end
                A1 = T(M)
                A2 = T(conj(permutedims(M))')
                dA1 = make_zero(A1)
                dA2 = make_zero(A2)
                dB1 = make_zero(B)
                dB2 = make_zero(B)
                dY1 = rand(TE, sizeB...)
                dY2 = copy(dY1)
                autodiff(
                    Reverse, f!, Duplicated(Y, dY1), Duplicated(A1, dA1), Duplicated(B, dB1)
                )
                autodiff(
                    Reverse, f!, Duplicated(Y, dY2), Duplicated(A2, dA2), Duplicated(B, dB2)
                )
                @test dA1.data ≈ dA2.data
                @test dB1 ≈ dB2
            end
        end
    end
end

@testset "rand and randn rules" begin
    # Distributed as x + unit normal + uniform
    struct MyDistribution
        x::Float64
    end

    Random.rand(rng::Random.AbstractRNG, d::MyDistribution) = d.x + randn() + rand()
    Random.rand(d::MyDistribution) = rand(Random.default_rng(), d)

    # Outer rand should be differentiated through, and inner rand and randn should be ignored.
    @test autodiff(Enzyme.Reverse, x -> rand(MyDistribution(x)), Active, Active(1.0)) == ((1.0,),)
end

end # InternalRules
