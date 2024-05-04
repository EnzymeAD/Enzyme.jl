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
    return first(sortperm(t, lt=lt)) * x
end

@testset "Sort rules" begin
    function f1(x)
        a = [1.0, 3.0, x]
        sort!(a)
        return a[2]
    end

    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1] == 1
    @test autodiff(Forward, f1, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1"=1.0, var"2"=2.0)
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 1
    @test autodiff(Forward, f1, Duplicated(4.0, 1.0))[1] == 0
    @test autodiff(Forward, f1, BatchDuplicated(4.0, (1.0, 2.0)))[1] == (var"1"=0.0, var"2"=0.0)
    @test autodiff(Reverse, f1, Active, Active(4.0))[1][1] == 0

    function f2(x)
        a = [1.0, -3.0, -x, -2x, x]
        sort!(a; rev=true, lt=(x, y) -> abs(x) < abs(y) || (abs(x) == abs(y) && x < y))
        return sum(a .* [1, 2, 3, 4, 5])
    end

    @test autodiff(Forward, f2, Duplicated(2.0, 1.0))[1] == -3
    @test autodiff(Forward, f2, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1"=-3.0, var"2"=-6.0)
    @test autodiff(Reverse, f2, Active, Active(2.0))[1][1] == -3

    function f3(x)
        a = [2.0, 2.5, x, 1.0]
        return partialsort(a, 2)
    end

    @test autodiff(Forward, f3, Duplicated(1.5, 1.0))[1] == 1.0
    @test autodiff(Forward, f3, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1"=1.0, var"2"=2.0)
    @test autodiff(Reverse, f3, Active(1.5))[1][1] == 1.0
    @test autodiff(Reverse, f3, Active(2.5))[1][1] == 0.0

    function f4(x)
        a = [2.0, 2.5, x, x / 2]
        y = partialsort(a, 1:2)
        return sum(y)
    end

    @test autodiff(Forward, f4, Duplicated(1.5, 1.0))[1] == 1.5
    @static if VERSION < v"1.7-" || VERSION >= v"1.8-"
        @test autodiff(Forward, f4, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1"=1.5, var"2"=3.0)
    end
    @test autodiff(Reverse, f4, Active(1.5))[1][1] == 1.5
    @test autodiff(Reverse, f4, Active(4.0))[1][1] == 0.5
    @test autodiff(Reverse, f4, Active(6.0))[1][1] == 0.0

    dd = Duplicated([TPair(1, 2), TPair(2, 3), TPair(0, 1)], [TPair(0, 0), TPair(0, 0), TPair(0, 0)])
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

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(\)}, Duplicated, Duplicated{typeof(A)}, Duplicated{typeof(b)})

    tape, primal, shadow = forward(Const(\), Duplicated(A, dA), Duplicated(b, db))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Duplicated(A, dA), Duplicated(b, db), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test dA ≈ (-z * transpose(y))
    @test db ≈ z

    db = zero(b)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(\)}, Duplicated, Const{typeof(A)}, Duplicated{typeof(b)})

    tape, primal, shadow = forward(Const(\), Const(A), Duplicated(b, db))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Const(A), Duplicated(b, db), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test db ≈ z

    dA = zero(A)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(\)}, Duplicated, Duplicated{typeof(A)}, Const{typeof(b)})

    tape, primal, shadow = forward(Const(\), Duplicated(A, dA), Const(b))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Duplicated(A, dA), Const(b), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test dA ≈ (-z * transpose(y))
end

@static if VERSION > v"1.8"
    @testset "cholesky" begin
        @testset "with wrapper arguments" begin
            @testset for Te in (Float64,), TS in (Symmetric, Hermitian), uplo in (:U, :L)
                @testset for TA in (Const, Duplicated), Tret in (Const, Duplicated)
                    _A = collect(exp(TS(rand(Te, 4, 4))))
                    A = TS(_A, uplo)
                    are_activities_compatible(Tret, TA) || continue
                    test_forward(cholesky, Tret, (A, TA))
                    test_reverse(cholesky, Tret, (A, TA))
                end
            end
        end
        @testset "without wrapper arguments" begin
            _square(A) = A * A'
            @testset for Te in (Float64,)
                @testset for TA in (Const, Duplicated), Tret in (Const, Duplicated)
                    A = rand(Te, 4, 4)
                    are_activities_compatible(Tret, TA) || continue
                    test_forward(cholesky ∘ _square, Tret, (A, TA))
                    test_reverse(cholesky ∘ _square, Tret, (A, TA))
                end
            end
        end
    end

    @testset "Linear solve for `Cholesky`" begin
        @testset for Te in (Float64, ComplexF64), uplo in ('L', 'U')
            C = Cholesky(I + rand(Te, 4, 4), uplo, 0)
            B = rand(Te, 4, 4)
            b = rand(Te, 4)
            @testset for TC in (Const, Duplicated, BatchDuplicated),
                         TB in (Const, Duplicated, BatchDuplicated),
                         Tret in (Const, Duplicated, BatchDuplicated)

                @testset "$(size(_B))" for _B in (B, b)
                    are_activities_compatible(Tret, TC, TB) || continue
                    # Non-uniform activities are disabled due to unresolved questions
                    # see https://github.com/EnzymeAD/Enzyme.jl/issues/1411
                    Tret == TC == TB || continue
                    test_forward(\, Tret, (C, TC), (_B, TB))
                    test_reverse(\, Tret, (C, TC), (_B, TB))
                end
            end
            @testset for TC in (Const, Duplicated, BatchDuplicated),
                         TB in (Const, Duplicated, BatchDuplicated),
                         Tret in (Const, Duplicated, BatchDuplicated)

                @testset "$(size(_B))" for _B in (B, b)
                    are_activities_compatible(Tret, TC, TB) || continue
                    # Non-uniform activities are disabled due to unresolved questions
                    # see https://github.com/EnzymeAD/Enzyme.jl/issues/1411
                    Tret == TC == TB || continue
                    test_forward(ldiv!, Tret, (C, TC), (_B, TB))
                    test_reverse(ldiv!, Tret, (C, TC), (_B, TB))
                end
            end
        end
    end

@testset "Linear solve for triangular matrices" begin
    @testset for T in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
        TE in (Float64, ComplexF64), sizeB in ((3,), (3, 3))
        n = sizeB[1]
        M = rand(TE, n, n)
        B = rand(TE, sizeB...)
        Y = zeros(TE, sizeB...)
        A = T(M)
        @testset "test through constructor" begin
            _A = T(A)
            function f!(Y, A, B, ::T) where T
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
            autodiff(Reverse, f!, Duplicated(Y, dY1), Duplicated(A1, dA1), Duplicated(B, dB1))
            autodiff(Reverse, f!, Duplicated(Y, dY2), Duplicated(A2, dA2), Duplicated(B, dB2))
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
