module InternalRules

using Enzyme
using Enzyme.EnzymeRules
using EnzymeTestUtils
using FiniteDifferences
using LinearAlgebra
using SparseArrays
using Test

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
@testset "Cholesky" begin
    function symmetric_definite(n :: Int=10)
        α = one(Float64)
        A = spdiagm(-1 => α * ones(n-1), 0 => 4 * ones(n), 1 => conj(α) * ones(n-1))
        b = A * Float64[1:n;]
        return A, b
    end

    function divdriver_NC(x, fact, b)
        res = fact\b
        x .= res
        return nothing
    end
    
    function ldivdriver_NC(x, fact, b)
        ldiv!(fact,b)
        x .= b
        return nothing
    end

    divdriver(x, A, b) = divdriver_NC(x, cholesky(A), b)
    divdriver_herm(x, A, b) = divdriver_NC(x, cholesky(Hermitian(A)), b)
    divdriver_sym(x, A, b) = divdriver_NC(x, cholesky(Symmetric(A)), b)    
    ldivdriver(x, A, b) = ldivdriver_NC(x, cholesky(A), b)
    ldivdriver_herm(x, A, b) = ldivdriver_NC(x, cholesky(Hermitian(A)), b)
    ldivdriver_sym(x, A, b) = ldivdriver_NC(x, cholesky(Symmetric(A)), b)

    # Test forward
    function fwdJdxdb(driver, A, b)
        adJ = zeros(size(A))
        dA = Duplicated(A, zeros(size(A)))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            dA.dval .= 0.0
            db.dval .= 0.0
            dx.dval .= 0.0
            db.dval[i] = 1.0
            Enzyme.autodiff(Forward, driver, dx, dA, db)
            adJ[i, :] = dx.dval
        end
        return adJ
    end

    function const_fwdJdxdb(driver, A, b)
        adJ = zeros(length(b), length(b))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            db.dval .= 0.0
            dx.dval .= 0.0
            db.dval[i] = 1.0
            Enzyme.autodiff(Forward, driver, dx, Const(A), db)
            adJ[i, :] = dx.dval
        end
        return adJ
    end

    function batchedfwdJdxdb(driver, A, b)
        n = length(b)
        function seed(i)
            x = zeros(n)
            x[i] = 1.0
            return x
        end
        adJ = zeros(size(A))
        dA = BatchDuplicated(A, ntuple(i -> zeros(size(A)), n))
        db = BatchDuplicated(b, ntuple(i -> seed(i), n))
        dx = BatchDuplicated(zeros(length(b)), ntuple(i -> zeros(length(b)), n))
        Enzyme.autodiff(Forward, driver, dx, dA, db)
        for i in 1:n
            adJ[i, :] = dx.dval[i]
        end
        return adJ
    end

    # Test reverse
    function revJdxdb(driver, A, b)
        adJ = zeros(size(A))
        dA = Duplicated(A, zeros(size(A)))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            dA.dval .= 0.0
            db.dval .= 0.0
            dx.dval .= 0.0
            dx.dval[i] = 1.0
            Enzyme.autodiff(Reverse, driver, dx, dA, db)
            adJ[i, :] = db.dval
        end
        return adJ
    end

    function const_revJdxdb(driver, A, b)
        adJ = zeros(length(b), length(b))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            db.dval .= 0.0
            dx.dval .= 0.0
            dx.dval[i] = 1.0
            Enzyme.autodiff(Reverse, driver, dx, Const(A), db)
            adJ[i, :] = db.dval
        end
        return adJ
    end

    function batchedrevJdxdb(driver, A, b)
        n = length(b)
        function seed(i)
            x = zeros(n)
            x[i] = 1.0
            return x
        end
        adJ = zeros(size(A))
        dA = BatchDuplicated(A, ntuple(i -> zeros(size(A)), n))
        db = BatchDuplicated(b, ntuple(i -> zeros(length(b)), n))
        dx = BatchDuplicated(zeros(length(b)), ntuple(i -> seed(i), n))
            Enzyme.autodiff(Reverse, driver, dx, dA, db)
        for i in 1:n
            adJ[i, :] .= db.dval[i]
        end
        return adJ
    end

    function Jdxdb(driver, A, b)
        db = zeros(length(b))
        J = zeros(length(b), length(b))
        for i in 1:length(b)
            db[i] = 1.0
            dx = A\db
            db[i] = 0.0
            J[i, :] = dx
        end
        return J
    end

    function JdxdA(driver, A, b)
        db = zeros(length(b))
        J = zeros(length(b), length(b))
        for i in 1:length(b)
            db[i] = 1.0
            dx = A\db
            db[i] = 0.0
            J[i, :] = dx
        end
        return J
    end
    
    @testset "Testing $op, $driver, $driver_NC" for (op, driver, driver_NC) in (
        (:\, divdriver, divdriver_NC),
        (:\, divdriver_herm, divdriver_NC),
        (:\, divdriver_sym, divdriver_NC),
        (:ldiv!, ldivdriver, ldivdriver_NC),
        (:ldiv!, ldivdriver_herm, ldivdriver_NC),
        (:ldiv!, ldivdriver_sym, ldivdriver_NC)
    )
        A, b = symmetric_definite(10)
        n = length(b)
        A = Matrix(A)
        x = zeros(n)
        x = driver(x, A, b)
        fdm = forward_fdm(2, 1);

        function b_one(b)
            _x = zeros(length(b))
            driver(_x,A,b)
            return _x
        end

        fdJ = op==:\ ? FiniteDifferences.jacobian(fdm, b_one, copy(b))[1] : nothing
        fwdJ = fwdJdxdb(driver, A, b)
        revJ = revJdxdb(driver, A, b)
        batchedrevJ = batchedrevJdxdb(driver, A, b)
        batchedfwdJ = batchedfwdJdxdb(driver, A, b)
        J = Jdxdb(driver, A, b)

        if op == :\
            @test isapprox(fwdJ, fdJ)
        end

        @test isapprox(fwdJ, revJ)
        @test isapprox(fwdJ, batchedrevJ)
        @test isapprox(fwdJ, batchedfwdJ)

        fwdJ = const_fwdJdxdb(driver_NC, cholesky(A), b)
        revJ = const_revJdxdb(driver_NC, cholesky(A), b)
        if op == :\
            @test isapprox(fwdJ, fdJ)
        end
        @test isapprox(fwdJ, revJ)

        function h(A, b)
            C = cholesky(A)
            b2 = copy(b)
            ldiv!(C, b2)
            @inbounds b2[1]
        end

        A = [1.3 0.5; 0.5 1.5]
        b = [1., 2.]
        dA = zero(A)
        Enzyme.autodiff(Reverse, h, Active, Duplicated(A, dA), Const(b))

        dA_sym = - (transpose(A) \ [1.0, 0.0]) * transpose(A \ b)
        @test isapprox((dA + dA') / 2, (dA_sym + dA_sym') / 2)
    end
    @testset "Unit test for `cholesky` (regression test for #1307)" begin
        # This test checks the `cholesky` rules without involving `ldiv!`
        function f(A)
            C = cholesky(A * adjoint(A))
            return sum(abs2, C.L * C.U)
        end
        @testset for TE in (Float64, ComplexF64)
            A = rand(TE, 3, 3)
            test_forward(f, Duplicated, (A, Duplicated))
            test_reverse(f, Active, (A, Duplicated))
            @testset "Compare against function bypassing `cholesky`" begin
                g(A) = sum(abs2, A * adjoint(A))
                # If C = cholesky(A * A'), we have A * A' ≈ C.L * C.U, so `g`
                # is essentially the same function as `f`, but bypassing `cholesky`.
                # We can therefore use this to check that we get the correct derivatives.
                @testset "Without wrapper" begin
                    @testset "Forward mode" begin
                        dA = rand(TE, size(A)...)
                        d1 = autodiff(Forward, f, Duplicated, Duplicated(A, dA))
                        d2 = autodiff(Forward, g, Duplicated, Duplicated(A, dA))
                        @test all(d1 .≈ d2)
                    end

                    @testset "Reverse mode" begin
                        dA1 = zero(A)
                        dA2 = zero(A)
                        autodiff(Reverse, f, Active, Duplicated(A, dA1))
                        autodiff(Reverse, g, Active, Duplicated(A, dA2))
                        @test dA1 ≈ dA2
                    end
                end
                if TE == Float64
                    function f_sym(A)
                        C = cholesky(Symmetric(A * adjoint(A)))
                        return sum(abs2, C.L * C.U)
                    end
                    g_sym(A) = sum(abs2, Symmetric(A * adjoint(A)))
                    function f_her(A)
                        C = cholesky(Hermitian(A * adjoint(A)))
                        return sum(abs2, C.L * C.U)
                    end
                    g_her(A) = sum(abs2, Hermitian(A * adjoint(A)))

                    @testset "Forward mode" begin
                        dA = rand(TE, size(A)...)
                        d1 = autodiff(Forward, f_sym, Duplicated, Duplicated(A, dA))
                        d2 = autodiff(Forward, g_sym, Duplicated, Duplicated(A, dA))
                        @test all(d1 .≈ d2)

                        d1 = autodiff(Forward, f_her, Duplicated, Duplicated(A, dA))
                        d2 = autodiff(Forward, g_her, Duplicated, Duplicated(A, dA))
                        @test all(d1 .≈ d2)
                    end

                    @testset "Reverse mode" begin
                        dA1 = zero(A)
                        dA2 = zero(A)
                        autodiff(Reverse, f_sym, Active, Duplicated(A, dA1))
                        autodiff(Reverse, g_sym, Active, Duplicated(A, dA2))
                        @test dA1 ≈ dA2

                        dA1 = zero(A)
                        dA2 = zero(A)
                        autodiff(Reverse, f_her, Active, Duplicated(A, dA1))
                        autodiff(Reverse, g_her, Active, Duplicated(A, dA2))
                        @test dA1 ≈ dA2
                    end
                end
            end
        end
    end
    @testset "Linear solve with and without `cholesky`" begin
        A = [3. 1.; 1. 2.]
        b = [1., 2.]
        dA1 = Duplicated(copy(A), zero(A))
        dA2 = Duplicated(copy(A), zero(A))
        autodiff(Reverse, (A, b) -> first(A\b), dA1, Const(b))
        autodiff(Reverse, (A, b) -> first(cholesky(A)\b), dA2, Const(b))
        @test dA1.dval ≈ dA2.dval
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
end # InternalRules
