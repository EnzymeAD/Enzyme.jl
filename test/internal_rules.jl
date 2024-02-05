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

    function divdriver(x, A, b)
        fact = cholesky(A)
        divdriver_NC(x, fact, b)
    end

    function divdriver_herm(x, A, b)
        fact = cholesky(Hermitian(A))
        divdriver_NC(x, fact, b)
    end

    function divdriver_sym(x, A, b)
        fact = cholesky(Symmetric(A))
        divdriver_NC(x, fact, b)
    end
    
    function ldivdriver(x, A, b)
        fact = cholesky(A)
        ldivdriver_NC(x, fact, b)
    end

    function ldivdriver_herm(x, A, b)
        fact = cholesky(Hermitian(A))
        ldivdriver_NC(x, fact, b)
    end

    function ldivdriver_sym(x, A, b)
        fact = cholesky(Symmetric(A))
        ldivdriver_NC(x, fact, b)
    end

    # Test forward
    function fwdJdxdb(driver, A, b)
        adJ = zeros(size(A))
        dA = Duplicated(A, zeros(size(A)))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            copyto!(dA.val, A)
            copyto!(db.val, b)
            fill!(dA.dval, 0.0)
            fill!(db.dval, 0.0)
            fill!(dx.dval, 0.0)
            db.dval[i] = 1.0
            Enzyme.autodiff(
                Forward,
                driver,
                dx,
                dA,
                db
            )
            adJ[i, :] = dx.dval
        end
        return adJ
    end

    function const_fwdJdxdb(driver, A, b)
        adJ = zeros(length(b), length(b))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            copyto!(db.val, b)
            fill!(db.dval, 0.0)
            fill!(dx.dval, 0.0)
            db.dval[i] = 1.0
            Enzyme.autodiff(
                Forward,
                driver,
                dx,
                A,
                db
            )
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
        Enzyme.autodiff(
            Forward,
            driver,
            dx,
            dA,
            db
        )
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
            copyto!(dA.val, A)
            copyto!(db.val, b)
            fill!(dA.dval, 0.0)
            fill!(db.dval, 0.0)
            fill!(dx.dval, 0.0)
            dx.dval[i] = 1.0
            Enzyme.autodiff(
                Reverse,
                driver,
                dx,
                dA,
                db
            )
            adJ[i, :] = db.dval
        end
        return adJ
    end

    function const_revJdxdb(driver, A, b)
        adJ = zeros(length(b), length(b))
        db = Duplicated(b, zeros(length(b)))
        dx = Duplicated(zeros(length(b)), zeros(length(b)))
        for i in 1:length(b)
            copyto!(db.val, b)
            fill!(db.dval, 0.0)
            fill!(dx.dval, 0.0)
            dx.dval[i] = 1.0
            Enzyme.autodiff(
                Reverse,
                driver,
                dx,
                A,
                db
            )
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
            Enzyme.autodiff(
                Reverse,
                driver,
                dx,
                dA,
                db
            )
        for i in 1:n
            adJ[i, :] .= db.dval[i]
        end
        return adJ
    end

    function Jdxdb(driver, A, b)
        x = A\b
        dA = zeros(size(A))
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
    
    @testset "Testing $op" for (op, driver, driver_NC) in (
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
        V = [1.0 0.0; 0.0 0.0]
        dA = zero(A)
        Enzyme.autodiff(Reverse, h, Active, Duplicated(A, dA), Const(b))

        dA_sym = - (transpose(A) \ [1.0, 0.0]) * transpose(A \ b)
        @test isapprox(dA, dA_sym)
    end
end

@testset "Linear solve for triangular matrices" begin
    @testset for T in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
        TE in (Float64, ComplexF64), sizeB in ((3,), (3, 3))
        M = rand(TE, 3, 3)
        B = rand(TE, sizeB...)
        Y = zeros(TE, sizeB...)
        A = T(M)
        @testset "test against EnzymeTestUtils through constructor" begin
            _A = T(A)
            function f!(Y, A, B, ::T) where T
                return ldiv!(Y, T(A), B)
            end
            for Tret in (Const, Active),
                TY in (Const, Duplicated),
                TA in (Const, Duplicated),
                TB in (Const, Duplicated)
                test_reverse(f!, Const, (Y, TY), (M, TA), (B, TB), (_A, Const))
            end
        end
    end
end
end
end # InternalRules
