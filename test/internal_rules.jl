module InternalRules

using Enzyme
using Enzyme.EnzymeRules
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
@testset "Cholesky" begin
    function symmetric_definite(n :: Int=10; FC=Float64)
    α = FC <: Complex ? FC(im) : one(FC)
    A = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
    b = A * FC[1:n;]
    return A, b
    end

    function b_one(b)
        _x = zeros(length(b))
        driver(_x,A,b)
        return _x
    end

    function driver(x, A, b)
        fact = cholesky(A)
        x .= fact\b
        return nothing
    end

    # Test forward
    function fwdJdxdb(A, b)
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

    # Test reverse
    function revJdxdb(A, b)
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

    function Jdxdb(A, b)
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

    function JdxdA(A, b)
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

    A, b = symmetric_definite(10, FC=Float64)
    A = Matrix(A)
    x = zeros(length(b))
    x = driver(x, A, b)
    fdm = forward_fdm(2, 1);
    fdJ = FiniteDifferences.jacobian(fdm, b_one, copy(b))[1]
    fwdJ = fwdJdxdb(A, b)
    revJ = revJdxdb(A, b)
    J = Jdxdb(A, b)

    @test isapprox(fwdJ, J)
    @test isapprox(fwdJ, fdJ)
    @test isapprox(fwdJ, revJ)
end # InternalRules
