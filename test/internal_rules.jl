module InternalRules

using Enzyme
using Enzyme.EnzymeRules
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
    @test autodiff(Forward, f4, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1"=1.5, var"2"=3.0)
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

    # Ensure multi dim doesn't crash
    function test2!(A)
        A .= A \ [1.0 0;0.0 1.0]
        return nothing
    end

    A = rand(2,2)
    dA = [1.0 0.0; 0.0 0.0]

    Enzyme.autodiff(
        Enzyme.Reverse,
        test2!,
        Enzyme.Duplicated(A,dA),
    )
end

function tr_solv(A, B, uplo, trans, diag, idx)
  B = copy(B)
  LAPACK.trtrs!(uplo, trans, diag, A, B)
  return @inbounds B[idx]
end


using FiniteDifferences
@testset "Reverse triangular solve" begin
	A = [0.7550523937508613 0.7979976952197996 0.29318222271218364; 0.4416768066117529 0.4335305304334933 0.8895389673238051; 0.07752980210005678 0.05978245503334367 0.4504482683752542]
	B = [0.10527381151977078 0.5450388247476627 0.3179106723232359 0.43919576779182357 0.20974326586875847; 0.7551160501548224 0.049772782182839426 0.09284926395551141 0.07862188927391855 0.17346407477062986; 0.6258040138863172 0.5928022963567454 0.24251650865340169 0.6626410383247967 0.32752198021506784]
    for idx in 1:15
    for uplo in ('L', 'U')
    for diag in ('N', 'U')
    for trans in ('N', 'T')
        dA = zero(A)
        dB = zero(B)	
        Enzyme.autodiff(Reverse, tr_solv, Duplicated(A, dA), Duplicated(B, dB), Const(uplo),Const(trans), Const(diag), Const(idx))
        fA = FiniteDifferences.grad(central_fdm(5, 1), A->tr_solv(A, B, uplo, trans, diag, idx), A)[1]
        fB = FiniteDifferences.grad(central_fdm(5, 1), B->tr_solv(A, B, uplo, trans, diag, idx), B)[1]

		if max(abs.(dA)...) >= 1e-10 || max(abs.(fA)...) >= 1e-10
			@test dA ≈ fA
		end
		if max(abs.(dB)...) >= 1e-10 || max(abs.(fB)...) >= 1e-10
			@test dB ≈ fB
		end
    end
    end
    end
    end
end

function chol_lower0(x)
  c = copy(x)
  C, info = LinearAlgebra.LAPACK.potrf!('L', c)
  return c[2,1]
end

function chol_upper0(x)
  c = copy(x)
  C, info = LinearAlgebra.LAPACK.potrf!('U', c)
  return c[1,2]
end

@testset "Cholesky PotRF" begin
    x = reshape([1.0, -0.10541615131279458, 0.6219810761363638, 0.293343219811946, -0.10541615131279458, 1.0, -0.05258941747718969, 0.34629296878264443, 0.6219810761363638, -0.05258941747718969, 1.0, 0.4692436399208845, 0.293343219811946, 0.34629296878264443, 0.4692436399208845, 1.0], 4, 4)
     dL = zero(x)
     dL[2, 1] = 1.0
 
     @test Enzyme.gradient(Reverse, chol_lower0, x)[1] ≈  [0.05270807565639164 0.0 0.0 0.0; 1.0000000000000024 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0] 
     
     @test Enzyme.gradient(Forward, chol_lower0, x)[1] ≈  [0.05270807565639164 0.0 0.0 0.0; 1.0000000000000024 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0] 

     @test FiniteDifferences.grad(central_fdm(5, 1), chol_lower0, x)[1] ≈ [0.05270807565639164 0.0 0.0 0.0; 1.0000000000000024 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
     
     @test Enzyme.gradient(Forward, chol_upper0, x)[1] ≈ [0.05270807565639728 0.9999999999999999 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
     @test Enzyme.gradient(Reverse, chol_upper0, x)[1] ≈ [0.05270807565639728 0.9999999999999999 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
     @test FiniteDifferences.grad(central_fdm(5, 1), chol_upper0, x)[1] ≈ [0.05270807565639728 0.9999999999999999 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
end


function tchol_lower(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('L', c)
    return c[row, col]
end
function tchol_upper(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('U', c)
    return c[row, col]
end

@testset "Cholesky PotRF 3x3" begin

    x = [1.0 0.13147601759884564 0.5282944836504488; 0.13147601759884564 1.0 0.18506733179093515; 0.5282944836504488 0.18506733179093515 1.0]
    for i in 1:size(x, 1)
        for j in 1:size(x, 2)
             reverse_grad  = Enzyme.gradient(Reverse, x -> tchol_lower(x, i, j), x)[1]
             forward_grad  = Enzyme.gradient(Forward, x -> tchol_lower(x, i, j), x)[1]
             finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tchol_lower(x, i, j), x)[1]
             @test reverse_grad  ≈ finite_diff 
             @test forward_grad  ≈ finite_diff 
             
             reverse_grad  = Enzyme.gradient(Reverse, x -> tchol_upper(x, i, j), x)[1]
             forward_grad  = Enzyme.gradient(Forward, x -> tchol_upper(x, i, j), x)[1]
             finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tchol_upper(x, i, j), x)[1]
             @test reverse_grad  ≈ finite_diff 
             @test forward_grad  ≈ finite_diff
        end
    end
end

function tcholsolv_lower(A, B, i)
    c = copy(B)
    C, info = LinearAlgebra.LAPACK.potrs!('L', A, c)
    return c[i]
end
function tcholsolv_upper(A, B, i)
    c = copy(B)
    C, info = LinearAlgebra.LAPACK.potrs!('U', A, c)
    return c[i]
end


@testset "Cholesky PotRS 3x5" begin

    x = [1.0 0.13147601759884564 0.5282944836504488; 0.13147601759884564 1.0 0.18506733179093515; 0.5282944836504488 0.18506733179093515 1.0]
    for i in 1:15
         B = [3.1 2.7 5.9 2.4 1.6; 7.9 8.2 1.3 9.4 5.5; 4.7 2.9 9.8 7.1 4.3]
         reverse_grad  = Enzyme.gradient(Reverse, Const(B -> tcholsolv_lower(x, B, i)), B)[1]
         # forward_grad  = Enzyme.gradient(Forward, B -> tcholsolv_lower(x, B, i), B)[1]
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), B -> tcholsolv_lower(x, B, i), B)[1]
         @test reverse_grad  ≈ finite_diff 
         # @test forward_grad  ≈ finite_diff 
         
         reverse_grad  = Enzyme.gradient(Reverse, Const(B -> tcholsolv_upper(x, B, i)), B)[1]
         # forward_grad  = Enzyme.gradient(Forward, B -> tcholsolv_upper(x, B, i), B))[1]
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), B -> tcholsolv_upper(x, B, i), B)[1]
         @test reverse_grad  ≈ finite_diff 
         # @test forward_grad  ≈ finite_diff

         reverse_grad  = Enzyme.gradient(Reverse, Const(x -> tcholsolv_lower(x, B, i)), x)[1]
         #forward_grad  = Enzyme.gradient(Forward, x -> tcholsolv_lower(x, B, i), x)[1]
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tcholsolv_lower(x, B, i), x)[1]
         @test reverse_grad  ≈ finite_diff 
         #@test forward_grad  ≈ finite_diff 
         # 
         reverse_grad  = Enzyme.gradient(Reverse, Const(x -> tcholsolv_upper(x, B, i)), x)[1]
         #forward_grad  = Enzyme.gradient(Forward, x -> tcholsolv_upper(x, B, i), x)[1]
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tcholsolv_upper(x, B, i), x)[1]
         @test reverse_grad  ≈ finite_diff 
         #@test forward_grad  ≈ finite_diff
    end
end

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
                Const(A),
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
                Const(A),
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
            A = copy(A)
            LinearAlgebra.LAPACK.potrf!('U', A)
            b2 = copy(b)
            LinearAlgebra.LAPACK.potrs!('U', A, b2)
            @inbounds b2[1]
        end

        A = [1.3 0.5; 0.5 1.5]
        b = [1., 2.]
        dA = zero(A)
        Enzyme.autodiff(Reverse, h, Active, Duplicated(A, dA), Const(b))
        # dA_fwd  = Enzyme.gradient(Forward, A->h(A, b), A)[1]
        dA_fd  = FiniteDifferences.grad(central_fdm(5, 1), A->h(A, b), A)[1]

        @test isapprox(dA, dA_fd)
    end
end

function chol_upper(x)
	x = reshape(x, 4, 4)
	x = parent(cholesky(Hermitian(x)).U)
	x = convert(typeof(x), UpperTriangular(x))
	return x[1,2]
end

@testset "Cholesky upper triangular v1" begin
	x = [1.0, -0.10541615131279458, 0.6219810761363638, 0.293343219811946, -0.10541615131279458, 1.0, -0.05258941747718969, 0.34629296878264443, 0.6219810761363638, -0.05258941747718969, 1.0, 0.4692436399208845, 0.293343219811946, 0.34629296878264443, 0.4692436399208845, 1.0]

    @test Enzyme.gradient(Forward, chol_upper, x)[1] ≈ [0.05270807565639728, 0.0, 0.0, 0.0, 0.9999999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    @test Enzyme.gradient(Reverse, chol_upper, x)[1] ≈ [0.05270807565639728, 0.0, 0.0, 0.0, 0.9999999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
end
 
using EnzymeTestUtils
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
            f!(Y, A, B, ::T) where {T} = ldiv!(Y, T(A), B)
            for TY in (Const, Duplicated, BatchDuplicated),
                TM in (Const, Duplicated, BatchDuplicated),
                TB in (Const, Duplicated, BatchDuplicated)
                are_activities_compatible(Const, TY, TM, TB) || continue
                test_reverse(f!, TY, (Y, TY), (M, TM), (B, TB), (_A, Const); atol = 1.0e-5, rtol = 1.0e-5)
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


@testset "Ranges" begin
    function f1(x)
        x = 25.0x
        ts = Array(Base.range_start_stop_length(0.0, x, 30))
        return sum(ts)
    end
    function f2(x)
        x = 25.0x
        ts = Array(Base.range_start_stop_length(0.0, 0.25, 30))
        return sum(ts) + x
    end
    function f3(x)
        ts = Array(Base.range_start_stop_length(x, 1.25, 30))
        return sum(ts)
    end
    @test Enzyme.autodiff(Forward, f1, Duplicated(0.1, 1.0)) == (374.99999999999994,)
    @test Enzyme.autodiff(Forward, f2, Duplicated(0.1, 1.0)) == (25.0,)
    @test Enzyme.autodiff(Forward, f3, Duplicated(0.1, 1.0)) == (15.0,)

    @test Enzyme.autodiff(Forward, f1, BatchDuplicated(0.1, (1.0, 2.0))) ==
          ((var"1" = 374.99999999999994, var"2" = 749.9999999999999),)
    @test Enzyme.autodiff(Forward, f2, BatchDuplicated(0.1, (1.0, 2.0))) ==
          ((var"1"=25.0, var"2"=50.0),)
    @test Enzyme.autodiff(Forward, f3, BatchDuplicated(0.1, (1.0, 2.0))) ==
          ((var"1"=15.0, var"2"=30.0),)

    @test Enzyme.autodiff(Reverse, f1,  Active, Active(0.1)) == ((375.0,),)
    @test Enzyme.autodiff(Reverse, f2,  Active, Active(0.1)) == ((25.0,),)
    @test Enzyme.autodiff(Reverse, f3,  Active, Active(0.1)) == ((15.0,),)
    
    # Batch active rule isnt setup
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f1(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((375.0,750.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f2(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((25.0,50.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f3(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((15.0,30.0)),)
end

@testset "Ranges 2" begin
    function f1(x)
        x = 25.0x
        ts = Array(0.0:x:3.0)
        return sum(ts)
    end
    function f2(x)
        x = 25.0x
        ts = Array(0.0:0.25:3.0)
        return sum(ts) + x
    end
    function f3(x)
        x = 25.0x
        ts = Array(x:0.25:3.0)
        return sum(ts)
    end
    function f4(x)
        x = 25.0x
        ts = Array(0.0:0.25:x)
        return sum(ts)
    end
    @test Enzyme.autodiff(Forward, f1, Duplicated(0.1, 1.0)) == (25.0,)
    @test Enzyme.autodiff(Forward, f2, Duplicated(0.1, 1.0)) == (25.0,)
    @test Enzyme.autodiff(Forward, f3, Duplicated(0.1, 1.0)) == (75.0,)
    @test Enzyme.autodiff(Forward, f4, Duplicated(0.12, 1.0)) == (0,)

    @test Enzyme.autodiff(Forward, f1, BatchDuplicated(0.1, (1.0, 2.0))) ==
          ((var"1"=25.0, var"2"=50.0),)
    @test Enzyme.autodiff(Forward, f2, BatchDuplicated(0.1, (1.0, 2.0))) ==
          ((var"1"=25.0, var"2"=50.0),)
    @test Enzyme.autodiff(Forward, f3, BatchDuplicated(0.1, (1.0, 2.0))) ==
          ((var"1"=75.0, var"2"=150.0),)
    @test Enzyme.autodiff(Forward, f4, BatchDuplicated(0.12, (1.0, 2.0))) ==
          ((var"1"=0.0, var"2"=0.0),)

    @test Enzyme.autodiff(Reverse, f1,  Active, Active(0.1)) == ((25.0,),)
    @test Enzyme.autodiff(Reverse, f2,  Active, Active(0.1)) == ((25.0,),)
    @test Enzyme.autodiff(Reverse, f3,  Active, Active(0.1)) == ((75.0,),)
    @test Enzyme.autodiff(Reverse, f4,  Active, Active(0.12)) == ((0.0,),)
    
    # Batch active rule isnt setup
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f1(x); nothing end,  Active(1.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((25.0,50.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f2(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((25.0,50.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f3(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((75.0,150.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f4(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((0.0,0.0)),)
end

@testset "SparseArrays spmatvec reverse rule" begin
    C = zeros(18)
    M = sprand(18, 9, 0.1)
    v = randn(9)
    α = 2.0
    β = 1.0

    for Tret in (Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated), 
        Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tret, Tv, Tα, Tβ) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (α, Tα), (β, Tβ))

    end


    for Tret in (Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated), bα in (true, false), bβ in (true, false)
        are_activities_compatible(Tret, Tret, Tv) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end
end

@testset "SparseArrays spmatmat reverse rule" begin
    C = zeros(18, 11)
    M = sprand(18, 9, 0.1)
    v = randn(9, 11)
    α = 2.0
    β = 1.0

    for Tret in (Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated), 
        Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tv, Tα, Tβ) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (α, Tα), (β, Tβ))
    end

    for Tret in (Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated), bα in (true, false), bβ in (true, false)
        are_activities_compatible(Tret, Tv) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end
end

end # InternalRules
