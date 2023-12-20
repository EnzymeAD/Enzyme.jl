function check(A,b)
  tA, tb = sparse_laplacian(4, FC=Float64)
  @test all(value.(tb) .== b)
  @test all(value.(tA) .== A)
end

function check_values(solver, A, b)
  x = solver(A,b; atol=atol, rtol=rtol)[1]
  db = Dual.(b)
  dx = solver(A,db; atol=atol, rtol=rtol)[1]
  @test all(dx .== x)
end

function check_jacobian(solver, A, b)
  adJ = ForwardDiff.jacobian(x -> solver(A, x; atol=atol, rtol=rtol)[1], b)
  fdm = central_fdm(8, 1);
  fdJ = FiniteDifferences.jacobian(fdm, x -> solver(A, x; atol=atol, rtol=rtol)[1], copy(b))
  @test all(isapprox.(adJ, fdJ[1]))
end

function check_derivatives_and_values_active_active(solver, A, b, x)
    fdm = central_fdm(8, 1);
    dualsA = copy(A)
    fill!(dualsA, 0.0)
    dualsA[1,1] = 1.0
    dA = ForwardDiff.Dual.(A, dualsA)
    check(A,b)

    dualsb = copy(b)
    fill!(dualsb, 0.0)
    dualsb[1] = 1.0
    db = ForwardDiff.Dual.(b, dualsb)
    dx, stats = solver(dA,db; atol=atol, rtol=rtol)

    all(isapprox(value.(dx), x))

    function A_one_one(x)
        _A = copy(A)
        _A[1,1] = x
        solver(_A,b; atol=atol, rtol=rtol)
    end

    function b_one(x)
        _b = copy(b)
        _b[1] = x
        solver(A,_b; atol=atol, rtol=rtol)
    end

    fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(A[1,1]))
    fdb = FiniteDifferences.jacobian(fdm, a -> b_one(a)[1], copy(b[1]))
    isapprox(value.(dx), x)
    fd =fda[1] + fdb[1]
    @test isapprox(partials.(dx,1), fd)
end

function check_derivatives_and_values_active_passive(solver, A, b, x)
    fdm = central_fdm(8, 1);
    dualsA = copy(A)
    fill!(dualsA, 0.0)
    dualsA[1,1] = 1.0
    dA = ForwardDiff.Dual.(A, dualsA)
    check(A,b)

    dx, stats = solver(dA,b; atol=atol, rtol=rtol)

    all(isapprox(value.(dx), x))

    function A_one_one(x)
        _A = copy(A)
        _A[1,1] = x
        solver(_A,b; atol=atol, rtol=rtol)
    end

    fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(A[1,1]))
    isapprox(value.(dx), x)
    @test isapprox(partials.(dx,1), fda[1])
end
