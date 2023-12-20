using Enzyme
using LinearAlgebra
using Random
using SparseArrays
using Test
using FiniteDifferences

function symmetric_definite(n :: Int=10; FC=Float64)
  α = FC <: Complex ? FC(im) : one(FC)
  A = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
  b = A * FC[1:n;]
  return A, b
end

function b_one(b)
    driver(A,b)
end

function driver(A, b)
    fact = cholesky(A)
    fact\b
end

# Test forward
function fwdJdxdb(A, b)
    adJ = zeros(size(A))
    ddA = Duplicated(A, zeros(size(A)))
    ddb = Duplicated(b, zeros(length(b)))
    for i in 1:length(b)
        copyto!(ddA.val, A)
        copyto!(ddb.val, b)
        fill!(ddA.dval, 0.0)
        fill!(ddb.dval, 0.0)
        ddb.dval[i] = 1.0
        grad = Enzyme.autodiff(
            Forward,
            driver,
            ddA,
            ddb
        )
        adJ[i, :] = grad[1]
    end
    return adJ
end

# Test reverse
function revJdxdb(A, b)
    adJ = zeros(size(A))
    ddA = Duplicated(A, zeros(size(A)))
    ddb = Duplicated(b, zeros(length(b)))
    for i in 1:length(b)
        copyto!(ddA.val, A)
        copyto!(ddb.val, b)
        fill!(ddA.dval, 0.0)
        fill!(ddb.dval, 0.0)
        ddb.dval[i] = 1.0
        grad = Enzyme.autodiff(
            Reverse,
            driver,
            ddA,
            ddb
        )
        adJ[i, :] = grad[1]
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

A, b = symmetric_definite(10, FC=Float64)
A = Matrix(A)
x = driver(A, b)
fdm = forward_fdm(2, 1);
fdJ = FiniteDifferences.jacobian(fdm, b_one, copy(b))[1]
fwdJ = fwdJdxdb(A, b)
J = Jdxdb(A, b)

@test isapprox(fwdJ, J)
@test isapprox(fwdJ, fdJ)

fact = cholesky(A)
x = fact\b

c = Matrix(fact)*x

isapprox(c, b)

cc = zeros(length(c))
mul!(cc, fact.U, x)
mul!(cc, fact.L, cc)

isapprox(cc, c)