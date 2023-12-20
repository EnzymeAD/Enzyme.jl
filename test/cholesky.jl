using Enzyme
# import .EnzymeRules: forward, reverse, augmented_primal
# using .EnzymeRules
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
        adJ[i, :] = dx.dval
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
x = zeros(length(b))
x = driver(x, A, b)
fdm = forward_fdm(2, 1);
fdJ = FiniteDifferences.jacobian(fdm, b_one, copy(b))[1]
fwdJ = fwdJdxdb(A, b)
revJ = revJdxdb(A, b)
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
