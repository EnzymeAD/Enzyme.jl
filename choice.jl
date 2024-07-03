
using Enzyme
using FiniteDifferences
using LinearAlgebra

function phi(x)
	y = tril(x)
	y[diagind(y)] ./= 2
	return y
end

function chol_lower(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('L', c)
    return c[row, col]
end

function chol_lower0(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('L', c)
    return c
end

function chol_upper(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('U', c)
    return c[row, col]
end

function cholesky_manual0(A, dA)
    dA2 = Symmetric(LowerTriangular(tril(dA)), :L)
    L = tril(A)
    @show L, dA2
    dL = L * phi(inv(L) * dA2 * inv(transpose(L)))
    # dL = L * phi(inv(L) * (dA2 +  * inv(transpose(L)))
    return dL
 end

x = [1.0 0.13147601759884564 0.5282944836504488; 0.13147601759884564 1.0 0.18506733179093515; 0.5282944836504488 0.18506733179093515 1.0]

Enzyme.API.printall!(true)
#for i in 1:size(x, 1)
#    for j in 1:size(x, 2)
const i=2
const j=2
  #       @show i, j
  #       reverse_grad  = Enzyme.gradient(Reverse, x -> chol_lower(x, i, j), x)
  #       forward_grad  = reshape(collect(Enzyme.gradient(Forward, x -> chol_lower(x, i, j), x)), size(x))
  #       finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> chol_lower(x, i, j), x)[1]
  #       @show reverse_grad
  #       @show forward_grad
  #       @show finite_diff

        dx = zero(x)
        dx[2,1]=1
        m = cholesky_manual0(x, dx)
        @show Enzyme.autodiff(Forward, x->chol_lower0(x, i, j), Duplicated(x, dx))[1]
        @show m
        @show m[i,j]
#    end
#end

