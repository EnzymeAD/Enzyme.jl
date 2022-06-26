using CUDA
using Enzyme
using Test

function mul_kernel(A)
    i = threadIdx().x
    if i <= length(A)
        A[i] *= A[i]
    end
    return nothing
end

function grad_mul_kernel(A, dA)
    Enzyme.autodiff_deferred(mul_kernel, Const, Duplicated(A, dA))
    return nothing
end

@testset "mul_kernel" begin
    A = CUDA.ones(64,)
    @cuda threads=length(A) mul_kernel(A)
    dA = similar(A)
    dA .= 1
    @cuda threads=length(A) grad_mul_kernel(A, dA)
    @test all(dA .== 2)
end

function val_kernel!(_, ::Val{N}) where N
    return nothing
end

function dval_kernel!(du, ::Val{N}) where N
    Enzyme.autodiff_deferred(val_kernel!, Const, du, Val(N))
    return nothing
end

# Test for https://github.com/EnzymeAD/Enzyme.jl/issues/358
@testset "Test val kernel" begin
    n = 10
    u = CUDA.rand(n)
    dzdu = CUDA.rand(n)
    @cuda threads=4 dval_kernel!(Duplicated(u, dzdu), Val(n))
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/367

relu(x) = ifelse(x<0, zero(x), x)
## Define CUDA kernel for dense layer
function dense!(
  feats_out, feats_in, W, b,
  ::Val{nfeat_out}, ::Val{nfeat_in}, ::Val{ndof}
) where {nfeat_out, nfeat_in, ndof}

  ## Each thread will update features for a single sample
  idof = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  ## Prevent out-of-bounds array access
  (idof > ndof) && return nothing

  ## Compute `feats_out`
  for i = 1:nfeat_out
    for k = 1:nfeat_in
      feats_out[i, idof] += W[i, k] * feats_in[k, idof]
    end

    feats_out[i, idof] = relu(feats_out[i, idof] + b[i])
  end

  return nothing
end # dense!

## Wrapper for Enzyme to differentiate `dense!`
function ddense!(
  dfeats_out, dfeats_in, dW, db,
  ::Val{nfeat_out}, ::Val{nfeat_in}, ::Val{ndof}
) where {nfeat_out, nfeat_in, ndof}

  Enzyme.autodiff_deferred(
    dense!,
    Const,
    dfeats_out, dfeats_in, dW, db,
    Val(nfeat_out), Val(nfeat_in), Val(ndof)
  )
  return nothing

end # ddense!

function call_ddense()
  nthread      = 32
  ndof         = 32
  nblock       = ceil(Int, ndof / nthread)
  nfeat_out    = 32
  nfeat_in     = 16
  feats_out    = zeros(nfeat_out, ndof)    |> cu
  feats_in     = rand(nfeat_in, ndof)      |> cu
  W            = rand(nfeat_out, nfeat_in) |> cu
  b            = rand(nfeat_out)           |> cu
  dzdfeats_out = rand(nfeat_out, ndof)     |> cu
  dfeats_out   = Duplicated(feats_out, dzdfeats_out)
  dzdfeats_in  = zero(feats_in)
  dfeats_in    = Duplicated(feats_in, dzdfeats_in)
  dzdW         = zero(W)
  dW           = Duplicated(W, dzdW)
  dzdb         = zero(b)
  db           = Duplicated(b, dzdb)

  @cuda threads=nthread blocks=nblock ddense!(
    dfeats_out, dfeats_in, dW, db,
    Val(nfeat_out), Val(nfeat_in), Val(ndof)
  )

end # call_ddense

@testset "DDense" begin
    call_ddense()
end
