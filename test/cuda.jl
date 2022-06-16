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
