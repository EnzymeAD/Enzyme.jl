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
