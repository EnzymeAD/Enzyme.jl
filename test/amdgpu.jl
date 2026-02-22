using AMDGPU
using Enzyme
using Test

function mul_kernel(A)
    i = workitemIdx().x
    if i <= length(A)
        A[i] *= A[i]
    end
    return nothing
end

function grad_mul_kernel(A, dA)
    autodiff_deferred(Reverse, Const(mul_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "mul_kernel" begin
    A = AMDGPU.ones(64)
    @roc groupsize = length(A) mul_kernel(A)
    A = AMDGPU.ones(64)
    dA = similar(A)
    dA .= 1
    @roc groupsize = length(A) grad_mul_kernel(A, dA)
    @test all(dA .== 2)
end

function exp_kernel(A)
    i = workitemIdx().x
    if i <= length(A)
        A[i] = exp(A[i])
    end
    return nothing
end

function grad_exp_kernel(A, dA)
    autodiff_deferred(Reverse, Const(exp_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "exp_kernel" begin
    A = AMDGPU.ones(64)
    @roc groupsize = length(A)  exp_kernel(A)
    A = AMDGPU.ones(64)
    dA = similar(A)
    dA .= 1
    @roc groupsize = length(A)  grad_exp_kernel(A, dA)
    @test all(dA .== exp(1.0f0))
end

function cos_kernel(A)
    i = workitemIdx().x
    if i <= length(A)
        A[i] = cos(A[i])
    end
    return nothing
end

function grad_cos_kernel(A, dA)
    autodiff_deferred(Reverse, Const(cos_kernel), Const, Duplicated(A, dA))
    return nothing
end

@testset "cos_kernel" begin
    A = AMDGPU.ones(64)
    @roc groupsize = length(A) cos_kernel(A)
    A = AMDGPU.ones(64)
    dA = similar(A)
    dA .= 1
    @roc groupsize = length(A) grad_cos_kernel(A, dA)
    @test all(dA .≈ -sin(1.0f0))
end
