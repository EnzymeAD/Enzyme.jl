using Test
using Enzyme
using KernelAbstractions

@kernel function square!(A)
    I = @index(Global, Linear)
    @inbounds A[I] *= A[I]
end

function square_caller(A)
    backend = get_backend(A)
    kernel = square!(backend)
    kernel(A, ndrange = size(A))
    KernelAbstractions.synchronize(backend)
    return
end


@kernel function mul!(A, B)
    I = @index(Global, Linear)
    @inbounds A[I] *= B
end

function mul_caller(A, B)
    backend = get_backend(A)
    kernel = mul!(backend)
    kernel(A, B, ndrange = size(A))
    KernelAbstractions.synchronize(backend)
    return
end

@testset "kernels" begin
    A = Array{Float64}(undef, 64)
    dA = Array{Float64}(undef, 64)

    A .= (1:1:64)
    dA .= 1

    Enzyme.autodiff(Reverse, square_caller, Duplicated(A, dA))
    @test all(dA .≈ (2:2:128))

    A .= (1:1:64)
    dA .= 1

    _, dB = Enzyme.autodiff(Reverse, mul_caller, Duplicated(A, dA), Active(1.2))[1]

    @test all(dA .≈ 1.2)
    @test dB ≈ sum(1:1:64)

    A .= (1:1:64)
    dA .= 1

    Enzyme.autodiff(Forward, square_caller, Duplicated(A, dA))
    @test all(dA .≈ 2:2:128)
end
