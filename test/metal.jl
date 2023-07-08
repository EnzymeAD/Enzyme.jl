using Metal
using Enzyme
using Test

function fun_cpu!(A, B, a)
    for ix ∈ axes(A, 1)
        A[ix] += a * B[ix] * Float32(100.65)
    end
    return nothing
end

function fun_gpu!(A, B, a)
    ix = Metal.thread_position_in_grid_1d()
    A[ix] += a * B[ix] * Float32(100.65)
    return nothing
end

function ∇_fun_cpu!(A, Ā, B, B̄, a)
    Enzyme.autodiff_deferred(Enzyme.Reverse, fun_cpu!, Enzyme.Const, DuplicatedNoNeed(A, Ā), DuplicatedNoNeed(B, B̄), Const(a))
    return nothing
end

function ∇_fun_gpu!(A_d, Ā_d, B_d, B̄_d, a)
    Enzyme.autodiff_deferred(Enzyme.Reverse, fun_gpu!, Enzyme.Const, DuplicatedNoNeed(A_d, Ā_d), DuplicatedNoNeed(B_d, B̄_d), Const(a))
    return nothing
end

@testset "Metal autodiff" begin
    N = 16

    A = rand(Float32, N)
    B = rand(Float32, N)
    a = Float32(6.5)

    A_d = MtlArray(copy(A))
    B_d = MtlArray(copy(B))

    Ā   = ones(Float32, size(A))
    B̄   = ones(Float32, size(B))
    Ā_d = Metal.ones(Float32, size(A_d))
    B̄_d = Metal.ones(Float32, size(B_d))

    ∇_fun_cpu!(A, Ā, B, B̄, a)

    
    # Metal.@sync @metal threads=N groups=1 fun_gpu!(A_d, B_d, a)
    Metal.@sync @metal threads=N groups=1 ∇_fun_gpu!(A_d, Ā_d, B_d, B̄_d, a)

    @test Ā_d ≈ Ā
    @test B̄_d ≈ B̄
    
end
