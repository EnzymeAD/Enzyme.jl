using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using SparseArrays
using Test

function test_sparse(M, v, α, β)
    tout = promote_type(eltype(M), eltype(v), typeof(α), typeof(β))
    if v isa AbstractVector
        C = zeros(tout, size(M, 1))
    else
        C = zeros(tout, size(M, 1), size(v, 2))
    end


    for Tret in (Duplicated, BatchDuplicated), TM in (Const, Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated),
            Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tret, TM, Tv, Tα, Tβ) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, TM), (v, Tv), (α, Tα), (β, Tβ))
    end

    for Tret in (Duplicated, BatchDuplicated), TM in (Const, Duplicated, BatchDuplicated),
            Tv in (Const, Duplicated, BatchDuplicated), bα in (true, false), bβ in (true, false)
        are_activities_compatible(Tret, Tret, TM, Tv) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end

    return test_reverse(LinearAlgebra.mul!, Const, (C, Const), (M, Const), (v, Const), (α, Active), (β, Active))
end

struct SparseWrapperScale{T}
    λ::T
end

Base.convert(::Type{Number}, x::SparseWrapperScale) = x.λ
Base.iszero(x::SparseWrapperScale) = iszero(x.λ)

struct SparseWrapperMatrix{A}
    A::A
end

struct SparseWrapperScaledMatrix
    λ::SparseWrapperScale{Float64}
    L::SparseWrapperMatrix{SparseMatrixCSC{Float64, Int}}
end

function sparse_wrapper_mul!(w, L::SparseWrapperScaledMatrix, v)
    iszero(L.λ) && return lmul!(false, w)
    α = convert(Number, L.λ)
    return mul!(w, L.L.A, v, α, false)
end

const sparse_wrapper_matrix = sparse(Float64[0.0 1.0; 0.0 0.0])
sparse_wrapper_nonconst_m = SparseWrapperMatrix(sparse_wrapper_matrix)

@testset "SparseArrays spmatvec reverse rule" begin
    Ts = ComplexF64

    M0 = [
        0.0   1.50614;
        0.0  -0.988357;
        0.0   0.0
    ]


    M = SparseMatrixCSC((M0 .+ 2im * M0))
    v = rand(Ts, 2)
    α = rand(Ts)
    β = rand(Ts)

    # Purely complex
    test_sparse(M, v, α, β)

    # Purely real
    test_sparse(real(M), real(v), real(α), real(β))

    # Now test mixed. We only need to test what the variables are active
    C = zeros(ComplexF64, size(M0, 1))
    TB = (Duplicated, BatchDuplicated)
    for T in TB
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (v, T), (α, Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (real(v), T), (α, Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (real(v), T), (α, Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (v, T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (v, T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (real(v), T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (real(v), T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (v, T), (α, Active), (real(β), Active))
    end
end

@testset "SparseArrays nested wrapper reverse rule" begin
    function f(p)
        u = [3.0, 4.0]
        du = similar(u)
        L = SparseWrapperScaledMatrix(SparseWrapperScale(-p[1]), sparse_wrapper_nonconst_m)
        sparse_wrapper_mul!(du, L, u)
        return sum(du)
    end

    p = [1.0]
    dp = Enzyme.make_zero(p)

    @test f(p) == -4.0
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), f, Active, Duplicated(p, dp))
    @test dp ≈ [-4.0]
end

@testset "SparseArrays spmatmat reverse rule" begin
    Ts = ComplexF64

    M0 = [
        0.0   1.50614;
        0.0  -0.988357;
        0.0   0.0
    ]


    M = SparseMatrixCSC((M0 .+ 2im * M0))
    v = rand(Ts, 2, 2)
    α = rand(Ts)
    β = rand(Ts)

    # Now all the code paths are already tested in the vector case so we just make sure that
    # general matrix multiplication works
    test_sparse(M, v, α, β)

end
