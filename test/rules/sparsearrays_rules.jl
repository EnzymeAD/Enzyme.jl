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
