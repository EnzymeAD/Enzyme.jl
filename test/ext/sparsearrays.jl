using Enzyme, Test

using SparseArrays

@testset "Gradient & SparseArrays" begin
    x = sparse([5.0, 0.0, 6.0])
    dx = Enzyme.gradient(Reverse, sum, x)[1]
    @test dx isa SparseVector
    @test dx ≈ [1, 0, 1]

    x = sparse([5.0 0.0 6.0])
    dx = Enzyme.gradient(Reverse, sum, x)[1]
    @test dx isa SparseMatrixCSC
    @test dx ≈ [1 0 1]
end

function sparse_eval(x::Vector{Float64})
    A = sparsevec([1, 1, 2, 3], [2.0*x[2]^3.0, 1.0-x[1], 2.0+x[3], -1.0])
    B = sparsevec([1, 1, 2, 3], [2.0*x[2], 1.0-x[1], 2.0+x[3], -1.0])
    C = A + B
    return A[1]
end

@testset "Type Unstable SparseArrays" begin
    x = [3.1, 2.7, 8.2]
    dx = [0.0, 0.0, 0.0]

    autodiff(Reverse, sparse_eval, Duplicated(x, dx))
    
    @test x ≈ [3.1, 2.7, 8.2]
    @test dx ≈ [-1.0, 43.74, 0]
end