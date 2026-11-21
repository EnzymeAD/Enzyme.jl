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
    A = sparsevec([1, 1, 2, 3], [2.0 * x[2]^3.0, 1.0 - x[1], 2.0 + x[3], -1.0])
    B = sparsevec([1, 1, 2, 3], [2.0 * x[2], 1.0 - x[1], 2.0 + x[3], -1.0])
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

@testset "spdiagm with heterogeneous pairs (#2108)" begin
    f1(p) = sum(nonzeros(spdiagm(0 => 1:11, 1 => p)))
    f2(p) = sum(spdiagm(12, 12, 0 => 1:11, 1 => p))
    f3(p) = sum(spdiagm(-1 => 2.0 .* p, 0 => 1:11, 1 => p))

    p = collect(0.1:0.1:1.0)

    for (f, expected) in ((f1, ones(10)), (f2, ones(10)), (f3, fill(3.0, 10)))
        dp = zeros(10)
        autodiff(Reverse, f, Active, Duplicated(p, dp))
        @test dp ≈ expected

        @test Enzyme.gradient(Reverse, f, p)[1] ≈ expected
        @test Enzyme.gradient(Forward, f, p)[1] ≈ expected
    end
end
