using Test
using Enzyme
using SparseArrays
using Logging

function dense_jacobian(f!, x, m)
    n = length(x)
    J = zeros(eltype(x), m, n)
    for j in 1:n
        dx = zeros(eltype(x), n)
        dx[j] = 1
        dy = zeros(eltype(x), m)
        autodiff(Forward, f!, Duplicated(zeros(eltype(x), m), dy), Duplicated(copy(x), dx))
        J[:, j] = dy
    end
    return J
end

function chain!(y, x)
    @inbounds for i in 1:(length(x) - 1)
        y[i] = x[i] * x[i + 1]
    end
    @inbounds y[end] = x[end]^2
    return nothing
end

function asymmetric!(y, x)
    @inbounds for i in 2:(length(x) - 1)
        y[i] = x[i + 1] - 2 * x[i]
    end
    return nothing
end

function stencil!(y, x)
    @inbounds for i in 3:(length(x) - 3)
        y[i] = -x[i - 2] - 3 * x[i] + x[i + 2] - 3 * sin(x[i + 3])
    end
    return nothing
end

function spring!(y, x)
    @inbounds for i in 1:(length(x) - 1)
        d = x[i + 1] - x[i]
        y[i] = (sqrt(d * d + 1) - 1)^2
    end
    return nothing
end

# Stores the partial sums of a reduction into the same element.
function banded!(y, x)
    n = length(x)
    @inbounds for i in 1:n
        y[i] = 0
        for j in max(1, i - 1):min(n, i + 2)
            y[i] += (i - j + 3) * x[j]^2
        end
    end
    return nothing
end

# Bounds checks add exits to the loop, which is then kept dense.
function checked!(y, x)
    for i in 1:(length(x) - 1)
        y[i] = x[i] * x[i + 1]
    end
    return nothing
end

@testset "sparse_jacobian" begin
    for (f!, T, n, m) in (
            (chain!, Float64, 7, 7),
            (chain!, Float32, 7, 7),
            (asymmetric!, Float64, 8, 8),
            (stencil!, Float64, 12, 12),
            (spring!, Float64, 9, 8),
        )
        x = T.(1:n) ./ 4
        J = @test_logs min_level = Logging.Warn Enzyme.sparse_jacobian(f!, zeros(T, m), x)
        @test J isa SparseMatrixCSC{T, Int}
        @test J ≈ dense_jacobian(f!, x, m)
        @test nnz(J) == count(!iszero, dense_jacobian(f!, x, m))
    end

    # Whether or not Enzyme sparsifies the reduction, the entries are exact.
    x = collect(1.0:7) ./ 4
    J = Enzyme.sparse_jacobian(banded!, zeros(7), x)
    @test J ≈ dense_jacobian(banded!, x, 7)

    y = zeros(3)
    Enzyme.sparse_jacobian(chain!, y, [1.0, 2.0, 3.0])
    @test y == [2.0, 6.0, 9.0]

    x = rand(6)
    J = @test_logs (:warn, r"could not sparsify") match_mode = :any Enzyme.sparse_jacobian(checked!, zeros(6), x)
    @test J ≈ dense_jacobian(checked!, x, 6)
end

# A `todense` pointer that reads a banded matrix and writes into a vector.
band_load(offset::Int, n::Int) = Float64(div(offset, 8) % n + 1)
band_store(val::Float64, offset::Int, out::Ptr{Float64}) = unsafe_store!(out, val, div(offset, 8) + 1)

function copy_through_todense(n::Int, out::Ptr{Float64})
    src = Enzyme.todense(Float64, band_load, (val, offset, n) -> nothing, n)
    dst = Enzyme.todense(Float64, (offset, out) -> 0.0, band_store, out)
    for i in 1:n
        unsafe_store!(dst, unsafe_load(src, i) * 2, i)
    end
    return n
end

@testset "todense" begin
    out = zeros(5)
    GC.@preserve out Enzyme.todense_call(copy_through_todense, 5, pointer(out))
    @test out == 2.0 .* (1:5)

    let a = rand()
        @test_throws ArgumentError Enzyme.todense(Float64, offset -> a, (val, offset) -> nothing)
        @test_throws ArgumentError Enzyme.sparse_accumulate(x -> a, 1)
    end
    @test_throws ArgumentError Enzyme.todense(Float64, band_load, band_store, [1.0])
end
