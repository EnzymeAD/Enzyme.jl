using Test
using Enzyme

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
