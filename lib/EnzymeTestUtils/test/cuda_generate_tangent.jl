using EnzymeTestUtils
using EnzymeTestUtils: map_fields_recursive
using CUDA
using Test

@testset "CUDA tangent generation" begin
    @testset "map_fields_recursive treats CuArrays as leaves" begin
        @testset for T in (Float64, ComplexF64)
            x = CuArray(randn(T, 4))
            y = CuArray(zeros(T, 4))
            res = map_fields_recursive(copyto!, y, x)
            @test res === y
            @test getfield(res, :data) === getfield(y, :data)
            @test Array(res) == Array(x)
        end
    end
end
