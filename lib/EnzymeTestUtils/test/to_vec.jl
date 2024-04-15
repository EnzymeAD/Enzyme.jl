using EnzymeTestUtils
using EnzymeTestUtils: to_vec
using Test

function test_to_vec(x)
    x_vec, from_vec = to_vec(x)
    @test x_vec isa DenseVector{<:AbstractFloat}
    x2 = from_vec(x_vec)
    @test typeof(x2) === typeof(x)
    return EnzymeTestUtils.test_approx(x2, x)
end

@testset "to_vec" begin
    @testset "BLAS floats" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
            x = randn(T)
            test_to_vec(x)
            if T <: Real
                @test to_vec(x)[1] == [x]
            else
                @test to_vec(x)[1] == [reim(x)...]
            end
        end
    end

    @testset "non-vectorizable cases" begin
        @testset for x in [Bool, (), true, 1, [2], (3, "string")]
            test_to_vec(x)
            @test isempty(to_vec(x)[1])
        end
    end

    @testset "array of floats" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                     sz in (2, (2, 3), (2, 3, 4))

            test_to_vec(randn(T, sz))
        end
    end

    @testset "struct" begin
        v = randn(2, 3)
        x = TestStruct(1, TestStruct("foo", v))
        test_to_vec(x)
        @test to_vec(x)[1] == vec(v)
    end

    @testset "nested array" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                     sz in (2, (2, 3), (2, 3, 4))

            test_to_vec([randn(T, sz) for _ in 1:10])
        end
    end

    @testset "tuple" begin
        v = randn(3)
        x = ("foo", 1, false, String, TestStruct(3.0, v))
        test_to_vec(x)
        @test to_vec(x)[1] == vcat(3.0, v)
    end

    @testset "namedtuple" begin
        x = (x="bar", y=randn(3), z=randn(), w=TestStruct(4.0, randn(2)))
        test_to_vec(x)
        @test to_vec(x)[1] == vcat(x.y, x.z, x.w.x, x.w.a)
    end

    @testset "dict" begin
        x = Dict(:a => randn(2), :b => randn(3))
        test_to_vec(x)
    end

    @testset "views of arrays" begin
        x = randn(2, 3)
        test_to_vec(reshape(x, 3, 2))
        test_to_vec(view(x, :, 1))
        test_to_vec(PermutedDimsArray(x, (2, 1)))
    end
end
