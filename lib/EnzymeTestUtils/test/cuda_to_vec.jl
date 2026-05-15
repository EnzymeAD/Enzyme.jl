using EnzymeTestUtils
using EnzymeTestUtils: to_vec
using CUDA
using Test

function test_to_vec(x)
    x_vec, from_vec = to_vec(x)
    @test x_vec isa CuVector{<:AbstractFloat}
    x2 = from_vec(x_vec)
    @test typeof(x2) === typeof(x)
    return EnzymeTestUtils.test_approx(x2, x)
end

@testset "to_vec" begin
    @testset "array of floats" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                sz in (2, (2, 3), (2, 3, 4))

            test_to_vec(CUDA.cuRAND.randn(T, sz))
        end
    end

    @testset "struct" begin
        v = CUDA.cuRAND.randn(2, 3)
        x = TestStruct(1, TestStruct("foo", v))
        test_to_vec(x)
        @test to_vec(x)[1] == vec(v)
    end

    @testset "incompletely initialized struct" begin
        x = CUDA.cuRAND.randn(2, 3)
        y = TestStruct2(x)
        v, from_vec = to_vec(y)
        @test v == vec(x)
        v2 = CUDA.cuRAND.randn(size(v))
        y2 = from_vec(v2)
        @test y2.x == reshape(v2, size(x))
        @test !isdefined(y2, :a)
    end

    @testset "mutable struct" begin
        @testset for k in (:a, :x)
            x = CUDA.cuRAND.randn(2, 3)
            y = MutableTestStruct()
            setfield!(y, k, x)
            @test isdefined(y, k)
            @test getfield(y, k) == x
            v, from_vec = to_vec(y)
            @test v == vec(x)
            v2 = CUDA.cuRAND.randn(size(v))
            y2 = from_vec(v2)
            @test getfield(y2, k) == reshape(v2, size(x))
            @test !isdefined(y2, k === :a ? :x : :a)
        end
    end

    @testset "nested array" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                sz in (2, (2, 3), (2, 3, 4))

            test_to_vec([CUDA.cuRAND.randn(T, sz) for _ in 1:10])
        end
    end

    @testset "dict" begin
        x = Dict(:a => CUDA.cuRAND.randn(2), :b => CUDA.cuRAND.randn(3))
        test_to_vec(x)
    end

    @testset "views of arrays" begin
        x = CUDA.cuRAND.randn(2, 3)
        test_to_vec(reshape(x, 3, 2))
        test_to_vec(view(x, :, 1))
    end

    @testset "subarrays" begin
        x = CUDA.cuRAND.randn(2, 3)
        # note: bottom right 2x2 submatrix ommited from y but will be present in v
        y = @views (x[:, 1], x[1, :])
        test_to_vec(y)
        v, from_vec = to_vec(y)
        @test v == vec(x)
        v2 = CUDA.cuRAND.randn(size(v))
        y2 = from_vec(v2)
        @test y2[1] == reshape(v2, size(x))[:, 1]
        @test y2[2] == reshape(v2, size(x))[1, :]
        @test Base.dataids(y2[1]) == Base.dataids(y2[2])
    end

    @testset "reshaped arrays share memory" begin
        struct MyContainer1
            a::Any
            b::Any
        end
        mutable struct MyContainer2
            a::Any
            b::Any
        end
        @testset for T in (MyContainer1, MyContainer2)
            x = CUDA.cuRAND.randn(2, 3)
            x2 = vec(x)
            y = T(x, x2)
            test_to_vec(y)
            v, from_vec = to_vec(y)
            @test v == x2
            y2 = from_vec(v)
            @test Base.dataids(y2.a) == Base.dataids(y2.b)
        end
    end
end
