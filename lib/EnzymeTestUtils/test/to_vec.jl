using EnzymeTestUtils
using EnzymeTestUtils: to_vec
using Test

function test_to_vec(x)
    x_vec, from_vec = to_vec(x)
    @test x_vec isa Vector{<:AbstractFloat}
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

        x = (TestStruct(1.0, 2.0), TestStruct(1.0, 2.0))
        v, from_vec = to_vec(x)
        @test v == [1.0, 2.0, 1.0, 2.0]
        @test from_vec(v) === x
    end

    @testset "incompletely initialized struct" begin
        x = randn(2, 3)
        y = TestStruct2(x)
        v, from_vec = to_vec(y)
        @test v == vec(x)
        v2 = randn(size(v))
        y2 = from_vec(v2)
        @test y2.x == reshape(v2, size(x))
        @test !isdefined(y2, :a)
    end

    @testset "mutable struct" begin
        @testset for k in (:a, :x)
            x = randn(2, 3)
            y = MutableTestStruct()
            setfield!(y, k, x)
            @test isdefined(y, k)
            @test getfield(y, k) == x
            v, from_vec = to_vec(y)
            @test v == vec(x)
            v2 = randn(size(v))
            y2 = from_vec(v2)
            @test getfield(y2, k) == reshape(v2, size(x))
            @test !isdefined(y2, k === :a ? :x : :a)
        end

        y = MutableTestStruct()
        y.x = randn()
        t = (y, y)
        v, from_vec = to_vec(t)
        @test v == [y.x]
        t2 = from_vec(v)
        @test t2[1] === t2[2]

        t = (y, deepcopy(y))
        v, from_vec = to_vec(t)
        @test v == [y.x, y.x]
        t2 = from_vec(v)
        @test t2[1].x == t2[2].x
        @test t2[1] !== t2[2]
    end

    @testset "nested array" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                     sz in (2, (2, 3), (2, 3, 4))

            test_to_vec([randn(T, sz) for _ in 1:10])
        end
    end

    @testset "partially defined array" begin
        @testset for i in 1:2
            x = Vector{Vector{Float64}}(undef, 2)
            x[i] = randn(5)
            v, from_vec = to_vec(x)
            @test v == x[i]
            v2 = randn(size(v))
            x2 = from_vec(v2)
            @test x2[i] == v2
            @test !isassigned(x2, 3 - i)
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

    @testset "subarrays" begin
        x = randn(2, 3)
        # note: bottom right 2x2 submatrix omitted from y but will be present in v
        y = @views (x[:, 1], x[1, :])
        test_to_vec(y)
        v, from_vec = to_vec(y)
        @test v == vec(x)
        v2 = randn(size(v))
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
            x = randn(2, 3)
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
