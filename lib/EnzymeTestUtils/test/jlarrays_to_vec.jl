using EnzymeTestUtils
using EnzymeTestUtils: to_vec
using JLArrays
using Test

include("helpers.jl")

function test_to_vec(x)
    x_vec, from_vec = to_vec(x)
    @test x_vec isa JLVector{<:AbstractFloat}
    x2 = from_vec(x_vec)
    @test typeof(x2) === typeof(x)
    return EnzymeTestUtils.test_approx(x2, x)
end

@testset "JLArrays to_vec" begin
    @testset "array of floats" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                sz in (2, (2, 3), (2, 3, 4))

            test_to_vec(JLArray(randn(T, sz)))
        end
    end
    @testset "struct" begin
        v = JLArray(randn(2, 3))
        x = TestStruct(1, TestStruct("foo", v))
        test_to_vec(x)
        @test to_vec(x)[1] == vec(v)
    end

    @testset "incompletely initialized struct" begin
        x = JLArray(randn(2, 3))
        y = TestStruct2(x)
        v, from_vec = to_vec(y)
        @test v == vec(x)
        v2 = JLArray(randn(size(v)))
        y2 = from_vec(v2)
        @test y2.x == reshape(v2, size(x))
        @test !isdefined(y2, :a)
    end

    @testset "mutable struct" begin
        @testset for k in (:a, :x)
            x = JLArray(randn(2, 3))
            y = MutableTestStruct()
            setfield!(y, k, x)
            @test isdefined(y, k)
            @test getfield(y, k) == x
            v, from_vec = to_vec(y)
            @test v == vec(x)
            v2 = JLArray(randn(size(v)...))
            y2 = from_vec(v2)
            @test getfield(y2, k) == reshape(v2, size(x))
            @test !isdefined(y2, k === :a ? :x : :a)
        end
    end

    @testset "nested array" begin
        @testset for T in (Float32, Float64, ComplexF32, ComplexF64),
                sz in (2, (2, 3), (2, 3, 4))

            test_to_vec([JLArray(randn(T, sz)) for _ in 1:10])
        end
    end

    @testset "dict" begin
        x = Dict(:a => JLArray(randn(2)), :b => JLArray(randn(3)))
        test_to_vec(x)
        # Float32 matches the eltype of the empty host vectors that a `Dict`'s constant
        # fields produce, which is the case where merging used to append the device data
        # into the host placeholder instead of keeping it on the device
        test_to_vec(Dict(:a => JLArray(randn(Float32, 2)), :b => JLArray(randn(Float32, 3))))
    end

    @testset "views of arrays" begin
        x = JLArray(randn(2, 3))
        test_to_vec(reshape(x, 3, 2))
        test_to_vec(view(x, :, 1))
    end

    @testset "merging host and device vectors" begin
        # `test_reverse` with an active return vectorizes the return value and the
        # arguments separately and merges the two. A scalar return vectorizes to a host
        # vector, so this is the one path where a host-backed and a device-backed vector
        # meet. `vcat` handles that by copying elementwise, which GPU arrays disallow.
        v = JLArray(Float32[1, 2, 3, 4])
        merged = EnzymeTestUtils.multi_tovec(true, (5.0f0, v))
        @test merged isa JLVector{Float32}
        @test Array(merged) == Float32[5, 1, 2, 3, 4]

        # the merge underneath it: a host `prev` must not pull the device data back to the
        # host, and must not be grown in place
        host = Float32[5]
        res, isnew = EnzymeTestUtils.append_or_merge((host, false), v)
        @test res isa JLVector{Float32}
        @test isnew
        @test Array(res) == Float32[5, 1, 2, 3, 4]
        @test host == Float32[5]

        # a device-backed `prev` flagged as a fresh allocation must not be `append!`ed
        # into either, since GPU arrays are not resizable
        res2, _ = EnzymeTestUtils.append_or_merge((v, true), Float32[5])
        @test res2 isa JLVector{Float32}
        @test Array(res2) == Float32[1, 2, 3, 4, 5]
        @test Array(v) == Float32[1, 2, 3, 4]
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
            x = JLArray(randn(2, 3))
            x2 = vec(x)
            y = T(x, x2)
            test_to_vec(y)
            v, from_vec = to_vec(y)
            @test v == x2
            y2 = from_vec(v)
            # `Base.dataids` is objectid-based for GPU arrays, so compare the device
            # pointers instead to check that the reconstruction shares memory
            @test pointer(y2.a) == pointer(y2.b)
        end
    end
end
