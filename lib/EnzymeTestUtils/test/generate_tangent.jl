using Test
using EnzymeTestUtils
using EnzymeTestUtils: rand_tangent, zero_tangent
using Quaternions

struct TestStruct2{X,A}
    x::X
    a::A
end

@testset "tangent generation" begin
    @testset "map_fields_recursive" begin
        x = (x=3.0, y=(a=4, b=:foo, c=[5.2]), z=:bar)
        y = EnzymeTestUtils.map_fields_recursive(x -> x .+ 1, x)
        @test y == (x=4.0, y=(a=4, b=:foo, c=[6.2]), z=:bar)
        z = (x=1.5, y=(a=4, b=:foo, c=[5.0]), z=:bar)
        w = EnzymeTestUtils.map_fields_recursive(x, z) do xi, zi
            return xi .* zi
        end
        @test w == (x=4.5, y=(a=4, b=:foo, c=[26.0]), z=:bar)
    end

    @testset "rand_tangent" begin
        @test rand_tangent(1) == 1
        @test rand_tangent(true) == true
        @test rand_tangent(false) == false
        @test rand_tangent(:foo) === :foo
        @test rand_tangent("bar") === "bar"
        @testset for T in (
            Float32, Float64, ComplexF32, ComplexF64, QuaternionF32, QuaternionF64
        )
            x = randn(T)
            @test rand_tangent(x) != x
            @test rand_tangent(x) isa T
            y = randn(T, 5)
            @test rand_tangent(y) != y
            @test rand_tangent(y) isa typeof(y)
        end
        x = TestStruct2(TestStruct2(:foo, TestStruct2(1, 3.0f0 + 1im)), [4.0, 5.0])
        y = rand_tangent(x)
        @test y.x.x == :foo
        @test y.x.a.x == 1
        @test y.x.a.a isa ComplexF32
        @test y.x.a.a != x.x.a.a
        @test y.a isa Vector{Float64}
        @test y.a != x.a
    end

    @testset "zero_tangent" begin
        @test zero_tangent(1) == 1
        @test zero_tangent(true) == true
        @test zero_tangent(false) == false
        @test zero_tangent(:foo) === :foo
        @test zero_tangent("bar") === "bar"
        @testset for T in (
            Float32, Float64, ComplexF32, ComplexF64, QuaternionF32, QuaternionF64
        )
            x = randn(T)
            @test zero_tangent(x) === zero(T)
            y = randn(T, 5)
            @test zero_tangent(y) == zero(y)
            @test zero_tangent(y) isa typeof(y)
        end
        x = TestStruct2(TestStruct2(:foo, TestStruct2(1, 3.0f0 + 1im)), [4.0, 5.0])
        y = zero_tangent(x)
        @test y.x.x == :foo
        @test y.x.a.x == 1
        @test y.x.a.a === zero(ComplexF32)
        @test y.a isa Vector{Float64}
        @test y.a == zero(x.a)
    end
end
