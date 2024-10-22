using Test
using EnzymeTestUtils
using EnzymeTestUtils: rand_tangent
using Enzyme
using Quaternions

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
        x = TestStruct(TestStruct(:foo, TestStruct(1, 3.0f0 + 1im)), [4.0, 5.0])
        y = rand_tangent(x)
        @test y.x.x == :foo
        @test y.x.a.x == 1
        @test y.x.a.a isa ComplexF32
        @test y.x.a.a != x.x.a.a
        @test y.a isa Vector{Float64}
        @test y.a != x.a
    end

    @testset "auto_activity" begin
        @test EnzymeTestUtils.auto_activity((1.0, Const)) === Const(1.0)
        @test EnzymeTestUtils.auto_activity((1.0, Active)) === Active(1.0)
        x = EnzymeTestUtils.auto_activity((1.2, Duplicated))
        @test x.val == 1.2
        @test x.dval !== 1.2
        x = EnzymeTestUtils.auto_activity((1.5, BatchDuplicated))
        @test x.val == 1.5
        @test length(x.dval) == 2
        @test x.dval[1] !== 1.5

        x = TestStruct(TestStruct(:foo, TestStruct(1, 3.0f0 + 1im)), [4.0, 5.0])
        dx = EnzymeTestUtils.auto_activity((x, Const))
        @test dx isa Const
        @test dx.val === x
        dx = EnzymeTestUtils.auto_activity((x, Active))
        @test dx isa Active
        @test dx.val === x
        dx = EnzymeTestUtils.auto_activity((x, Duplicated))
        @test dx isa Duplicated
        @test dx.val === x
        dx = EnzymeTestUtils.auto_activity((x, BatchDuplicated))
        @test dx isa BatchDuplicated
        @test dx.val === x
        @test length(dx.dval) == 2
    end
end
