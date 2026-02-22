using Test
using EnzymeTestUtils
using EnzymeTestUtils: test_approx
using MetaTesting

function make_struct(a, b, c, d, e, f)
    x = TestStruct(a, :x)
    return TestStruct([b, x], (; c, x, y = [d, e], z = "foo", f))
end

@testset "test_approx" begin
    @testset "numbers" begin
        test_approx(1, 1)
        test_approx(2, 2.0)
        test_approx(2.0, 2 + 1.0e-9; atol = 1.1e-9)
        @test fails(() -> test_approx(2.0, 2 + 1.0e-9; atol = 1.0e-9))
        test_approx(1.0, 1.0 + 1.0e-9; rtol = 1.1e-9)
        @test fails(() -> test_approx(1.0, 1.0 + 1.0e-9; rtol = 1.0e-9))
    end
    @testset "arrays" begin
        test_approx([1, 2], [1, 2])
        test_approx([1, 2], [1, 2 + 1.0e-9]; atol = 1.1e-9)
        @test fails(() -> test_approx([1, 2], [1, 2 + 1.0e-9]; atol = 1.0e-9))
        test_approx([0, 1], [0, 1 + 1.0e-9]; rtol = 1.1e-9)
        @test fails(() -> test_approx([0, 1], [0, 1 + 1.0e-9]; rtol = 1.0e-9))
        @test errors(() -> test_approx([1, 2], [1, 2, 3]))
    end
    @testset "tuples" begin
        test_approx((1, 2), (1, 2))
        test_approx((1, 2), (1, 2 + 1.0e-9); atol = 1.1e-9)
        @test fails(() -> test_approx((1, 2), (1, 2 + 1.0e-9); atol = 1.0e-9))
        test_approx((0, 1), (0, 1 + 1.0e-9); rtol = 1.1e-9)
        @test fails(() -> test_approx((0, 1), (0, 1 + 1.0e-9); rtol = 1.0e-9))
        @test fails(() -> test_approx((1, 2), (1, 2, 3)))
    end
    @testset "type" begin
        test_approx(Bool, Bool)
        test_approx(String, String)
        @test fails(() -> test_approx(Bool, String))
    end
    @testset "dict" begin
        x1 = Dict(:x => randn(3), :y => randn(2))
        x2 = Dict(:x => copy(x1[:x]), :y => copy(x1[:y]))
        test_approx(x1, x2)
        for i in eachindex(x2[:x]), err in (1.0e-2, 1.0e-9)
            y = copy(x1[:x])
            y[i] += rand((-1, 1)) * err
            x2[:x] = y
            test_approx(x1, x2; atol = err * 1.1)
            @test fails() do
                return test_approx(x1, x2; atol = err * 0.9)
            end
        end
        x2[:x] = vcat(x1[:x], 1.0)
        @test errors() do
            return test_approx(x1, x2; atol = err * 0.9)
        end
    end
    @testset "non-numeric types" begin
        test_approx(:x, :x)
        @test fails(() -> test_approx(:x, :y))
        test_approx("foo", "foo")
        @test fails(() -> test_approx("foo", "bar"))
        test_approx([:x, :y], [:x, :y])
        @test fails(() -> test_approx([:x, :y], [:x, :z]))
    end
    @testset "nested structures" begin
        x = randn(6)
        for i in eachindex(x), err in (1.0e-2, 1.0e-9)
            y = copy(x)
            y[i] += rand((-1, 1)) * err
            test_approx(make_struct(x...), make_struct(y...); atol = err * 1.1)
            @test fails() do
                test_approx(make_struct(x...), make_struct(y...); atol = err * 0.9)
            end
        end
    end
end
