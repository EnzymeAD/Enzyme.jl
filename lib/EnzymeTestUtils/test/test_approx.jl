using Test
using EnzymeTestUtils
using EnzymeTestUtils: test_approx
using MetaTesting

function make_struct(a, b, c, d, e, f)
    x = TestStruct(a, :x)
    return TestStruct([b, x], (; c=c, x=x, y=[d, e], z="foo", f=f))
end

@testset "test_approx" begin
    @testset "numbers" begin
        test_approx(1, 1)
        test_approx(2, 2.0)
        test_approx(2.0, 2 + 1e-9; atol=1.1e-9)
        @test fails(() -> test_approx(2.0, 2 + 1e-9; atol=1e-9))
        test_approx(1.0, 1.0 + 1e-9; rtol=1.1e-9)
        @test fails(() -> test_approx(1.0, 1.0 + 1e-9; rtol=1e-9))
    end
    @testset "arrays" begin
        test_approx([1, 2], [1, 2])
        test_approx([1, 2], [1, 2 + 1e-9]; atol=1.1e-9)
        @test fails(() -> test_approx([1, 2], [1, 2 + 1e-9]; atol=1e-9))
        test_approx([0, 1], [0, 1 + 1e-9]; rtol=1.1e-9)
        @test fails(() -> test_approx([0, 1], [0, 1 + 1e-9]; rtol=1e-9))
        @test errors(() -> test_approx([1, 2], [1, 2, 3]))
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
        for i in eachindex(x), err in (1e-2, 1e-9)
            y = copy(x)
            y[i] += rand((-1, 1)) * err
            test_approx(make_struct(x...), make_struct(y...); atol=err * 1.1)
            @test fails() do
                test_approx(make_struct(x...), make_struct(y...); atol=err * 0.9)
            end
        end
    end
end
