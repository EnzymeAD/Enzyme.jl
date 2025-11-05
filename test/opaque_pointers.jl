using Enzyme
using Test

# Tests for opaque pointer support (Julia 1.12+)
# These tests target code paths that were failing with opaque pointers
# Specifically exercises eltype(value_type(...)) calls that we fixed

@testset "Opaque pointer support" begin
    # Test basic scalar differentiation (exercises propagate_returned! path)
    f1(x) = x * x + 2x
    @test autodiff(Reverse, f1, Active, Active(3.0))[1][1] ≈ 8.0
    @test autodiff(Forward, f1, Duplicated(3.0, 1.0))[1] ≈ 8.0

    # Test multiple arguments (exercises custom rules path)
    f2(x, y) = x * y
    res = autodiff(Reverse, f2, Active, Active(2.0), Active(3.0))[1]
    @test res[1] ≈ 3.0
    @test res[2] ≈ 2.0
    @test autodiff(Forward, f2, Duplicated(2.0, 1.0), Const(3.0))[1] ≈ 3.0
    @test autodiff(Forward, f2, Const(2.0), Duplicated(3.0, 1.0))[1] ≈ 2.0

    # Test composition (exercises getparent and bitcast operations)
    f3(x) = sin(x * x)
    @test autodiff(Reverse, f3, Active, Active(1.0))[1][1] ≈ 2.0 * cos(1.0)
    @test autodiff(Forward, f3, Duplicated(1.0, 1.0))[1] ≈ 2.0 * cos(1.0)

    # Test higher order derivatives (exercises multiple code paths)
    f4(x) = x^3
    df4(x) = autodiff(Forward, f4, Duplicated(x, 1.0))[1]
    @test autodiff(Forward, df4, Duplicated(2.0, 1.0))[1] ≈ 12.0  # f''(2) = 6*2 = 12

    # Test batch mode (exercises BatchDuplicated handling)
    f5(x) = x * x * x
    tup = autodiff(Forward, f5, BatchDuplicated(2.0, (1.0, 2.0, 3.0)))[1]
    @test tup[1] ≈ 12.0   # 3*2^2 * 1
    @test tup[2] ≈ 24.0   # 3*2^2 * 2
    @test tup[3] ≈ 36.0   # 3*2^2 * 3

    # Test exponential and trigonometric functions (exercises LLVM intrinsics)
    @test autodiff(Reverse, exp, Active, Active(1.0))[1][1] ≈ exp(1.0)
    @test autodiff(Reverse, cos, Active, Active(0.5))[1][1] ≈ -sin(0.5)
    @test autodiff(Forward, tanh, Duplicated(1.0, 1.0))[1] ≈ (1 - tanh(1.0)^2)

    # Test nested calls (exercises inline handling)
    f6(x) = exp(sin(x))
    @test autodiff(Reverse, f6, Active, Active(0.5))[1][1] ≈ exp(sin(0.5)) * cos(0.5)

    if VERSION >= v"1.12-"
        @info "✓ Opaque pointer tests passed on Julia $(VERSION)"
    end
end
