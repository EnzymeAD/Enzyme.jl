using Enzyme
using Test

provided_square(x) = x * x
provided_square_gradient(x) = (x * x, 2x)
provided_sum_squares(x) = sum(abs2, x)
provided_sum_squares_gradient(x) = (sum(abs2, x), 2 .* x)
provided_scaled(f, x) = 3 * f(x)
provided_first_derivative(f, x) = only(only(autodiff(Reverse, Const(f), Active, Active(x))))
provided_bad_shape(x) = (sum(abs2, x), zeros(length(x) + 1))
provided_bad_value(x) = ([sum(abs2, x)], 2 .* x)
provided_nonfinite_gradient(x) = (zero(x), oftype(x, Inf))

struct ReusedGradient{T}
    buffer::T
end
function (provider::ReusedGradient)(x)
    provider.buffer .= 2 .* x
    return (sum(abs2, x), provider.buffer)
end

@testset "Supplied gradient ($T)" for T in (Float32, Float64)
    scalar = WithGradient(provided_square, provided_square_gradient)
    @test scalar(T(3)) == T(9)
    @test autodiff(Reverse, Const(scalar), Active, Active(T(3))) == ((T(6),),)
    @test autodiff(ReverseWithPrimal, Const(scalar), Active, Active(T(3))) == ((T(6),), T(9))
    @test autodiff(Forward, Const(scalar), Duplicated(T(3), T(2))) == (T(12),)
    @test autodiff(ForwardWithPrimal, Const(scalar), Duplicated(T(3), T(2))) == (T(12), T(9))
    @test autodiff(Reverse, provided_scaled, Active, Const(scalar), Active(T(3)))[1][2] == T(18)
    @test autodiff(Forward, provided_first_derivative, Const(scalar), Duplicated(T(3), one(T))) == (T(2),)

    array = WithGradient(provided_sum_squares, provided_sum_squares_gradient)
    x = T[1, -2, 3]
    original = copy(x)
    dx = fill(T(7), 3)
    @test array(x) == T(14)
    @test gradient(Reverse, Const(array), x) == (T[2, -4, 6],)
    @test autodiff(Forward, Const(array), Duplicated(x, T[2, 3, 4])) == (T(16),)
    @test autodiff(ForwardWithPrimal, Const(array), Duplicated(x, T[2, 3, 4])) == (T(16), T(14))
    autodiff(Reverse, provided_scaled, Active, Const(array), Duplicated(x, dx))
    @test dx == T[13, -5, 25]
    @test x == original

    @testset "Batch width $width" for width in (1, 2, 3)
        tangents = ntuple(i -> fill(T(i), 3), width)
        expected = ntuple(i -> T(4i), width)
        derivatives = autodiff(Forward, Const(array), BatchDuplicated(x, tangents))
        @test (width == 1 ? derivatives : Tuple(only(derivatives))) == expected
        derivatives, primal = autodiff(ForwardWithPrimal, Const(array), BatchDuplicated(x, tangents))
        @test (width == 1 ? (derivatives,) : Tuple(derivatives)) == expected
        @test primal == T(14)

        # Split reverse accepts independent output seeds and preserves existing adjoints.
        shadows = ntuple(i -> fill(T(i), 3), width)
        arg = BatchDuplicated(x, shadows)
        mode = Enzyme.ReverseSplitWidth(ReverseSplitWithPrimal, Val(width))
        fw, rv = autodiff_thunk(mode, Const{typeof(array)}, Active, typeof(arg))
        tape, value, shadow = fw(Const(array), arg)
        @test value == T(14)
        seeds = ntuple(i -> T(2i), width)
        rv(Const(array), arg, width == 1 ? only(seeds) : seeds, tape)
        for i in 1:width
            @test shadows[i] == T(i) .+ T(2i) .* (2 .* x)
        end
    end

    @testset "Tape owns its gradient" begin
        provider = ReusedGradient(zeros(T, 3))
        reusable = WithGradient(provided_sum_squares, provider)
        arg = Duplicated(copy(x), zeros(T, 3))
        fw, rv = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(reusable)}, Active, typeof(arg))
        tape, value, shadow = fw(Const(reusable), arg)
        provider(fill(T(8), 3))
        fill!(arg.val, T(9))
        rv(Const(reusable), arg, T(3), tape)
        @test arg.dval == 6 .* x
        @test provider.buffer == fill(T(16), 3)
    end

    @testset "Validation and inactive inputs" begin
        @test_throws DimensionMismatch gradient(Reverse, Const(WithGradient(provided_sum_squares, provided_bad_shape)), x)
        @test_throws ArgumentError gradient(Reverse, Const(WithGradient(provided_sum_squares, provided_bad_value)), x)
        @test autodiff(ReverseWithPrimal, Const(array), Const, Const(x)) == ((nothing,), T(14))
        @test gradient(Reverse, Const(array), T[]) == (T[],)
        @test gradient(Reverse, Const(array), reshape(x, 1, 3)) == (reshape(2 .* x, 1, 3),)

        inactive = copy(x)
        @test autodiff(set_runtime_activity(Forward), Const(array), Duplicated(inactive, inactive)) == (zero(T),)
        autodiff(set_runtime_activity(Reverse), Const(array), Active, Duplicated(inactive, inactive))
        @test inactive == x

        nonfinite = WithGradient(zero, provided_nonfinite_gradient)
        @test autodiff(set_strong_zero(Forward), Const(nonfinite), Duplicated(T(3), zero(T))) == (zero(T),)
    end
end
