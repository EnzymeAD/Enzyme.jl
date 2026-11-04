using Enzyme
using EnzymeTestUtils
import Random
using Test

function store!(y, f, x)
    y[] = f(x)
    return nothing
end

# Derivatives of `f` at `x` for the seeds 1 and 2 of its result
function batch_reverse(f, x)
    y = BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))
    return Enzyme.autodiff(Reverse, store!, y, Const(f), Active(x))[1][3]
end

@testset "Ranges" begin
    function f1(x)
        x = 25.0x
        ts = Array(Base.range_start_stop_length(0.0, x, 30))
        return sum(ts)
    end
    function f2(x)
        x = 25.0x
        ts = Array(Base.range_start_stop_length(0.0, 0.25, 30))
        return sum(ts) + x
    end
    function f3(x)
        ts = Array(Base.range_start_stop_length(x, 1.25, 30))
        return sum(ts)
    end
    @test Enzyme.autodiff(Forward, f1, Duplicated(0.1, 1.0))[1] ≈ 375.0
    @test Enzyme.autodiff(Forward, f2, Duplicated(0.1, 1.0)) == (25.0,)
    @test Enzyme.autodiff(Forward, f3, Duplicated(0.1, 1.0))[1] ≈ 15.0

    res = Enzyme.autodiff(Forward, f1, BatchDuplicated(0.1, (1.0, 2.0)))
    @test res[1][1] ≈ 375.0
    @test res[1][2] ≈ 750.0

    @test Enzyme.autodiff(Forward, f2, BatchDuplicated(0.1, (1.0, 2.0))) ==
        ((var"1" = 25.0, var"2" = 50.0),)
    res = Enzyme.autodiff(Forward, f3, BatchDuplicated(0.1, (1.0, 2.0)))
    @test res[1][1] ≈ 15.0
    @test res[1][2] ≈ 30.0

    @test Enzyme.autodiff(Reverse, f1, Active, Active(0.1)) == ((375.0,),)
    @test Enzyme.autodiff(Reverse, f2, Active, Active(0.1)) == ((25.0,),)
    @test Enzyme.autodiff(Reverse, f3, Active, Active(0.1))[1][1] == 15.0

    @test batch_reverse(f1, 0.1) == (375.0, 750.0)
    @test batch_reverse(f2, 0.1) == (25.0, 50.0)
    @test all(batch_reverse(f3, 0.1) .≈ (15.0, 30.0))
end

@testset "Ranges 2" begin
    function f1(x)
        x = 25.0x
        ts = Array(0.0:x:3.0)
        return sum(ts)
    end
    function f2(x)
        x = 25.0x
        ts = Array(0.0:0.25:3.0)
        return sum(ts) + x
    end
    function f3(x)
        x = 25.0x
        ts = Array(x:0.25:3.0)
        return sum(ts)
    end
    function f4(x)
        x = 25.0x
        ts = Array(0.0:0.25:x)
        return sum(ts)
    end
    @test Enzyme.autodiff(Forward, f1, Duplicated(0.1, 1.0)) == (25.0,)
    @test Enzyme.autodiff(Forward, f2, Duplicated(0.1, 1.0)) == (25.0,)
    @test Enzyme.autodiff(Forward, f3, Duplicated(0.1, 1.0)) == (75.0,)
    @test Enzyme.autodiff(Forward, f4, Duplicated(0.12, 1.0)) == (0,)

    @test Enzyme.autodiff(Forward, f1, BatchDuplicated(0.1, (1.0, 2.0))) ==
        ((var"1" = 25.0, var"2" = 50.0),)
    @test Enzyme.autodiff(Forward, f2, BatchDuplicated(0.1, (1.0, 2.0))) ==
        ((var"1" = 25.0, var"2" = 50.0),)
    @test Enzyme.autodiff(Forward, f3, BatchDuplicated(0.1, (1.0, 2.0))) ==
        ((var"1" = 75.0, var"2" = 150.0),)
    @test Enzyme.autodiff(Forward, f4, BatchDuplicated(0.12, (1.0, 2.0))) ==
        ((var"1" = 0.0, var"2" = 0.0),)

    @test Enzyme.autodiff(Reverse, f1, Active, Active(0.1)) == ((25.0,),)
    @test Enzyme.autodiff(Reverse, f2, Active, Active(0.1)) == ((25.0,),)
    @test Enzyme.autodiff(Reverse, f3, Active, Active(0.1)) == ((75.0,),)
    @test Enzyme.autodiff(Reverse, f4, Active, Active(0.12)) == ((0.0,),)

    @test batch_reverse(f1, 0.1) == (25.0, 50.0)
    @test batch_reverse(f2, 0.1) == (25.0, 50.0)
    @test batch_reverse(f3, 0.1) == (75.0, 150.0)
    @test batch_reverse(f4, 0.12) == (0.0, 0.0)
end

@testset "Ranges from start, step and length" begin
    # Exactly-rational values make Base build these ranges from integers
    scaled(x) = sum(sin.(x * (1:5)))
    dscaled(x) = sum(k -> k * cos(x * k), 1:5)
    shifted(x) = sum(sin.(x .- (1:5)))
    dshifted(x) = sum(k -> cos(x - k), 1:5)
    divided(x) = sum(sin.((1:5) / x))
    ddivided(x) = sum(k -> -k * cos(k / x) / x^2, 1:5)
    stepped(x) = sum(sin.(range(-1.0; step = x, length = 5)))
    dstepped(x) = sum(k -> k * cos(-1.0 + k * x), 0:4)
    @testset for x in (0.5, -0.25, 3.0, 0.1, 2.0943951023931957)
        @testset for (f, df) in ((scaled, dscaled), (shifted, dshifted), (divided, ddivided), (stepped, dstepped))
            @test Enzyme.autodiff(Reverse, f, Active, Active(x))[1][1] ≈ df(x)
            @test Enzyme.autodiff(Forward, f, Duplicated(x, 1.0))[1] ≈ df(x)
            res = Enzyme.autodiff(Forward, f, BatchDuplicated(x, (1.0, 2.0)))[1]
            @test res[1] ≈ df(x)
            @test res[2] ≈ 2df(x)
            res = batch_reverse(f, x)
            @test res[1] ≈ df(x)
            @test res[2] ≈ 2df(x)
        end
    end

    scaled32(x) = sum(sin.(x * (1:5)))
    @test Enzyme.autodiff(Reverse, scaled32, Active, Active(0.5f0))[1][1] ≈ Float32(dscaled(0.5))
    @test Enzyme.autodiff(Forward, scaled32, Duplicated(0.5f0, 1.0f0))[1] ≈ Float32(dscaled(0.5))

    g(a, s) = sum(sin.(range(a; step = s, length = 5)))
    @testset for Ta in (Const, Active), Ts in (Const, Active)
        test_reverse(g, Active, (0.5, Ta), (-0.25, Ts))
    end
    @testset for Ta in (Const, Duplicated), Ts in (Const, Duplicated)
        test_forward(g, Duplicated, (0.5, Ta), (-0.25, Ts))
    end
end

@testset "Range offsets" begin
    # Base stores the element of smallest magnitude, which is not the first one here
    colon_step(x) = sum(sin.(-1.0:x:1.0))
    dcolon_step(x) = sum(k -> k * cos(-1.0 + k * x), 0:(length(-1.0:x:1.0) - 1))
    colon_start(x) = sum(sin.(x:0.5:3.0))
    dcolon_start(x) = sum(k -> cos(x + 0.5k), 0:(length(x:0.5:3.0) - 1))
    stop_length(x) = sum(sin.(range(x, 1.0; length = 5)))
    dstop_length(x) = sum(k -> (1 - k / 4) * cos(x + k * (1.0 - x) / 4), 0:4)
    start_length(x) = sum(sin.(range(-1.0, x; length = 5)))
    dstart_length(x) = sum(k -> k / 4 * cos(-1.0 + k * (x + 1.0) / 4), 0:4)
    cases = (
        (colon_step, dcolon_step, (0.5, 0.25, 0.3)),
        (colon_start, dcolon_start, (-0.5, -1.25, 0.5)),
        (stop_length, dstop_length, (-0.5, 0.5, 2.0943951023931957)),
        (start_length, dstart_length, (-0.5, 0.5, 2.0943951023931957)),
    )
    @testset for (f, df, xs) in cases, x in xs
        @test Enzyme.autodiff(Reverse, f, Active, Active(x))[1][1] ≈ df(x)
        @test Enzyme.autodiff(Forward, f, Duplicated(x, 1.0))[1] ≈ df(x)
    end

    single(x) = sum(sin.(range(x; step = 0.5, length = 1)))
    single_stop(x) = sum(sin.(range(x, x; length = 1)))
    @test Enzyme.autodiff(Reverse, single, Active, Active(0.5))[1][1] ≈ cos(0.5)
    @test Enzyme.autodiff(Reverse, single_stop, Active, Active(0.5))[1][1] ≈ cos(0.5)
    @test Enzyme.autodiff(Forward, single_stop, Duplicated(0.5, 1.0))[1] ≈ cos(0.5)
end

@testset "Ranges with an inactive result" begin
    f(x) = length(x:0.5:3.0) * x
    @test Enzyme.autodiff(Reverse, f, Active, Active(0.5))[1][1] == 6.0
    @test Enzyme.autodiff(Forward, f, Duplicated(0.5, 1.0))[1] == 6.0
end

@testset "Arithmetic on ranges" begin
    # Twice-precision arithmetic truncates the bits of the range's step
    summed(x) = sum(x * (1:5))
    dsummed(x) = 15.0
    scaled(x) = sum(sin.(x * (0.0:0.5:2.0)))
    dscaled(x) = sum(k -> 0.5k * cos(0.5k * x), 0:4)
    divided(x) = sum(abs2, (-1.0:0.5:1.0) / x)
    ddivided(x) = -5 / x^3
    added(x) = sum(sin.(range(-1.0; step = x, length = 5) + (0.0:0.25:1.0)))
    dadded(x) = sum(k -> k * cos(-1.0 + k * (x + 0.25)), 0:4)
    stop_length(x) = sum(range(-1.0, x; length = 5))
    dstop_length(x) = 2.5
    @testset for x in (0.5, -0.25, 2.0943951023931957)
        @testset for (f, df) in ((summed, dsummed), (scaled, dscaled), (divided, ddivided), (added, dadded), (stop_length, dstop_length))
            @test Enzyme.autodiff(Reverse, f, Active, Active(x))[1][1] ≈ df(x)
            @test Enzyme.autodiff(Forward, f, Duplicated(x, 1.0))[1] ≈ df(x)
            res = Enzyme.autodiff(Forward, f, BatchDuplicated(x, (1.0, 2.0)))[1]
            @test res[1] ≈ df(x)
            @test res[2] ≈ 2df(x)
            res = batch_reverse(f, x)
            @test res[1] ≈ df(x)
            @test res[2] ≈ 2df(x)
        end
    end
    @test Enzyme.autodiff(Reverse, x -> sum(0.0:x:1.0), Active, Active(0.25))[1][1] ≈ 10.0
end
