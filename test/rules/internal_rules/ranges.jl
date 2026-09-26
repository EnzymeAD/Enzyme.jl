using Enzyme
using EnzymeTestUtils
import Random
using Test

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

    # Batch active rule isnt setup
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f1(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((375.0,750.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f2(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((25.0,50.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f3(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((15.0,30.0)),)
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

    # Batch active rule isnt setup
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f1(x); nothing end,  Active(1.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((25.0,50.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f2(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((25.0,50.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f3(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((75.0,150.0)),)
    # @test Enzyme.autodiff(Reverse, (x, y) -> begin y[] = f4(x); nothing end,  Active(0.1), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(2.0)))) == (((0.0,0.0)),)
end
