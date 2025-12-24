using Enzyme
using EnzymeTestUtils
import Random, LinearAlgebra
using Test

struct TPair
    a::Float64
    b::Float64
end

function sorterrfn(t, x)
    function lt(a, b)
        return a.a < b.a
    end
    return first(sortperm(t, lt = lt)) * x
end

@testset "Sort rules" begin
    function f1(x)
        a = [1.0, 3.0, x]
        sort!(a)
        return a[2]
    end

    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1] == 1
    @test autodiff(Forward, f1, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1" = 1.0, var"2" = 2.0)
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 1
    @test autodiff(Forward, f1, Duplicated(4.0, 1.0))[1] == 0
    @test autodiff(Forward, f1, BatchDuplicated(4.0, (1.0, 2.0)))[1] == (var"1" = 0.0, var"2" = 0.0)
    @test autodiff(Reverse, f1, Active, Active(4.0))[1][1] == 0

    function f2(x)
        a = [1.0, -3.0, -x, -2x, x]
        sort!(a; rev = true, lt = (x, y) -> abs(x) < abs(y) || (abs(x) == abs(y) && x < y))
        return sum(a .* [1, 2, 3, 4, 5])
    end

    @test autodiff(Forward, f2, Duplicated(2.0, 1.0))[1] == -3
    @test autodiff(Forward, f2, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1" = -3.0, var"2" = -6.0)
    @test autodiff(Reverse, f2, Active, Active(2.0))[1][1] == -3

    function f3(x)
        a = [2.0, 2.5, x, 1.0]
        return partialsort(a, 2)
    end

    @test autodiff(Forward, f3, Duplicated(1.5, 1.0))[1] == 1.0
    @test autodiff(Forward, f3, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1" = 1.0, var"2" = 2.0)
    @test autodiff(Reverse, f3, Active(1.5))[1][1] == 1.0
    @test autodiff(Reverse, f3, Active(2.5))[1][1] == 0.0

    function f4(x)
        a = [2.0, 2.5, x, x / 2]
        y = partialsort(a, 1:2)
        return sum(y)
    end

    @test autodiff(Forward, f4, Duplicated(1.5, 1.0))[1] == 1.5
    @test autodiff(Forward, f4, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1" = 1.5, var"2" = 3.0)
    @test autodiff(Reverse, f4, Active(1.5))[1][1] == 1.5
    @test autodiff(Reverse, f4, Active(4.0))[1][1] == 0.5
    @test autodiff(Reverse, f4, Active(6.0))[1][1] == 0.0

    dd = Duplicated([TPair(1, 2), TPair(2, 3), TPair(0, 1)], [TPair(0, 0), TPair(0, 0), TPair(0, 0)])
    res = Enzyme.autodiff(Reverse, sorterrfn, dd, Active(1.0))

    @test res[1][2] ≈ 3
    @test dd.dval[1].a ≈ 0
    @test dd.dval[1].b ≈ 0
    @test dd.dval[2].a ≈ 0
    @test dd.dval[2].b ≈ 0
    @test dd.dval[3].a ≈ 0
    @test dd.dval[3].b ≈ 0
end

@testset "rand and randn rules" begin
    # Distributed as x + unit normal + uniform
    struct MyDistribution
        x::Float64
    end

    Random.rand(rng::Random.AbstractRNG, d::MyDistribution) = d.x + randn() + rand()
    Random.rand(d::MyDistribution) = rand(Random.default_rng(), d)

    # Outer rand should be differentiated through, and inner rand and randn should be ignored.
    @test autodiff(Enzyme.Reverse, x -> rand(MyDistribution(x)), Active, Active(1.0)) == ((1.0,),)
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
    @test Enzyme.autodiff(Forward, f3, Duplicated(0.1, 1.0))[1] == 15.0

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

@testset "hypot rules" begin
    @testset "forward" begin
        @testset for RT in (Const, DuplicatedNoNeed, Duplicated),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated),
                Tz in (Const, Duplicated),
                Txs in (Const, Duplicated)

            x, y, z, xs = 2.0, 3.0, 5.0, 17.0
            test_forward(hypot, RT, (x, Tx), (y, Ty))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))

            x, y, z, xs = 2.0 + 7.0im, 3.0 + 11.0im, 5.0 + 13.0im, 17.0 + 19.0im
            test_forward(hypot, RT, (x, Tx), (y, Ty))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_forward(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))
        end
    end
    @testset "reverse" begin
        @testset for RT in (Active,),
                Tx in (Const, Active),
                Ty in (Const, Active),
                Tz in (Const, Active),
                Txs in (Const, Active)

            x, y, z, xs = 2.0, 3.0, 5.0, 17.0
            test_reverse(hypot, RT, (x, Tx), (y, Ty))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))

            x, y, z, xs = 2.0 + 7.0im, 3.0 + 11.0im, 5.0 + 13.0im, 17.0 + 19.0im
            test_reverse(hypot, RT, (x, Tx), (y, Ty))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz))
            test_reverse(hypot, RT, (x, Tx), (y, Ty), (z, Tz), (xs, Txs))
        end
    end
end

@testset "(matrix) det" begin
    @testset "forward" begin
        @testset for RT in (Const,DuplicatedNoNeed,Duplicated,),
                     Tx in (Const,Duplicated,)
            xr = [4.0 3.0; 2.0 1.0]
            test_forward(LinearAlgebra.det, RT, (xr, Tx))

            xc = [4.0+0.0im 3.0; 2.0-0.0im 1.0]
            test_forward(LinearAlgebra.det, RT, (xc, Tx))
        end
    end
    @testset "reverse" begin
        @testset for RT in (Const, Active,), Tx in (Const, Duplicated,)
            x = [4.0 3.0; 2.0 1.0]
            test_reverse(LinearAlgebra.det, RT, (x, Tx))

            x = [4.0+0.0im 3.0; 2.0-0.0im 1.0]
            test_reverse(LinearAlgebra.det, RT, (x, Tx))
        end
    end
end
