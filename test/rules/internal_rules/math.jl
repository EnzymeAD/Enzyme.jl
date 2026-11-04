using Enzyme
using EnzymeTestUtils
import Random
using Test

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
        @testset for RT in (Const, Active),
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

function hypot_store!(out, args...)
    out[] = hypot(args...)
    return nothing
end
hypot_dx(x, y) = autodiff(Reverse, hypot, Active(x), Active(y))[1][1]
hypot_dy(x, y) = autodiff(Reverse, hypot, Active(x), Active(y))[1][2]
hypot_forward_y(x, y) = autodiff(Forward, hypot, Const(x), Duplicated(y, one(y)))[1]

hypot_guard(x) = 0.1 <= 0.35hypot(x + 2, 3) ? x^2 : x
hypot_guard_reverse(x) = autodiff(Reverse, hypot_guard, Active(x))[1][1]
hypot_branch(x, xs...) = hypot(x, xs...) > 1 ? x^2 : x
function hypot_branch_store!(out, x, xs...)
    out[] = hypot_branch(x, xs...)
    return nothing
end

@testset "hypot in inactive branch predicates" begin
    @test autodiff(ReverseWithPrimal, hypot_guard, Active(2.0)) == ((4.0,), 4.0)
    @test autodiff(Forward, hypot_guard_reverse, Duplicated(2.0, 1.0))[1] == 2.0
    for T in (Float32, Float64), n in 1:2, x in (zero(T), T(0.25), T(2))
        args = ntuple(Returns(T(0.25)), n)
        expected = x > 1 ? 2x : one(x)
        @test autodiff(Reverse, hypot_branch, Active(x), map(Const, args)...)[1] == (expected, ntuple(Returns(nothing), n)...)
        out = BatchDuplicated(Ref(zero(T)), (Ref(one(T)), Ref(T(-2))))
        derivatives = autodiff(Reverse, hypot_branch_store!, out, Active(x), map(Const, args)...)[1]
        @test out.val[] == hypot_branch(x, args...)
        @test derivatives == (nothing, (expected, -2expected), ntuple(Returns(nothing), n)...)
    end
end

@testset "hypot analytic derivatives" begin
    @testset "scales and signs" begin
        for (x, y) in (
                (3.0, 4.0), (1.0e8, 1.0), (1.0e155, 1.0),
                (3.0e200, 4.0e200), (3.0e-200, 4.0e-200), (0.0, 4.0),
                (3.0f0, 4.0f0), (1.0f8, 1.0f0), (3.0f30, 4.0f30),
                (3.0f-30, 4.0f-30), (Float16(3000), Float16(4000)),
            )
            for (a, b) in ((x, y), (y, x), (-x, y), (x, -y))
                h = hypot(a, b)
                expected = (a / h, b / h)
                for mode in (Forward, Reverse)
                    actual = Enzyme.gradient(mode, Const(hypot), a, b)
                    @test all(isapprox.(actual, expected; rtol = 4eps(typeof(h)), atol = 0))
                end
                @test autodiff(Forward, hypot, Duplicated(a, one(a)), Const(b))[1] ≈ expected[1]
                @test autodiff(Reverse, hypot, Const(a), Active(b))[1][2] ≈ expected[2]
            end
        end
    end
    @testset "seed scaling and batching" begin
        for args in (
                (3.0e200, 4.0e200), (3.0e-200, 4.0e-200),
                (3.0e200, 4.0e200, 12.0e200), (3.0e-200, 4.0e-200, 12.0e-200),
            )
            h = hypot(args...)
            seeds = map(x -> Duplicated(x, x), args)
            @test autodiff(Forward, hypot, seeds...)[1] ≈ h
            batch = map(x -> BatchDuplicated(x, (x, -x)), args)
            actual, primal = autodiff(ForwardWithPrimal, hypot, batch...)
            @test primal == h
            @test actual[1] ≈ h
            @test actual[2] ≈ -h
            out = BatchDuplicated(Ref(0.0), (Ref(h), Ref(-h)))
            reverse = autodiff(Reverse, hypot_store!, out, map(Active, args)...)[1]
            @test out.val[] == h
            for (dx, x) in zip(Base.tail(reverse), args)
                @test dx[1] ≈ x
                @test dx[2] ≈ -x
            end
        end
        x, y = 1.0e155, 1.0
        @test Tuple(autodiff(Forward, hypot, Const(x), BatchDuplicated(y, (1.0, 2.0)))[1]) == (1.0e-155, 2.0e-155)
        @test Tuple(autodiff(Forward, hypot, BatchDuplicated(x, (1.0, 2.0)), Const(y))[1]) == (1.0, 2.0)
        # The partial itself underflows, but its product with the seed does not.
        for args in ((1.0e300, 1.0e-300), (1.0e300, 1.0e-300, 0.0))
            annotations = (Const(args[1]), Duplicated(args[2], 1.0e300), map(Const, args[3:end])...)
            @test autodiff(Forward, hypot, annotations...)[1] ≈ 1.0e-300
            out = BatchDuplicated(Ref(0.0), (Ref(1.0e300), Ref(-1.0e300)))
            reverse = autodiff(Reverse, hypot_store!, out, map(Active, args)...)[1]
            @test reverse[3][1] ≈ 1.0e-300
            @test reverse[3][2] ≈ -1.0e-300
        end
    end
    @testset "mixed precision and zero" begin
        @test autodiff(Reverse, hypot, Active(3.0f0), Active(4.0))[1] === (0.6f0, 0.8)
        @test autodiff(Forward, hypot, Duplicated(3.0f0, 1.0f0), Const(4.0))[1] ≈ 0.6
        for mode in (Forward, Reverse)
            @test Enzyme.gradient(mode, Const(hypot), 0.0, 0.0) == (0.0, 0.0)
        end
    end
    @testset "nested modes" begin
        for (x, y) in ((3.0, 4.0), (1.0e8, 1.0))
            h = hypot(x, y)
            @test autodiff(Forward, hypot_dx, Const(x), Duplicated(y, 1.0))[1] ≈ -(x / h) * (y / h) / h
            @test autodiff(Forward, hypot_dy, Const(x), Duplicated(y, 1.0))[1] ≈ (x / h)^2 / h
            @test autodiff(Reverse, hypot_forward_y, Const(x), Active(y))[1][2] ≈ (x / h)^2 / h
        end
    end
end
