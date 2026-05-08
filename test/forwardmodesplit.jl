using Enzyme
using Test

import Enzyme: API

# ── helpers ──────────────────────────────────────────────────────────────────

# Finite-difference first derivative for real scalars
function fd(f, x; h = 1e-5)
    (f(x + h) - f(x - h)) / (2h)
end

# ── basic scalar tests ────────────────────────────────────────────────────────

@testset "ForwardModeSplit – scalar NoPrimal" begin
    f(x) = x * x

    aug, deriv = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(f)},
        Duplicated,
        Duplicated{Float64},
    )

    for x in [0.0, 1.0, -2.5, 3.14]
        tape, _, _ = aug(Const(f), Duplicated(x, 1.0))
        (shadow,) = deriv(Const(f), Duplicated(x, 1.0), tape)
        @test shadow ≈ fd(f, x)
    end
end

@testset "ForwardModeSplit – scalar WithPrimal" begin
    f(x) = x * x

    aug, deriv = autodiff_thunk(
        ForwardSplitWithPrimal,
        Const{typeof(f)},
        Duplicated,
        Duplicated{Float64},
    )

    for x in [0.0, 1.0, -2.5, 3.14]
        tape, primal_aug, _ = aug(Const(f), Duplicated(x, 1.0))
        @test primal_aug ≈ f(x)

        shadow, primal_deriv = deriv(Const(f), Duplicated(x, 1.0), tape)
        @test shadow      ≈ fd(f, x)
        @test primal_deriv ≈ f(x)
    end
end

@testset "ForwardModeSplit – matches plain ForwardMode" begin
    f(x) = sin(x) * exp(x / 2)

    aug, deriv = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(f)},
        Duplicated,
        Duplicated{Float64},
    )

    for x in [-2.0, 0.0, 0.5, 1.0, 3.0]
        # Reference from plain Forward mode
        ref_shadow = autodiff(Forward, f, Duplicated(x, 1.0))[1]

        tape, _, _ = aug(Const(f), Duplicated(x, 1.0))
        (shadow,) = deriv(Const(f), Duplicated(x, 1.0), tape)

        @test shadow ≈ ref_shadow
    end
end

# ── multi-argument ────────────────────────────────────────────────────────────

@testset "ForwardModeSplit – multi-argument" begin
    g(x, y) = x * y + y^2

    # Differentiate w.r.t. x (y is Const)
    aug_x, deriv_x = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(g)},
        Duplicated,
        Duplicated{Float64},
        Const{Float64},
    )

    for (x, y) in [(2.0, 3.0), (-1.0, 4.0), (0.0, 1.0)]
        tape, _, _ = aug_x(Const(g), Duplicated(x, 1.0), Const(y))
        (dg_dx,) = deriv_x(Const(g), Duplicated(x, 1.0), Const(y), tape)
        @test dg_dx ≈ y  # ∂g/∂x = y
    end

    # Differentiate w.r.t. y (x is Const)
    aug_y, deriv_y = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(g)},
        Duplicated,
        Const{Float64},
        Duplicated{Float64},
    )

    for (x, y) in [(2.0, 3.0), (-1.0, 4.0), (0.0, 1.0)]
        tape, _, _ = aug_y(Const(g), Const(x), Duplicated(y, 1.0))
        (dg_dy,) = deriv_y(Const(g), Const(x), Duplicated(y, 1.0), tape)
        @test dg_dy ≈ x + 2y  # ∂g/∂y = x + 2y
    end
end

# ── Const return (no shadow) ─────────────────────────────────────────────────

@testset "ForwardModeSplit – Const return" begin
    h(x) = 42.0  # constant function

    aug, deriv = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(h)},
        Const,
        Duplicated{Float64},
    )

    tape, _, _ = aug(Const(h), Duplicated(1.0, 1.0))
    res = deriv(Const(h), Duplicated(1.0, 1.0), tape)
    # Const return → nothing returned from derivative pass (empty tuple or nothing)
    @test res === nothing || res === (nothing,) || res == (nothing,) || isempty(res)
end

# ── mutating function (tape correctness) ─────────────────────────────────────

@testset "ForwardModeSplit – mutating primal" begin
    function fill_sq!(y, x)
        y[1] = x[1]^2
        y[2] = x[1] * x[2]
        return nothing
    end

    y    = [0.0, 0.0]
    dy   = [0.0, 0.0]
    x    = [3.0, 4.0]
    dx   = [1.0, 0.0]  # seed: d/dx[1]

    aug, deriv = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(fill_sq!)},
        Const,
        Duplicated{Vector{Float64}},
        Duplicated{Vector{Float64}},
    )

    tape, _, _ = aug(
        Const(fill_sq!),
        Duplicated(y, dy),
        Duplicated(x, dx),
    )
    deriv(
        Const(fill_sq!),
        Duplicated(y, dy),
        Duplicated(x, dx),
        tape,
    )

    # dy/dx[1]: d(x[1]^2)/dx[1] = 2*x[1] = 6, d(x[1]*x[2])/dx[1] = x[2] = 4
    @test dy[1] ≈ 6.0
    @test dy[2] ≈ 4.0
end

# ── thunk caching ─────────────────────────────────────────────────────────────

@testset "ForwardModeSplit – thunk caching" begin
    f(x) = x^3

    aug1, deriv1 = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(f)},
        Duplicated,
        Duplicated{Float64},
    )
    aug2, deriv2 = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(f)},
        Duplicated,
        Duplicated{Float64},
    )

    # Same thunk types (cached)
    @test typeof(aug1)   === typeof(aug2)
    @test typeof(deriv1) === typeof(deriv2)

    # Still produces correct results
    tape, _, _ = aug1(Const(f), Duplicated(2.0, 1.0))
    (shadow,)  = deriv1(Const(f), Duplicated(2.0, 1.0), tape)
    @test shadow ≈ 3 * 2.0^2  # f'(2) = 12
end

# ── guess_activity ────────────────────────────────────────────────────────────

@testset "ForwardModeSplit – guess_activity" begin
    @test Enzyme.guess_activity(Float64,  ForwardSplitNoPrimal)  == Duplicated{Float64}
    @test Enzyme.guess_activity(Float32,  ForwardSplitNoPrimal)  == Duplicated{Float32}
    @test Enzyme.guess_activity(Int,      ForwardSplitNoPrimal)  == Const{Int}
    @test Enzyme.guess_activity(String,   ForwardSplitNoPrimal)  == Const{String}
end

# ── error cases ───────────────────────────────────────────────────────────────

@testset "ForwardModeSplit – Active return errors" begin
    f(x) = x^2
    @test_throws ErrorException autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(f)},
        Active,  # Active not allowed in forward mode
        Duplicated{Float64},
    )
end

# ── ForwardSplitWidth ─────────────────────────────────────────────────────────

@testset "ForwardModeSplit – width-2 batch" begin
    f(x) = x^2

    mode2 = ForwardSplitWidth(ForwardSplitNoPrimal, Val(2))

    aug, deriv = autodiff_thunk(
        mode2,
        Const{typeof(f)},
        BatchDuplicated,
        BatchDuplicated{Float64, 2},
    )

    x   = 3.0
    dx  = (1.0, 2.0)  # two simultaneous seeds

    tape, _, _ = aug(Const(f), BatchDuplicated(x, dx))
    (shadows,) = deriv(Const(f), BatchDuplicated(x, dx), tape)

    # f'(x) = 2x = 6; batch: (1*6, 2*6) = (6, 12)
    @test shadows[1] ≈ 6.0
    @test shadows[2] ≈ 12.0
end

# ── convert mode ─────────────────────────────────────────────────────────────

@testset "ForwardModeSplit – convert to CDerivativeMode" begin
    @test convert(API.CDerivativeMode, ForwardSplitNoPrimal)  === API.DEM_ForwardModeSplit
    @test convert(API.CDerivativeMode, ForwardSplitWithPrimal) === API.DEM_ForwardModeSplit
end
