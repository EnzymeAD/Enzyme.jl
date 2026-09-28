using Enzyme
using Test

import Enzyme: API, EnzymeRules

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

# ── loops (tape caches inside loops) ─────────────────────────────────────────

@testset "ForwardModeSplit – loop" begin
    function sumsq(x)
        r = 0.0
        for i in eachindex(x)
            r += x[i] * x[i]
        end
        return r
    end

    aug, deriv = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(sumsq)},
        Duplicated,
        Duplicated{Vector{Float64}},
    )

    x  = [3.0, 4.0]
    dx = [1.0, 1.0]
    tape, _, _ = aug(Const(sumsq), Duplicated(x, dx))
    (shadow,) = deriv(Const(sumsq), Duplicated(x, dx), tape)
    @test shadow ≈ 14.0
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

# ── custom forward rules and unsupported handled calls ──────────────────────
# Custom forward rules are split when libEnzyme supports split forward call
# handlers. Other runtime-handled calls, such as dynamic dispatch, error instead
# of crashing.

const FWDSPLIT_RULES = "enzyme_custom" in Enzyme.Compiler.FWDSPLIT_HANDLED

function fwdsplit_rule_result(config, primal, shadow)
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return shadow isa Tuple ? BatchDuplicated(primal, shadow) : Duplicated(primal, shadow)
    elseif EnzymeRules.needs_shadow(config)
        return shadow
    elseif EnzymeRules.needs_primal(config)
        return primal
    end
    return nothing
end

@noinline fwdsplit_rule_f(x) = x^2
function EnzymeRules.forward(config, ::Const{typeof(fwdsplit_rule_f)}, ::Type, x::Duplicated)
    return fwdsplit_rule_result(config, fwdsplit_rule_f(x.val), 2 * x.val * x.dval)
end
function EnzymeRules.forward(config, ::Const{typeof(fwdsplit_rule_f)}, ::Type, x::BatchDuplicated)
    return fwdsplit_rule_result(config, fwdsplit_rule_f(x.val), map(dx -> 2 * x.val * dx, x.dval))
end

@noinline fwdsplit_rule_sum(x) = sum(x)
function EnzymeRules.forward(config, ::Const{typeof(fwdsplit_rule_sum)}, ::Type, x::Duplicated)
    # Deliberately not the true derivative, to check that the rule is used.
    return fwdsplit_rule_result(config, fwdsplit_rule_sum(x.val), 10 * sum(x.dval))
end

@noinline function fwdsplit_rule_mut!(x)
    x[1] *= 2
    return x[1]
end
function EnzymeRules.forward(config, ::Const{typeof(fwdsplit_rule_mut!)}, ::Type, x::Duplicated)
    x.val[1] *= 2
    x.dval[1] *= 2
    return fwdsplit_rule_result(config, x.val[1], x.dval[1])
end

struct FwdSplitBox
    v::Any
end


@testset "ForwardModeSplit – custom forward rules" begin
    h(x) = fwdsplit_rule_f(x) + 3x
    if FWDSPLIT_RULES
        aug, deriv = autodiff_thunk(ForwardSplitWithPrimal, Const{typeof(h)}, Duplicated, Duplicated{Float64})
        tape, primal, _ = aug(Const(h), Duplicated(3.0, 1.0))
        @test primal ≈ 18.0
        @test deriv(Const(h), Duplicated(3.0, 1.0), tape) == (9.0, 18.0)

        aug, deriv = autodiff_thunk(ForwardSplitWidth(ForwardSplitNoPrimal, Val(2)), Const{typeof(h)}, BatchDuplicated, BatchDuplicated{Float64, 2})
        tape, _, _ = aug(Const(h), BatchDuplicated(3.0, (1.0, 2.0)))
        shadows = deriv(Const(h), BatchDuplicated(3.0, (1.0, 2.0)), tape)[1]
        @test shadows[1] ≈ 9.0
        @test shadows[2] ≈ 18.0

        k(x) = fwdsplit_rule_sum(x) * 2
        aug, deriv = autodiff_thunk(ForwardSplitNoPrimal, Const{typeof(k)}, Duplicated, Duplicated{Vector{Float64}})
        x = [1.0, 2.0]; dx = [1.0, 3.0]
        tape, _, _ = aug(Const(k), Duplicated(x, dx))
        @test deriv(Const(k), Duplicated(x, dx), tape)[1] ≈ 80.0

        # The rule would repeat the side effects of the primal.
        m(x) = fwdsplit_rule_mut!(x) + 1
        @test_throws Enzyme.Compiler.ForwardModeSplitUnsupportedException autodiff_thunk(
            ForwardSplitNoPrimal, Const{typeof(m)}, Duplicated, Duplicated{Vector{Float64}},
        )
    else
        @test_throws Enzyme.Compiler.ForwardModeSplitUnsupportedException autodiff_thunk(
            ForwardSplitNoPrimal, Const{typeof(h)}, Duplicated, Duplicated{Float64},
        )
    end
end

@testset "ForwardModeSplit – unsupported handled calls" begin
    dyn(b, x) = b.v(x)
    @test_throws Enzyme.Compiler.ForwardModeSplitUnsupportedException autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(dyn)},
        Duplicated,
        Const{FwdSplitBox},
        Duplicated{Float64},
    )

    # Dynamic dispatch on constant data only is fine.
    dyninact(b, x) = (b.v(2.0); x * x)
    aug, deriv = autodiff_thunk(
        ForwardSplitNoPrimal,
        Const{typeof(dyninact)},
        Duplicated,
        Const{FwdSplitBox},
        Duplicated{Float64},
    )
    tape, _, _ = aug(Const(dyninact), Const(FwdSplitBox(sin)), Duplicated(1.0, 1.0))
    @test deriv(Const(dyninact), Const(FwdSplitBox(sin)), Duplicated(1.0, 1.0), tape)[1] ≈ 2.0
end

# ── split forward rules ──────────────────────────────────────────────────────
# forward_augmented runs the primal once and forward_tangent propagates shadows
# from its tape. The pair also provides a synthesized forward rule.

const FWDSPLIT_CALLS = Ref(0)

@noinline function fwdsplit_sq(x)
    FWDSPLIT_CALLS[] += 1
    return x^2
end
function EnzymeRules.forward_augmented(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_sq)}, ::Type, x::Union{Duplicated, BatchDuplicated})
    p = fwdsplit_sq(x.val)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? p : nothing, nothing, (x.val, p))
end
function EnzymeRules.forward_tangent(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_sq)}, ::Type, tape, x::Duplicated)
    xv, p = tape
    return fwdsplit_rule_result(config, p, 2 * xv * x.dval)
end
function EnzymeRules.forward_tangent(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_sq)}, ::Type, tape, x::BatchDuplicated)
    xv, p = tape
    return fwdsplit_rule_result(config, p, map(dx -> 2 * xv * dx, x.dval))
end

# Mutates its argument, which a forward rule cannot split.
@noinline function fwdsplit_scalesum!(x, a)
    x .*= a
    return sum(x)
end
function EnzymeRules.forward_augmented(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_scalesum!)}, ::Type, x::Duplicated, a::Const)
    s = fwdsplit_scalesum!(x.val, a.val)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? s : nothing, nothing, s)
end
function EnzymeRules.forward_tangent(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_scalesum!)}, ::Type, s, x::Duplicated, a::Const)
    x.dval .*= a.val
    return fwdsplit_rule_result(config, s, sum(x.dval))
end

# Returns an array: its shadow may already be needed by the augmented pass.
@noinline fwdsplit_dbl(x) = x .* 2
function EnzymeRules.forward_augmented(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_dbl)}, ::Type, x::Duplicated)
    y = fwdsplit_dbl(x.val)
    shadow = EnzymeRules.needs_shadow(config) ? zero(y) : nothing
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? y : nothing, shadow, (y, shadow))
end
function EnzymeRules.forward_tangent(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_dbl)}, ::Type, tape, x::Duplicated)
    y, shadow = tape
    if EnzymeRules.needs_shadow(config)
        return fwdsplit_rule_result(config, y, x.dval .* 2)
    end
    shadow .= x.dval .* 2
    return EnzymeRules.needs_primal(config) ? y : nothing
end

@noinline fwdsplit_kwsq(x; scale = 1.0) = scale * x^2
function EnzymeRules.forward_augmented(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_kwsq)}, ::Type, x::Duplicated; scale = 1.0)
    return EnzymeRules.AugmentedReturn(EnzymeRules.needs_primal(config) ? fwdsplit_kwsq(x.val; scale) : nothing, nothing, x.val)
end
function EnzymeRules.forward_tangent(config::EnzymeRules.FwdSplitConfig, ::Const{typeof(fwdsplit_kwsq)}, ::Type, xv, x::Duplicated; scale = 1.0)
    return fwdsplit_rule_result(config, scale * xv^2, 2 * scale * xv * x.dval)
end

@testset "ForwardModeSplit – split forward rules" begin
    h(x) = fwdsplit_sq(x) + 3x
    m(x) = fwdsplit_scalesum!(x, 2.0) + 1
    k(x) = sum(fwdsplit_dbl(x))
    kw(x) = fwdsplit_kwsq(x; scale = 3.0)

    @test EnzymeRules.has_frule_from_sig(Tuple{typeof(fwdsplit_sq), Float64})

    # Forward rules synthesized from the split rules.
    FWDSPLIT_CALLS[] = 0
    @test autodiff(ForwardWithPrimal, h, Duplicated(3.0, 1.0)) == (9.0, 18.0)
    @test FWDSPLIT_CALLS[] == 1
    @test autodiff(Forward, h, BatchDuplicated(3.0, (1.0, 2.0)))[1] == (var"1" = 9.0, var"2" = 18.0)
    x = [1.0, 2.0]; dx = [1.0, 10.0]
    @test autodiff(Forward, m, Duplicated(x, dx))[1] ≈ 22.0
    @test x == [2.0, 4.0]
    @test dx == [2.0, 20.0]
    @test autodiff(Forward, k, Duplicated([1.0, 2.0], [1.0, 10.0]))[1] ≈ 22.0
    @test autodiff(Forward, kw, Duplicated(2.0, 1.0))[1] ≈ 12.0

    if FWDSPLIT_RULES
        FWDSPLIT_CALLS[] = 0
        aug, deriv = autodiff_thunk(ForwardSplitWithPrimal, Const{typeof(h)}, Duplicated, Duplicated{Float64})
        tape, primal, _ = aug(Const(h), Duplicated(3.0, 1.0))
        @test primal ≈ 18.0
        # The primal needed by the derivative pass comes from the tape.
        @test deriv(Const(h), Duplicated(3.0, 1.0), tape) == (9.0, 18.0)
        @test FWDSPLIT_CALLS[] == 1

        aug, deriv = autodiff_thunk(ForwardSplitWidth(ForwardSplitNoPrimal, Val(2)), Const{typeof(h)}, BatchDuplicated, BatchDuplicated{Float64, 2})
        tape, _, _ = aug(Const(h), BatchDuplicated(3.0, (1.0, 2.0)))
        shadows = deriv(Const(h), BatchDuplicated(3.0, (1.0, 2.0)), tape)[1]
        @test shadows[1] ≈ 9.0
        @test shadows[2] ≈ 18.0

        aug, deriv = autodiff_thunk(ForwardSplitNoPrimal, Const{typeof(m)}, Duplicated, Duplicated{Vector{Float64}})
        x = [1.0, 2.0]; dx = [1.0, 10.0]
        tape, _, _ = aug(Const(m), Duplicated(x, dx))
        @test x == [2.0, 4.0]
        @test deriv(Const(m), Duplicated(x, dx), tape)[1] ≈ 22.0
        @test x == [2.0, 4.0]
        @test dx == [2.0, 20.0]

        aug, deriv = autodiff_thunk(ForwardSplitNoPrimal, Const{typeof(k)}, Duplicated, Duplicated{Vector{Float64}})
        x = [1.0, 2.0]; dx = [1.0, 10.0]
        tape, _, _ = aug(Const(k), Duplicated(x, dx))
        @test deriv(Const(k), Duplicated(x, dx), tape)[1] ≈ 22.0

        aug, deriv = autodiff_thunk(ForwardSplitNoPrimal, Const{typeof(kw)}, Duplicated, Duplicated{Float64})
        tape, _, _ = aug(Const(kw), Duplicated(2.0, 1.0))
        @test deriv(Const(kw), Duplicated(2.0, 1.0), tape)[1] ≈ 12.0
    else
        @test_throws Enzyme.Compiler.ForwardModeSplitUnsupportedException autodiff_thunk(
            ForwardSplitNoPrimal, Const{typeof(h)}, Duplicated, Duplicated{Float64},
        )
    end
end
