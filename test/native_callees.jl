using Enzyme, Test

# Derivative-free callees called natively instead of being emitted (Julia 1.10 and later).

struct NCStructure
    idx::Vector{Int}
    n::Int
end
@noinline nc_structure(n::Int) = NCStructure(collect(1:n), n)

function nc_loss(x::Vector{Float64})
    s = nc_structure(length(x))
    acc = 0.0
    for i in s.idx
        acc += x[i]^2
    end
    return acc
end

# `_sortperm` on integers takes and returns only boxed values, so Julia compiles it with
# the boxed `jl_fptr_args` ABI and not a specialized entry: it must stay emitted.
nc_sortperm(x) = (p = sortperm([3, 1, 2]); x * p[1])

@static if VERSION < v"1.12-" || v"1.12-beta3" <= VERSION
    @testset "Native derivative-free callees" begin
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test autodiff(Forward, nc_loss, Duplicated(x, ones(3)))[1] ≈ 12.0

        @test autodiff(Reverse, nc_sortperm, Active, Active(2.0))[1][1] ≈ 2.0
        @test autodiff(Forward, nc_sortperm, Duplicated(2.0, 1.0))[1] ≈ 2.0
    end
end

# Primals of functions with custom rules. Enzyme calls the rule when one applies. When none
# does and the call is constant (its function, arguments and return are all constant),
# Enzyme calls the primal as is; on Julia 1.12 and 1.13 natively instead of emitted. Any
# other call without a rule throws a `MethodError`.

using Enzyme: EnzymeRules

# Rules for active arguments only, as ChainRules imports make them: no rule applies when
# every argument is constant, and the return is constant.
nc_ruled(x) = x^2

function EnzymeRules.forward(config, func::Const{typeof(nc_ruled)}, ::Type{<:Union{Duplicated, DuplicatedNoNeed}}, x::Duplicated)
    d = 100 * x.dval
    return EnzymeRules.needs_primal(config) ? Duplicated(func.val(x.val), d) : d
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(nc_ruled)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(nc_ruled)}, dret::Active, tape, x::Active)
    return (100 * dret.val,)
end

# The first call goes through the rule, the second calls the primal.
nc_ruled_use(x, c) = nc_ruled(x) + x * nc_ruled(c)
# An active call only: the rule, whose derivative is 100.
nc_ruled_active(x) = 2 * nc_ruled(x)

# A rule for constant arguments only: it applies to a constant call, and Enzyme calls it
# rather than the primal. No rule applies to an active call, which throws.
nc_construle(x) = x^2
function EnzymeRules.forward(config, ::Const{typeof(nc_construle)}, ::Type{<:Const}, x::Const)
    return EnzymeRules.needs_primal(config) ? 100 * x.val : nothing
end
nc_construle_use(x, c) = x * nc_construle(c)
nc_construle_active(x) = 2 * nc_construle(x)

# A return with GC roots, returned through an sret buffer. Activity analysis gives the
# buffer a shadow although the argument is constant, so the return is not constant: the
# call needs a rule, and none applies. Its stub is still compiled (see
# `bind_ruled_stub!`).
struct NCRooted
    v::Vector{Float64}
    s::Float64
end
nc_rooted(c) = NCRooted([c, 2c], 3c)

function EnzymeRules.forward(config, ::Const{typeof(nc_rooted)}, ::Type, c::Duplicated)
    error("nc_rooted: the rule is not expected to run")
end
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(nc_rooted)}, ::Type, c::Active)
    error("nc_rooted: the rule is not expected to run")
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(nc_rooted)}, dret, tape, c::Active)
    error("nc_rooted: the rule is not expected to run")
end

nc_rooted_use(x, c) = (r = nc_rooted(c); x * r.s * r.v[2])

# A return with GC roots and inline data, from a function Julia does not inline, so that
# its stub, which returns through the sret and roots of its own parameters, stays a call.
# As for `nc_rooted`, the return is not constant.
@noinline nc_mixed(c) = (view(c, :), (1, 2, 3, 4))

function EnzymeRules.forward(config, ::Const{typeof(nc_mixed)}, ::Type, c::Duplicated)
    error("nc_mixed: the rule is not expected to run")
end
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(nc_mixed)}, ::Type, c::Duplicated)
    error("nc_mixed: the rule is not expected to run")
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(nc_mixed)}, dret, tape, c::Duplicated)
    error("nc_mixed: the rule is not expected to run")
end

nc_mixed_use(x, c) = (r = nc_mixed(c); x * r[1][2] * r[2][4])

# A keyword call of a function with a rule.
nc_kw(x; scale = 1.0) = scale * x^2
function EnzymeRules.forward(config, ::Const{typeof(nc_kw)}, ::Type, x::Duplicated; scale = 1.0)
    error("nc_kw: the rule is not expected to run")
end
nc_kw_use(x, c) = x * nc_kw(c; scale = 3.0)

# A reverse rule only: forward over reverse differentiates through the primal, which the
# inner reverse differentiation calls with constant arguments.
nc_rronly(x) = x^2

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(nc_rronly)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (x.val,))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(nc_rronly)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (2 * xv * dret.val,)
end

nc_rronly_use(a, c) = a * nc_rronly(c)
nc_rronly_grad(c) = autodiff_deferred(Reverse, Const(nc_rronly_use), Active, Active(1.0), Const(c))[1][1]

# Called natively, the primal is ordinary Julia code: `ignore_derivatives` is the
# identity, and `within_autodiff()` is false. Emitted, `EnzymeInterpreter` folds
# `within_autodiff()` to true, as in any code Enzyme compiles.
nc_within(x) = Enzyme.within_autodiff() ? 2x : 3x
function EnzymeRules.forward(config, ::Const{typeof(nc_within)}, ::Type, x::Duplicated)
    error("nc_within: the rule is not expected to run")
end
nc_within_use(x, c) = x * nc_within(c)

nc_ignore(x) = Enzyme.ignore_derivatives(x)^2
function EnzymeRules.forward(config, ::Const{typeof(nc_ignore)}, ::Type, x::Duplicated)
    error("nc_ignore: the rule is not expected to run")
end
nc_ignore_use(x, c) = x * nc_ignore(c)

@testset "Primals of functions with rules, in constant calls" begin
    # d/dx = 100 (the rule) + c^2 (the primal)
    @test autodiff(Forward, nc_ruled_use, Duplicated(3.0, 1.0), Const(2.0))[1] ≈ 104.0
    @test autodiff(ForwardWithPrimal, nc_ruled_use, Duplicated(3.0, 1.0), Const(2.0)) == (104.0, 21.0)
    @test autodiff(Reverse, nc_ruled_use, Active, Active(3.0), Const(2.0))[1][1] ≈ 104.0

    # d/dx = 3c^2
    @test autodiff(Forward, nc_kw_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ 12.0

    @test nc_rronly_grad(3.0) ≈ 9.0
    # d/dc c^2
    @test autodiff(Forward, nc_rronly_grad, Duplicated(3.0, 1.0))[1] ≈ 6.0

    # d/dx = 3c natively, 2c emitted
    native = v"1.12-beta3" <= VERSION < v"1.14-"
    @test autodiff(Forward, nc_within_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ (native ? 6.0 : 4.0)
end

# Emitted, the primal leaves `ignore_derivatives` to the Enzyme pipeline, which lowers it
# only in the code it differentiates.
@static if v"1.12-beta3" <= VERSION < v"1.14-"
    @testset "Natively called primals of functions with rules are ordinary Julia code" begin
        # d/dx = c^2
        @test autodiff(Forward, nc_ignore_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ 4.0
    end
end

@testset "Rules of functions with rules still apply" begin
    # An active call goes through the rule.
    @test autodiff(Forward, nc_ruled_active, Duplicated(3.0, 1.0))[1] ≈ 200.0
    @test autodiff(Reverse, nc_ruled_active, Active, Active(3.0))[1][1] ≈ 200.0

    # A rule for constant arguments applies to a constant call: d/dx = 100c
    @test autodiff(Forward, nc_construle_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ 200.0
    # No rule applies to an active call.
    @test_throws MethodError autodiff(Forward, nc_construle_active, Duplicated(2.0, 1.0))

    # A call whose return is not constant needs a rule, although its arguments are.
    @test_throws MethodError autodiff(Forward, nc_rooted_use, Duplicated(1.5, 1.0), Const(2.0))
    # Before Julia 1.12 the rule handler rejects the mixed activity of the return first
    # (see `enzyme_custom_setup_ret`).
    rooted_rev_error = VERSION < v"1.12" ? Enzyme.Compiler.MixedReturnException : MethodError
    @test_throws rooted_rev_error autodiff(Reverse, nc_rooted_use, Active, Active(1.5), Const(2.0))
    c = [1.0, 2.0]
    @test_throws MethodError autodiff(ForwardWithPrimal, nc_mixed_use, Duplicated(1.5, 1.0), Const(c))
    @test_throws MethodError autodiff(ReverseWithPrimal, nc_mixed_use, Active, Active(1.5), Const(c))
end
