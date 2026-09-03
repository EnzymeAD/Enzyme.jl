using Enzyme
using Test
using Enzyme.EnzymeRules

# How a rule reaches the calling module follows its inlining annotation. `@inline` rules
# are emitted into the calling module and always-inlined. `@noinline` rules are called
# through their natively compiled entry point. Unannotated rules follow Julia's inlining
# heuristics.

cube_inline(x) = x^3
cube_noinline(x) = x^3
cube_default(x) = x^3
cube_big(x) = x^3

@inline function EnzymeRules.forward(config, ::Const{typeof(cube_inline)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(cube_inline(x.val), 10 * 3 * x.val^2 * x.dval)
end

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_noinline)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(cube_noinline(x.val), 10 * 3 * x.val^2 * x.dval)
end

function EnzymeRules.forward(config, ::Const{typeof(cube_default)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(cube_default(x.val), 10 * 3 * x.val^2 * x.dval)
end

# This rule is too large for Julia's inlining heuristics.
function EnzymeRules.forward(config, ::Const{typeof(cube_big)}, ::Type{<:Duplicated}, x::Duplicated)
    acc = 0.0
    for i in 1:64
        acc += sin(x.val + i) * cos(x.val - i) + exp(-abs(x.val)) * log1p(abs(x.val) + i)
        acc += atan(x.val, i) + sqrt(abs(x.val) + i) + tanh(x.val * i) - expm1(-abs(x.val) - i)
    end
    return Duplicated(cube_big(x.val) + 0 * acc, 10 * 3 * x.val^2 * x.dval)
end

@noinline function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(cube_noinline)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (x.val,))
end

@noinline function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(cube_noinline)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (10 * 3 * xv^2 * dret.val,)
end

# This argument holds both a tracked pointer and data. On 1.12+ it has inline roots.
struct Scale
    v::Vector{Float64}
    c::Float64
end

scale_dot(p::Scale, x) = p.c * p.v[1] * x

@noinline function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(scale_dot)}, ::Type{<:Active}, p::Const{Scale}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? func.val(p.val, x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (p.val.c * p.val.v[1],))
end

@noinline function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(scale_dot)}, dret::Active, tape, p::Const{Scale}, x::Active
    )
    return (nothing, 10 * tape[1] * dret.val)
end

function scale_twice(p, x)
    y = scale_dot(p, x)
    p.v[1] = 5.0
    return y + scale_dot(p, x)
end

# In a loop, the roots of the overwritten argument come back from the tape.
function scale_loop(p, x)
    s = 0.0
    for i in 1:3
        s += scale_dot(p, x)
        p.v[1] += 1.0
    end
    return s
end

@testset "rule inlining annotations" begin
    for f in (cube_inline, cube_noinline, cube_default, cube_big)
        @test autodiff(ForwardWithPrimal, f, Duplicated(2.0, 1.0)) == (120.0, 8.0)
    end
    @test autodiff(Reverse, cube_noinline, Active(2.0))[1][1] == 120.0
    # 10 * (2 * 3) + 10 * (2 * 5). The argument is overwritten between the two calls.
    @test autodiff(Reverse, scale_twice, Const(Scale([3.0], 2.0)), Active(1.5))[1][2] == 160.0
    # 10 * 2 * (3 + 4 + 5)
    @test autodiff(Reverse, scale_loop, Const(Scale([3.0], 2.0)), Active(1.5))[1][2] == 240.0

    @static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
        world = Base.get_world_counter()
        C = EnzymeRules.FwdConfig{true, true, 1, false, false}
        function convention(f)
            TT = Tuple{C, Const{typeof(f)}, Type{Duplicated{Float64}}, Duplicated{Float64}}
            mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(EnzymeRules.forward), TT, world)
            ci = Enzyme.Compiler.rule_codeinst(mi, world)
            return Enzyme.Compiler.rule_call_convention(mi, ci)
        end
        @test convention(cube_inline) === :inline
        @test convention(cube_noinline) === :call
        @test convention(cube_default) === :inline
        @test convention(cube_big) === :call
    end
end

# A natively called rule is compiled like any other Julia code. `ignore_derivatives`
# is the identity there, and `within_autodiff()` is false. A rule emitted into the
# differentiated function is inferred by Enzyme's interpreter, where
# `within_autodiff()` is true.

cube_ignore(x) = x^3
cube_within(x) = x^3

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_ignore)}, ::Type{<:Duplicated}, x::Duplicated)
    xv = Enzyme.ignore_derivatives(x.val)
    return Duplicated(xv^3, 10 * 3 * xv^2 * x.dval)
end

@noinline function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(cube_ignore)}, ::Type{<:Active}, x::Active)
    xv = Enzyme.ignore_derivatives(x.val)
    primal = EnzymeRules.needs_primal(config) ? func.val(xv) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (xv,))
end

@noinline function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(cube_ignore)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (10 * 3 * xv^2 * Enzyme.ignore_derivatives(dret.val),)
end

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_within)}, ::Type{<:Duplicated}, x::Duplicated)
    sign = Enzyme.within_autodiff() ? 1.0 : -1.0
    return Duplicated(x.val^3, sign * 10 * 3 * x.val^2 * x.dval)
end

@testset "natively called rules are ordinary Julia code" begin
    @test autodiff(ForwardWithPrimal, cube_ignore, Duplicated(2.0, 1.0)) == (120.0, 8.0)
    @test autodiff(Reverse, cube_ignore, Active(2.0))[1][1] == 120.0
    @static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
        @test autodiff(ForwardWithPrimal, cube_within, Duplicated(2.0, 1.0)) == (-120.0, 8.0)
    else
        @test autodiff(ForwardWithPrimal, cube_within, Duplicated(2.0, 1.0)) == (120.0, 8.0)
    end
end
