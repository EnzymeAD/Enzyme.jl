using Enzyme
using Test
using Enzyme.EnzymeRules

# How a rule reaches the calling module follows its inlining annotation: `@inline` rules
# are emitted into it and always-inlined, `@noinline` rules are called through their
# natively compiled entry point, and unannotated rules follow Julia's inlining heuristics.

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

# large enough that Julia's heuristics would not inline it
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

@testset "rule inlining annotations" begin
    for f in (cube_inline, cube_noinline, cube_default, cube_big)
        @test autodiff(ForwardWithPrimal, f, Duplicated(2.0, 1.0)) == (120.0, 8.0)
    end
    @test autodiff(Reverse, cube_noinline, Active(2.0))[1][1] == 120.0

    @static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
        world = Base.get_world_counter()
        interp = Enzyme.Compiler.primal_interp_world(Forward, world)
        C = EnzymeRules.FwdConfig{true, true, 1, false, false}
        function convention(f)
            TT = Tuple{C, Const{typeof(f)}, Type{Duplicated{Float64}}, Duplicated{Float64}}
            mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(EnzymeRules.forward), TT, world)
            src = Core.Compiler.typeinf_code(interp, mi, true)
            return Enzyme.Compiler.rule_call_convention(src)
        end
        @test convention(cube_inline) === :inline
        @test convention(cube_noinline) === :call
        @test convention(cube_default) === :inline
        @test convention(cube_big) === :call
    end
end
