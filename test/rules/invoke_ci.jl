using Enzyme
using Test
using Enzyme.EnzymeRules

# `invoke(f, ci::CodeInstance, args...)` to a function with a custom rule must apply the
# rule. It must not differentiate the invoked code. Julia 1.13 lowers such calls to a
# direct `:invoke` of the CodeInstance. Julia 1.12 leaves them as runtime `invoke` calls,
# which Enzyme does not support either way.

sq_invoke_ci(x) = x * x

function EnzymeRules.forward(config, ::Const{typeof(sq_invoke_ci)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(sq_invoke_ci(x.val), 100 * 2 * x.val * x.dval)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(sq_invoke_ci)}, ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(sq_invoke_ci)}, dret::Active, tape, x::Active
    )
    return (100 * 2 * x.val * dret.val,)
end

@static if VERSION >= v"1.13-"
    const CC = Core.Compiler
    const sq_mi = CC.specialize_method(which(sq_invoke_ci, (Float64,)), Tuple{typeof(sq_invoke_ci), Float64}, Core.svec())
    const sq_ci = CC.typeinf_ext_toplevel(CC.NativeInterpreter(Base.get_world_counter()), sq_mi, CC.SOURCE_MODE_ABI)

    invoke_only(x) = invoke(sq_invoke_ci, sq_ci, x)
    # Call the same signature twice, with distinct arguments: once dispatched,
    # once invoked.
    dispatch_and_invoke(x) = sq_invoke_ci(x) + invoke(sq_invoke_ci, sq_ci, 2x)

    @testset "invoke with a CodeInstance" begin
        @test sq_ci isa Core.CodeInstance
        @test invoke_only(3.0) == 9.0
        @test dispatch_and_invoke(3.0) == 45.0

        @test autodiff(ForwardWithPrimal, invoke_only, Duplicated(3.0, 1.0)) == (600.0, 9.0)
        @test autodiff(Reverse, invoke_only, Active(3.0))[1][1] == 600.0

        # The rule on x gives 600. The rule on 2x, with the chain factor 2, gives 2 * 1200.
        @test autodiff(ForwardWithPrimal, dispatch_and_invoke, Duplicated(3.0, 1.0)) == (3000.0, 45.0)
        @test autodiff(Reverse, dispatch_and_invoke, Active(3.0))[1][1] == 3000.0
    end
end

# The callee of `invoke(f, ci, args...)` need not have a singleton type: a
# callable struct and a closure instance carry data. Their rules apply too.

struct Layer
    w::Float64
end
(m::Layer)(x) = m.w * x * x

function EnzymeRules.forward(config, m::Const{Layer}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(m.val(x.val), 100 * 2 * m.val.w * x.val * x.dval)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, m::Const{Layer}, ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? m.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, m::Const{Layer}, dret::Active, tape, x::Active
    )
    return (100 * 2 * m.val.w * x.val * dret.val,)
end

make_scaler(w) = x -> w * x * x
const scaler = make_scaler(2.0)

function EnzymeRules.forward(config, f::Const{typeof(scaler)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(f.val(x.val), 100 * 2 * f.val.w * x.val * x.dval)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, f::Const{typeof(scaler)}, ::Type{<:Active}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? f.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, f::Const{typeof(scaler)}, dret::Active, tape, x::Active
    )
    return (100 * 2 * f.val.w * x.val * dret.val,)
end

@static if VERSION >= v"1.13-"
    const layer_mi = CC.specialize_method(which(Layer(1.0), (Float64,)), Tuple{Layer, Float64}, Core.svec())
    const layer_ci = CC.typeinf_ext_toplevel(CC.NativeInterpreter(Base.get_world_counter()), layer_mi, CC.SOURCE_MODE_ABI)
    const scaler_mi = CC.specialize_method(which(scaler, (Float64,)), Tuple{typeof(scaler), Float64}, Core.svec())
    const scaler_ci = CC.typeinf_ext_toplevel(CC.NativeInterpreter(Base.get_world_counter()), scaler_mi, CC.SOURCE_MODE_ABI)

    invoke_layer(m, x) = invoke(m, layer_ci, x)
    invoke_scaler(x) = invoke(scaler, scaler_ci, x)

    @testset "invoke with a CodeInstance, non-singleton callee" begin
        @test layer_ci isa Core.CodeInstance
        @test scaler_ci isa Core.CodeInstance
        @test invoke_layer(Layer(2.0), 3.0) == 18.0
        @test invoke_scaler(3.0) == 18.0

        # 100 * 2 * w * x = 1200 at w = 2, x = 3
        @test autodiff(ForwardWithPrimal, invoke_layer, Const(Layer(2.0)), Duplicated(3.0, 1.0)) == (1200.0, 18.0)
        @test autodiff(Reverse, invoke_layer, Const(Layer(2.0)), Active(3.0))[1][2] == 1200.0
        @test autodiff(ForwardWithPrimal, invoke_scaler, Duplicated(3.0, 1.0)) == (1200.0, 18.0)
        @test autodiff(Reverse, invoke_scaler, Active(3.0))[1][1] == 1200.0
    end
end
