using Enzyme
using Test
using Enzyme.EnzymeRules

# `invoke(f, ci::CodeInstance, args...)` to a function with a custom rule must apply the
# rule rather than differentiate the invoked code. Julia 1.13 lowers such calls to a
# direct `:invoke` of the CodeInstance; 1.12 leaves them as runtime `invoke` calls, which
# Enzyme does not support either way.

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
    # a dispatched and an invoked call to the same signature, with distinct arguments
    dispatch_and_invoke(x) = sq_invoke_ci(x) + invoke(sq_invoke_ci, sq_ci, 2x)

    @testset "invoke with a CodeInstance" begin
        @test sq_ci isa Core.CodeInstance
        @test invoke_only(3.0) == 9.0
        @test dispatch_and_invoke(3.0) == 45.0

        @test autodiff(ForwardWithPrimal, invoke_only, Duplicated(3.0, 1.0)) == (600.0, 9.0)
        @test autodiff(Reverse, invoke_only, Active(3.0))[1][1] == 600.0

        # rule on x (600) plus rule on 2x with the chain factor 2 (2 * 1200)
        @test autodiff(ForwardWithPrimal, dispatch_and_invoke, Duplicated(3.0, 1.0)) == (3000.0, 45.0)
        @test autodiff(Reverse, dispatch_and_invoke, Active(3.0))[1][1] == 3000.0
    end
end
