module UnreadAndMixedCustomRuleArgs

using Enzyme
using Enzyme.EnzymeCore: EnzymeRules
using Test

struct EasyRuleUnreadArg
    x::Float64
    v::Vector{Float64}
end

@noinline easy_unread_arg(p::EasyRuleUnreadArg, w::Vector{Float64}) = 2.0 * w[1]
@noinline easy_partial_arg(p::EasyRuleUnreadArg, w::Vector{Float64}) = p.x * w[1]

EnzymeRules.@easy_rule(
    easy_unread_arg(p::EasyRuleUnreadArg, w::Vector{Float64}),
    (@Constant(p), [2.0]),
)

EnzymeRules.@easy_rule(
    easy_partial_arg(p::EasyRuleUnreadArg, w::Vector{Float64}),
    (@Constant(p), [p.x]),
)

easy_unread_loss(q) = easy_unread_arg(EasyRuleUnreadArg(q[1], [3.0]), q)
easy_unread_split_loss(q) = Base.inferencebarrier(f -> f(EasyRuleUnreadArg(2.0, [3.0]), q))(easy_unread_arg)
easy_partial_loss(q) = easy_partial_arg(EasyRuleUnreadArg(3.0, [5.0]), q)

@testset "Custom rule setup with readnone struct/roots parameters" begin
    @test Enzyme.gradient(Reverse, easy_unread_loss, [4.0])[1] ≈ [2.0]
    @test Enzyme.gradient(Enzyme.set_runtime_activity(Reverse), easy_unread_loss, [4.0])[1] ≈ [2.0]
    @test Enzyme.gradient(Reverse, easy_unread_split_loss, [4.0])[1] ≈ [2.0]
    @test Enzyme.gradient(Enzyme.set_runtime_activity(Reverse), easy_partial_loss, [4.0])[1] ≈ [3.0]
    @test Enzyme.gradient(Forward, easy_unread_loss, [4.0])[1] ≈ [2.0]
    @test Enzyme.gradient(Forward, easy_partial_loss, [4.0])[1] ≈ [3.0]
end

struct MixedConstInlineArg
    scale::Float64
    weights::Vector{Float64}
end

@noinline function mixed_const_inline_eval(p::MixedConstInlineArg, u::Vector{Float64})
    return p.weights .* (p.scale .* u)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig, ::Const{typeof(mixed_const_inline_eval)},
        ::Type{RT}, p::Annotation{MixedConstInlineArg}, u::Annotation{Vector{Float64}}
    ) where {RT}
    res = p.val.weights .* (p.val.scale .* u.val)
    dp = p isa Const ? zero(res) : (p.dval.weights .* (p.val.scale .* u.val) .+ p.val.weights .* (p.dval.scale .* u.val))
    du = u isa Const ? zero(res) : p.val.weights .* (p.val.scale .* u.dval)
    dres = dp .+ du
    if EnzymeRules.needs_primal(config)
        return EnzymeRules.needs_shadow(config) ? Duplicated(res, dres) : res
    else
        return EnzymeRules.needs_shadow(config) ? dres : nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(mixed_const_inline_eval)},
        RTA::Type{<:Duplicated}, p::Annotation{MixedConstInlineArg}, u::Annotation{Vector{Float64}}
    )
    res = p.val.weights .* (p.val.scale .* u.val)
    dres = EnzymeRules.needs_shadow(config) ? zero(res) : nothing
    tape = (dres, copy(p.val.weights), copy(u.val))
    return EnzymeRules.augmented_rule_return_type(config, RTA)(
        EnzymeRules.needs_primal(config) ? res : nothing, dres, tape
    )
end

const seen_rev_dscale = Ref(-1.0)

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1}, ::Const{typeof(mixed_const_inline_eval)},
        ::Type{<:Duplicated}, tape, p::Annotation{MixedConstInlineArg}, u::Annotation{Vector{Float64}}
    )
    dres, w, uval = tape
    rta = EnzymeRules.runtime_activity(config)
    if p isa MixedDuplicated
        seen_rev_dscale[] = p.dval[].scale
        if !(rta && p.dval[].weights === p.val.weights)
            p.dval[].weights .+= dres .* (p.val.scale .* uval)
        end
        dp = p.dval[]
        p.dval[] = MixedConstInlineArg(dp.scale + sum(dres .* (w .* uval)), dp.weights)
    end
    if !(u isa Const) && !(rta && u.dval === u.val)
        u.dval .+= dres .* (p.val.scale .* w)
    end
    fill!(dres, 0.0)
    return (nothing, nothing)
end

mixed_const_inline_active_weights(w, u) = sum(mixed_const_inline_eval(MixedConstInlineArg(2.0, w), u))
mixed_const_inline_const_prob(u) = sum(mixed_const_inline_eval(MixedConstInlineArg(2.0, [3.0, 5.0]), u))

@testset "Mixed struct with constant inline floats and active roots" begin
    w = [3.0, 5.0]
    dw = zeros(2)
    u = [4.0, 7.0]
    du = zeros(2)
    seen_rev_dscale[] = -1.0
    Enzyme.autodiff(
        Reverse, mixed_const_inline_active_weights, Active,
        Duplicated(w, dw), Duplicated(u, du)
    )
    @test seen_rev_dscale[] == 0.0
    @test dw ≈ [8.0, 14.0]
    @test du ≈ [6.0, 10.0]

    fill!(dw, 0.0)
    fill!(du, 0.0)
    seen_rev_dscale[] = -1.0
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Reverse), mixed_const_inline_active_weights, Active,
        Duplicated(w, dw), Duplicated(u, du)
    )
    @test seen_rev_dscale[] == 0.0
    @test dw ≈ [8.0, 14.0]
    @test du ≈ [6.0, 10.0]

    @test Enzyme.gradient(Reverse, mixed_const_inline_const_prob, [1.0, 2.0])[1] ≈ [6.0, 10.0]
    @test Enzyme.gradient(Enzyme.set_runtime_activity(Reverse), mixed_const_inline_const_prob, [1.0, 2.0])[1] ≈ [6.0, 10.0]

    # Forward mode with constant inline `2.0` (`p.dval.scale` must be zeroed to `0.0`, not `2.0`!)
    fwd_d = Enzyme.autodiff(
        Forward, mixed_const_inline_active_weights, Duplicated,
        Duplicated([3.0, 5.0], [1.0, -1.0]), Duplicated([4.0, 7.0], [0.5, 2.0])
    )
    @test fwd_d[1] ≈ 17.0
end

end # module UnreadAndMixedCustomRuleArgs
