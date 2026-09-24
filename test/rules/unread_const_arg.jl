using Enzyme
using Test
using Enzyme.EnzymeRules

# A rule may read an argument which the function it is defined for never touches.
# The argument must reach the rule also when the call sits behind a function that
# is not inlined, https://github.com/EnzymeAD/Enzyme.jl/issues/3570

struct RuleState
    codes::Vector{Int8}
    visits::Vector{Int}
end

unread_abs(state, value) = abs(value)

function EnzymeRules.forward(config, ::Const{typeof(unread_abs)}, RT::Type, state::Const{RuleState}, value::Duplicated)
    @assert state.val.codes[1] == 7
    state.val.visits[1] += 1
    primal = abs(value.val)
    derivative = ifelse(value.val < 0, -value.dval, value.dval)
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(primal, derivative)
    elseif needs_primal(config)
        return primal
    elseif needs_shadow(config)
        return derivative
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(config, ::Const{typeof(unread_abs)}, RT::Type{<:Active}, state::Const{RuleState}, value::Active)
    @assert state.val.codes[1] == 7
    state.val.visits[1] += 1
    primal = needs_primal(config) ? abs(value.val) : nothing
    return AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(config, ::Const{typeof(unread_abs)}, dret::Active, tape, state::Const{RuleState}, value::Active)
    @assert state.val.codes[1] == 7
    state.val.visits[1] += 1
    return (nothing, ifelse(value.val < 0, -dret.val, dret.val))
end

@noinline unread_helper(state, input) = unread_abs(state, input[1])

function unread_sink!(out, state, input)
    out[1] = unread_helper(state, input)
    return nothing
end

@testset "Rule reads an argument the primal does not" begin
    state = RuleState(Int8[7], [0])
    input = Float32[-2]
    dinput = Float32[1]
    out = zeros(Float32, 1)
    dout = zeros(Float32, 1)

    autodiff(Forward, unread_sink!, Const, Duplicated(out, dout), Const(state), Duplicated(input, dinput))
    @test out[1] == 2
    @test dout[1] == -1
    @test state.visits[1] == 1

    dinput .= 0
    dout .= 1
    autodiff(Reverse, unread_sink!, Const, Duplicated(out, dout), Const(state), Duplicated(input, dinput))
    @test dinput[1] == -1
    @test state.visits[1] == 3

    dinput .= 0
    dout .= 1
    fwd, rev = autodiff_thunk(
        ReverseSplitNoPrimal, Const{typeof(unread_sink!)}, Const,
        Duplicated{typeof(out)}, Const{RuleState}, Duplicated{typeof(input)}
    )
    tape, _, _ = fwd(Const(unread_sink!), Duplicated(out, dout), Const(state), Duplicated(input, dinput))
    rev(Const(unread_sink!), Duplicated(out, dout), Const(state), Duplicated(input, dinput), tape)
    @test dinput[1] == -1
    @test state.visits[1] == 5
end
