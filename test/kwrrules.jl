module KWReverseRules

using Enzyme
using Enzyme.EnzymeRules
using Test

function f_kw(x; kwargs...)
    x^2
end

import .EnzymeRules: augmented_primal, reverse
using .EnzymeRules

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f_kw)}, ::Type{<:Active}, x::Active; kwargs...)
    @show kwargs
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::ConfigWidth{1}, ::Const{typeof(f_kw)}, dret::Active, tape, x::Active; kwargs...)
    @show kwargs # TODO do we want them here?
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

@test Enzyme.autodiff(Enzyme.Reverse, f_kw, Active(2.0))[1][1] ≈ 104.0

# TODO: autodiff wrapper with kwargs support
g(x, y) = f_kw(x; val=y)

@test Enzyme.autodiff(Enzyme.Reverse, g, Active(2.0), Const(42.0))[1][1] ≈ 104.0

end # KWReverseRules

