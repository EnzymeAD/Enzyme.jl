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


function f_kw2(x; kwargs...)
    x^2
end

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f_kw2)}, ::Type{<:Active}, x::Active)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::ConfigWidth{1}, ::Const{typeof(f_kw2)}, dret::Active, tape, x::Active)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

# Test that this errors due to missing kwargs in rule definition
g2(x, y) = f_kw2(x; val=y)
@test_throws ErrorException autodiff(Reverse, g2, Active(2.0), Const(42.0))[1][1]


function f_kw3(x; val=nothing)
    x^2
end

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f_kw3)}, ::Type{<:Active}, x::Active; dval=nothing)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::ConfigWidth{1}, ::Const{typeof(f_kw3)}, dret::Active, tape, x::Active; dval=nothing)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

# Test that this errors due to missing kwargs in rule definition
g3(x, y) = f_kw3(x; val=y)
@test_throws MethodError autodiff(Reverse, g3, Active(2.0), Const(42.0))[1][1]

function f_kw4(x; y=2.0)
    x*y
end

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f_kw4)}, ::Type{<:Active}, x::Active; y)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::ConfigWidth{1}, ::Const{typeof(f_kw4)}, dret::Active, tape, x::Active; y)
    return (1000*y+2*x.val*dret.val,)
end

# Test that this errors due to missing kwargs in rule definition
g4(x, y) = f_kw4(x; y)
@test autodiff(Reverse, g4, Active(2.0), Const(42.0))[1][1] ≈ 42004.0
@test_throws ErrorException autodiff(Reverse, g4, Active(2.0), Active(42.0))[1]

end # KWReverseRules

