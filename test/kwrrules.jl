module KWReverseRules

using Enzyme
using Enzyme.EnzymeRules
using Test

function f_kw(x; kwargs...)
    x^2
end

import .EnzymeRules: augmented_primal, reverse
using .EnzymeRules

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(f_kw)}, ::Type{<:Active}, x::Active; kwargs...)
    @assert length(overwritten(config)) == 2
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(f_kw)}, dret::Active, tape, x::Active; kwargs...)
    # TODO do we want kwargs here?
    @assert length(overwritten(config)) == 2
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

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(f_kw2)}, ::Type{<:Active}, x::Active)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(f_kw2)}, dret::Active, tape, x::Active)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

# Test that this errors due to missing kwargs in rule definition
g2(x, y) = f_kw2(x; val=y)
@test_throws MethodError autodiff(Reverse, g2, Active(2.0), Const(42.0))[1][1]


function f_kw3(x; val=nothing)
    x^2
end

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(f_kw3)}, ::Type{<:Active}, x::Active; dval=nothing)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(f_kw3)}, dret::Active, tape, x::Active; dval=nothing)
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

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(f_kw4)}, ::Type{<:Active}, x::Active; y)
    @assert length(overwritten(config)) == 2
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(f_kw4)}, dret::Active, tape, x::Active; y)
    @assert length(overwritten(config)) == 2
    return (1000*y+2*x.val*dret.val,)
end

# Test that this errors due to missing kwargs in rule definition
g4(x, y) = f_kw4(x; y)
@test autodiff(Reverse, g4, Active(2.0), Const(42.0))[1][1] ≈ 42004.0
@test_throws Enzyme.Compiler.EnzymeRuntimeException autodiff(Reverse, g4, Active(2.0), Active(42.0))[1]

struct Closure2
    v::Vector{Float64}
    str::String
end

function (cl::Closure2)(x; width=7)
    val = cl.v[1] * x * width
    cl.v[1] = 0.0
    return val
end

function wrapclos(cl, x)
    cl(x; width=9)
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{Closure2},
    ::Type{<:Active}, args::Vararg{Active,N}; width=7) where {N}
    vec = copy(func.val.v)
    pval = func.val(args[1].val)
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{Closure2},
    dret::Active, tape, args::Vararg{Active,N}; width=7) where {N}
    dargs = ntuple(Val(N)) do i
        7 * args[1].val * dret.val + tape[1] * 1000 + width * 100000
    end
    return dargs
end

@testset "KWClosure rule" begin
    cl = Closure2([3.14], "3.14")
    res = autodiff(Reverse, wrapclos, Active, Const(cl), Active(2.7))[1][2]
    @test res ≈ 7 * 2.7 + 3.14 * 1000 + 9 * 100000
    @test cl.v[1] ≈ 0.0
end

end # KWReverseRules

