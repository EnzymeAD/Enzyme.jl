module ReverseRules

using Enzyme
using Enzyme: EnzymeRules
using LinearAlgebra
using Test

f(x) = x^2

function f_ip(x)
   x[1] *= x[1]
   return nothing
end

import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

Enzyme.API.printall!(true)

struct Closure
    v::Vector{Float64}
end

function (cl::Closure)(x)
    val = cl.v[1] * x
    cl.v[1] = 0.0
    return val
end


function EnzymeRules.augmented_primal(config::ConfigWidth{1}, func::Const{Closure},
    ::Type{<:Active}, args::Vararg{Active,N}) where {N}
    vec = copy(func.val.v)
    pval = func.val(args[1].val)
    primal = if EnzymeRules.needs_primal(config)
        pval
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, vec)
end

function EnzymeRules.reverse(config::ConfigWidth{1}, func::Const{Closure},
    dret::Active, tape, args::Vararg{Active,N}) where {N}

    @show tape
    @show dret
    @show args
    dargs = ntuple(Val(N)) do i
        fval = 7 * args[1].val * dret.val + tape[1] * 1000
        @show fval
        fval
    end
    return dargs
end

@testset "Closure rule" begin
    cl = Closure([3.14])
    res = autodiff(Reverse, cl, Active, Active(2.7))[1][1]
    @test res ≈ 7 * 2.7 + 3.14 * 1000
    @test cl[1] ≈ 0.0
end

end # ReverseRules
