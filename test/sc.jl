@testset "sc" begin

using Enzyme
using Enzyme: EnzymeRules
using LinearAlgebra

f(x) = x^2

function f_ip(x)
   x[1] *= x[1]
   return nothing
end

import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

struct Closure
    v::Vector{Float64}
end

function (cl::Closure)(x)
    val = cl.v[1] * x
    cl.v[1] = 0.0
    return val
end


function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{Closure},
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

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{Closure},
    dret::Active, tape, args::Vararg{Active,N}) where {N}

    dargs = ntuple(Val(N)) do i
        7 * args[1].val * dret.val + tape[1] * 1000
    end
    return dargs
end

@testset "Closure rule" begin
    cl = Closure([3.14])
    res = autodiff(Reverse, cl, Active, Active(2.7))[1][1]
    @test res ≈ 7 * 2.7 + 3.14 * 1000
    @test cl.v[1] ≈ 0.0
end

end # testset "sc"
