module ReverseRules

using Enzyme
using Enzyme: EnzymeRules
using Test

f(x) = x^2

function f_ip(x)
   x[1] *= x[1]
   return nothing
end

import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule, has_rrule_from_sig
using .EnzymeRules

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active}, x::Active)
    if needs_primal(config)
        return (func.val(x.val), nothing)
    else
        return (nothing, nothing)
    end
end

function reverse(config::ConfigWidth{1}, ::Const{typeof(f)}, dret::Active, tape, x::Active)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

function augmented_primal(::Config{false, false, 1}, func::Const{typeof(f_ip)}, ::Type{<:Const}, x::Duplicated)
    v = x.val[1]
    x.val[1] *= v
    return (nothing, v)
end

function reverse(::Config{false, false, 1}, ::Const{typeof(f_ip)}, ::Const, tape, x::Duplicated)
    x.dval[1] = 100 + x.dval[1] * tape
    return ()
end

@testset "has_rrule" begin
    @test has_rrule_from_sig(Base.signature_type(f, Tuple{Float64}))
    @test has_rrule_from_sig(Base.signature_type(f_ip, Tuple{Vector{Float64}}))
end


@testset "Custom Reverse Rules" begin
    @test Enzyme.autodiff(Enzyme.Reverse, f, Active(2.0))[1] ≈ 104.0
    @test Enzyme.autodiff(Enzyme.Reverse, x->f(x)^2, Active(2.0))[1] ≈ 42.0

    x = [2.0]
    dx = [1.0]
    
    Enzyme.autodiff(Enzyme.Reverse, f_ip, Duplicated(x, dx))
    
    @test x ≈ [4.0]
    @test dx ≈ [102.0]
end

end # ReverseRules
