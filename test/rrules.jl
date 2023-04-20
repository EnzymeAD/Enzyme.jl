module ReverseRules

using Enzyme
using Enzyme: EnzymeRules
using Test

f(x) = x^2

function f_ip(x)
   x[1] *= x[1]
   return nothing
end

import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active}, x::Active)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
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
    return AugmentedReturn(nothing, nothing, v)
end

function reverse(::Config{false, false, 1}, ::Const{typeof(f_ip)}, ::Type{<:Const}, tape, x::Duplicated)
    x.dval[1] = 100 + x.dval[1] * tape
    return (nothing,)
end

@testset "has_rrule" begin
    @test has_rrule_from_sig(Base.signature_type(f, Tuple{Float64}))
    @test has_rrule_from_sig(Base.signature_type(f_ip, Tuple{Vector{Float64}}))
end


@testset "Custom Reverse Rules" begin
    @test Enzyme.autodiff(Enzyme.Reverse, f, Active(2.0))[1][1] ≈ 104.0
    @test Enzyme.autodiff(Enzyme.Reverse, x->f(x)^2, Active(2.0))[1][1] ≈ 42.0

    x = [2.0]
    dx = [1.0]
    
    Enzyme.autodiff(Enzyme.Reverse, f_ip, Duplicated(x, dx))
    
    @test x ≈ [4.0]
    @test dx ≈ [102.0]
end

function alloc_sq(x)
    return Ref(x*x)
end

function h(x)
    alloc_sq(x)[]
end

function h2(x)
    y = alloc_sq(x)[]
    y * y
end

function augmented_primal(config, func::Const{typeof(alloc_sq)}, ::Type{<:Annotation}, x::Active{T}) where T
    primal = nothing
    # primal
    if needs_primal(config)
        primal = func.val(x.val)
    end

    shadref = Ref{T}(0)

    shadow = nothing
    # shadow
    if needs_shadow(config)
        shadow = shadref
    end
    
    return AugmentedReturn(primal, shadow, shadref)
end

function reverse(config, ::Const{typeof(alloc_sq)}, ::Type{<:Annotation}, tape, x::Active)
    if needs_primal(config)
        return (10*2*x.val*tape[],)
    else
        return (1000*2*x.val*tape[],)
    end
end

@testset "Shadow" begin
    @test Enzyme.autodiff(Reverse, h, Active(3.0)) == ((6000.0,),)
    @test Enzyme.autodiff(ReverseWithPrimal, h, Active(3.0))  == ((60.0,), 9.0)
    @test Enzyme.autodiff(Reverse, h2, Active(3.0))  == ((1080.0,),)
end

q(x) = x^2
function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(q)}, ::Type{<:Active}, x::Active)
    tape = (Ref(2.0), Ref(3.4))
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, tape)
    else
        return AugmentedReturn(nothing, nothing, tape)
    end
end

function reverse(config::ConfigWidth{1}, ::Const{typeof(q)}, dret::Active, tape, x::Active)
    @test tape[1][] == 2.0
    @test tape[2][] == 3.4
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

@testset "Byref Tape" begin
    @test Enzyme.autodiff(Enzyme.Reverse, q, Active(2.0))[1][1] ≈ 104.0
end

end # ReverseRules
