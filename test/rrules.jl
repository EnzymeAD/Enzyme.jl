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

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active}, x::Active)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(f)}, dret::Active, tape, x::Active)
    if needs_primal(config)
        return (10+2*x.val*dret.val,)
    else
        return (100+2*x.val*dret.val,)
    end
end

function augmented_primal(::RevConfig{false, false, 1}, func::Const{typeof(f_ip)}, ::Type{<:Const}, x::Duplicated)
    v = x.val[1]
    x.val[1] *= v
    return AugmentedReturn(nothing, nothing, v)
end

function reverse(::RevConfig{false, false, 1}, ::Const{typeof(f_ip)}, ::Type{<:Const}, tape, x::Duplicated)
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

function augmented_primal(config::RevConfigWidth{2}, func::Const{typeof(f)}, ::Type{<:Active}, x::Active)
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, nothing)
    else
        return AugmentedReturn(nothing, nothing, nothing)
    end
end

function reverse(config::RevConfigWidth{2}, ::Const{typeof(f)}, dret::Active, tape, x::Active)
    return ((10+2*x.val*dret.val,100+2*x.val*dret.val,))
end

function fip_2(out, in)
    out[] = f(in[])
    nothing
end

@testset "Batch ActiveReverse Rules" begin
    out = BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(3.0)))
    in = BatchDuplicated(Ref(2.0), (Ref(0.0), Ref(0.0)))
    # TODO: Not yet supported: Enzyme custom rule of batch size=2, and active return EnzymeCore.Active{Float64}
    @test_throws Enzyme.Compiler.EnzymeRuntimeException Enzyme.autodiff(Enzyme.Reverse, fip_2, out, in)
    @test_broken in.dvals[1][] ≈ 104.0
    @test_broken in.dvals[1][] ≈ 42.0
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
function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(q)}, ::Type{<:Active}, x::Active)
    tape = (Ref(2.0), Ref(3.4))
    if needs_primal(config)
        return AugmentedReturn(func.val(x.val), nothing, tape)
    else
        return AugmentedReturn(nothing, nothing, tape)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(q)}, dret::Active, tape, x::Active)
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

foo(x::Complex) = 2x

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(foo)},
    ::Type{<:Active},
    x
)
    r = func.val(x.val)
    if EnzymeRules.needs_primal(config)
        primal = func.val(x.val)
    else
        primal = nothing
    end
    if EnzymeRules.needs_shadow(config)
        shadow = zero(r)
    else
        shadow = nothing
    end
    tape = nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(foo)},
    dret,
    tape,
    y
)
    return (dret.val+13.0im,)
end

@testset "Complex values" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(foo)}, Active, Active{ComplexF64})
    z = 1.0+3im
    grad_u = rev(Const(foo), Active(z), 1.0 + 0.0im, fwd(Const(foo), Active(z))[1])[1][1]
    @test grad_u ≈ 1.0+13.0im
end

_scalar_dot(x, y) = conj(x) * y

function _dot(X::StridedArray{T}, Y::StridedArray{T}) where {T<:Union{Real,Complex}}
    return mapreduce(_scalar_dot, +, X, Y)
end

function augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(_dot)},
    ::Type{<:Union{Const,Active}},
    X::Duplicated{<:StridedArray{T}},
    Y::Duplicated{<:StridedArray{T}},
) where {T<:Union{Real,Complex}}
    r = func.val(X.val, Y.val)
    primal = needs_primal(config) ? r : nothing
    shadow = needs_shadow(config) ? zero(r) : nothing
    tape = (copy(X.val), copy(Y.val))
    return AugmentedReturn(primal, shadow, tape)
end

function reverse(
    ::RevConfigWidth{1},
    ::Const{typeof(_dot)},
    dret::Union{Active,Type{<:Const}},
    tape,
    X::Duplicated{<:StridedArray{T}},
    Y::Duplicated{<:StridedArray{T}},
) where {T<:Union{Real,Complex}}
    if !(dret isa Type{<:Const})
        Xtape, Ytape = tape
        X.dval .+= dret.val .* Ytape
        Y.dval .+= dret.val .* Xtape
    end
    return (nothing, nothing)
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/761
@testset "Correct primal computation for custom `dot`" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        n = 10
        x, y = randn(T, n), randn(T, n);
        ∂x, ∂y = map(zero, (x, y));
        val_exp = _dot(x, y)
        _, val = autodiff(
            ReverseWithPrimal, _dot, Const, Duplicated(x, ∂x), Duplicated(y, ∂y),
        )
        @test val ≈ val_exp
    end
end

function cmyfunc!(y, x) 
    y .= x
    nothing
end

function cprimal(x0, y0)
    x = copy(x0)
    y = copy(y0)
    for j in 1:2
        cmyfunc!(y, x)
        x .+= y
    end
    return @inbounds x[1]
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(cmyfunc!)}, ::Type{<:Const},
    y::Duplicated, x::Duplicated)
    cmyfunc!(y.val, x.val)
    tape = (copy(x.val), 3)
    return AugmentedReturn(nothing, nothing, tape)
end

const seen = Set()
function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(cmyfunc!)}, ::Type{<:Const}, tape,
    y::Duplicated,  x::Duplicated)
    xval = tape[1] 
    p = pointer(xval)
    @show p, seen
    @assert !in(p, seen)
    push!(seen, p)
    return (nothing, nothing)
end

@testset "Force caching on sret primal" begin
    x = fill(2.0, 3)
    y = fill(2.0, 3)
    dx = zero(x)
    dy = zero(y)
    autodiff(Reverse, Const(cprimal), Active, Duplicated(x, dx), Duplicated(y, dy))
end

function remultr(arg)
    arg * arg
end

function EnzymeRules.augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(remultr)},
    ::Type{<:Active}, args::Vararg{Active,N}) where {N}
    primal = if EnzymeRules.needs_primal(config)
        func.val(args[1].val)
    else
        nothing
    end
    return AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(config::RevConfigWidth{1}, func::Const{typeof(remultr)},
    dret::Active, tape, args::Vararg{Active,N}) where {N}

    dargs = ntuple(Val(N)) do i
        7 * args[1].val * dret.val
    end
    return dargs
end

function plaquette_sum(U)
    p = eltype(U)(0)

    for site in 1:length(U)
        p += remultr(@inbounds U[site])
    end

    return real(p)
end


@testset "No caching byref julia" begin
    U = Complex{Float64}[3.0 + 4.0im]
    dU = Complex{Float64}[0.0]

    autodiff(Reverse, plaquette_sum, Active, Duplicated(U, dU))

    @test dU[1] ≈ 7 * ( 3.0 + 4.0im )
end

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
    res = autodiff(Reverse, Const(cl), Active, Active(2.7))[1][1]
    @test res ≈ 7 * 2.7 + 3.14 * 1000
    @test cl.v[1] ≈ 0.0
end


function times2(wt_y)
    return wt_y*2
end
function EnzymeRules.augmented_primal(config, ::Const{typeof(times2)}, FA, x)
    return EnzymeRules.AugmentedReturn(2*x.val, nothing, nothing)
end
function EnzymeRules.reverse(config, ::Const{typeof(times2)}, FA, tape, arg)
    return (46.7*FA.val,)
end


function times2_ar(x)
	n = length(x)
    res = Vector{Float64}(undef, n)
	i = 1
	while true
        @inbounds res[i] = @inbounds times2(@inbounds x[i])
		if i == n
			break
		end
		i+=1
    end
    return res[3]::Float64
end

@testset "Zero diffe result" begin
    vals = [2.7, 5.6, 7.8, 12.2]
    dvals = zero(vals)
    Enzyme.autodiff(Reverse, times2_ar, Duplicated(vals, dvals))
    @test dvals ≈ [0., 0., 46.7, 0.]
end

unstabletape(x) = x^2

function augmented_primal(config::RevConfigWidth{1}, func::Const{typeof(unstabletape)}, ::Type{<:Active}, x::Active)
    tape = if x.val < 3
        400
    else
        (x.val +7 ) * 10
    end
    if needs_primal(config)
        return AugmentedReturn{eltype(x), Nothing, typeof(tape)}(func.val(x.val), nothing, tape)
    else
        return AugmentedReturn{Nothing, Nothing, typeof(tape)}(nothing, nothing, tape)
    end
end

function reverse(config::RevConfigWidth{1}, ::Const{typeof(unstabletape)}, dret, tape, x::Active{T}) where T
    return (T(tape)::T,)
end

unstabletapesq(x) = unstabletape(x)^2

@testset "Unstable Tape" begin
    @test Enzyme.autodiff(Enzyme.Reverse, unstabletape, Active(2.0))[1][1] ≈ 400.0
    @test Enzyme.autodiff(Enzyme.ReverseWithPrimal, unstabletape, Active(2.0))[1][1] ≈ 400.0
    @test Enzyme.autodiff(Enzyme.Reverse, unstabletape, Active(5.0))[1][1] ≈ (5.0 + 7) * 10
    @test Enzyme.autodiff(Enzyme.ReverseWithPrimal, unstabletape, Active(5.0))[1][1] ≈ (5.0 + 7) * 10

    @test Enzyme.autodiff(Enzyme.Reverse, unstabletapesq, Active(2.0))[1][1] ≈ (400.0)
    @test Enzyme.autodiff(Enzyme.ReverseWithPrimal, unstabletapesq, Active(2.0))[1][1] ≈ (400.0)
    @test Enzyme.autodiff(Enzyme.Reverse, unstabletapesq, Active(5.0))[1][1] ≈ ((5.0 + 7) * 10)
    @test Enzyme.autodiff(Enzyme.ReverseWithPrimal, unstabletapesq, Active(5.0))[1][1] ≈ ((5.0 + 7) * 10)
end

include("mixedrrule.jl")
end # ReverseRules
