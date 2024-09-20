using Enzyme, Test


make3() = (1.0, 2.0, 3.0)

@testset "Deferred and deferred thunk" begin
    function dot(A)
        return A[1] * A[1] + A[2] * A[2] 
    end
    dA = zeros(2)
    A = [3.0, 5.0]
    thunk_dA, def_dA = copy(dA), copy(dA)
    def_A, thunk_A = copy(A), copy(A)
    primal = Enzyme.autodiff(ReverseWithPrimal, dot, Active, Duplicated(A, dA))[2]
    @test primal == 34.0
    primal = Enzyme.autodiff_deferred(ReverseWithPrimal, Const(dot), Active, Duplicated(def_A, def_dA))[2]
    @test primal == 34.0

    dup = Duplicated(thunk_A, thunk_dA)
    TapeType = Enzyme.EnzymeCore.tape_type(
        ReverseSplitWithPrimal,
        Const{typeof(dot)}, Active, Duplicated{typeof(thunk_A)}
    )
    @test Tuple{Float64,Float64}  === TapeType
    Ret = Active
    fwd, rev = Enzyme.autodiff_deferred_thunk(
        ReverseSplitWithPrimal,
        TapeType,
        Const{typeof(dot)},
        Ret,
        Duplicated{typeof(thunk_A)}
    )
    tape, primal, _  = fwd(Const(dot), dup)
    @test isa(tape, Tuple{Float64,Float64})
    rev(Const(dot), dup, 1.0, tape)
    @test all(primal == 34)
    @test all(dA .== [6.0, 10.0])
    @test all(dA .== def_dA)
    @test all(dA .== thunk_dA)

    function kernel(len, A)
        for i in 1:len
            A[i] *= A[i]
        end
    end

    A = Array{Float64}(undef, 64)
    dA = Array{Float64}(undef, 64)

    A .= (1:1:64)
    dA .= 1

    function aug_fwd(ctx, f::FT, ::Val{ModifiedBetween}, args...) where {ModifiedBetween, FT}
        TapeType = Enzyme.tape_type(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
        forward, reverse = Enzyme.autodiff_deferred_thunk(ReverseSplitModified(ReverseSplitWithPrimal, Val(ModifiedBetween)), TapeType, Const{Core.Typeof(f)}, Const, Const{Core.Typeof(ctx)}, map(Core.Typeof, args)...)
        forward(Const(f), Const(ctx), args...)[1]
        return nothing
    end

    ModifiedBetween = Val((false, false, true))

    aug_fwd(64, kernel, ModifiedBetween, Duplicated(A, dA))

end

@testset "Deferred upgrade" begin
    function gradsin(x)
        return gradient(Reverse, sin, x)[1]
    end
    res = Enzyme.gradient(Reverse, gradsin, 3.1)[1]
    @test res ≈ -sin(3.1)
end


function deadarg_pow(z::T, i) where {T<:Real}
    zabs = abs(z)
    if sign(z) < zero(T)
        return (zabs^i) * (cos(T(π) * i) + sin(T(π) * i)im)
    end
    return zabs^i + zero(T)im
end

function deadargtest(n)
    wp = 1 + deadarg_pow(-n, 0.5)

    deadarg_pow(-n, 0.5)

    return real(wp)
end

@testset "Dead arg elim" begin
    res = autodiff(Enzyme.ReverseWithPrimal, deadargtest, Active, Active(0.25))
    @test res[2] ≈ 1.0
end

@testset "Taylor series tests" begin

# Taylor series for `-log(1-x)`
# eval at -log(1-1/2) = -log(1/2)
function euroad(f::T) where T
    g = zero(T)
    for i in 1:10^7
        g += f^i / i
    end
    return g
end

euroad′(x) = first(autodiff(Reverse, euroad, Active, Active(x)))[1]

@test euroad(0.5) ≈ -log(0.5) # -log(1-x)
@test euroad′(0.5) ≈ 2.0 # d/dx -log(1-x) = 1/(1-x)
test_scalar(euroad, 0.5)
end
@noinline function womylogpdf(X::AbstractArray{<:Real})
  map(womylogpdf, X)
end

function womylogpdf(x::Real)
    (x - 2)
end


function wologpdf_test(x)
    return womylogpdf(x)
end

@testset "Ensure writeonly deduction combines with capture" begin
    res = Enzyme.autodiff(Enzyme.Forward, wologpdf_test, Duplicated([0.5], [0.7]))
    @test res[1] ≈ [0.7]
end

@testset "Nested AD" begin
    tonest(x,y) = (x + y)^2

    @test autodiff(Forward, (x,y) -> autodiff(Forward, Const(tonest), Duplicated(x, 1.0), Const(y))[1], Const(1.0), Duplicated(2.0, 1.0))[1] ≈ 2.0
end


let
    function loadsin2(xp)
        x = @inbounds xp[1]
        @inbounds xp[1] = 0.0
        sin(x)
    end
    global invsin2
    function invsin2(xp)
        xp = Base.invokelatest(convert, Vector{Float64}, xp)
        loadsin2(xp)
    end
    x = [2.0]
end

function grad_closure(f, x)
    function noretval(x,res)
        y = f(x)
        copyto!(res,y)
        return nothing
    end
    n = length(x)
    dx = zeros(n)
    y  = zeros(n)
    dy = zeros(n)
    dy[1] = 1.0

    autodiff(Reverse, Const(noretval), Duplicated(x,dx), Duplicated(y, dy))
    return dx
end

@testset "Closure" begin
    x = [2.0,6.0]
    dx = grad_closure(x->[x[1], x[2]], x)
    @test dx == [1.0, 0.0]
end

@testset "Null init tape" begin
    struct Leaf
        params::NamedTuple
    end

    function LeafF(n::Leaf)::Float32
        y = first(n.params.b2)
        r = convert(Tuple{Float32}, (y,))
        return r[1]
    end

    ps =
        (
            b2 = 1.0f0,
        )

    grads =
        (
            b2 = 0.0f0,
        )

    t1 = Leaf(ps)
    t1Grads = Leaf(grads)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((false, true))), Const{typeof(LeafF)}, Active, Duplicated{Leaf})
    tape, primal, shadow = forward(Const(LeafF), Duplicated(t1, t1Grads))


    struct Foo2{X,Y}
        x::X
        y::Y
    end

    test_f(f::Foo2) = f.x^2
    res = autodiff(Reverse, test_f, Active(Foo2(3.0, :two)))[1][1]
    @test res.x ≈ 6.0
    @test res.y == nothing
end

@testset "BoxFloat" begin
    function boxfloat(x)
        x = ccall(:jl_box_float64, Any, (Float64,), x)
        (sin(x)::Float64 + x)::Float64
    end
    @test 0.5838531634528576 ≈ Enzyme.autodiff(Reverse, boxfloat, Active, Active(2.0))[1][1]
    @test 0.5838531634528576 ≈ Enzyme.autodiff(Forward, boxfloat, Duplicated, Duplicated(2.0, 1.0))[1]
    res = Enzyme.autodiff(Forward, boxfloat, BatchDuplicated, BatchDuplicated(2.0, (1.0, 2.0)))[1]
    @test 0.5838531634528576 ≈ res[1]
    @test 1.1677063269057153 ≈ res[2]
end

"""
    J(ν, z) := ∑ (−1)^k / Γ(k+1) / Γ(k+ν+1) * (z/2)^(ν+2k)
"""
function mybesselj(ν, z, atol=1e-8)
    k = 0
    s = (z/2)^ν / factorial(ν)
    out = s
    while abs(s) > atol
        k += 1
        s *= (-1) / k / (k+ν) * (z/2)^2
        out += s
    end
    out
end
mybesselj0(z) = mybesselj(0, z)
mybesselj1(z) = mybesselj(1, z)

@testset "Bessel" begin
    autodiff(Reverse, mybesselj, Active, Const(0), Active(1.0))
    autodiff(Reverse, mybesselj, Active, Const(0), Active(1.0))
    autodiff(Forward, mybesselj, Const(0), Duplicated(1.0, 1.0))
    autodiff(Forward, mybesselj, Const(0), Duplicated(1.0, 1.0))
    @testset "besselj0/besselj1" for x in (1.0, -1.0, 0.0, 0.5, 10, -17.1,) # 1.5 + 0.7im)
        test_scalar(mybesselj0, x, rtol=1e-5, atol=1e-5)
        test_scalar(mybesselj1, x, rtol=1e-5, atol=1e-5)
    end
end

# Ensure that this returns an error, and not a crash
# https://github.com/EnzymeAD/Enzyme.jl/issues/368
abstract type TensorProductBasis <: Function end

struct LegendreBasis <: TensorProductBasis
    n::Int
end

function (basis::LegendreBasis)(x)
    return x
end

struct MyTensorLayer
    model::Array{TensorProductBasis}
end

function fn(layer::MyTensorLayer, x)
    model = layer.model
    return model[1](x)
end

const nn = MyTensorLayer([LegendreBasis(10)])

function dxdt_pred(x)
  return fn(nn, x)
end

@testset "AbstractType calling convention" begin
    # TODO get rid of runtime activity
    @test 1.0 ≈ Enzyme.autodiff(set_runtime_activity(Reverse), dxdt_pred, Active(1.0))[1][1]
end

function fillsum(x)
    a = similar(rand(3, 3))
    fill!(a, x)
    return sum(a)
end

@testset "Fill sum" begin
    res = autodiff(Forward, fillsum, Duplicated(2.0, 1.0))[1]
    @test 9.0 ≈ res
end

@inline function myquantile(v::AbstractVector, p::Real; alpha)
    n = length(v)
    
    m = 1.0 + p * (1.0 - alpha - 1.0)
    aleph = n*p + oftype(p, m)
    j = clamp(trunc(Int, aleph), 1, n-1)
    γ = clamp(aleph - j, 0, 1)

    if n == 1
        a = @inbounds v[1]
        b = @inbounds v[1]
    else
        a = @inbounds v[j]
        b = @inbounds v[j + 1]
    end
    
    return a + γ*(b-a)
end

function fquantile(x)
    v = [1.0, x]
    return @inbounds (map(y->myquantile(v, y, alpha=1.), [0.7]))[1]
end

@testset "Attributor issues" begin

    cor = fquantile(2.0)
    res = autodiff(ForwardWithPrimal, fquantile, Duplicated,Duplicated(2.0, 1.0))
    @test cor ≈ res[2]
    @test 0.7 ≈ res[1]

end

@testset "hmlstm" begin
    sigm(x)  = @fastmath 1 / (1 + exp(-x))
    @fastmath function hmlstm_update_c_scalar(z, zb, c, f, i, g)
        if z == 1.0f0 # FLUSH
            return sigm(i) * tanh(g)
        elseif zb == 0.0f0 # COPY
            return c
        else # UPDATE
            return sigm(f) * c + sigm(i) * tanh(g)
        end
    end

    N = 64
    Z = round.(rand(Float32, N))
    Zb = round.(rand(Float32, N))
    C = rand(Float32, N, N)
    F = rand(Float32, N, N)
    I = rand(Float32, N, N)
    G = rand(Float32, N, N)

    function broadcast_hmlstm(out, Z, Zb, C, F, I, G)
        out .= hmlstm_update_c_scalar.(Z, Zb, C, F, I, G)
        return nothing
    end

    ∇C = zeros(Float32, N, N)
    ∇F = zeros(Float32, N, N)
    ∇I = zeros(Float32, N, N)
    ∇G = zeros(Float32, N, N)

    # TODO(wsmoses): Check after updating Enzyme_jll
    # autodiff(broadcast_hmlstm, Const,
    #          Const(zeros(Float32, N, N)), Const(Z), Const(Zb),
    #          Duplicated(C, ∇C), Duplicated(F, ∇F), Duplicated(I, ∇I), Duplicated(G, ∇G))
    # fwddiff(broadcast_hmlstm, Const,
    #          Const(zeros(Float32, N, N)), Const(Z), Const(Zb),
    #          Duplicated(C, ∇C), Duplicated(F, ∇F), Duplicated(I, ∇I), Duplicated(G, ∇G))
end

@testset "No speculation" begin
	mutable struct SpecFoo

		iters::Int
		a::Float64
		b::Vector{Float64}

	end

	function f(Foo)
		for i = 1:Foo.iters

			c = -1.0

			if Foo.a < 0.0
				X = (-Foo.a)^0.25
				c = 2*log(X)
			end

			# set b equal to desired result
			Foo.b[1] = 1.0 / c

			return nothing
		end
	end

	foo  = SpecFoo(1, 1.0, zeros(Float64, 1))
	dfoo = SpecFoo(0, 0.0, zeros(Float64, 1))

	# should not throw a domain error, which
	# will occur if the pow is mistakenly speculated
	Enzyme.autodiff(Reverse, f, Duplicated(foo, dfoo))
end

genlatestsin(x)::Float64 = Base.invokelatest(sin, x)
function genlatestsinx(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    Base.invokelatest(sin, x)::Float64 + 1
end

function loadsin(xp)
    x = @inbounds xp[1]
    @inbounds xp[1] = 0.0
    sin(x)
end
function invsin(xp)
    xp = Base.invokelatest(convert, Vector{Float64}, xp)
    loadsin(xp)
end

@testset "generic" begin
    @test -0.4161468365471424 ≈ Enzyme.autodiff(Reverse, genlatestsin, Active, Active(2.0))[1][1]
    @test -0.4161468365471424 ≈ Enzyme.autodiff(Forward, genlatestsin, Duplicated(2.0, 1.0))[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(Reverse, genlatestsinx, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

    x = [2.0]
    dx = [0.0]
    Enzyme.autodiff(Reverse, invsin, Active, Duplicated(x, dx))
    @test 0 ≈ x[1]
    @test -0.4161468365471424 ≈ dx[1]

	function inactive_gen(x)
		n = 1
		for k in 1:2
			y = falses(n)
		end
		return x
	end
    @test 1.0 ≈ Enzyme.autodiff(Reverse, inactive_gen, Active, Active(1E4))[1][1]
	@test 1.0 ≈ Enzyme.autodiff(Forward, inactive_gen, Duplicated(1E4, 1.0))[1]

    function whocallsmorethan30args(R)
        temp = diag(R)     
         R_inv = [temp[1] 0. 0. 0. 0. 0.; 
             0. temp[2] 0. 0. 0. 0.; 
             0. 0. temp[3] 0. 0. 0.; 
             0. 0. 0. temp[4] 0. 0.; 
             0. 0. 0. 0. temp[5] 0.; 
         ]
    
        return sum(R_inv)
    end
    
    R = zeros(6,6)    
    dR = zeros(6, 6)

    @static if VERSION ≥ v"1.10-"
        @test_broken autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    else
        autodiff(Reverse, whocallsmorethan30args, Active, Duplicated(R, dR))
    	@test 1.0 ≈ dR[1, 1]
    	@test 1.0 ≈ dR[2, 2]
    	@test 1.0 ≈ dR[3, 3]
    	@test 1.0 ≈ dR[4, 4]
    	@test 1.0 ≈ dR[5, 5]
    	@test 0.0 ≈ dR[6, 6]
    end
end

@testset "invoke" begin
    @noinline apply(@nospecialize(func)) = func()

    function invtest(arr)
        function f()
           arr[1] *= 5.0
           nothing
        end
        apply(f)
    end

    x  = [2.0]
    dx = [1.0]

    Enzyme.autodiff(Reverse, invtest, Duplicated(x, dx))

    @test 10.0 ≈ x[1]
    @test 5.0 ≈ dx[1]
end

@testset "Extract Tuple for Reverse" begin
    autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(solve_cubic_eq)}, Const, Duplicated{Vector{Complex{Float64}}})
end

@testset "Forward on Reverse" begin

	function speelpenning(y, x)
		ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
								  y, x, 2 * 8)
		return nothing
	end

	x = [0.5, 0.3]
	y = zeros(2)
    dx = ones(2)
    rx = zeros(2)
    drx = zeros(2)
    dy = zeros(2)
    ry = ones(2)
    dry = zeros(2)

    function foo(y, dy, x, dx)
        autodiff(Reverse, speelpenning, Const, Duplicated(y, dy), Duplicated(x, dx))
        return nothing
    end

    autodiff(Forward, foo, Duplicated(x, dx), Duplicated(rx, drx), Duplicated(y, dy), Duplicated(ry, dry))
end


struct DensePE
    n_inp::Int
    W::Matrix{Float64}
end

struct NNPE
    layers::Tuple{DensePE, DensePE}
end


function set_paramsPE(nn, params)
    i = 1
    for l in nn.layers
        W = l.W # nn.layers[1].W
        Base.copyto!(W, reshape(view(params,i:(i+length(W)-1)), size(W)))
    end
end

@testset "Illegal phi erasure" begin
    # just check that it compiles
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(set_paramsPE)}, Const, Duplicated{NNPE}, Duplicated{Vector{Float64}})
    @test fwd !== nothing
    @test rev !== nothing
    nn = NNPE( ( DensePE(1, Matrix{Float64}(undef, 4, 4)), DensePE(1, Matrix{Float64}(undef, 4, 4)) ) )
    dnn = NNPE( ( DensePE(1, Matrix{Float64}(undef, 4, 4)), DensePE(1, Matrix{Float64}(undef, 4, 4)) ) )
    l = Vector{Float64}(undef, 32)
    dl = Vector{Float64}(undef, 32)
    fwd(Const(set_paramsPE), Duplicated(nn, dnn), Duplicated(l, dl))
end

@testset "Copy Broadcast arg" begin
	x = Float32[3]
	w = Float32[1]
	dw = zero(w)

	function inactiveArg(w, x, cond)
	   if cond
		  x = copy(x)
	   end
	  @inbounds w[1] * x[1]
	end

	Enzyme.autodiff(Reverse, inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))

    @test x ≈ [3.0]
    @test w ≈ [1.0]
    @test dw ≈ [3.0]

    x = Float32[3]

    function loss(w, x, cond)
      dest = Array{Float32}(undef, 1)
      r = cond ? copy(x) : x
      res = @inbounds w[1] * r[1]
      @inbounds dest[1] = res
      res
    end

    dw = Enzyme.autodiff(Reverse, loss, Active, Active(1.0), Const(x), Const(false))[1]

    @test x ≈ [3.0]
    @test dw[1] ≈ 3.0

    c = ones(3)
    inner(e) = c .+ e
    fres = Enzyme.autodiff(Enzyme.Forward, Const(inner), Duplicated{Vector{Float64}}, Duplicated([0., 0., 0.], [1., 1., 1.]))[1]
    @test c ≈ [1.0, 1.0, 1.0]
    @test fres ≈ [1.0, 1.0, 1.0]
end

@testset "View Splat" begin
	function getloc(locs, i)
		loss = 0.0
		if i==1
			x, y = 0.0, 0.0
		else
		# correct
			# x, y = locs[1,i-1], locs[2,i-1]
		# incorrect
		x, y = @inbounds locs[:,i-1]
		end
		loss += y
		return loss
	end

	x0 = ones(2, 9)
	din = zeros(2, 9)
	Enzyme.autodiff(Reverse, getloc, Duplicated(x0, din), Const(2))
	@test din[1, 1] ≈ 0.0
	@test din[2, 1] ≈ 1.0
end

@testset "Large dynamic tape" begin

	function ldynloss(X, Y, ps, bs)
		ll = 0.0f0
		for (x, y) in zip(X, Y)
			yhat = ps * x .+ bs
			ll += (yhat[1] - y)^2
		end
		return ll
	end

	ps = randn(Float32, (1, 5))
	bs = randn(Float32)

	X = map(x->rand(Float32, 5), 1:1000)
	Y = map(x->rand(Float32), 1:1000)

	grads = zero(ps)
	for epoch=1:1000
		fill!(grads, 0)
		autodiff(Reverse, ldynloss, Const(X), Const(Y), Duplicated(ps, grads), Active(bs))
	end

end

function absset(out, x)
    @inbounds out[1] = (x,)
    return nothing
end

@testset "Tape Width" begin
    struct Roo
        x::Float64
        bar::String63
    end

    struct Moo
        x::Float64
        bar::String63
    end

    function g(f)
        return f.x*5.0
    end

    res = autodiff(Reverse, g, Active, Active(Roo(3.0, "a")))[1][1]

    @test res.x == 5.0

    res = autodiff(Reverse, g, Active, Active(Moo(3.0, "a")))[1][1]

    @test res.x == 5.0
end

struct GDoubleField{T}
    this_field_does_nothing::T
    b::T
end

GDoubleField() = GDoubleField{Float64}(0.0, 1.0)
function fexpandempty(vec)
    x = vec[1]
    empty = []
    d = GDoubleField(empty...)
    return x ≤ d.b ? x * d.b : zero(x)
end

@testset "Constant Complex return" begin
    vec = [0.5]
    @test Enzyme.gradient(Enzyme.Reverse, fexpandempty, vec)[1] ≈ [1.0]
    @test Enzyme.gradient(Enzyme.Forward, fexpandempty, vec)[1] ≈ [1.0]
end

struct HarmonicAngle
    k::Float64
    t0::Float64
end

function harmonic_g(a, coords_i)
    return (a.k) * a.t0
end

function harmonic_f!(inter_list, coords, inters)
    si = 0.0
    for (i, b) in zip(inter_list, inters)
        si += harmonic_g(b, coords[i])
    end
    return si
end

@testset "Decay preservation" begin
    inters = [HarmonicAngle(1.0, 0.1), HarmonicAngle(2.0, 0.3)]
    inter_list = [1, 3]
    dinters = [HarmonicAngle(0.0, 0.0), HarmonicAngle(0.0, 0.0)]
    coords   = [(1.0, 2.0, 3.0), (1.1, 2.1, 3.1), (1.2, 2.2, 3.2)]
    d_coords = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    autodiff(
        Reverse,
        harmonic_f!,
        Active,
        Const(inter_list),
        Duplicated(coords, d_coords),
        Duplicated(inters, dinters),
    )

    @test dinters[1].k ≈ 0.1 
    @test dinters[1].t0 ≈ 1.0 
    @test dinters[2].k ≈ 0.3 
    @test dinters[2].t0 ≈ 2.0 
end

function objective!(x, loss, R)
    for i in 1:1000
        y = zeros(3)
        y[1] = R[1,1] * x[1] + R[1,2] * x[2] + R[1,3] * x[3]

        loss[] = y[1]
    end
    return nothing
end;

@testset "Static tape allocation" begin
    x = zeros(3)
    R = [1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0]
    loss = Ref(0.0)
    dloss = Ref(1.0)

    autodiff(Reverse, objective!, Duplicated(x, zero(x)), Duplicated(loss, dloss), Const(R))

    @test loss[] ≈ 0.0
    @show dloss[] ≈ 0.0
end



@inline function uns_mymean(f, A, ::Type{T}, c) where T
    c && return Base.inferencebarrier(nothing)
    x1 = f(@inbounds A[1]) / 1
    return @inbounds A[1][1]
end

function uns_sum2(x::Array{T})::T where T
    op = Base.add_sum
    itr = x
    y = iterate(itr)::Tuple{T, Int}
    v = y[1]::T
    while true
        y = iterate(itr, y[2])
        y === nothing && break
        v = (v + y[1])::T
    end
    return v
end

function uns_ad_forward(scale_diag::Vector{T}, c) where T 
    ccall(:jl_, Cvoid, (Any,), scale_diag) 
    res = uns_mymean(uns_sum2, [scale_diag,], T, c)
	return res
end

@testset "Split box float32" begin
    q = ones(Float32, 1)
    dx = make_zero(q)
    res, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        uns_ad_forward,
        Enzyme.Active,
        Enzyme.Duplicated(q, dx),
        Enzyme.Const(false),
    )
    @test dx ≈ Float32[1.0]
    q = ones(Float64, 1)
    dx = make_zero(q)
    res, y = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        uns_ad_forward,
        Enzyme.Active,
        Enzyme.Duplicated(q, dx),
        Enzyme.Const(false),
    )
    @test dx ≈ Float64[1.0]
end


