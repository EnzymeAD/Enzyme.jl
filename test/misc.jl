using Enzyme, Test, Random
using FiniteDifferences
using InlineStrings

@isdefined(UTILS) || include("utils.jl")


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

    @test autodiff(Forward, (x,y) -> autodiff_deferred(Forward, tonest, Duplicated(x, 1.0), Const(y))[1], Const(1.0), Duplicated(2.0, 1.0))[1] ≈ 2.0
end

@testset "Nested Type Error" begin
    nested_f(x) = sum(tanh, x)

    function nested_df!(dx, x)
        make_zero!(dx)
        autodiff_deferred(Reverse, nested_f, Active, Duplicated(x, dx))
        return nothing
    end

    function nested_hvp!(hv, v, x)
        make_zero!(hv)
        autodiff(Forward, nested_df!, Const, Duplicated(make_zero(x), hv), Duplicated(x, v))
        return nothing
    end

    x = [0.5]

    # primal: sanity check
    @test nested_f(x) ≈ sum(tanh, x)

    # gradient: works
    dx = make_zero(x)
    nested_df!(dx, x)

    @test dx ≈ (sech.(x).^2)

    v = first(onehot(x))
    hv = make_zero(v)
    nested_hvp!(hv, v, x)
end

@testset "Generic Active Union Return" begin

    function generic_union_ret(A)
            if 0 < length(A)
                @inbounds A[1]
            else
                nothing
                Base._InitialValue()
            end
    end

    function outergeneric(weights::Vector{Float64})::Float64
        v = generic_union_ret(Base.inferencebarrier(weights))
        return v::Float64
    end

    weights = [0.2]
    dweights = [0.0]

    autodiff(Reverse, outergeneric, Duplicated(weights, dweights))

    @test dweights[1] ≈ 1.
end

@testset "Null init union" begin
    @noinline function unionret(itr, cond)
        if cond
            return Base._InitialValue()
        else
            return itr[1]
        end
    end

    function fwdunion(data::Vector{Float64})::Real
        unionret(data, false)
    end

    data = ones(Float64, 500)
    ddata = zeros(Float64, 500)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((false, true))), Const{typeof(fwdunion)}, Active, Duplicated{Vector{Float64}})
    tape, primal, shadow = forward(Const(fwdunion), Duplicated(data, ddata))

	function firstimpl(itr)
		v = firstfold(itr)
		@assert !(v isa Base._InitialValue)
		return v
	end

	function firstfold(itr)
		op, itr = Base._xfadjoint(Base.BottomRF(Base.add_sum), Base.Generator(Base.identity, itr))
		y = iterate(itr)
		init = Base._InitialValue()
		y === nothing && return init
		v = op(init, y[1])
		return v
	end

	function smallrf(weights::Vector{Float64}, data::Vector{Float64})::Float64
		itr1 = (weight for (weight, mean) in zip(weights, weights))

		itr2 = (firstimpl(itr1) for x in data)

		firstimpl(itr2)
	end

	data = ones(Float64, 1)

	weights = [0.2]
	dweights = [0.0]
    # Technically this test doesn't need runtimeactivity since the closure combo of active itr1 and const data
    # doesn't use any of the const data values, but now that we error for activity confusion, we need to
    # mark runtimeActivity to let this pass
    Enzyme.API.runtimeActivity!(true)
    Enzyme.autodiff(Enzyme.Reverse, Const(smallrf), Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    @test dweights[1] ≈ 1.

    function invokesum(weights::Vector{Float64}, data::Vector{Float64})::Float64
        sum(
            sum(
                weight
                for (weight, mean) in zip(weights, weights)
            )
            for x in data
        )
    end

    data = ones(Float64, 20)

    weights = [0.2, 0.8]
    dweights = [0.0, 0.0]

    Enzyme.autodiff(Enzyme.Reverse, invokesum, Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    Enzyme.API.runtimeActivity!(false)
    @test dweights[1] ≈ 20.
    @test dweights[2] ≈ 20.
end

@testset "Compare against" begin
    x = 3.0
    fd = central_fdm(5, 1)(sin, x)

    @test fd ≈ cos(x)
    @test fd ≈ first(autodiff(Reverse, sin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, sin, Duplicated(x, 1.0)))

    x = 0.2 + sin(3.0)
    fd = central_fdm(5, 1)(asin, x)

    @test fd ≈ 1/sqrt(1-x*x)
    @test fd ≈ first(autodiff(Reverse, asin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, asin, Duplicated(x, 1.0)))
    test_scalar(asin, x)

    function foo(x)
        a = sin(x)
        b = 0.2 + a
        c = asin(b)
        return c
    end

    x = 3.0
    fd = central_fdm(5, 1)(foo, x)

    @test fd ≈ cos(x)/sqrt(1-(0.2+sin(x))*(0.2+sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1.0)))
    test_scalar(foo, x)

    # Input type shouldn't matter
    x = 3
    @test fd ≈ cos(x)/sqrt(1-(0.2+sin(x))*(0.2+sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    # They do matter for duplicated, which can't be auto promoted
    # @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1)))

    f74(a, c) = a * √c
    @test √3 ≈ first(autodiff(Reverse, f74, Active, Active(2), Const(3)))[1]
    @test √3 ≈ first(autodiff(Forward, f74, Duplicated(2.0, 1.0), Const(3)))
end

@testset "BoxFloat" begin
    function boxfloat(x)
        x = ccall(:jl_box_float64, Any, (Float64,), x)
        (sin(x)::Float64 + x)::Float64
    end
    @test 0.5838531634528576 ≈ Enzyme.autodiff(Reverse, boxfloat, Active, Active(2.0))[1][1]
    @test 0.5838531634528576 ≈ Enzyme.autodiff(Forward, boxfloat, DuplicatedNoNeed, Duplicated(2.0, 1.0))[1]
    res = Enzyme.autodiff(Forward, boxfloat, BatchDuplicatedNoNeed, BatchDuplicated(2.0, (1.0, 2.0)))[1]
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
    Enzyme.API.runtimeActivity!(true)
    @test 1.0 ≈ Enzyme.autodiff(Reverse, dxdt_pred, Active(1.0))[1][1]
    Enzyme.API.runtimeActivity!(false)
end


mutable struct RTGData
	x
end

@noinline function rtg_sub(V, cv)
	return cv
end

@noinline function rtg_cast(cv)
	return cv
end

function rtg_f(V,@nospecialize(cv))
	s = rtg_sub(V, Base.inferencebarrier(cv))::RTGData
	s = rtg_cast(Base.inferencebarrier(s.x))::Float64
	return s
end

@testset "RuntimeActivity generic call" begin
    Enzyme.API.runtimeActivity!(true)
    res = autodiff(Forward, rtg_f, Duplicated, Duplicated([0.2], [1.0]), Const(RTGData(3.14)))
    @test 3.14 ≈ res[1]
    @test 0.0 ≈ res[2]
    Enzyme.API.runtimeActivity!(false)
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
    res = autodiff(Forward, fquantile, Duplicated,Duplicated(2.0, 1.0))
    @test cor ≈ res[1]
    @test 0.7 ≈ res[2]
end

@testset "DiffTest" begin
    include("DiffTests.jl")

    n = 1 + rand()
    x, y = 1 .+ rand(5, 5), 1 .+ rand(5)
    A, B = 1 .+ rand(5, 5), 1 .+ rand(5, 5)

    # f returns Number
    @testset "Number to Number" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
        test_scalar(f, n; rtol=1e-6, atol=1e-6)
    end

    @testset "Vector to Number" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
        test_matrix_to_number(f, y; rtol=1e-6, atol=1e-6)
    end

    @testset "Matrix to Number" for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
        test_matrix_to_number(f, x; rtol=1e-6, atol=1e-6)
    end

    # TODO(vchuravy/wsmoses): Enable these tests
    # for f in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS
    #     @test isa(f(A, B, x), Number)
    # end

    # # f returns Array

    # for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f(n), Array)
    # end

    # for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f(A), Array)
    #     @test isa(f(y), Array)
    # end

    # for f in DiffTests.MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A), Array)
    # end

    # for f in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS
    #     @test isa(f(A, B), Array)
    # end

    # # f! returns Nothing

    # for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    #     @test isa(f!(y, x), Nothing)
    # end

    # for f! in DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS
    #     @test isa(f!(y, n), Nothing)
    # end

end

@testset "IO" begin

    function printsq(x)
        println(x)
        x*x
    end

    @test 4.6 ≈ first(autodiff(Reverse, printsq, Active, Active(2.3)))[1]
    @test 4.6 ≈ first(autodiff(Forward, printsq, Duplicated(2.3, 1.0)))

    function tostring(x)
        string(x)
        x*x
    end

    @test 4.6 ≈ first(autodiff(Reverse, tostring, Active, Active(2.3)))[1]
    @test 4.6 ≈ first(autodiff(Forward, tostring, Duplicated(2.3, 1.0)))
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

@testset "Dynamic Val Construction" begin
    dyn_f(::Val{D}) where D = prod(D)
    dyn_mwe(x, t) = x / dyn_f(Val(t))

    @test 0.5 ≈ Enzyme.autodiff(Reverse, dyn_mwe, Active, Active(1.0), Const((1, 2)))[1][1]
end

@testset "No inference" begin
    c = 5.0
    @test 5.0 ≈ autodiff(Reverse, (A,)->c * A, Active, Active(2.0))[1][1]
    @test 5.0 ≈ autodiff(Forward, (A,)->c * A, Duplicated(2.0, 1.0))[1]
end

@testset "Type-instable capture" begin
    L = Array{Float64, 1}(undef, 2)

    F = [1.0, 0.0]

    function main()
        t = 0.0

        function cap(m)
            t = m
        end

        @noinline function inner(F, cond)
            if cond
                genericcall(F)
            end
        end

        function tobedifferentiated(F, cond)
            inner(F, cond)
            # Force an apply generic
            -t
            nothing
        end
        autodiff(Reverse, Const(tobedifferentiated), Duplicated(F, L), Const(false))
        autodiff(Forward, Const(tobedifferentiated), Duplicated(F, L), Const(false))
    end

    main()
end

@testset "Type" begin
    function foo(in::Ptr{Cvoid}, out::Ptr{Cvoid})
        markType(Float64, in)
        ccall(:memcpy,Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), out, in, 8)
    end

    x = [2.0]
    y = [3.0]
    dx = [5.0]
    dy = [7.0]

    @test markType(x) === nothing
    @test markType(zeros(Float32, 64)) === nothing
    @test markType(view(zeros(64), 16:32)) === nothing

    GC.@preserve x y begin
        foo(Base.unsafe_convert(Ptr{Cvoid}, x), Base.unsafe_convert(Ptr{Cvoid}, y))
    end

    GC.@preserve x y dx dy begin
      autodiff(Reverse, foo,
                Duplicated(Base.unsafe_convert(Ptr{Cvoid}, x), Base.unsafe_convert(Ptr{Cvoid}, dx)),
                Duplicated(Base.unsafe_convert(Ptr{Cvoid}, y), Base.unsafe_convert(Ptr{Cvoid}, dy)))
    end
end

function solve_cubic_eq(poly::AbstractVector{Complex{T}}) where T
    a1  =  1 / @inbounds poly[1]
    E1  = 2*a1
    E12 =  E1*E1
    s1 = log(E12)
    return nothing
end

@testset "Extract Tuple for Reverse" begin
    autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(solve_cubic_eq)}, Const, Duplicated{Vector{Complex{Float64}}})
end


@testset "GetField" begin
    mutable struct MyType
       x::Float64
    end

    getfield_idx(v, idx) = ccall(:jl_get_nth_field_checked, Any, (Any, UInt), v, idx)

    function gf(v::MyType, fld::Symbol)
       x = getfield(v, fld)
       x = x::Float64
       2 * x
    end

    function gf(v::MyType, fld::Integer)
       x = getfield_idx(v, fld)
       x = x::Float64
       2 * x
    end

    function gf2(v::MyType, fld::Integer, fld2::Integer)
       x = getfield_idx(v, fld)
       y = getfield_idx(v, fld2)
       x + y
    end

    function gf2(v::MyType, fld::Symbol, fld2::Symbol)
       x = getfield(v, fld)
       y = getfield(v, fld2)
       x + y
    end

    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf, Active, Duplicated(mx, dx), Const(:x))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0


    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf, Active, Duplicated(mx, dx), Const(0))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0


    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf2, Active, Duplicated(mx, dx), Const(:x), Const(:x))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0

    mx = MyType(3.0)
    dx = MyType(0.0)

    Enzyme.autodiff(Reverse, gf2, Active, Duplicated(mx, dx), Const(0), Const(0))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0

    function forbatch(v, fld::Symbol, out)
        x = getfield(v, fld)
        x = x::Float64
        out[] = 2 * x
        nothing
    end
    function forbatch(v, fld::Integer, out)
        x = getfield_idx(v, fld)
        x = x::Float64
        out[] = 2 * x
        nothing
    end

    mx = MyType(3.0)
    dx = MyType(0.0)
    dx2 = MyType(0.0)

    Enzyme.autodiff(Reverse, forbatch, Const, BatchDuplicated(mx, (dx, dx2)), Const(:x), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(3.14))))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0
    @test dx2.x ≈ 6.28

    mx = MyType(3.0)
    dx = MyType(0.0)
    dx2 = MyType(0.0)

    Enzyme.autodiff(Reverse, forbatch, Const, BatchDuplicated(mx, (dx, dx2)), Const(0), BatchDuplicated(Ref(0.0), (Ref(1.0), Ref(3.14))))
    @test mx.x ≈ 3.0
    @test dx.x ≈ 2.0
    @test dx2.x ≈ 6.28

    mutable struct MyType2
       x::Float64
       y::Float64
    end

    function sf2(v::MyType2, fld, fld2)
       x = getfield(v, fld)
       x = x::Float64
       r = 2 * x
       x = setfield!(v, fld2, r)
       return nothing
    end

    mt2 = MyType2(3.0, 642.0)
    dmt2 = MyType2(1.2, 541.0)

    Enzyme.autodiff(Forward, sf2, Duplicated(mt2, dmt2), Const(:x), Const(:y))
    @test mt2.x ≈ 3.0
    @test mt2.y ≈ 6.0
    @test dmt2.x ≈ 1.2
    @test dmt2.y ≈ 2.4

    function sf_for2(v, fld, fld2, x)
       setfield!(v, fld, 0.0)
       for i in 1:100
            setfield!(v, fld2, getfield(v, fld)::Float64 + x * i)
       end
       return getfield(v, fld)::Float64
    end

    mt2 = MyType2(0.0, 0.0)
    dmt2 = MyType2(0.0, 0.0)

    adres = Enzyme.autodiff(Reverse, sf_for2, Duplicated(mt2, dmt2), Const(:x), Const(:x), Active(3.1))
    @test adres[1][4] ≈ 5050.0

    mutable struct MyType3
       x::Base.RefValue{Float64}
       y::Base.RefValue{Float64}
    end

    function sf_for3(v, fld, fld2, x)
       setfield!(v, fld, Ref(0.0))
       for i in 1:100
            setfield!(v, fld2, Base.Ref((getfield(v, fld)::Base.RefValue{Float64})[] + x * i))
       end
       return (getfield(v, fld)::Base.RefValue{Float64})[]
    end

    mt3 = MyType3(Ref(0.0), Ref(0.0))
    dmt3 = MyType3(Ref(0.0), Ref(0.0))

    adres = Enzyme.autodiff(Reverse, sf_for3, Duplicated(mt3, dmt3), Const(:x), Const(:x), Active(3.1))
    @test adres[1][4] ≈ 5050.0
    
    mutable struct MyTypeM
       x::Float64
       y
    end

    @noinline function unstable_mul(x, y)
        return (x*y)::Float64
    end

    function gf3(y, v::MyTypeM, fld::Symbol)
       x = getfield(v, fld)
       unstable_mul(x, y)
    end

    function gf3(y, v::MyTypeM, fld::Integer)
       x = getfield_idx(v, fld)
       unstable_mul(x, y)
    end
    
    mx = MyTypeM(3.0, 1)
    res = Enzyme.autodiff(Reverse, gf3, Active, Active(2.7), Const(mx), Const(:x))
    @test mx.x ≈ 3.0
    @test res[1][1] ≈ 3.0
    
    mx = MyTypeM(3.0, 1)
    res = Enzyme.autodiff(Reverse, gf3, Active, Active(2.7), Const(mx), Const(0))
    @test mx.x ≈ 3.0
    @test res[1][1] ≈ 3.0
end


struct GFUniform{T}
    a::T
    b::T
end
GFlogpdf(d::GFUniform, ::Real) = -log(d.b - d.a)

struct GFNormal{T}
    μ::T
    σ::T
end
GFlogpdf(d::GFNormal, x::Real) = -(x - d.μ)^2 / (2 * d.σ^2)

struct GFProductDist{V}
    dists::V
end
function GFlogpdf(d::GFProductDist, x::Vector)
    dists = d.dists
    s = zero(eltype(x))
    for i in eachindex(x)
	s += GFlogpdf(dists[i], x[i])
    end
    return s
end

struct GFNamedDist{Names, D<:NamedTuple{Names}}
    dists::D
end

function GFlogpdf(d::GFNamedDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = d.dists
    return mapreduce((dist, acc) -> GFlogpdf(dist, acc), +, dists, vt)
end


@testset "Getfield with reference" begin
    Enzyme.API.runtimeActivity!(true)

    d = GFNamedDist((;a = GFNormal(0.0, 1.0), b = GFProductDist([GFUniform(0.0, 1.0), GFUniform(0.0, 1.0)])))
    p = (a = 1.0, b = [0.5, 0.5])
    dp = Enzyme.make_zero(p)
    GFlogpdf(d, p)
    autodiff(Reverse, GFlogpdf, Active, Const(d), Duplicated(p, dp))
    Enzyme.API.runtimeActivity!(false)
end

   
function indirectfltret(a)::DataType
    a[] *= 2
    return Float64
end
@testset "Partial return information" begin
    d = Duplicated(Ref(3.0), Ref(0.0))
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(indirectfltret)}, Const{DataType}, typeof(d))

    tape, primal, shadow = fwd(Const(indirectfltret), d)
    @test tape == nothing
    @test primal == Float64
    @test shadow == nothing
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

@testset "Union return" begin
    function unionret(a, out, cond)
        if cond
            out[] = a
        end
    end

    out = Ref(0.0)
    dout = Ref(1.0)
    @test 2.0 ≈ Enzyme.autodiff(Reverse, unionret, Active, Active(2.0), Duplicated(out, dout), Const(true))[1][1]
end

struct MyFlux
end

@testset "Union i8" begin
    args = (
        Val{(false, false, false)},
        Val(1),
        Val((true, true, true)),
        Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
        Base.getindex,
        nothing,
        ((nothing,), MyFlux()),
        ((nothing,), MyFlux()),
        1,
        nothing
    )
    
    nt1 = Enzyme.Compiler.runtime_generic_augfwd(args...)
    @test nt1[1] == (nothing,)
    @test nt1[2] == (nothing,)
    
    args2 = (
        Val{(false, false, false)},
        Val(1),
        Val((true, true, true)),
        Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
        Base.getindex,
        nothing,
        ((nothing,), MyFlux()),
        ((nothing,), MyFlux()),
        2,
        nothing
    )
    
    nt = Enzyme.Compiler.runtime_generic_augfwd(args2...)
    @test nt[1] == MyFlux()
    @test nt[2] == MyFlux()
end

@testset "Batched inactive" begin
    augres = Enzyme.Compiler.runtime_generic_augfwd(Val{(false, false, false)}, Val(2), Val((true, true, true)),
                                                    Val(Enzyme.Compiler.AnyArray(2+Int(2))),
                                ==, nothing, nothing,
                                :foo, nothing, nothing,
                                :bar, nothing, nothing)

    Enzyme.Compiler.runtime_generic_rev(Val{(false, false, false)}, Val(2), Val((true, true, true)), augres[end],
                                ==, nothing, nothing,
                                :foo, nothing, nothing,
                                :bar, nothing, nothing)
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
        autodiff_deferred(Reverse, speelpenning, Const, Duplicated(y, dy), Duplicated(x, dx))
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

@testset "Union return getproperty" begin
	using Enzyme

	struct DOSData
		interp_func
	end

	function get_dos(Ef=0.)
		return x->x+Ef
	end

	struct MyMarcusHushChidseyDOS
		A::Float64
		dos::DOSData
	end

	mhcd = MyMarcusHushChidseyDOS(0.3,  DOSData(get_dos()));

	function myintegrand(V, a_r)
		function z(E)
			dos = mhcd.dos

			interp = dos.interp_func

			res = interp(V)

			return res
		end
		return z
	end

	function f2(V)
		fn = myintegrand(V, 1.0)

		fn(0.0)
	end

    Enzyme.API.runtimeActivity!(true)
    res = autodiff(Forward, Const(f2), Duplicated, Duplicated(0.2, 1.0))
    Enzyme.API.runtimeActivity!(false)
    @test res[1] ≈ 0.2
    # broken as the return of an apply generic is {primal, primal}
    # but since the return is abstractfloat doing the 
    @test res[2] ≈ 1.0
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

@inline extract_bc(bc, ::Val{:north}) = (bc.north)
@inline extract_bc(bc, ::Val{:top}) = (bc.top)

function permute_boundary_conditions(boundary_conditions)
    sides = [:top, :north] # changing the order of these actually changes the error
    boundary_conditions = Tuple(extract_bc(boundary_conditions, Val(side)) for side in sides)

    return nothing
end

@testset "Extract abstype" begin

    parameters = (a = 1, b = 0.1)

    bc   = (north=1, top=tuple(parameters, tuple(:c)))
    d_bc = Enzyme.make_zero(bc)
    Enzyme.API.looseTypeAnalysis!(true)

    dc²_dκ = autodiff(Enzyme.Reverse,
                      permute_boundary_conditions,
                      Duplicated(bc, d_bc))

    Enzyme.API.looseTypeAnalysis!(false)
end

@testset "Static activity" begin

    struct Test2{T}
        obs::T
    end

    function test(t, x)
        o = t.obs
        y = (x .- o)
        yv = @inbounds y[1]
        return yv*yv
    end

    obs = [1.0]
    t = Test2(obs)

    x0 = [0.0]
    dx0 = [0.0]

    autodiff(Reverse, test, Const(t), Duplicated(x0, dx0))

    @test obs[1] ≈ 1.0
    @test x0[1] ≈ 0.0
    @test dx0[1] ≈ -2.0

end

@testset "Const Activity through intermediate" begin
    struct RHS_terms
        eta1::Vector{Float64}
        u_t::Vector{Float64}
        eta_t::Vector{Float64}
    end

    @noinline function comp_u_v_eta_t(rhs)
        Base.unsafe_copyto!(rhs.eta_t, 1, rhs.u_t, 1, 1)
        return nothing
    end

    function advance(eta, rhs)

        @inbounds rhs.eta1[1] = @inbounds eta[1]

        comp_u_v_eta_t(rhs)

        @inbounds eta[1] = @inbounds rhs.eta_t[1]

        return nothing

    end

    rhs_terms = RHS_terms(zeros(1), zeros(1), zeros(1))

    u_v_eta = Float64[NaN]
    ad_eta = zeros(1)

    autodiff(Reverse, advance,
        Duplicated(u_v_eta, ad_eta),
        Const(rhs_terms),
    )
    @test ad_eta[1] ≈ 0.0
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

@testset "Type preservation" begin
    # Float16 fails due to #870
    for T in (Float64, Float32, #=Float16=#)
        res = autodiff(Reverse, x -> x * 2.0, Active, Active(T(1.0)))[1][1]
        @test res isa T
        @test res == 2
    end
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
    @test Enzyme.gradient(Enzyme.Reverse, fexpandempty, vec)[1] ≈ 1.0
    @test Enzyme.gradient(Enzyme.Forward, fexpandempty, vec)[1] ≈ 1.0
end

const CUmemoryPool2 = Ptr{Float64} 

struct CUmemPoolProps2
    reserved::NTuple{31,Char}
end

mutable struct CuMemoryPool2
    handle::CUmemoryPool2
end

function ccall_macro_lower(func, rettype, types, args, nreq)
    # instead of re-using ccall or Expr(:foreigncall) to perform argument conversion,
    # we need to do so ourselves in order to insert a jl_gc_safe_enter|leave
    # just around the inner ccall

    cconvert_exprs = []
    cconvert_args = []
    for (typ, arg) in zip(types, args)
        var = gensym("$(func)_cconvert")
        push!(cconvert_args, var)
        push!(cconvert_exprs, quote
            $var = Base.cconvert($(esc(typ)), $(esc(arg)))
        end)
    end

    unsafe_convert_exprs = []
    unsafe_convert_args = []
    for (typ, arg) in zip(types, cconvert_args)
        var = gensym("$(func)_unsafe_convert")
        push!(unsafe_convert_args, var)
        push!(unsafe_convert_exprs, quote
            $var = Base.unsafe_convert($(esc(typ)), $arg)
        end)
    end

    quote
        $(cconvert_exprs...)

        $(unsafe_convert_exprs...)

        ret = ccall($(esc(func)), $(esc(rettype)), $(Expr(:tuple, map(esc, types)...)),
                    $(unsafe_convert_args...))
    end
end

macro gcsafe_ccall(expr)
    ccall_macro_lower(Base.ccall_macro_parse(expr)...)
end

function cuMemPoolCreate2(pool, poolProps)
    # CUDA.initialize_context()
    #CUDA.
    gc_state = @ccall(jl_gc_safe_enter()::Int8)
    @gcsafe_ccall cuMemPoolCreate(pool::Ptr{CUmemoryPool2},
                                          poolProps::Ptr{CUmemPoolProps2})::Cvoid
    @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
end

function cual()
        props = Ref(CUmemPoolProps2( 
            ntuple(i->Char(0), 31)
        ))
        handle_ref = Ref{CUmemoryPool2}()
        cuMemPoolCreate2(handle_ref, props)

        CuMemoryPool2(handle_ref[])
end

@testset "Unused shadow phi rev" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(cual)}, Duplicated)
end


const SEED = 42
const N_SAMPLES = 500
const N_COMPONENTS = 4

const rnd = Random.MersenneTwister(SEED)
const data = randn(rnd, N_SAMPLES)
const params0 = [rand(rnd, N_COMPONENTS); randn(rnd, N_COMPONENTS); 2rand(rnd, N_COMPONENTS)]

# ========== Objective function ==========
normal_pdf(x::Real, mean::Real, var::Real) =
    exp(-(x - mean)^2 / (2var)) / sqrt(2π * var)

normal_pdf(x, mean, var) =
    exp(-(x - mean)^2 / (2var)) / sqrt(2π * var)

# original objective (doesn't work)
function mixture_loglikelihood1(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[K+1:2K], params[2K+1:end]
    mat = normal_pdf.(data, means', stds' .^2) # (N, K)
    sum(mat .* weights', dims=2) .|> log |> sum
end

# another form of original objective (doesn't work)
function mixture_loglikelihood2(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[K+1:2K], params[2K+1:end]
    mat = normal_pdf.(data, means', stds' .^2) # (N, K)
    obj_true = sum(
        sum(
            weight * normal_pdf(x, mean, std^2)
            for (weight, mean, std) in zip(weights, means, stds)
        ) |> log
        for x in data
    )
end

# objective re-written by me
function mixture_loglikelihood3(params::AbstractVector{<:Real}, data::AbstractVector{<:Real})::Real
    K = length(params) ÷ 3
    weights, means, stds = @views params[1:K], params[K+1:2K], params[2K+1:end]
    mat = normal_pdf.(data, means', stds' .^2) # (N, K)

    obj = zero(eltype(mat))
    for x in data
        obj_i = zero(eltype(mat))
        for (weight, mean, std) in zip(weights, means, stds)
            obj_i += weight * normal_pdf(x, mean, std^2)
        end
        obj += log(obj_i)
    end
    return obj
end

const objective1 = params -> mixture_loglikelihood1(params, data)
const objective2 = params -> mixture_loglikelihood2(params, data)
const objective3 = params -> mixture_loglikelihood3(params, data)

@testset "Type unsstable return" begin
    expected =  [289.7308495620467,
                199.27559524985728,
                 236.6894577756876,
                 292.0612340227955,
                  -9.429799389881452,
                  26.722295646439047,
                  -1.9180355546752244,
                  37.98749089573396,
                 -24.095620148778277,
                 -13.935687326484112,
                 -38.00044665702692,
                 12.87712891527131]
    @test expected ≈ Enzyme.gradient(Reverse, objective1, params0)
    # objective2 fails from runtime activity requirements
    # @test expected ≈ Enzyme.gradient(Reverse, objective2, params0)
    @test expected ≈ Enzyme.gradient(Reverse, objective3, params0)
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

