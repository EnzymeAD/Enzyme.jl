using Enzyme, Test
using Random


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

@testset "Struct return" begin
    x = [2.0]
    dx = [0.0]
    @test Enzyme.autodiff(Reverse, invsin2, Active, Duplicated(x, dx)) == ((nothing,),)
    @test dx[1] == -0.4161468365471424
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
    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), Const(smallrf), Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
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

    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), invokesum, Enzyme.Duplicated(weights, dweights), Enzyme.Const(data))
    @test dweights[1] ≈ 20.
    @test dweights[2] ≈ 20.
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


function assured_err(x)
    throw(AssertionError("foo"))
end

@testset "UnionAll" begin
    @test_throws AssertionError Enzyme.autodiff(Reverse, assured_err, Active, Active(2.0))
end

struct MyFlux
end

@testset "Union i8" begin
    args = (
        Val{(false, false, false)},
        Val(false),
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
        Val(false),
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

    res = autodiff(set_runtime_activity(ForwardWithPrimal), Const(f2), Duplicated, Duplicated(0.2, 1.0))
    @test res[2] ≈ 0.2
    # broken as the return of an apply generic is {primal, primal}
    # but since the return is abstractfloat doing the 
    @test res[1] ≈ 1.0
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

@testset "Type preservation" begin
    # Float16 fails due to #870
    for T in (Float64, Float32, #=Float16=#)
        res = autodiff(Reverse, x -> x * 2.0, Active, Active(T(1.0)))[1][1]
        @test res isa T
        @test res == 2
    end
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
    @test expected ≈ Enzyme.gradient(Reverse, objective1, params0)[1]
    # objective2 fails from runtime activity requirements
    # @test expected ≈ Enzyme.gradient(Reverse, objective2, params0)[1]
    @test expected ≈ Enzyme.gradient(Reverse, objective3, params0)[1]
end

