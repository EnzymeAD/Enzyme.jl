using Enzyme, Test
using Statistics, Random

@isdefined(UTILS_INCLUDE) || include("utils.jl")


@testset "Base functions" begin
    f1(x) = prod(ntuple(i -> i * x, 3))
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 72
    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1]   == 72

    f2(x) = x * something(nothing, 2)
    @test autodiff(Reverse, f2, Active, Active(1.0))[1][1] == 2
    @test autodiff(Forward, f2, Duplicated(1.0, 1.0))[1]   == 2

    f3(x) = x * sum(unique([x, 2.0, 2.0, 3.0]))
    @test autodiff(Reverse, f3, Active, Active(1.0))[1][1] == 7
    @test autodiff(Forward, f3, Duplicated(1.0, 1.0))[1]   == 7

    for rf in (reduce, foldl, foldr)
        f4(x) = rf(*, [1.0, x, x, 3.0])
        @test autodiff(Reverse, f4, Active, Active(2.0))[1][1] == 12
        @test autodiff(Forward, f4, Duplicated(2.0, 1.0))[1]   == 12
    end

    f5(x) = sum(accumulate(+, [1.0, x, x, 3.0]))
    @test autodiff(Reverse, f5, Active, Active(2.0))[1][1] == 5
    @test autodiff(Forward, f5, Duplicated(2.0, 1.0))[1]   == 5

    f6(x) = x |> inv |> abs
    @test autodiff(Reverse, f6, Active, Active(-2.0))[1][1] == 1/4
    @test autodiff(Forward, f6, Duplicated(-2.0, 1.0))[1]   == 1/4

    f7(x) = (inv ∘ abs)(x)
    @test autodiff(Reverse, f7, Active, Active(-2.0))[1][1] == 1/4
    @test autodiff(Forward, f7, Duplicated(-2.0, 1.0))[1]   == 1/4

    f8(x) = x * count(i -> i > 1, [0.5, x, 1.5])
    @test autodiff(Reverse, f8, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f8, Duplicated(2.0, 1.0))[1]   == 2

    function f9(x)
        y = []
        foreach(i -> push!(y, i^2), [1.0, x, x])
        return sum(y)
    end
    @test autodiff(Reverse, f9, Active, Active(2.0))[1][1] == 8
    @test autodiff(Forward, f9, Duplicated(2.0, 1.0))[1]   == 8

    f10(x) = hypot(x, 2x)
    @test autodiff(Reverse, f10, Active, Active(2.0))[1][1] == sqrt(5)
end

@testset "Dict" begin
    params = Dict{Symbol, Float64}()
    dparams = Dict{Symbol, Float64}()

    params[:var] = 10.0
    dparams[:var] = 0.0

    f_dict(params, x) = params[:var] * x

    @test autodiff(Reverse, f_dict, Const(params), Active(5.0)) == ((nothing, 10.0,),)
    @test autodiff(Reverse, f_dict, Duplicated(params, dparams), Active(5.0)) == ((nothing, 10.0,),)
    @test dparams[:var] == 5.0


    mutable struct MD
        v::Float64
        d::Dict{Symbol, MD}
    end

    # TODO without Float64 on return
    # there is a potential phi bug
    function sum_rec(d::Dict{Symbol,MD})::Float64
        s = 0.0
        for k in keys(d)
            s += d[k].v
            s += sum_rec(d[k].d)
        end
        return s
    end

    par = Dict{Symbol, MD}()
    par[:var] = MD(10.0, Dict{Symbol, MD}())
    par[:sub] = MD(2.0, Dict{Symbol, MD}(:a=>MD(3.0, Dict{Symbol, MD}())))

    dpar = Dict{Symbol, MD}()
    dpar[:var] = MD(0.0, Dict{Symbol, MD}())
    dpar[:sub] = MD(0.0, Dict{Symbol, MD}(:a=>MD(0.0, Dict{Symbol, MD}())))

    # TODO
    # autodiff(Reverse, sum_rec, Duplicated(par, dpar))
    # @show par, dpar, sum_rec(par)
    # @test dpar[:var].v ≈ 1.0
    # @test dpar[:sub].v ≈ 1.0
    # @test dpar[:sub].d[:a].v ≈ 1.0
end

@testset "Bithacks" begin
    function fneg(x::Float64)
        xptr = reinterpret(Int64, x)
        y = Int64(-9223372036854775808)
        out = y ⊻ xptr;
        return reinterpret(Float64, out)
    end
    @test autodiff(Reverse, fneg, Active, Active(2.0))[1][1] ≈ -1.0
    @test autodiff(Forward, fneg, Duplicated(2.0, 1.0))[1] ≈ -1.0
    function expor(x::Float64)
        xptr = reinterpret(Int64, x)
        y = UInt64(4607182418800017408)
        out = y | xptr;
        return reinterpret(Float64, out)
    end
    @test autodiff(Reverse, expor, Active, Active(0.42))[1][1] ≈ 4.0
    @test autodiff(Forward, expor, Duplicated(0.42, 1.0))[1] ≈ 4.0
end

@testset "SinCos" begin
	function sumsincos(theta)
		a, b = sincos(theta)
		return a + b
	end
    test_scalar(sumsincos, 1.0, rtol=1e-5, atol=1e-5)
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
    d = GFNamedDist((;a = GFNormal(0.0, 1.0), b = GFProductDist([GFUniform(0.0, 1.0), GFUniform(0.0, 1.0)])))
    p = (a = 1.0, b = [0.5, 0.5])
    dp = Enzyme.make_zero(p)
    GFlogpdf(d, p)
    autodiff(set_runtime_activity(Reverse), GFlogpdf, Active, Const(d), Duplicated(p, dp))
end

@testset "Random" begin
    f_rand(x) = x*rand()
    f_randn(x, N) = x*sum(randn(N))
    @test 0 <= autodiff(Reverse, f_rand, Active, Active(1.0))[1][1] < 1
    @test !iszero(autodiff(Reverse, f_randn, Active, Active(1.0), Const(64))[1][1])
    @test iszero(autodiff(Reverse, x -> rand(), Active, Active(1.0))[1][1])
    @test iszero(autodiff(Reverse, (x, N) -> sum(randn(N)), Active, Active(1.0), Const(64))[1][1])
    @test autodiff(Reverse, x -> x * sum(randcycle(5)), Active, Active(1.0))[1][1] == 15
    @test autodiff(Reverse, x -> x * sum(randperm( 5)), Active, Active(1.0))[1][1] == 15
    @test autodiff(Reverse, x -> x * sum(shuffle(1:5)), Active, Active(1.0))[1][1] == 15
end

@testset "Statistics" begin
    f1(x) = var([x, 2.0, 3.0])
    @test autodiff(Reverse, f1, Active, Active(0.0))[1][1] ≈ -5/3
    @test autodiff(Forward, f1, Duplicated(0.0, 1.0))[1]   ≈ -5/3

    f2(x) = varm([x, 2.0, 3.0], 5/3)
    @test autodiff(Reverse, f2, Active, Active(0.0))[1][1] ≈ -5/3
    @test autodiff(Forward, f2, Duplicated(0.0, 1.0))[1]   ≈ -5/3

    f3(x) = std([x, 2.0, 3.0])
    @test autodiff(Reverse, f3, Active, Active(0.0))[1][1] ≈ -0.54554472559
    @test autodiff(Forward, f3, Duplicated(0.0, 1.0))[1]   ≈ -0.54554472559

    f4(x) = stdm([x, 2.0, 3.0], 5/3)
    @test autodiff(Reverse, f4, Active, Active(0.0))[1][1] ≈ -0.54554472559
    @test autodiff(Forward, f4, Duplicated(0.0, 1.0))[1]   ≈ -0.54554472559

    f5(x) = cor([2.0, x, 1.0], [1.0, 2.0, 3.0])
    @test autodiff(Reverse, f5, Active, Active(4.0))[1][1] ≈ 0.11690244120
    @test autodiff(Forward, f5, Duplicated(4.0, 1.0))[1]   ≈ 0.11690244120

    f6(x) = cov([2.0, x, 1.0])
    @test autodiff(Reverse, f6, Active, Active(4.0))[1][1] ≈ 5/3
    @test autodiff(Forward, f6, Duplicated(4.0, 1.0))[1]   ≈ 5/3

    f7(x) = median([2.0, 1.0, x])
    @test autodiff(Reverse, f7, Active, Active(1.5))[1][1] == 1
    @test autodiff(Forward, f7, Duplicated(1.5, 1.0))[1]   == 1
    @test autodiff(Reverse, f7, Active, Active(2.5))[1][1] == 0
    @test autodiff(Forward, f7, Duplicated(2.5, 1.0))[1]   == 0

    f8(x) = middle([2.0, x, 1.0])
    @test autodiff(Reverse, f8, Active, Active(2.5))[1][1] == 0.5
    @test autodiff(Forward, f8, Duplicated(2.5, 1.0))[1]   == 0.5
    @test autodiff(Reverse, f8, Active, Active(1.5))[1][1] == 0
    @test autodiff(Forward, f8, Duplicated(1.5, 1.0))[1]   == 0

    f9(x) = sum(quantile([1.0, x], [0.5, 0.7]))
    @test autodiff(Reverse, f9, Active, Active(2.0))[1][1] == 1.2
    @test autodiff(Forward, f9, Duplicated(2.0, 1.0))[1]   == 1.2
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

