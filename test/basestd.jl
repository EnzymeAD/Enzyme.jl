using Enzyme, Test, Statistics, Random

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

# dot product (https://github.com/EnzymeAD/Enzyme.jl/issues/495)
@testset "Dot product" for T in (Float32, Float64)
    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (y) -> mapreduce(x -> x*x, +, y), Duplicated(xx, grads))
    @test xx .* 2 == grads

    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (x) -> sum(x .* x), Duplicated(xx, grads))
    @test xx .* 2 == grads

    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (x) -> x' * x, Duplicated(xx, grads))
    @test xx .* 2 == grads
end

@testset "SinCos" begin
	function sumsincos(theta)
		a, b = sincos(theta)
		return a + b
	end
    test_scalar(sumsincos, 1.0, rtol=1e-5, atol=1e-5)
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

