using Enzyme, Test
using FiniteDifferences

@isdefined(UTILS) || include("utils.jl")


@testset "Simple tests" begin
    make3() = (1.0, 2.0, 3.0)

    f1(x) = 1.0 + x
    f2(x) = x*x
    @test autodiff(Reverse, f1, Active, Active(1.0))[1][1] ≈ 1.0
    @test autodiff(Forward, f1, DuplicatedNoNeed, Duplicated(1.0, 1.0))[1] ≈ 1.0
    @test autodiff(Forward, f1, Duplicated, Duplicated(1.0, 1.0))[2] ≈ 1.0
    @test autodiff(Reverse, f2, Active, Active(1.0))[1][1] ≈ 2.0
    @test autodiff(Forward, f2, Duplicated(1.0, 1.0))[1] ≈ 2.0
    tup = autodiff(Forward, f2, BatchDuplicated(1.0, (1.0, 2.0, 3.0)))[1]
    @test tup[1] ≈ 2.0
    @test tup[2] ≈ 4.0
    @test tup[3] ≈ 6.0
    tup = autodiff(Forward, f2, BatchDuplicatedFunc{Float64, 3, typeof(make3)}(1.0))[1]
    @test tup[1] ≈ 2.0
    @test tup[2] ≈ 4.0
    @test tup[3] ≈ 6.0
    @test autodiff(Reverse, tanh, Active, Active(1.0))[1][1] ≈ 0.41997434161402606939
    @test autodiff(Forward, tanh, Duplicated(1.0, 1.0))[1] ≈ 0.41997434161402606939
    @test autodiff(Reverse, tanh, Active, Active(1.0f0))[1][1] ≈ Float32(0.41997434161402606939)
    @test autodiff(Forward, tanh, Duplicated(1.0f0, 1.0f0))[1] ≈ Float32(0.41997434161402606939)

    for T in (Float64, Float32, Float16)
        res = autodiff(Reverse, tanh, Active, Active(T(1)))[1][1]
        @test res isa T
        cmp = if T == Float64
            T(0.41997434161402606939)
        else
            T(0.41997434161402606939f0)
        end
        @test res ≈ cmp
        res = autodiff(Forward, tanh, Duplicated(T(1), T(1)))[1]
        @test res isa T
        @test res ≈ cmp
    end

    test_scalar(f1, 1.0)
    test_scalar(f2, 1.0)
    test_scalar(log2, 1.0)
    test_scalar(log1p, 1.0)

    test_scalar(log10, 1.0)
    test_scalar(Base.acos, 0.9)

    test_scalar(Base.atan, 0.9)

    res = autodiff(Reverse, Base.atan, Active, Active(0.9), Active(3.4))[1]
    @test res[1] ≈ 3.4 / (0.9 * 0.9 + 3.4 * 3.4)
    @test res[2] ≈ -0.9 / (0.9 * 0.9 + 3.4 * 3.4)

    test_scalar(cbrt, 1.0)
    test_scalar(cbrt, 1.0f0; rtol = 1.0e-5, atol = 1.0e-5)
    test_scalar(Base.sinh, 1.0)
    test_scalar(Base.cosh, 1.0)
    test_scalar(Base.sinc, 2.2)
    test_scalar(Base.FastMath.sinh_fast, 1.0)
    test_scalar(Base.FastMath.cosh_fast, 1.0)
    test_scalar(Base.FastMath.exp_fast, 1.0)
    test_scalar(Base.exp10, 1.0)
    test_scalar(Base.exp2, 1.0)
    test_scalar(Base.expm1, 1.0)
    test_scalar(x->rem(x, 1), 0.7)
    test_scalar(x->rem2pi(x,RoundDown), 0.7)
    test_scalar(x->fma(x,x+1,x/3), 2.3)
    test_scalar(sqrt, 1.7+2.1im)
    
    @test autodiff(Forward, sincos, Duplicated(1.0, 1.0))[1][1] ≈ cos(1.0)

    @test autodiff(Reverse, (x)->log(x), Active(2.0)) == ((0.5,),)

    a = [3.14]
    da = [0.0]
    sumcopy(x) = sum(copy(x))
    autodiff(Reverse, sumcopy, Duplicated(a, da))
    @test da[1] ≈ 1.0

    da = [2.7]
    @test autodiff(Forward, sumcopy, Duplicated(a, da))[1] ≈ 2.7

    da = [0.0]
    sumdeepcopy(x) = sum(deepcopy(x))
    autodiff(Reverse, sumdeepcopy, Duplicated(a, da))
    @test da[1] ≈ 1.0

    da = [2.7]
    @test autodiff(Forward, sumdeepcopy, Duplicated(a, da))[1] ≈ 2.7

end

@testset "Simple Complex tests" begin
    mul2(z) = 2 * z
    square(z) = z * z

    z = 1.0+1.0im

    @test_throws ErrorException autodiff(Reverse, mul2, Active, Active(z))
    @test_throws ErrorException autodiff(ReverseWithPrimal, mul2, Active, Active(z))
    @test autodiff(ReverseHolomorphic, mul2, Active, Active(z))[1][1] ≈ 2.0 + 0.0im
    @test autodiff(ReverseHolomorphicWithPrimal, mul2, Active, Active(z))[1][1] ≈ 2.0 + 0.0im
    @test autodiff(ReverseHolomorphicWithPrimal, mul2, Active, Active(z))[2] ≈ 2 * z

    z = 3.4 + 2.7im
    @test autodiff(ReverseHolomorphic, square, Active, Active(z))[1][1] ≈ 2 * z
    @test autodiff(ReverseHolomorphic, identity, Active, Active(z))[1][1] ≈ 1

    @test autodiff(ReverseHolomorphic, Base.inv, Active, Active(3.0 + 4.0im))[1][1] ≈ 0.0112 + 0.0384im

    mul3(z) = Base.inferencebarrier(2 * z)

    @test_throws ErrorException autodiff(ReverseHolomorphic, mul3, Active, Active(z))
    @test_throws ErrorException autodiff(ReverseHolomorphic, mul3, Active{Complex}, Active(z))

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sum, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 1.0

    sumsq(x) = sum(x .* x)

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 2 * (3.4 + 2.7im)

    sumsq2(x) = sum(abs2.(x))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq2, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 2 * (3.4 + 2.7im)

    sumsq2C(x) = Complex{Float64}(sum(abs2.(x)))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq2C, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 3.4 - 2.7im

    sumsq3(x) = sum(x .* conj(x))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq3, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 3.4 - 2.7im

    sumsq3R(x) = Float64(sum(x .* conj(x)))
    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, sumsq3R, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 3.4 + 2.7im
    @test dvals[1] ≈ 2 * (3.4 + 2.7im)

    function setinact(z)
        z[1] *= 2
        nothing
    end

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setinact, Const, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0


    function setinact2(z)
        z[1] *= 2
        return 0.0+1.0im
    end

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setinact2, Const, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setinact2, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0


    function setact(z)
        z[1] *= 2
        return z[1]
    end

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setact, Const, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 0.0

    vals = Complex{Float64}[3.4 + 2.7im]
    dvals = Complex{Float64}[0.0]
    autodiff(ReverseHolomorphic, setact, Active, Duplicated(vals, dvals))
    @test vals[1] ≈ 2 * (3.4 + 2.7im)
    @test dvals[1] ≈ 2.0

    function upgrade(z)
        z = ComplexF64(z)
        return z*z
    end
    @test autodiff(ReverseHolomorphic, upgrade, Active, Active(3.1))[1][1] ≈ 6.2
end

@testset "Simple Exception" begin
    f_simple_exc(x, i) = ccall(:jl_, Cvoid, (Any,), x[i])
    y = [1.0, 2.0]
    f_x = zero.(y)
    @test_throws BoundsError autodiff(Reverse, f_simple_exc, Duplicated(y, f_x), Const(0))
end

@testset "Simple tests" begin
    g(x) = real((x + im)*(1 - im*x))
    @test first(autodiff(Reverse, g, Active, Active(2.0))[1]) ≈ 2.0
    @test first(autodiff(Forward, g, Duplicated(2.0, 1.0))) ≈ 2.0
    @test first(autodiff(Reverse, g, Active, Active(3.0))[1]) ≈ 2.0
    @test first(autodiff(Forward, g, Duplicated(3.0, 1.0))) ≈ 2.0
    test_scalar(g, 2.0)
    test_scalar(g, 3.0)
    test_scalar(Base.inv, 3.0 + 4.0im)
end

