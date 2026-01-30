using Enzyme
using Statistics
using Test

include("common.jl")

make3() = (1.0, 2.0, 3.0)

@testset "Simple tests" begin
    f1(x) = 1.0 + x
    f2(x) = x * x
    @test autodiff(Reverse, f1, Active, Active(1.0))[1][1] ≈ 1.0
    @test autodiff(Forward, f1, Duplicated, Duplicated(1.0, 1.0))[1] ≈ 1.0
    @test autodiff(ForwardWithPrimal, f1, Duplicated, Duplicated(1.0, 1.0))[1] ≈ 1.0
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
        if T == Float16 && Sys.isapple()
            continue
        end
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
    test_scalar(x -> rem(x, 1), 0.7)
    test_scalar(x -> rem2pi(x, RoundDown), 0.7)
    test_scalar(x -> fma(x, x + 1, x / 3), 2.3)
    test_scalar(sqrt, 1.7 + 2.1im)

    @test autodiff(Forward, sincos, Duplicated(1.0, 1.0))[1][1] ≈ cos(1.0)

    @test autodiff(Reverse, (x) -> log(x), Active(2.0)) == ((0.5,),)

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

    z = 1.0 + 1.0im

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

    @test_throws MethodError autodiff(ReverseHolomorphic, mul3, Active, Active(z))
    @test_throws MethodError autodiff(ReverseHolomorphic, mul3, Active{Complex}, Active(z))

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
        return 0.0 + 1.0im
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
        return z * z
    end
    @test autodiff(ReverseHolomorphic, upgrade, Active, Active(3.1))[1][1] ≈ 6.2
end

@testset "Simple Exception" begin
    f_simple_exc(x, i) = x[i]
    y = [1.0, 2.0]
    f_x = zero.(y)
    @test_throws BoundsError autodiff(Reverse, f_simple_exc, Duplicated(y, f_x), Const(0))
end

@testset "Simple tests" begin
    g(x) = real((x + im) * (1 - im * x))
    @test first(autodiff(Reverse, g, Active, Active(2.0))[1]) ≈ 2.0
    @test first(autodiff(Forward, g, Duplicated(2.0, 1.0))) ≈ 2.0
    @test first(autodiff(Reverse, g, Active, Active(3.0))[1]) ≈ 2.0
    @test first(autodiff(Forward, g, Duplicated(3.0, 1.0))) ≈ 2.0
    test_scalar(g, 2.0)
    test_scalar(g, 3.0)
    test_scalar(Base.inv, 3.0 + 4.0im)
end

abstract type AbsFwdType end

# Two copies of the same type.
struct FwdNormal1{T <: Real} <: AbsFwdType
    σ::T
end

struct FwdNormal2{T <: Real} <: AbsFwdType
    σ::T
end

fwdlogpdf(d) = d.σ

function simple_absactfunc(x)
    dists = AbsFwdType[FwdNormal1{Float64}(1.0)]
    return @inbounds dists[1].σ
end

@testset "Simple Forward Mode active runtime activity" begin
    res = Enzyme.autodiff(set_runtime_activity(Enzyme.ForwardWithPrimal), Enzyme.Const(simple_absactfunc), Duplicated{Float64}, Duplicated(2.7, 3.1))
    @test res[1] == 0.0
    @test res[2] == 1.0

    res = Enzyme.autodiff(set_runtime_activity(Enzyme.Forward), Enzyme.Const(simple_absactfunc), Duplicated{Float64}, Duplicated(2.7, 3.1))
    @test res[1] == 0.0


    @static if VERSION < v"1.11-"
    else
        res = Enzyme.autodiff(Enzyme.ForwardWithPrimal, Enzyme.Const(simple_absactfunc), Duplicated{Float64}, Duplicated(2.7, 3.1))
        @test res[1] == 0.0
        @test res[2] == 1.0

        res = Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(simple_absactfunc), Duplicated{Float64}, Duplicated(2.7, 3.1))
        @test res[1] == 0.0
    end
end

function absactfunc(x)
    dists = AbsFwdType[FwdNormal1{Float64}(1.0), FwdNormal2{Float64}(x)]
    res = Vector{Float64}(undef, 2)
    for i in 1:length(dists)
        @inbounds res[i] = fwdlogpdf(dists[i])
    end
    return @inbounds res[1] + @inbounds res[2]
end

@testset "Forward Mode active runtime activity" begin
    res = Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(absactfunc), Duplicated(2.7, 3.1))
    @test res[1] ≈ 3.1

    res = Enzyme.autodiff(set_runtime_activity(Enzyme.Forward), Enzyme.Const(absactfunc), Duplicated(2.7, 3.1))
    @test res[1] ≈ 3.1
end

# dot product (https://github.com/EnzymeAD/Enzyme.jl/issues/495)
@testset "Dot product" for T in (Float32, Float64)
    xx = rand(T, 10)
    grads = zeros(T, size(xx))
    autodiff(Reverse, (y) -> mapreduce(x -> x * x, +, y), Duplicated(xx, grads))
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

@testset "Compare against" begin
    x = 3.0
    fd = central_fdm(5, 1)(sin, x)

    @test fd ≈ cos(x)
    @test fd ≈ first(autodiff(Reverse, sin, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, sin, Duplicated(x, 1.0)))

    x = 0.2 + sin(3.0)
    fd = central_fdm(5, 1)(asin, x)

    @test fd ≈ 1 / sqrt(1 - x * x)
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

    @test fd ≈ cos(x) / sqrt(1 - (0.2 + sin(x)) * (0.2 + sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1.0)))
    test_scalar(foo, x)

    # Input type shouldn't matter
    x = 3
    @test fd ≈ cos(x) / sqrt(1 - (0.2 + sin(x)) * (0.2 + sin(x)))
    @test fd ≈ first(autodiff(Reverse, foo, Active, Active(x)))[1]
    # They do matter for duplicated, which can't be auto promoted
    # @test fd ≈ first(autodiff(Forward, foo, Duplicated(x, 1)))

    f74(a, c) = a * √c
    @test √3 ≈ first(autodiff(Reverse, f74, Active, Active(2), Const(3)))[1]
    @test √3 ≈ first(autodiff(Forward, f74, Duplicated(2.0, 1.0), Const(3)))
end

@testset "SinCos" begin
    function sumsincos(theta)
        a, b = sincos(theta)
        return a + b
    end
    test_scalar(sumsincos, 1.0, rtol = 1.0e-5, atol = 1.0e-5)
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
function mybesselj(ν, z, atol = 1.0e-8)
    k = 0
    s = (z / 2)^ν / factorial(ν)
    out = s
    while abs(s) > atol
        k += 1
        s *= (-1) / k / (k + ν) * (z / 2)^2
        out += s
    end
    return out
end
mybesselj0(z) = mybesselj(0, z)
mybesselj1(z) = mybesselj(1, z)

@testset "Bessel" begin
    autodiff(Reverse, mybesselj, Active, Const(0), Active(1.0))
    autodiff(Reverse, mybesselj, Active, Const(0), Active(1.0))
    autodiff(Forward, mybesselj, Const(0), Duplicated(1.0, 1.0))
    autodiff(Forward, mybesselj, Const(0), Duplicated(1.0, 1.0))
    @testset "besselj0/besselj1" for x in (1.0, -1.0, 0.0, 0.5, 10, -17.1) # 1.5 + 0.7im)
        test_scalar(mybesselj0, x, rtol = 1.0e-5, atol = 1.0e-5)
        test_scalar(mybesselj1, x, rtol = 1.0e-5, atol = 1.0e-5)
    end
end

@testset "Base functions" begin
    f1(x) = prod(ntuple(i -> i * x, 3))
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 72
    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1] == 72

    f2(x) = x * something(nothing, 2)
    @test autodiff(Reverse, f2, Active, Active(1.0))[1][1] == 2
    @test autodiff(Forward, f2, Duplicated(1.0, 1.0))[1] == 2

    f3(x) = x * sum(unique([x, 2.0, 2.0, 3.0]))
    @test autodiff(Reverse, f3, Active, Active(1.0))[1][1] == 7
    @test autodiff(Forward, f3, Duplicated(1.0, 1.0))[1] == 7

    for rf in (reduce, foldl, foldr)
        f4(x) = rf(*, [1.0, x, x, 3.0])
        @test autodiff(Reverse, f4, Active, Active(2.0))[1][1] == 12
        @test autodiff(Forward, f4, Duplicated(2.0, 1.0))[1] == 12
    end

    f5(x) = sum(accumulate(+, [1.0, x, x, 3.0]))
    @test autodiff(Reverse, f5, Active, Active(2.0))[1][1] == 5
    @test autodiff(Forward, f5, Duplicated(2.0, 1.0))[1] == 5

    f6(x) = x |> inv |> abs
    @test autodiff(Reverse, f6, Active, Active(-2.0))[1][1] == 1 / 4
    @test autodiff(Forward, f6, Duplicated(-2.0, 1.0))[1] == 1 / 4

    f7(x) = (inv ∘ abs)(x)
    @test autodiff(Reverse, f7, Active, Active(-2.0))[1][1] == 1 / 4
    @test autodiff(Forward, f7, Duplicated(-2.0, 1.0))[1] == 1 / 4

    f8(x) = x * count(i -> i > 1, [0.5, x, 1.5])
    @test autodiff(Reverse, f8, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f8, Duplicated(2.0, 1.0))[1] == 2

    Enzyme.API.strictAliasing!(false)
    function f9(x)
        y = []
        foreach(i -> push!(y, i^2), [1.0, x, x])
        return sum(y)
    end
    @test autodiff(Reverse, f9, Active, Active(2.0))[1][1] == 8
    @test autodiff(Forward, f9, Duplicated(2.0, 1.0))[1] == 8

    Enzyme.API.strictAliasing!(true)
    f10(x) = hypot(x, 2x)
    @test autodiff(Reverse, f10, Active, Active(2.0))[1][1] == sqrt(5)
    @test autodiff(Forward, f10, Duplicated(2.0, 1.0))[1] == sqrt(5)

    f11(x) = x * sum(LinRange(x, 10.0, 6))
    @test autodiff(Reverse, f11, Active, Active(2.0))[1][1] == 42
    @test autodiff(Forward, f11, Duplicated(2.0, 1.0))[1] == 42

    f12(x, k) = get(Dict(1 => 1.0, 2 => x, 3 => 3.0), k, 1.0)
    @test autodiff(Reverse, f12, Active, Active(2.0), Const(2))[1] == (1.0, nothing)
    @test autodiff(Forward, f12, Duplicated(2.0, 1.0), Const(2)) == (1.0,)
    @test autodiff(Reverse, f12, Active, Active(2.0), Const(3))[1] == (0.0, nothing)
    @test autodiff(Forward, f12, Duplicated(2.0, 1.0), Const(3)) == (0.0,)
    @test autodiff(Reverse, f12, Active, Active(2.0), Const(4))[1] == (0.0, nothing)
    @test autodiff(Forward, f12, Duplicated(2.0, 1.0), Const(4)) == (0.0,)

    f13(x) = muladd(x, 3, x)
    @test autodiff(Reverse, f13, Active, Active(2.0))[1][1] == 4
    @test autodiff(Forward, f13, Duplicated(2.0, 1.0))[1] == 4

    f14(x) = x * cmp(x, 3)
    @test autodiff(Reverse, f14, Active, Active(2.0))[1][1] == -1
    @test autodiff(Forward, f14, Duplicated(2.0, 1.0))[1] == -1

    f15(x) = x * argmax([1.0, 3.0, 2.0])
    @test autodiff(Reverse, f15, Active, Active(3.0))[1][1] == 2
    @test autodiff(Forward, f15, Duplicated(3.0, 1.0))[1] == 2

    f16(x) = evalpoly(2, (1, 2, x))
    @test autodiff(Reverse, f16, Active, Active(3.0))[1][1] == 4
    @test autodiff(Forward, f16, Duplicated(3.0, 1.0))[1] == 4

    f17(x) = @evalpoly(2, 1, 2, x)
    @test autodiff(Reverse, f17, Active, Active(3.0))[1][1] == 4
    @test autodiff(Forward, f17, Duplicated(3.0, 1.0))[1] == 4

    f18(x) = widemul(x, 5.0f0)
    @test autodiff(Reverse, f18, Active, Active(2.0f0))[1][1] == 5
    @test autodiff(Forward, f18, Duplicated(2.0f0, 1.0f0))[1] == 5

    f19(x) = copysign(x, -x)
    @test autodiff(Reverse, f19, Active, Active(2.0))[1][1] == -1
    @test autodiff(Forward, f19, Duplicated(2.0, 1.0))[1] == -1

    f20(x) = sum([ifelse(i > 5, i, zero(i)) for i in [x, 2x, 3x, 4x]])
    @test autodiff(Reverse, f20, Active, Active(2.0))[1][1] == 7
    @test autodiff(Forward, f20, Duplicated(2.0, 1.0))[1] == 7

    function f21(x)
        nt = (a = x, b = 2x, c = 3x)
        return nt.c
    end
    @test autodiff(Reverse, f21, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f21, Duplicated(2.0, 1.0))[1] == 3

    f22(x) = sum(fill(x, (3, 3)))
    @test autodiff(Reverse, f22, Active, Active(2.0))[1][1] == 9
    @test autodiff(Forward, f22, Duplicated(2.0, 1.0))[1] == 9

    function f23(x)
        a = similar(rand(3, 3))
        fill!(a, x)
        return sum(a)
    end
    @test autodiff(Reverse, f23, Active, Active(2.0))[1][1] == 9
    @test autodiff(Forward, f23, Duplicated(2.0, 1.0))[1] == 9

    function f24(x)
        try
            return 3x
        catch
            return 2x
        end
    end
    @test autodiff(Reverse, f24, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f24, Duplicated(2.0, 1.0))[1] == 3

    function f25(x)
        try
            sqrt(-1.0)
            return 3x
        catch
            return 2x
        end
    end
    @test autodiff(Reverse, f25, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f25, Duplicated(2.0, 1.0))[1] == 2

    f26(x) = circshift([1.0, 2x, 3.0], 1)[end]
    @test autodiff(Reverse, f26, Active, Active(2.0))[1][1] == 2
    @test autodiff(Forward, f26, Duplicated(2.0, 1.0))[1] == 2

    f27(x) = repeat([x 3x], 3)[2, 2]
    @test autodiff(Reverse, f27, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f27, Duplicated(2.0, 1.0))[1] == 3

    f28(x) = x * sum(trues(4, 3))
    @test autodiff(Reverse, f28, Active, Active(2.0))[1][1] == 12
    @test autodiff(Forward, f28, Duplicated(2.0, 1.0))[1] == 12

    f29(x) = sum(Set([1.0, x, 2x, x]))
    @static if VERSION ≥ v"1.11-"
        @test autodiff(set_runtime_activity(Reverse), f29, Active, Active(2.0))[1][1] == 3
        @test autodiff(set_runtime_activity(Forward), f29, Duplicated(2.0, 1.0))[1] == 3
    else
        @test autodiff(Reverse, f29, Active, Active(2.0))[1][1] == 3
        @test autodiff(Forward, f29, Duplicated(2.0, 1.0))[1] == 3
    end

    f30(x) = reverse([x 2.0 3x])[1]
    @test autodiff(Reverse, f30, Active, Active(2.0))[1][1] == 3
    @test autodiff(Forward, f30, Duplicated(2.0, 1.0))[1] == 3
end

@testset "Taylor series tests" begin

    # Taylor series for `-log(1-x)`
    # eval at -log(1-1/2) = -log(1/2)
    function euroad(f::T) where {T}
        g = zero(T)
        for i in 1:(10^7)
            g += f^i / i
        end
        return g
    end

    euroad′(x) = first(autodiff(Reverse, euroad, Active, Active(x)))[1]

    @test euroad(0.5) ≈ -log(0.5) # -log(1-x)
    @test euroad′(0.5) ≈ 2.0 # d/dx -log(1-x) = 1/(1-x)
    test_scalar(euroad, 0.5)
end
@testset "Statistics" begin
    f1(x) = var([x, 2.0, 3.0])
    @test autodiff(Reverse, f1, Active, Active(0.0))[1][1] ≈ -5 / 3
    @test autodiff(Forward, f1, Duplicated(0.0, 1.0))[1] ≈ -5 / 3

    f2(x) = varm([x, 2.0, 3.0], 5 / 3)
    @test autodiff(Reverse, f2, Active, Active(0.0))[1][1] ≈ -5 / 3
    @test autodiff(Forward, f2, Duplicated(0.0, 1.0))[1] ≈ -5 / 3

    f3(x) = std([x, 2.0, 3.0])
    @test autodiff(Reverse, f3, Active, Active(0.0))[1][1] ≈ -0.54554472559
    @test autodiff(Forward, f3, Duplicated(0.0, 1.0))[1] ≈ -0.54554472559

    f4(x) = stdm([x, 2.0, 3.0], 5 / 3)
    @test autodiff(Reverse, f4, Active, Active(0.0))[1][1] ≈ -0.54554472559
    @test autodiff(Forward, f4, Duplicated(0.0, 1.0))[1] ≈ -0.54554472559

    f5(x) = cor([2.0, x, 1.0], [1.0, 2.0, 3.0])
    @test autodiff(Reverse, f5, Active, Active(4.0))[1][1] ≈ 0.1169024412
    @test autodiff(Forward, f5, Duplicated(4.0, 1.0))[1] ≈ 0.1169024412

    f6(x) = cov([2.0, x, 1.0])
    @test autodiff(Reverse, f6, Active, Active(4.0))[1][1] ≈ 5 / 3
    @test autodiff(Forward, f6, Duplicated(4.0, 1.0))[1] ≈ 5 / 3

    f7(x) = median([2.0, 1.0, x])
    @test autodiff(Reverse, f7, Active, Active(1.5))[1][1] == 1
    @test autodiff(Forward, f7, Duplicated(1.5, 1.0))[1] == 1
    @test autodiff(Reverse, f7, Active, Active(2.5))[1][1] == 0
    @test autodiff(Forward, f7, Duplicated(2.5, 1.0))[1] == 0

    f8(x) = middle([2.0, x, 1.0])
    @test autodiff(Reverse, f8, Active, Active(2.5))[1][1] == 0.5
    @test autodiff(Forward, f8, Duplicated(2.5, 1.0))[1] == 0.5
    @test autodiff(Reverse, f8, Active, Active(1.5))[1][1] == 0
    @test autodiff(Forward, f8, Duplicated(1.5, 1.0))[1] == 0

    f9(x) = sum(quantile([1.0, x], [0.5, 0.7]))
    @test autodiff(Reverse, f9, Active, Active(2.0))[1][1] == 1.2
    @test autodiff(Forward, f9, Duplicated(2.0, 1.0))[1] == 1.2
end

@testset "hvcat_fill" begin
    ar = Matrix{Float64}(undef, 2, 3)
    dar = [1.0 2.0 3.0; 4.0 5.0 6.0]

    res = first(Enzyme.autodiff(Reverse, Base.hvcat_fill!, Const, Duplicated(ar, dar), Active((1, 2.2, 3, 4.4, 5, 6.6))))

    @test res[2][1] == 0
    @test res[2][2] ≈ 2.0
    @test res[2][3] ≈ 0
    @test res[2][4] ≈ 4.0
    @test res[2][5] ≈ 0
    @test res[2][6] ≈ 6.0
end

function named_deepcopy(x, nt)
    nt2 = deepcopy(nt)
    return nt2.a + x[1]
end

@testset "Deepcopy" begin
    nt = (a = 0.0,)
    x = [0.5]

    @test Enzyme.gradient(Forward, named_deepcopy, x, Const(nt))[1] ≈ [1.0]
    @test Enzyme.gradient(Reverse, named_deepcopy, x, Const(nt))[1] ≈ [1.0]
end

@testset "Duplicated" begin
    x = Ref(1.0)
    y = Ref(2.0)

    ∇x = Ref(0.0)
    ∇y = Ref(0.0)

    autodiff(Reverse, (a, b) -> a[] * b[], Active, Duplicated(x, ∇x), Duplicated(y, ∇y))

    @test ∇y[] == 1.0
    @test ∇x[] == 2.0
end

@testset "Nested Type Error" begin
    nested_f(x) = sum(tanh, x)

    function nested_df!(dx, x)
        make_zero!(dx)
        autodiff_deferred(Reverse, Const(nested_f), Active, Duplicated(x, dx))
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

    @test dx ≈ (sech.(x) .^ 2)

    v = first(onehot(x))
    hv = make_zero(v)
    nested_hvp!(hv, v, x)
end

const CONST_VAL = 2.0
f_const_global(x) = x^2 * CONST_VAL

MUTABLE_VAL = 2.0
f_mutable_global(x) = x^2 * MUTABLE_VAL

TYPED_VAL::Float64 = 2.0
f_typed_global(x) = x^2 * TYPED_VAL

@testset "Globals" begin
    @test Enzyme.autodiff(Reverse, f_const_global, Active, Active(3.0))[1][1] ≈ 12.0
    @test Enzyme.autodiff(Reverse, f_mutable_global, Active, Active(3.0))[1][1] ≈ 12.0
    @test Enzyme.autodiff(Reverse, f_typed_global, Active, Active(3.0))[1][1] ≈ 12.0
end
