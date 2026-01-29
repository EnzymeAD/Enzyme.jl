using Enzyme
using InlineStrings
using LinearAlgebra
using Statistics
using Test
using GPUCompiler

@testset "GC" begin
    function gc_alloc(x)  # Basically g(x) = x^2
        a = Array{Float64, 1}(undef, 10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end
    @test autodiff(Reverse, gc_alloc, Active, Active(5.0))[1][1] ≈ 10
    @test autodiff(Forward, gc_alloc, Duplicated(5.0, 1.0))[1] ≈ 10

    A = Float64[2.0, 3.0]
    B = Float64[4.0, 5.0]
    dB = Float64[0.0, 0.0]
    f = (X, Y) -> sum(X .* Y)
    Enzyme.autodiff(Reverse, f, Active, Const(A), Duplicated(B, dB))

    function gc_copy(x)  # Basically g(x) = x^2
        a = x * ones(10)
        for n in 1:length(a)
            a[n] = x^2
        end
        return mean(a)
    end

    @test Enzyme.autodiff(Reverse, gc_copy, Active, Active(5.0))[1][1] ≈ 10
    @test Enzyme.autodiff(Forward, gc_copy, Duplicated(5.0, 1.0))[1] ≈ 10
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

    ps = (
        b2 = 1.0f0,
    )

    grads = (
        b2 = 0.0f0,
    )

    t1 = Leaf(ps)
    t1Grads = Leaf(grads)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitModified(ReverseSplitNoPrimal, Val((false, true))), Const{typeof(LeafF)}, Active, Duplicated{Leaf})
    tape, primal, shadow = forward(Const(LeafF), Duplicated(t1, t1Grads))

    struct Foo2{X, Y}
        x::X
        y::Y
    end

    test_f(f::Foo2) = f.x^2
    res = autodiff(Reverse, test_f, Active(Foo2(3.0, :two)))[1][1]
    @test res.x ≈ 6.0
    @test res.y == nothing
end

@testset "GCPreserve" begin
    function f(x, y)
        GC.@preserve x y begin
            @ccall memcpy(x::Ptr{Float64}, y::Ptr{Float64}, 8::Csize_t)::Cvoid
        end
        nothing
    end
    autodiff(Reverse, f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
    autodiff(Forward, f, Duplicated([1.0], [0.0]), Duplicated([1.0], [0.0]))
end

@testset "GCPreserve2" begin
    function f!(a_out, a_in)
        a_out[1:(end - 1)] .= a_in[2:end]
        return nothing
    end
    a_in = rand(4)
    a_out = a_in

    shadow_a_out = ones(4)
    shadow_a_in = shadow_a_out

    autodiff(Reverse, f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))

    @test shadow_a_in ≈ Float64[0.0, 1.0, 1.0, 2.0]
    @test shadow_a_out ≈ Float64[0.0, 1.0, 1.0, 2.0]

    autodiff(Forward, f!, Const, Duplicated(a_out, shadow_a_out), Duplicated(a_in, shadow_a_in))

    @test shadow_a_in ≈ Float64[1.0, 1.0, 2.0, 2.0]
    @test shadow_a_out ≈ Float64[1.0, 1.0, 2.0, 2.0]
end

@testset "GC Sret" begin
    @noinline function _get_batch_statistics(x)
        batchmean = @inbounds x[1]
        return (x, x)
    end

    @noinline function _normalization_impl(x)
        _stats = _get_batch_statistics(x)
        return x
    end

    function gcloss(x)
        _normalization_impl(x)[1]
        return nothing
    end

    x = randn(10)
    dx = zero(x)

    Enzyme.autodiff(Reverse, gcloss, Duplicated(x, dx))
end

typeunknownvec = Float64[]

@testset "GC Sret 2" begin

    struct AGriddedInterpolation{K <: Tuple{Vararg{AbstractVector}}} <: AbstractArray{Float64, 1}
        knots::K
        v::Int
    end

    function AGriddedInterpolation(A::AbstractArray{Float64, 1})
        knots = (A,)
        use(A)
        AGriddedInterpolation{typeof(knots)}(knots, 2)
    end

    function ainterpolate(A::AbstractArray{Float64, 1})
        AGriddedInterpolation(A)
    end

    function cost(C::Vector{Float64})
        zs = typeunknownvec
        ainterpolate(zs)
        return nothing
    end

    A = Float64[]
    dA = Float64[]
    @test_throws Base.UndefVarError autodiff(Reverse, cost, Const, Duplicated(A, dA))
end

@testset "No Decayed / GC" begin
    @noinline function deduplicate_knots!(knots)
        last_knot = first(knots)
        for i in eachindex(knots)
            if i == 1
                continue
            end
            if knots[i] == last_knot
                @warn knots[i]
                @inbounds knots[i] *= knots[i]
            else
                last_knot = @inbounds knots[i]
            end
        end
    end

    function cost(C::Vector{Float64})
        deduplicate_knots!(C)
        @inbounds C[1] = 0
        return nothing
    end
    A = Float64[1, 3, 3, 7]
    dA = Float64[1, 1, 1, 1]
    @test_warn "3.0" autodiff(Reverse, cost, Const, Duplicated(A, dA))
    @test dA ≈ [0.0, 1.0, 6.0, 1.0]
end

@testset "Split GC" begin
    @noinline function bmat(x)
        data = [x]
        return data
    end

    function f(x::Float64)
        @inbounds return bmat(x)[1]
    end
    @test 1.0 ≈ autodiff(Reverse, f, Active(0.1))[1][1]
    @test 1.0 ≈ autodiff(Forward, f, Duplicated(0.1, 1.0))[1]
end

@testset "Recursive GC" begin
    function modf!(a)
        as = [zero(a) for _ in 1:2]
        a .+= sum(as)
        return nothing
    end

    a = rand(5)
    da = zero(a)
    autodiff(Reverse, modf!, Duplicated(a, da))
end

@testset "Dict" begin
    params = Dict{Symbol, Float64}()
    dparams = Dict{Symbol, Float64}()

    params[:var] = 10.0
    dparams[:var] = 0.0

    f_dict(params, x) = params[:var] * x

    @test autodiff(Reverse, f_dict, Const(params), Active(5.0)) == ((nothing, 10.0),)
    @test autodiff(Reverse, f_dict, Duplicated(params, dparams), Active(5.0)) == ((nothing, 10.0),)
    @test dparams[:var] == 5.0


    mutable struct MD
        v::Float64
        d::Dict{Symbol, MD}
    end

    # TODO without Float64 on return
    # there is a potential phi bug
    function sum_rec(d::Dict{Symbol, MD})::Float64
        s = 0.0
        for k in keys(d)
            s += d[k].v
            s += sum_rec(d[k].d)
        end
        return s
    end

    par = Dict{Symbol, MD}()
    par[:var] = MD(10.0, Dict{Symbol, MD}())
    par[:sub] = MD(2.0, Dict{Symbol, MD}(:a => MD(3.0, Dict{Symbol, MD}())))

    dpar = Dict{Symbol, MD}()
    dpar[:var] = MD(0.0, Dict{Symbol, MD}())
    dpar[:sub] = MD(0.0, Dict{Symbol, MD}(:a => MD(0.0, Dict{Symbol, MD}())))

    # TODO
    # autodiff(Reverse, sum_rec, Duplicated(par, dpar))
    # @show par, dpar, sum_rec(par)
    # @test dpar[:var].v ≈ 1.0
    # @test dpar[:sub].v ≈ 1.0
    # @test dpar[:sub].d[:a].v ≈ 1.0
end


const julia_typed_pointers = GPUCompiler.JuliaContext() do ctx
    GPUCompiler.supports_typed_pointers(ctx)
end


let
    function loadsin2(xp)
        x = @inbounds xp[1]
        @inbounds xp[1] = 0.0
        return sin(x)
    end
    global invsin2
    function invsin2(xp)
        xp = Base.invokelatest(convert, Vector{Float64}, xp)
        return loadsin2(xp)
    end
    x = [2.0]
end

@testset "Struct return" begin
    x = [2.0]
    dx = [0.0]
    @test Enzyme.autodiff(Reverse, invsin2, Active, Duplicated(x, dx)) == ((nothing,),)
    if julia_typed_pointers
        @test dx[1] == -0.4161468365471424
    else
        @test_broken dx[1] == -0.4161468365471424
    end
end

function grad_closure(f, x)
    function noretval(x, res)
        y = f(x)
        copyto!(res, y)
        return nothing
    end
    n = length(x)
    dx = zeros(n)
    y = zeros(n)
    dy = zeros(n)
    dy[1] = 1.0

    autodiff(Reverse, Const(noretval), Duplicated(x, dx), Duplicated(y, dy))
    return dx
end

@testset "Closure" begin
    x = [2.0, 6.0]
    dx = grad_closure(x -> [x[1], x[2]], x)
    @test dx == [1.0, 0.0]
end

struct MyClosure{A}
    a::A
end

function (mc::MyClosure)(x)
    # computes x^2 using internal storage
    mc.a[1] = x
    return mc.a[1]^2
end

@testset "Batch Closure" begin
    g = MyClosure([0.0])
    g_and_dgs = BatchDuplicated(g, (make_zero(g), make_zero(g)))
    x_and_dxs = BatchDuplicated(3.0, (5.0, 7.0))
    autodiff(Forward, g_and_dgs, BatchDuplicated, x_and_dxs)  # error
end

@testset "GetField" begin
    mutable struct MyType
        x::Float64
    end

    getfield_idx(v, idx) = @ccall jl_get_nth_field_checked(v::Any, idx::UInt)::Any

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
        return (x * y)::Float64
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

struct GFNamedDist{Names, D <: NamedTuple{Names}}
    dists::D
end

function GFlogpdf(d::GFNamedDist{N}, x::NamedTuple{N}) where {N}
    vt = values(x)
    dists = d.dists
    return mapreduce((dist, acc) -> GFlogpdf(dist, acc), +, dists, vt)
end

@testset "Getfield with reference" begin
    d = GFNamedDist((; a = GFNormal(0.0, 1.0), b = GFProductDist([GFUniform(0.0, 1.0), GFUniform(0.0, 1.0)])))
    p = (a = 1.0, b = [0.5, 0.5])
    dp = Enzyme.make_zero(p)
    GFlogpdf(d, p)
    autodiff(set_runtime_activity(Reverse), GFlogpdf, Active, Const(d), Duplicated(p, dp))
end

@testset "Higher order rules" begin
    sqr(x) = x * x
    power(x, n) = x^n

    function objective(x)
        (x1, x2, x3, x4) = x
        objvar = -4 - -(((((((((((((sqr(x1) + sqr(x2)) + sqr(x3 + x4)) + x3) + sqr(sin(x3))) + sqr(x1) * sqr(x2)) + x4) + sqr(sin(x3))) + sqr(-1 + x4)) + sqr(sqr(x2))) + sqr(sqr(x3) + sqr(x1 + x4))) + sqr(((-4 + sqr(sin(x4))) + sqr(x2) * sqr(x3)) + x1)) + power(sin(x4), 4)))
        return objvar
    end

    x0 = [0.0, 2.0, -1.0, 2.0]

    res = Enzyme.jacobian(Forward, Const(Enzyme.gradient), Const(Reverse), Const(objective), x0)

    @test res[3][1][1] ≈ [64.0, 8.0, -32.0, 50.48639500938415]
    @test res[3][2][1] ≈ [8.0, 85.30728724172722, -77.2291489669089, -6.0544199624634265]
    @test res[3][3][1] ≈ [-32.0, -77.2291489669089, 169.56456162072033, -1.8911600750731472]
    @test res[3][4][1] ≈ [50.48639500938415, -6.0544199624634265, -1.891160075073147, 53.967425651780005]
end

@testset "Bithacks" begin
    function fneg(x::Float64)
        xptr = reinterpret(Int64, x)
        y = Int64(-9223372036854775808)
        out = y ⊻ xptr
        return reinterpret(Float64, out)
    end
    @test autodiff(Reverse, fneg, Active, Active(2.0))[1][1] ≈ -1.0
    @test autodiff(Forward, fneg, Duplicated(2.0, 1.0))[1] ≈ -1.0
    function expor(x::Float64)
        xptr = reinterpret(Int64, x)
        y = UInt64(4607182418800017408)
        out = y | xptr
        return reinterpret(Float64, out)
    end
    @test autodiff(Reverse, expor, Active, Active(0.42))[1][1] ≈ 4.0
    @test autodiff(Forward, expor, Duplicated(0.42, 1.0))[1] ≈ 4.0
end

function bc0_test_function(ps)
    z = view(ps, 26:30)
    C = Matrix{Float64}(undef, 5, 1)
    C .= z
    return C[1]
end

@noinline function bc1_bcs2(x, y)
    x != y && error(2)
    return x
end

@noinline function bc1_affine_normalize(x::AbstractArray)
    _axes = bc1_bcs2(axes(x), axes(x))
    dest = similar(Array{Float32}, _axes)
    bc = convert(Broadcast.Broadcasted{Nothing}, Broadcast.instantiate(Base.broadcasted(+, x, x)))
    copyto!(dest, bc)
    return x
end

function bc1_loss_function(x)
    return bc1_affine_normalize(x)[1]
end

function bc2_affine_normalize(
        ::typeof(identity), x::AbstractArray, xmean, xvar,
        scale::AbstractArray, bias::AbstractArray, epsilon::Real
    )
    _scale = @. scale / sqrt(xvar + epsilon)
    _bias = @. bias - xmean * _scale
    return @. x * _scale + _bias
end

function bc2_loss_function(x, scale, bias)
    x_ = reshape(x, 6, 6, 3, 2, 2)
    scale_ = reshape(scale, 1, 1, 3, 2, 1)
    bias_ = reshape(bias, 1, 1, 3, 2, 1)

    xmean = mean(x_, dims = (1, 2, 5))
    xvar = var(x_, corrected = false, mean = xmean, dims = (1, 2, 5))

    return sum(abs2, bc2_affine_normalize(identity, x_, xmean, xvar, scale_, bias_, 1.0e-5))
end

@testset "Broadcast noalias" begin
    x = ones(30)

    @static if VERSION < v"1.11-"
        autodiff(Reverse, bc0_test_function, Active, Const(x))
    else
        # TODO
        @test_broken autodiff(Reverse, bc0_test_function, Active, Const(x))
    end

    x = rand(Float32, 2, 3)
    Enzyme.autodiff(Reverse, bc1_loss_function, Duplicated(x, zero(x)))

    x = rand(Float32, 6, 6, 6, 2)
    sc = rand(Float32, 6)
    bi = rand(Float32, 6)
    Enzyme.autodiff(
        Reverse, bc2_loss_function, Active, Duplicated(x, Enzyme.make_zero(x)),
        Duplicated(sc, Enzyme.make_zero(sc)), Duplicated(bi, Enzyme.make_zero(bi))
    )
end

@testset "View Splat" begin
    function getloc(locs, i)
        loss = 0.0
        if i == 1
            x, y = 0.0, 0.0
        else
            # correct
            # x, y = locs[1,i-1], locs[2,i-1]
            # incorrect
            x, y = @inbounds locs[:, i - 1]
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

@testset "View Vars" begin

    x = [Float32(0.25)]
    dx = [Float32(0.0)]
    rng = Base.UnitRange{Int64}(1, 0)

    f = Const(Base.SubArray{T, N, P, I, L} where {L} where {I} where {P} where {N} where {T})
    a1 = Const(Base.IndexLinear())
    a2 = Duplicated(x, dx)
    a3 = Const((rng,))
    a4 = Const((true,))

    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal,
        typeof(f),
        Duplicated,
        typeof(a1),
        typeof(a2),
        typeof(a3),
        typeof(a4)
    )

    res = fwd(f, a1, a2, a3, a4)
    @test res[2].indices == (rng,)
    @test res[3].indices == (rng,)
    @test res[2].offset1 == 0
    @test res[3].offset1 == 0
    @test res[2].stride1 == 1
    @test res[3].stride1 == 1

    x = [Float32(0.25)]
    dx = [Float32(0.0)]
    rng = Base.UnitRange{Int64}(1, 0)

    f = Const(Base.SubArray{T, N, P, I, L} where {L} where {I} where {P} where {N} where {T})
    a1 = Const(Base.IndexLinear())
    a2 = Duplicated(x, dx)
    a3 = Const((rng,))
    a4 = Const((true,))

    fwd, rev = autodiff_thunk(
        set_runtime_activity(ReverseSplitWithPrimal),
        typeof(f),
        Duplicated,
        typeof(a1),
        typeof(a2),
        typeof(a3),
        typeof(a4)
    )

    res = fwd(f, a1, a2, a3, a4)
    @test res[2].indices == (rng,)
    @test res[3].indices == (rng,)
    @test res[2].offset1 == 0
    @test res[3].offset1 == 0
    @test res[2].stride1 == 1
    @test res[3].stride1 == 1
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

    @static if VERSION < v"1.11-" || VERSION >= v"1.12"
        Enzyme.autodiff(Reverse, inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))

        @test x ≈ [3.0]
        @test w ≈ [1.0]
        @test dw ≈ [3.0]
    else
        # TODO broken should not throw
        @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError Enzyme.autodiff(Reverse, inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))
        Enzyme.autodiff(set_runtime_activity(Reverse), inactiveArg, Active, Duplicated(w, dw), Const(x), Const(false))
    end

    x = Float32[3]

    function loss(w, x, cond)
        dest = Array{Float32}(undef, 1)
        r = cond ? copy(x) : x
        res = @inbounds w[1] * r[1]
        @inbounds dest[1] = res
        res
    end

    @static if VERSION < v"1.11-" || VERSION >= v"1.12"
        dw = Enzyme.autodiff(Reverse, loss, Active, Active(1.0), Const(x), Const(false))[1]

    else
        # TODO broken should not throw
        @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError Enzyme.autodiff(Reverse, loss, Active, Active(1.0), Const(x), Const(false))[1]
        dw = Enzyme.autodiff(set_runtime_activity(Reverse), loss, Active, Active(1.0), Const(x), Const(false))[1]
    end

    @test x ≈ [3.0]
    @test dw[1] ≈ 3.0

    c = ones(3)
    inner(e) = c .+ e

    fres = Enzyme.autodiff(Enzyme.Forward, Const(inner), Duplicated{Vector{Float64}}, Duplicated([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))[1]
    @test c ≈ [1.0, 1.0, 1.0]
    @test fres ≈ [1.0, 1.0, 1.0]
end

@testset "Arrays are double pointers" begin
    @noinline function func_scalar(X)
        return X
    end

    function timsteploop_scalar(FH1)
        G = Float64[FH1]
        k1 = @inbounds func_scalar(G[1])
        return k1
    end
    @test Enzyme.autodiff(Reverse, timsteploop_scalar, Active(2.0))[1][1] ≈ 1.0
    @test Enzyme.autodiff(Forward, timsteploop_scalar, Duplicated(2.0, 1.0))[1] ≈ 1.0

    @noinline function func(X)
        return @inbounds X[1]
    end
    function timsteploop(FH1)
        G = Float64[FH1]
        k1 = func(G)
        return k1
    end
    @test Enzyme.autodiff(Reverse, timsteploop, Active(2.0))[1][1] ≈ 1.0
    @test Enzyme.autodiff(Forward, timsteploop, Duplicated(2.0, 1.0))[1] ≈ 1.0
end

@noinline function prt_sret(A)
    A[1] *= 2
    return (A, A[2])
end

@noinline function sretf(A2, x, c)
    return x[3] = c * A2[3]
end

@noinline function batchdecaysret0(x, A, b)
    A2, c = prt_sret(A)
    sretf(A2, x, c)
    return nothing
end

function batchdecaysret(x, A, b)
    batchdecaysret0(x, A, b)
    A[2] = 0
    return nothing
end

@testset "Batch Reverse sret fix" begin
    Enzyme.autodiff(
        Reverse, batchdecaysret,
        BatchDuplicated(ones(3), (ones(3), ones(3))),
        BatchDuplicated(ones(3), (ones(3), ones(3))),
        BatchDuplicated(ones(3), (ones(3), ones(3)))
    )
end

@testset "Batch Generics" begin
    function mul2ip(y)
        y[1] *= 2
        return nothing
    end

    function fwdlatestfooip(y)
        Base.invokelatest(mul2ip, y)
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    dx2 = [10.0, 20.0, 30.0]

    res = Enzyme.autodiff(Forward, fwdlatestfooip, Const, BatchDuplicated(x, (dx, dx2)))
    @test 2.0 ≈ dx[1]
    @test 20.0 ≈ dx2[1]

    function mul2(y)
        return y[1] * 2
    end

    function fwdlatestfoo(y)
        Base.invokelatest(mul2, y)
    end

    x = [1.0, 2.0, 3.0]
    dx = [1.0, 1.0, 1.0]
    dx2 = [10.0, 20.0, 30.0]

    res = Enzyme.autodiff(ForwardWithPrimal, fwdlatestfoo, BatchDuplicated, BatchDuplicated(x, (dx, dx2)))

    @test 2.0 ≈ res[1][1]
    @test 20.0 ≈ res[1][2]
    @test 2.0 ≈ res[2][1]

    res = Enzyme.autodiff(Forward, fwdlatestfoo, BatchDuplicated, BatchDuplicated(x, (dx, dx2)))
    @test 2.0 ≈ res[1][1]
    @test 20.0 ≈ res[1][2]


    function revfoo(out, x)
        out[] = x * x
        nothing
    end

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(10.0)

    res = Enzyme.autodiff(Reverse, revfoo, BatchDuplicated(out, (dout, dout2)), Active(2.0))[1][2]
    @test 4.0 ≈ res[1]
    @test 40.0 ≈ res[2]
    @test 0.0 ≈ dout[]
    @test 0.0 ≈ dout2[]

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(10.0)

    function rev_lq(y)
        return y * y
    end
    function revfoo2(out, x)
        out[] = Base.invokelatest(rev_lq, x)::Float64
        nothing
    end
    res = Enzyme.autodiff(Reverse, revfoo2, BatchDuplicated(out, (dout, dout2)), Active(2.0))[1][2]
    @test 4.0 ≈ res[1]
    @test 40.0 ≈ res[2]
    @test 0.0 ≈ dout[]
    @test 0.0 ≈ dout2[]

end

function batchgf(out, args)
    res = 0.0
    x = Base.inferencebarrier((args[1][1],))
    for v in x
        v = v::Float64
        res += v
        break
    end
    out[] = res
    return nothing
end

@testset "Batch Getfield" begin
    x = [(2.0, 3.0)]
    dx = [(0.0, 0.0)]
    dx2 = [(0.0, 0.0)]
    dx3 = [(0.0, 0.0)]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    dout3 = Ref(5.0)
    Enzyme.autodiff(Reverse, batchgf, Const, BatchDuplicatedNoNeed(out, (dout, dout2, dout3)), BatchDuplicated(x, (dx, dx2, dx3)))
    @test dx[1][1] ≈ 1.0
    @test dx[1][2] ≈ 0.0
    @test dx2[1][1] ≈ 3.0
    @test dx2[1][2] ≈ 0.0
    @test dx3[1][1] ≈ 5.0
    @test dx2[1][2] ≈ 0.0
end

@testset "Batch Forward" begin
    square(x) = x * x
    bres = autodiff(Forward, square, BatchDuplicated, BatchDuplicated(3.0, (1.0, 2.0, 3.0)))
    @test length(bres) == 1
    @test length(bres[1]) == 3
    @test bres[1][1] ≈ 6.0
    @test bres[1][2] ≈ 12.0
    @test bres[1][3] ≈ 18.0

    bres = autodiff(Forward, square, BatchDuplicated, BatchDuplicated(3.0 + 7.0im, (1.0 + 0im, 2.0 + 0im, 3.0 + 0im)))
    @test bres[1][1] ≈ 6.0 + 14.0im
    @test bres[1][2] ≈ 12.0 + 28.0im
    @test bres[1][3] ≈ 18.0 + 42.0im

    squareidx(x) = x[1] * x[1]
    inp = Float32[3.0]

    # Shadow offset is not the same as primal so following doesn't work
    # d_inp = Float32[1.0, 2.0, 3.0]
    # autodiff(Forward, squareidx, BatchDuplicated, BatchDuplicated(view(inp, 1:1), (view(d_inp, 1:1), view(d_inp, 2:2), view(d_inp, 3:3))))

    d_inp = (Float32[1.0], Float32[2.0], Float32[3.0])
    bres = autodiff(Forward, squareidx, BatchDuplicated, BatchDuplicated(inp, d_inp))
    @test bres[1][1] ≈ 6.0
    @test bres[1][2] ≈ 12.0
    @test bres[1][3] ≈ 18.0
end

@testset "Batch Reverse" begin
    function refbatchbwd(out, x)
        v = x[]
        out[1] = v
        out[2] = v * v
        out[3] = v * v * v
        nothing
    end

    dxs = (Ref(0.0), Ref(0.0), Ref(0.0))
    out = Float64[0, 0, 0]
    x = Ref(2.0)

    autodiff(Reverse, refbatchbwd, BatchDuplicated(out, Enzyme.onehot(out)), BatchDuplicated(x, dxs))
    @test dxs[1][] ≈ 1.0
    @test dxs[2][] ≈ 4.0
    @test dxs[3][] ≈ 12.0

    function batchbwd(out, v)
        out[1] = v
        out[2] = v * v
        out[3] = v * v * v
        nothing
    end

    bres = Enzyme.autodiff(Reverse, batchbwd, BatchDuplicated(out, Enzyme.onehot(out)), Active(2.0))[1]
    @test length(bres) == 2
    @test length(bres[2]) == 3
    @test bres[2][1] ≈ 1.0
    @test bres[2][2] ≈ 4.0
    @test bres[2][3] ≈ 12.0

    times2(x) = x * 2
    xact = BatchDuplicated([1.0, 2.0, 3.0, 4.0, 5.0], (zeros(5), zeros(5)))
    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(times2)}, BatchDuplicated, typeof(xact))

    tape, primal, shadow = forward(Const(times2), xact)
    dy1 = [0.07, 0.011, 0.013, 0.017, 0.019]
    dy2 = [0.23, 0.029, 0.031, 0.037, 0.041]
    copyto!(shadow[1], dy1)
    copyto!(shadow[2], dy2)
    r = pullback(Const(times2), xact, tape)
    @test xact.dval[1] ≈ dy1 * 2
    @test xact.dval[2] ≈ dy2 * 2
end

@testset "Batched inactive" begin
    augres = Enzyme.Compiler.runtime_generic_augfwd(
        Val{(false, false, false)}, Val(false), Val(false), Val(2), Val((true, true, true)),
        Val(Enzyme.Compiler.AnyArray(2 + Int(2))),
        ==, nothing, nothing,
        :foo, nothing, nothing,
        :bar, nothing, nothing
    )

    Enzyme.Compiler.runtime_generic_rev(
        Val{(false, false, false)}, Val(false), Val(false), Val(2), Val((true, true, true)), augres[end],
        ==, nothing, nothing,
        :foo, nothing, nothing,
        :bar, nothing, nothing
    )
end

@testset "Uncached batch sizes" begin
    genericsin(x) = Base.invokelatest(sin, x)
    res = Enzyme.autodiff(Forward, genericsin, BatchDuplicated(2.0, NTuple{10, Float64}((Float64(i) for i in 1:10))))[1]
    for (i, v) in enumerate(res)
        @test v ≈ i * -0.4161468365471424
    end
    @assert length(res) == 10
    res = Enzyme.autodiff(Forward, genericsin, BatchDuplicated(2.0, NTuple{40, Float64}((Float64(i) for i in 1:40))))[1]
    for (i, v) in enumerate(res)
        @test v ≈ i * -0.4161468365471424
    end
    @assert length(res) == 40
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

    Enzyme.autodiff(Enzyme.Reverse, outergeneric, Enzyme.Duplicated(weights, dweights))

    @test dweights[1] ≈ 1.0
end


function Valuation1(z, Ls1)
    @inbounds Ls1[1] = sum(Base.inferencebarrier(z))
    return nothing
end
@testset "Active setindex!" begin
    v = ones(5)
    dv = zero(v)

    DV1 = Float32[0]
    DV2 = Float32[1]

    Enzyme.autodiff(Reverse, Valuation1, Duplicated(v, dv), Duplicated(DV1, DV2))
    @test dv[1] ≈ 1.0

    DV1 = Float32[0]
    DV2 = Float32[1]
    v = ones(5)
    dv = zero(v)
    dv[1] = 1.0
    Enzyme.autodiff(Forward, Valuation1, Duplicated(v, dv), Duplicated(DV1, DV2))
    @test DV2[1] ≈ 1.0
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
    @test dweights[1] ≈ 1.0

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
    @test dweights[1] ≈ 20.0
    @test dweights[2] ≈ 20.0
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
        Val(false),
        Val(1),
        Val((true, true, true)),
        Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
        Base.getindex,
        nothing,
        ((nothing,), MyFlux()),
        ((nothing,), MyFlux()),
        1,
        nothing,
    )

    nt1 = Enzyme.Compiler.runtime_generic_augfwd(args...)
    @test nt1[1] == (nothing,)
    @test nt1[2] == (nothing,)

    args2 = (
        Val{(false, false, false)},
        Val(false),
        Val(false),
        Val(1),
        Val((true, true, true)),
        Base.Val(NamedTuple{(Symbol("1"), Symbol("2"), Symbol("3")), Tuple{Any, Any, Any}}),
        Base.getindex,
        nothing,
        ((nothing,), MyFlux()),
        ((nothing,), MyFlux()),
        2,
        nothing,
    )

    nt = Enzyme.Compiler.runtime_generic_augfwd(args2...)
    @test nt[1] == MyFlux()
    @test nt[2] == MyFlux()
end

@testset "Union return getproperty" begin
    struct DOSData
        interp_func
    end

    function get_dos(Ef = 0.0)
        return x -> x + Ef
    end

    struct MyMarcusHushChidseyDOS
        A::Float64
        dos::DOSData
    end

    mhcd = MyMarcusHushChidseyDOS(0.3, DOSData(get_dos()))

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

@testset "Method errors" begin
    fwd = Enzyme.autodiff_thunk(Forward, Const{typeof(sum)}, Duplicated, Duplicated{Vector{Float64}})
    @test_throws Enzyme.Compiler.ThunkCallError fwd(ones(10))
    @test_throws Enzyme.Compiler.ThunkCallError fwd(Duplicated(ones(10), ones(10)))
    @test_throws Enzyme.Compiler.ThunkCallError fwd(Const(first), Duplicated(ones(10), ones(10)))
    # TODO
    # @test_throws MethodError fwd(Const(sum), Const(ones(10)))
    fwd(Const(sum), Duplicated(ones(10), ones(10)))
end

@testset "Exception" begin

    f_no_derv(x) = ccall("extern doesnotexist", llvmcall, Float64, (Float64,), x)
    @test_throws Enzyme.Compiler.EnzymeNoDerivativeError autodiff(Reverse, f_no_derv, Active, Active(0.5))

    f_union(cond, x) = cond ? x : 0
    g_union(cond, x) = f_union(cond, x) * x

    # This only works as a test in < 1.12 as we actually optimize away the issue in later LLVM's
    if VERSION < v"1.12"
        if sizeof(Int) == sizeof(Int64)
            @test_throws Enzyme.Compiler.IllegalTypeAnalysisException autodiff(Reverse, g_union, Active, Const(true), Active(1.0))
        else
            @test_throws Enzyme.Compiler.IllegalTypeAnalysisException autodiff(Reverse, g_union, Active, Const(true), Active(1.0f0))
        end
    end

    # TODO: Add test for NoShadowException
end

@testset "UndefVar" begin
    function f_undef(x, y)
        if x
            undefinedfnthowmagic()
        end
        y
    end
    @test 1.0 ≈ autodiff(Reverse, f_undef, Const(false), Active(2.14))[1][2]
    @test_throws Base.UndefVarError autodiff(Reverse, f_undef, Const(true), Active(2.14))

    @test 1.0 ≈ autodiff(Forward, f_undef, Const(false), Duplicated(2.14, 1.0))[1]
    @test_throws Base.UndefVarError autodiff(Forward, f_undef, Const(true), Duplicated(2.14, 1.0))
end

@testset "Return GC error" begin
    t = 0.0

    function tobedifferentiated(cond, a)::Float64
        if cond
            t + t
        else
            0.0
        end
    end

    @test 0.0 ≈ autodiff(Reverse, tobedifferentiated, Const(true), Active(2.1))[1][2]
    @test 0.0 ≈ autodiff(Forward, tobedifferentiated, Const(true), Duplicated(2.1, 1.0))[1]

    function tobedifferentiated2(cond, a)::Float64
        if cond
            a + t
        else
            0.0
        end
    end

    @test 1.0 ≈ autodiff(Reverse, tobedifferentiated2, Const(true), Active(2.1))[1][2]
    @test 1.0 ≈ autodiff(Forward, tobedifferentiated2, Const(true), Duplicated(2.1, 1.0))[1]

    @noinline function copy(dest, p1, cond)
        bc = convert(Broadcast.Broadcasted{Nothing}, Broadcast.instantiate(p1))

        if cond
            return nothing
        end

        bc2 = Broadcast.preprocess(dest, bc)
        @inbounds    dest[1] = bc2[1]

        nothing
    end

    function mer(F, F_H, cond)
        p1 = Base.broadcasted(Base.identity, F_H)
        copy(F, p1, cond)

        # Force an apply generic
        flush(stdout)
        nothing
    end

    L_H = Array{Float64, 1}(undef, 2)
    L = Array{Float64, 1}(undef, 2)

    F_H = [1.0, 0.0]
    F = [1.0, 0.0]

    autodiff(Reverse, mer, Duplicated(F, L), Duplicated(F_H, L_H), Const(true))
    autodiff(Forward, mer, Duplicated(F, L), Duplicated(F_H, L_H), Const(true))
end

@testset "No speculation" begin
    mutable struct SpecFoo

        iters::Int
        a::Float64
        b::Vector{Float64}

    end

    function f(Foo)
        for i in 1:Foo.iters

            c = -1.0

            if Foo.a < 0.0
                X = (-Foo.a)^0.25
                c = 2 * log(X)
            end

            # set b equal to desired result
            Foo.b[1] = 1.0 / c

            return nothing
        end
    end

    foo = SpecFoo(1, 1.0, zeros(Float64, 1))
    dfoo = SpecFoo(0, 0.0, zeros(Float64, 1))

    # should not throw a domain error, which
    # will occur if the pow is mistakenly speculated
    Enzyme.autodiff(Reverse, f, Duplicated(foo, dfoo))
end

@testset "Nested AD" begin
    tonest(x, y) = (x + y)^2

    @test autodiff(Forward, (x, y) -> autodiff(Forward, Const(tonest), Duplicated(x, 1.0), Const(y))[1], Const(1.0), Duplicated(2.0, 1.0))[1] ≈ 2.0
end

catsin(x::Number) = hcat(sin.(x .* [1, 2]))

function inner_reverse(x)
    return Enzyme.jacobian(Enzyme.Reverse, catsin, x)[1]
end

@testset "Nested AD With Allocation" begin
    @test Enzyme.autodiff(Enzyme.Forward, inner_reverse, Enzyme.Duplicated(3.1, 2.7))[1] ≈ reshape([-0.11226778856988433 0.8973655504289612 ], (2, 1))
end

@testset "Hessian" begin
    function origf(x::Array{Float64}, y::Array{Float64})
        y[1] = x[1] * x[1] + x[2] * x[1]
        return nothing
    end

    function grad(x, dx, y, dy)
        Enzyme.autodiff(Reverse, Const(origf), Duplicated(x, dx), DuplicatedNoNeed(y, dy))
        nothing
    end

    x = [2.0, 2.0]
    y = Vector{Float64}(undef, 1)
    dx = [0.0, 0.0]
    dy = [1.0]

    grad(x, dx, y, dy)

    vx = ([1.0, 0.0], [0.0, 1.0])
    hess = ([0.0, 0.0], [0.0, 0.0])
    dx2 = [0.0, 0.0]
    dy = [1.0]

    Enzyme.autodiff(
        Enzyme.Forward, grad,
        Enzyme.BatchDuplicated(x, vx),
        Enzyme.BatchDuplicated(dx2, hess),
        Const(y),
        Const(dy)
    )

    @test dx ≈ dx2
    @test hess[1][1] ≈ 2.0
    @test hess[1][2] ≈ 1.0
    @test hess[2][1] ≈ 1.0
    @test hess[2][2] ≈ 0.0

    function f_ip(x, tmp)
        tmp .= x ./ 2
        return dot(tmp, x)
    end

    function f_gradient_deferred!(dx, x, tmp)
        dtmp = make_zero(tmp)
        autodiff_deferred(Reverse, Const(f_ip), Active, Duplicated(x, dx), Duplicated(tmp, dtmp))
        return nothing
    end

    function f_hvp!(hv, x, v, tmp)
        dx = make_zero(x)
        btmp = make_zero(tmp)
        autodiff(
            Forward,
            f_gradient_deferred!,
            Duplicated(dx, hv),
            Duplicated(x, v),
            Duplicated(tmp, btmp),
        )
        return nothing
    end

    x = [1.0]
    v = [-1.0]
    hv = make_zero(v)
    tmp = similar(x)

    f_hvp!(hv, x, v, tmp)
    @test hv ≈ [-1.0]
end

@noinline function womylogpdf(X::AbstractArray{<:Real})
    return map(womylogpdf, X)
end

function womylogpdf(x::Real)
    return (x - 2)
end


function wologpdf_test(x)
    return womylogpdf(x)
end

@testset "Ensure writeonly deduction combines with capture" begin
    res = Enzyme.autodiff(Enzyme.Forward, wologpdf_test, Duplicated([0.5], [0.7]))
    @test res[1] ≈ [0.7]
end

function objective!(x, loss, R)
    for i in 1:1000
        y = zeros(3)
        y[1] = R[1, 1] * x[1] + R[1, 2] * x[2] + R[1, 3] * x[3]

        loss[] = y[1]
    end
    return nothing
end

@testset "Static tape allocation" begin
    x = zeros(3)
    R = [
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]
    loss = Ref(0.0)
    dloss = Ref(1.0)

    autodiff(Reverse, objective!, Duplicated(x, zero(x)), Duplicated(loss, dloss), Const(R))

    @test loss[] ≈ 0.0
    @test dloss[] ≈ 0.0
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

    X = map(x -> rand(Float32, 5), 1:1000)
    Y = map(x -> rand(Float32), 1:1000)

    grads = zero(ps)
    for epoch in 1:1000
        fill!(grads, 0)
        autodiff(Reverse, ldynloss, Const(X), Const(Y), Duplicated(ps, grads), Active(bs))
    end

end

@testset "Static activity" begin
    struct Test2{T}
        obs::T
    end

    function test(t, x)
        o = t.obs
        y = (x .- o)
        yv = @inbounds y[1]
        return yv * yv
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

    autodiff(
        Reverse, advance,
        Duplicated(u_v_eta, ad_eta),
        Const(rhs_terms),
    )
    @test ad_eta[1] ≈ 0.0
end

function absset(out, x)
    @inbounds out[1] = (x,)
    return nothing
end

@testset "Abstract Array element type" begin
    out = Tuple{Any}[(9.7,)]
    dout = Tuple{Any}[(4.3,)]

    autodiff(
        Enzyme.Forward,
        absset,
        Duplicated(out, dout),
        Duplicated(3.1, 2.4)
    )
    @test dout[1][1] ≈ 2.4
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
        return f.x * 5.0
    end

    res = autodiff(Reverse, g, Active, Active(Roo(3.0, "a")))[1][1]

    @test res.x == 5.0

    res = autodiff(Reverse, g, Active, Active(Moo(3.0, "a")))[1][1]

    @test res.x == 5.0
end

@testset "Type preservation" begin
    # Float16 fails due to #870
    for T in (Float64, Float32 #=Float16=#)
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
    @test Enzyme.gradient(Enzyme.Reverse, fexpandempty, vec)[1] ≈ [1.0]
    @test Enzyme.gradient(Enzyme.Forward, fexpandempty, vec)[1] ≈ [1.0]
end
const CUmemoryPool2 = Ptr{Float64}

struct CUmemPoolProps2
    reserved::NTuple{31, Char}
end

mutable struct CuMemoryPool2
    handle::CUmemoryPool2
end

function ccall_macro_lower(func, rettype, types, args, gcsafe_or_nreq...)
    # instead of re-using ccall or Expr(:foreigncall) to perform argument conversion,
    # we need to do so ourselves in order to insert a jl_gc_safe_enter|leave
    # just around the inner ccall

    cconvert_exprs = []
    cconvert_args = []
    for (typ, arg) in zip(types, args)
        var = gensym("$(func)_cconvert")
        push!(cconvert_args, var)
        push!(
            cconvert_exprs, quote
                $var = Base.cconvert($(esc(typ)), $(esc(arg)))
            end
        )
    end

    unsafe_convert_exprs = []
    unsafe_convert_args = []
    for (typ, arg) in zip(types, cconvert_args)
        var = gensym("$(func)_unsafe_convert")
        push!(unsafe_convert_args, var)
        push!(
            unsafe_convert_exprs, quote
                $var = Base.unsafe_convert($(esc(typ)), $arg)
            end
        )
    end

    return quote
        $(cconvert_exprs...)

        $(unsafe_convert_exprs...)

        ret = ccall(
            $(esc(func)), $(esc(rettype)), $(Expr(:tuple, map(esc, types)...)),
            $(unsafe_convert_args...)
        )
    end
end

macro gcsafe_ccall(expr)
    return ccall_macro_lower(Base.ccall_macro_parse(expr)...)
end

function cuMemPoolCreate2(pool, poolProps)
    # CUDA.initialize_context()
    #CUDA.
    gc_state = @ccall(jl_gc_safe_enter()::Int8)
    @gcsafe_ccall cuMemPoolCreate(
        pool::Ptr{CUmemoryPool2},
        poolProps::Ptr{CUmemPoolProps2}
    )::Cvoid
    return @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
end

function cual()
    props = Ref(
        CUmemPoolProps2(
            ntuple(i -> Char(0), 31)
        )
    )
    handle_ref = Ref{CUmemoryPool2}()
    cuMemPoolCreate2(handle_ref, props)

    return CuMemoryPool2(handle_ref[])
end

@testset "Unused shadow phi rev" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(cual)}, Duplicated)
end
