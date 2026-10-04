using Enzyme, Test

@noinline function mixedmul(tup::T) where {T}
    return tup[1] * tup[2][1]
end

function outmixedmul(x::Float64)
    vec = [x]
    tup = (x, vec)
    return Base.inferencebarrier(mixedmul)(tup)::Float64
end

function outmixedmul2(res, x::Float64)
    vec = [x]
    tup = (x, vec)
    return res[] = Base.inferencebarrier(mixedmul)(tup)::Float64
end

@testset "Basic Mixed Activity" begin
    @test 6.2 ≈ Enzyme.autodiff(Reverse, outmixedmul, Active, Active(3.1))[1][1]
end

@testset "Byref Mixed Activity" begin
    res = Ref(4.7)
    dres = Ref(1.0)
    @test 6.2 ≈ Enzyme.autodiff(Reverse, outmixedmul2, Const, Duplicated(res, dres), Active(3.1))[1][2]
end

@testset "Batched Byref Mixed Activity" begin
    res = Ref(4.7)
    dres = Ref(1.0)
    dres2 = Ref(3.0)
    sig = Enzyme.autodiff(Reverse, outmixedmul2, Const, BatchDuplicated(res, (dres, dres2)), Active(3.1))
    @test 6.2 ≈ sig[1][2][1]
    @test 3 * 6.2 ≈ sig[1][2][2]
end

function tupmixedmul(x::Float64)
    vec = [x]
    tup = (x, Base.inferencebarrier(vec))
    return Base.inferencebarrier(mixedmul)(tup)::Float64
end

@testset "Tuple Mixed Activity" begin
    @test 6.2 ≈ Enzyme.autodiff(Reverse, tupmixedmul, Active, Active(3.1))[1][1]
end

function outtupmixedmul(res, x::Float64)
    vec = [x]
    tup = (x, Base.inferencebarrier(vec))
    return res[] = Base.inferencebarrier(mixedmul)(tup)::Float64
end

@testset "Byref Tuple Mixed Activity" begin
    res = Ref(4.7)
    dres = Ref(1.0)
    @test 6.2 ≈ Enzyme.autodiff(Reverse, outtupmixedmul, Const, Duplicated(res, dres), Active(3.1))[1][2]
end

@testset "Batched Byref Tuple Mixed Activity" begin
    res = Ref(4.7)
    dres = Ref(1.0)
    dres2 = Ref(3.0)
    sig = Enzyme.autodiff(Reverse, outtupmixedmul, Const, BatchDuplicated(res, (dres, dres2)), Active(3.1))
    @test 6.2 ≈ sig[1][2][1]
    @test 3 * 6.2 ≈ sig[1][2][2]
end

struct Foobar
    x::Int
    y::Int
    z::Int
    q::Int
    r::Float64
end

function bad_abi(fb)
    v = fb.x
    throw(AssertionError("saw bad val $v"))
end

@testset "Mixed PrimalError" begin
    @test_throws AssertionError autodiff(Reverse, bad_abi, MixedDuplicated(Foobar(2, 3, 4, 5, 6.0), Ref(Foobar(2, 3, 4, 5, 6.0))))
end


function flattened_unique_values(tupled)
    flattened = flatten_tuple(tupled)

    return nothing
end

@inline flatten_tuple(a::Tuple) = tuple(inner_flatten_tuple(a[1])..., inner_flatten_tuple(a[2:end])...)
@inline flatten_tuple(a::Tuple{<:Any}) = tuple(inner_flatten_tuple(a[1])...)

@inline inner_flatten_tuple(a) = tuple(a)
@inline inner_flatten_tuple(a::Tuple) = flatten_tuple(a)
@inline inner_flatten_tuple(a::Tuple{}) = ()


struct Center end

struct Field{LX}
    grid::Float64
    data::Float64
end

@testset "Mixed Unstable Return" begin
    grid = 1.0
    data = 2.0
    f1 = Field{Center}(grid, data)
    f2 = Field{Center}(grid, data)
    f3 = Field{Center}(grid, data)
    f4 = Field{Center}(grid, data)
    f5 = Field{Nothing}(grid, data)
    thing = (f1, f2, f3, f4, f5)
    dthing = Enzyme.make_zero(thing)

    dedC = autodiff(
        Enzyme.Reverse,
        flattened_unique_values,
        Duplicated(thing, dthing)
    )
end


function literalrt(x)
    y = Base.inferencebarrier(x * x)
    y2 = Base.inferencebarrier(x * x * x)
    return (y, y2)
end

@testset "Literal RT mismatch" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(literalrt)}, Active{Tuple{Float64, Float64}}, Active{Float64})

    tape, = fwd(Const(literalrt), Active(3.1))

    x = 3.1
    @test rev(Const(literalrt), Active(3.1), (2.7, 0.2), tape)[1][1] ≈ 2 * x * 2.7 + 3 * x * x * 0.2

end

function literalrt_mixed(x)
    y = Base.inferencebarrier(x * x)
    y2 = Base.inferencebarrier([x * x * x])
    return (y, y2)
end

@testset "Mixed Literal RT mismatch" begin
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(literalrt_mixed)}, MixedDuplicated{Tuple{Float64, Vector{Float64}}}, Active{Float64})

    tape, prim, shad = fwd(Const(literalrt_mixed), Active(3.1))

    shad[][2][1] = 0.2

    x = 3.1
    @test rev(Const(literalrt_mixed), Active(3.1), (2.7, shad[][2]), tape)[1][1] ≈ 2 * x * 2.7 + 3 * x * x * 0.2
end

struct DMixedForTest
    a::Vector{Float64}
    b::Float64
end

function lp_mixed_for_test(d::DMixedForTest, x::AbstractVector{<:Real})
    return d.a[1] * x[1]
end

@testset "Mixed activity with Base.Fix1" begin
    dd = DMixedForTest([1.0], 0.0)
    f(params) = Base.Fix1(lp_mixed_for_test, dd)(params)
    res = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f, [0.5])
    @test res[1] ≈ [1.0]
end

struct MixedWidthParams
    α::Float64
    β::Float64
end
mixed_width_f!(out, p) = (out[1] = p.α * 2 + p.β * 3; nothing)

@testset "MixedDuplicated in a batched thunk is rejected" begin
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(mixed_width_f!)}, Const{Nothing},
        BatchDuplicated{Vector{Float64}, 2}, MixedDuplicated{MixedWidthParams},
    )
    bd = BatchDuplicated(zeros(1), ([1.0], [10.0]))
    md = MixedDuplicated(MixedWidthParams(1.0, 1.0), Ref(MixedWidthParams(0.0, 0.0)))
    @test_throws ErrorException fwd(Const(mixed_width_f!), bd, md)

    # The batched annotation works
    r1 = Ref(MixedWidthParams(0.0, 0.0))
    r2 = Ref(MixedWidthParams(0.0, 0.0))
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(mixed_width_f!)}, Const{Nothing},
        BatchDuplicated{Vector{Float64}, 2}, BatchMixedDuplicated{MixedWidthParams, 2},
    )
    bmd = BatchMixedDuplicated(MixedWidthParams(1.0, 1.0), (r1, r2))
    tape = fwd(Const(mixed_width_f!), bd, bmd)[1]
    rev(Const(mixed_width_f!), bd, bmd, tape)
    @test r1[] == MixedWidthParams(2.0, 3.0)
    @test r2[] == MixedWidthParams(20.0, 30.0)
end

@testset "MixedDuplicated with pointer shadows" begin
    p = MixedWidthParams(1.0, 1.0)
    dp = [MixedWidthParams(0.0, 0.0), MixedWidthParams(0.0, 0.0)]
    GC.@preserve dp begin
        autodiff(Reverse, mixed_width_f!, Const, Duplicated(zeros(1), [1.0]), MixedDuplicated(p, pointer(dp)))
        @test dp[1] == MixedWidthParams(2.0, 3.0)
        @test dp[2] == MixedWidthParams(0.0, 0.0)

        fill!(dp, MixedWidthParams(0.0, 0.0))
        autodiff(
            Reverse, mixed_width_f!, Const, BatchDuplicated(zeros(1), ([1.0], [10.0])),
            BatchMixedDuplicated(p, (pointer(dp, 1), pointer(dp, 2))),
        )
        @test dp == [MixedWidthParams(2.0, 3.0), MixedWidthParams(20.0, 30.0)]

        # Split mode, with the pointer shadow in the thunk type
        fill!(dp, MixedWidthParams(0.0, 0.0))
        md = MixedDuplicated(p, pointer(dp))
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(mixed_width_f!)}, Const{Nothing},
            Duplicated{Vector{Float64}}, typeof(md),
        )
        d = Duplicated(zeros(1), [1.0])
        tape = fwd(Const(mixed_width_f!), d, md)[1]
        rev(Const(mixed_width_f!), d, md, tape)
        @test dp[1] == MixedWidthParams(2.0, 3.0)

        # A thunk for `MixedDuplicated{T}` takes a `RefValue{T}` shadow
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(mixed_width_f!)}, Const{Nothing},
            Duplicated{Vector{Float64}}, MixedDuplicated{MixedWidthParams},
        )
        @test_throws Enzyme.Compiler.ThunkCallError fwd(Const(mixed_width_f!), d, md)
    end
end
