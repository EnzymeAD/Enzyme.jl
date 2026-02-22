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
