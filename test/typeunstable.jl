using Enzyme, Test

@eval construct_splatnew(T, fields) = $(Expr(:splatnew, :T, :fields))

struct ActivePair
	x::Float32
	y::Float64
end

function toactivepair(x, y)
	tup = Base.inferencebarrier((x, y))
	pair = construct_splatnew(ActivePair, tup)
	(pair.x * pair.y)::Float64
end

struct VectorPair
	x::Vector{Float32}
	y::Vector{Float64}
end


function tovectorpair(x, y)
	tup = Base.inferencebarrier((x, y))
	pair = construct_splatnew(VectorPair, tup)
	(pair.x[1] * pair.y[1])::Float64
end


function toactivepair!(res, x, y)
	tup = Base.inferencebarrier((x[1], y[1]))
	pair = construct_splatnew(ActivePair, tup)
	res[] = (pair.x * pair.y)::Float64
	nothing
end

function tovectorpair!(res, x, y)
	tup = Base.inferencebarrier((x, y))
	pair = construct_splatnew(VectorPair, tup)
	res[] = (pair.x[1] * pair.y[1])::Float64
	nothing
end

@testset "Reverse Unstable newstructt" begin
	res = Enzyme.autodiff(Reverse, toactivepair, Active(2.7f0), Active(3.1))
	@test res[1][1] ≈ 3.1f0
	@test res[1][2] ≈ 2.700000047683716

	x = Float32[2.7f0]
	dx = Float32[0.0f0]
	y = Float64[3.1]
	dy = Float64[0.0]

	Enzyme.autodiff(Reverse, tovectorpair, Duplicated(x, dx), Duplicated(y, dy))
	@test dx[1] ≈ 3.1f0
	@test dy[1] ≈ 2.700000047683716

	x = Float32[2.7f0]
	dx = Float32[0.0f0]
	dx2 = Float32[0.0f0]
	y = Float64[3.1]
	dy = Float64[0.0]
	dy2 = Float64[0.0]

	res = Ref(0.0)
	dres = Ref(1.0)
	dres2 = Ref(3.0)

	Enzyme.autodiff(Reverse, toactivepair!, BatchDuplicated(res, (dres, dres2)), BatchDuplicated(x, (dx, dx2)), BatchDuplicated(y, (dy, dy2)))

	@test dx[1] ≈ 3.1f0
	@test dy[1] ≈ 2.700000047683716

	@test dx2[1] ≈ 3.1f0 * 3
	@test dy2[1] ≈ 2.700000047683716 * 3


	x = Float32[2.7f0]
	dx = Float32[0.0f0]
	dx2 = Float32[0.0f0]
	y = Float64[3.1]
	dy = Float64[0.0]
	dy2 = Float64[0.0]

	res = Ref(0.0)
	dres = Ref(1.0)
	dres2 = Ref(3.0)

	Enzyme.autodiff(Reverse, tovectorpair!, BatchDuplicated(res, (dres, dres2)), BatchDuplicated(x, (dx, dx2)), BatchDuplicated(y, (dy, dy2)))

	@test dx[1] ≈ 3.1f0
	@test dy[1] ≈ 2.700000047683716

	@test dx2[1] ≈ 3.1f0 * 3
	@test dy2[1] ≈ 2.700000047683716 * 3

end

@testset "Forward Unstable newstructt" begin
	res = Enzyme.autodiff(Forward, toactivepair, Duplicated(2.7f0, 2.0f0), Duplicated(3.1, 3.0))
	@test res[1] ≈ 2.7f0 * 3.0 + 2.0f0 * 3.1
	res = Enzyme.autodiff(Forward, toactivepair, BatchDuplicated(2.7f0, (2.0f0, 5.0f0)), BatchDuplicated(3.1, (3.0, 7.0)))
	@test res[1][1] ≈ 2.7f0 * 3.0 + 2.0f0 * 3.1
	@test res[1][2] ≈ 2.7f0 * 7.0 + 5.0f0 * 3.1	
end

struct InsFwdNormal1{T<:Real}
	σ::T
end

struct InsFwdNormal2{T<:Real}
	σ::T
end

insfwdlogpdf(d, x) = d.σ

function insfwdfunc(x)
    dists = [InsFwdNormal1{Float64}(1.0), InsFwdNormal2{Float64}(1.0)]
    return sum(Base.Fix2(insfwdlogpdf, x), dists)
end

@testset "Forward Batch Constant insertion" begin
    res = Enzyme.gradient(Enzyme.Forward, insfwdfunc, [0.5, 0.7])[1]
    @test res ≈ [0.0, 0.0]
end

function use(x)
    use(x[1]) * use(x[2])
end
function use(x::Vector{Float64})
    @inbounds x[1]
end
function tupq(x, y)
    res = (Base.inferencebarrier(x), y)
    use(res)::Float64
end

@testset "Runtime Activity Tuple Construction" begin
    x = [2.0]
    y = [3.0]
    dy = [0.0]

    @test_throws Enzyme.Compiler.EnzymeRuntimeActivityError Enzyme.autodiff(Reverse, tupq, Const(x), Duplicated(y, dy))

    x = [2.0]
    y = [3.0]
    dy = [0.0]
    Enzyme.autodiff(set_runtime_activity(Reverse), tupq, Const(x), Duplicated(y, dy))
    
    @test x ≈ [2.0]
    @test y ≈ [3.0]
    @test dy ≈ [2.0]
end

@noinline function setone(rany)
   rany[] = 1.0
   nothing
end

function typeunstable_constant_shadow()
        rany = Ref{Any}()
        setone(rany)
        return rany[]
end

@testset "Zero type unstable shadow" begin
   fwd, _ = autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(typeunstable_constant_shadow)}, Active{Float64})
   _, _, shad = fwd(Const(typeunstable_constant_shadow))
end

kwcallee(x; saveat) = x + saveat # callee takes a keyword argument
typeunstable_kwloss(x, r) = kwcallee(x; saveat = r[])

@testset "Inactive argument to new struct" begin
	@test Enzyme.gradient(Enzyme.Reverse, typeunstable_kwloss, 2.7, Enzyme.Const(Base.RefValue{Any}(3.1)))[1] ≈ 1.0
end

@noinline newstruct_runtime_any()::Any = Base.inferencebarrier(Ref(1.0))

mutable struct NewstructConstField{T}
    v::Float64
    inner::T
end

function newstruct_const_active_field(x)
    w = NewstructConstField(0.0, newstruct_runtime_any())
    w.v = sum(abs2, x)
    return w.v
end

@testset "Runtime newstruct with constant active-typed field" begin
    g = Enzyme.gradient(
        set_runtime_activity(Reverse), newstruct_const_active_field, [3.0, 1.0]
    )[1]
    @test g ≈ [6.0, 2.0]
end

@noinline mutwrap_runtime_any()::Any = Base.inferencebarrier(Ref(1.0))

mutable struct MutWrapGeneric{T}
    v::Float64
    inner::T
end

@noinline mutwrap_use(w, x) = w.v * @inbounds x[1]

function mutwrap_generic_call(x)
    w = MutWrapGeneric(0.0, mutwrap_runtime_any())
    w.v = x[2]
    f = Base.inferencebarrier(mutwrap_use)
    return (f(w, x))::Float64
end

@testset "Mutable runtime newstruct shadow passed to generic call" begin
    g = Enzyme.gradient(
        set_runtime_activity(Reverse), mutwrap_generic_call, [3.0, 5.0]
    )[1]
    @test g ≈ [5.0, 3.0]
end

# ODEProblem-shaped immutable struct, parameterized by `iip`.
struct Prob3246{iip, U, T, P}
    u0::U      # inactive array
    tspan::T   # inactive isbits tuple
    p::P       # active array
end
Prob3246{iip}(u0, tspan, p) where {iip} =
    Prob3246{iip, typeof(u0), typeof(tspan), typeof(p)}(u0, tspan, p)

@noinline build_3246(u0, tspan, p, flag) = Prob3246{flag}(u0, tspan, p)

function loss_3246(p, flag)
    sum(abs2, build_3246([1.0, 2.0], (0.0, 1.0), p, flag).p)
end

@testset "Issue 3246 HVP with runtime activity" begin
    RA_R = Enzyme.set_runtime_activity(Enzyme.Reverse)
    RA_F = Enzyme.set_runtime_activity(Enzyme.Forward)
    
    FLAG = Ref(true)
    runtime_iip() = FLAG[]
    
    grad(x) = Enzyme.gradient(RA_R, p -> loss_3246(p, runtime_iip()), x)[1]
    
    x0 = [1.0, 2.0, 3.0]
    v = [1.0, 0.0, 0.0]
    
    res = Enzyme.autodiff(RA_F, Enzyme.Const(grad), Enzyme.Duplicated(x0, v))
    @test res[1] ≈ [2.0, 0.0, 0.0]
end

struct TypeUnstableGetfieldBox{T1, T2}
    a::T1
    b::T2
end

@noinline function getfield_unstable_fn(x, idx)
    box = TypeUnstableGetfieldBox(1.0, x)
    val = getfield(box, idx)
    return val[1] * val[2]
end

@testset "Forward type-unstable getfield" begin
    idx = Base.inferencebarrier(2)
    res = Enzyme.autodiff(set_runtime_activity(Forward), getfield_unstable_fn, Duplicated([2.0, 3.0], [1.0, 0.0]), Const(idx))
    @test res[1] ≈ 3.0
end

struct MiniProblem3280{U, P, K}
    u0::U
    tspan::Tuple{Float64, Float64}
    p::P
    kwargs::K
end

function loss_3280(p)
    prob = MiniProblem3280([1.0], (0.0, 1.0), p, pairs((;)))
    active_field = Enzyme.Compiler.idx_jl_getfield_aug(
        Val(NamedTuple{(1,)}), prob, Val{2}, Val(false))
    return sum(abs2, active_field)
end

@testset "idx_jl_getfield_aug calling convention (Issue 3280 reproducer)" begin
    p = [0.5]
    dp = zero(p)
    Enzyme.autodiff(set_runtime_activity(Reverse), Enzyme.Const(loss_3280), Enzyme.Active,
        Enzyme.Duplicated(p, dp))
    @test dp ≈ [1.0]
end