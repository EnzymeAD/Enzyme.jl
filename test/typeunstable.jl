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
  @test autodiff(Reverse, typeunstable_constant_shadow, Active)[1] == 0.0
end

