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
	@show Enzyme.autodiff(Reverse, toactivepair, Active(2.7f0), Active(3.1))

	x = Float32[2.7f0]
	dx = Float32[0.0f0]
	y = Float64[3.1]
	dy = Float64[0.0]

	Enzyme.autodiff(Reverse, toactivepair2, Duplicated(x, dx), Duplicated(y, dy))
	@show dx, dy

	x = Float32[2.7f0]
	dx = Float32[0.0f0]
	y = Float64[3.1]
	dy = Float64[0.0]

	Enzyme.autodiff(Reverse, tovectorpair, Duplicated(x, dx), Duplicated(y, dy))
	@show dx, dy



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

	@show dx, dy, dx2, dy2


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

	@show dx, dy, dx2, dy2
end

@testset "Forward Unstable newstructt" begin
	res = Enzyme.autodiff(Forward, toactivepair, Duplicated(2.7f0, 2.0), Duplicated(3.1, 3.0))
	@show res
	res = Enzyme.autodiff(Forward, toactivepair, BatchDuplicated(2.7f0, (2.0, 5.0)), BatchDuplicated(3.1, (3.0, 7.0)))
end