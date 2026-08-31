using Enzyme, Test

struct T{A,B,C}
	eval_sol::A
	a::B
	stage::C
end

function (c::T)()
	@inbounds c.eval_sol[1][1][1] = 2.1
        return nothing
end
@testset "Nested Struct Ordering" begin
	stage = 1
	a = zeros(2)
	eval_sol = ([zeros(2)],)

	loss! = T(eval_sol, a, stage)

	Enzyme.autodiff(Forward, Duplicated(loss!, deepcopy(loss!)))
end

struct Outer{B}
    M::Int
    inner::Tuple{Vector{Float64}}
    y::B
end

function work!(u, cache)
    y_ = [cache.y[] for _ in 1:cache.M]
    copyto!(y_[1], u)
    nothing
end

function (o::Outer)(u)
    work!(u, o)
    nothing
end

@testset "Nested Struct Ordering 2" begin
    cache = Outer(1, (rand(0),), Ref(zeros(2)))
    Enzyme.autodiff(Forward, Duplicated(cache, cache) , Duplicated(zeros(2), zeros(2)))
end


struct MyCache
    M::Int
    kwargs::NamedTuple       # abstract NamedTuple — UnionAll, not DataType
    data::Vector{Float64}    # needed so closure is not ghost/constant
end

function (c::MyCache)(resid, u)
    resid[1] = u[1] * c.data[1]
    nothing
end

@testset "Abstract struct arg" begin
	nt = (a = 1,)
	cache = MyCache(2, nt, [1.0, 2.0])

	Enzyme.autodiff(
	    Enzyme.Forward,
	    Enzyme.Duplicated(cache, cache),
	    Enzyme.Duplicated(zeros(1), zeros(1)),
	    Enzyme.Duplicated(zeros(2), zeros(2))
	)
end

# On Julia 1.12+, a by-value read of a large inline immutable field containing GC pointers
# out of a heap object (`f = prob.f`) is lowered by loading the pointer words into the
# separate inline-roots array and poisoning the corresponding words of the value's own stack
# slot with 0xff (`llvm.memset ... i8 -1` / `store i64 -1`). That slot carries no type
# information of its own, so type analysis is left with holes exactly at the pointer fields
# unless Enzyme recovers the slot's Julia type from the object it was copied out of.
# xref https://github.com/EnzymeAD/Enzyme.jl/issues/3433
struct PoisonSlotInner
    p1::Vector{Float64}
    p2::Vector{Float64}
end

struct PoisonSlotFn
    n::Int
    o1::PoisonSlotInner
    o2::PoisonSlotInner
    o3::PoisonSlotInner
    o4::PoisonSlotInner
    m::Int
end

mutable struct PoisonSlotProb
    f::PoisonSlotFn
    u0::Vector{Float64}
end

# Recursive so that it survives LLVM inlining and keeps an interior pointer of the slot
# escaping, which is what stops the poisoning writes from being eliminated as dead.
@noinline function poison_slot_inner_dot(a::PoisonSlotInner, x::Vector{Float64}, k::Int)
    s = 0.0
    for i in eachindex(x)
        s += a.p1[i] * x[i] + a.p2[i] * x[i]
    end
    if k > 0
        s += poison_slot_inner_dot(a, x, k - 1)
    end
    return s
end

@noinline function poison_slot_use(f::PoisonSlotFn, x::Vector{Float64})
    return poison_slot_inner_dot(f.o1, x, 1) + f.n * x[1] + f.m * x[2]
end

function poison_slot_loss(prob::PoisonSlotProb, x::Vector{Float64})
    f = prob.f
    return poison_slot_use(f, x)
end

@testset "By-value inline struct read out of heap object" begin
    inner = PoisonSlotInner([1.0, 2.0], [3.0, 4.0])
    prob = PoisonSlotProb(
        PoisonSlotFn(3, inner, inner, inner, inner, 5),
        [0.0, 0.0],
    )
    x = [1.0, 2.0]
    dx = zero(x)
    autodiff(
        set_runtime_activity(Reverse),
        poison_slot_loss,
        Active,
        Const(prob),
        Duplicated(x, dx),
    )
    # d/dx of 2 * sum(p1 .* x .+ p2 .* x) + n * x[1] + m * x[2]
    @test dx ≈ [2 * (1.0 + 3.0) + 3, 2 * (2.0 + 4.0) + 5]
end
