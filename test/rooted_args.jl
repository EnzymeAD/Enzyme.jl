using Enzyme, Test

struct T{A, B, C}
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
    return nothing
end

function (o::Outer)(u)
    work!(u, o)
    return nothing
end

@testset "Nested Struct Ordering 2" begin
    cache = Outer(1, (rand(0),), Ref(zeros(2)))
    Enzyme.autodiff(Forward, Duplicated(cache, cache), Duplicated(zeros(2), zeros(2)))
end


struct MyCache
    M::Int
    kwargs::NamedTuple       # abstract NamedTuple — UnionAll, not DataType
    data::Vector{Float64}    # needed so closure is not ghost/constant
end

function (c::MyCache)(resid, u)
    resid[1] = u[1] * c.data[1]
    return nothing
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
