using Enzyme, Test
using StaticArrays

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


# On Julia 1.13 (LLVM 20) a loop over an SVector argument is strength-reduced into a
# pointer induction variable over the addrspace(11) argument; `nodecayed_phis!` used
# to fail on the resulting phi ("Could not analyze garbage collection behavior").
# Reduced from OrdinaryDiffEqCore.initialize_saveat (SciML integration test).
function saveat_times(saveat, tspan)
    out = Float64[]
    t0, tf = tspan
    tdir = sign(tf - t0)
    tdir_t0 = tdir * t0
    tdir_tf = tdir * tf
    for t in saveat
        tdir_t = tdir * t
        tdir_t0 < tdir_t ≤ tdir_tf && push!(out, tdir_t)
    end
    return out
end
saveat_count(x, saveat, tspan) = sum(x) * length(saveat_times(saveat, tspan))

# Same failure iterating a `zip` of NTuples (an addrspace(11) `Iterators.Zip`
# argument of `_foldl_impl`). Reduced from Molly's periodic torsion force.
struct Force4{T}
    fi::SVector{3, T}
    fj::SVector{3, T}
    fk::SVector{3, T}
    fl::SVector{3, T}
end
Base.:+(a::Force4, b::Force4) = Force4(a.fi + b.fi, a.fj + b.fj, a.fk + b.fk, a.fl + b.fl)
struct TorsionForce{V}
    ab::V
    bc::V
end
function (t::TorsionForce)((periodicity, phase, k))
    d = -k * periodicity * sin(periodicity * phase)
    fi = d * t.ab
    fl = -d * t.bc
    v = (d / 2) * fi - (d / 3) * fl
    return Force4(fi, v - fi, -v - fl, fl)
end
function torsion_sum(ab, bc, periodicities::NTuple{6, Int}, phases::NTuple{6, Float64}, ks::NTuple{6, Float64})
    return sum(TorsionForce(ab, bc), zip(periodicities, phases, ks))
end
function torsion_total(x, periodicities, phases, ks)
    fs = torsion_sum(SVector(x[1], x[2], x[3]), SVector(x[3], x[2], x[1]), periodicities, phases, ks)
    return sum(fs.fi) + 2 * sum(fs.fj) + 3 * sum(fs.fk) + 4 * sum(fs.fl)
end

function central_difference_gradient(x, periodicities, phases, ks)
    h = 1.0e-6
    g = similar(x)
    for i in eachindex(x)
        xp = copy(x)
        xm = copy(x)
        xp[i] += h
        xm[i] -= h
        g[i] = (torsion_total(xp, periodicities, phases, ks) - torsion_total(xm, periodicities, phases, ks)) / (2h)
    end
    return g
end

@testset "Pointer phis over addrspace(11) arguments" begin
    saveat = SA[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    tspan = (0.0, 3.0)
    x = [1.0, 2.0, 3.0]
    dx = zeros(3)
    autodiff(Reverse, saveat_count, Active, Duplicated(x, dx), Const(saveat), Const(tspan))
    @test dx ≈ fill(Float64(length(saveat_times(saveat, tspan))), 3)

    periodicities = ntuple(identity, 6)
    phases = ntuple(i -> i / 2, 6)
    ks = ntuple(i -> i / 3, 6)
    x = [1.0, 2.0, 3.0]
    dx = zeros(3)
    autodiff(Reverse, torsion_total, Active, Duplicated(x, dx), Const(periodicities), Const(phases), Const(ks))
    fd = central_difference_gradient(x, periodicities, phases, ks)
    @test dx ≈ fd rtol = 1.0e-6
end
