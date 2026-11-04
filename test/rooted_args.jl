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

# A phi joins the by-reference argument with a GC object copied from it, which
# nodecayed_phis! cannot yet root (SciMLSensitivity.jl#1696). Reaching the phi's
# block throws at runtime instead, so paths that skip it still work.
# Once such phis are supported, the throwing case needs another trigger.
struct UnrootedPhiArg
    x::Float64
    flag::Bool
end

@noinline function unrooted_phi_sum(s::UnrootedPhiArg, n::Int)
    acc = 0.0
    for i in 1:n
        t = s.flag ? deepcopy(s) : s
        acc += t.x
    end
    return acc + 2 * s.x
end

unrooted_phi_loss(x, n) = unrooted_phi_sum(UnrootedPhiArg(x, true), n)

@testset "Unrootable phi becomes a runtime error" begin
    @test autodiff(Reverse, unrooted_phi_loss, Active, Active(2.0), Const(0))[1][1] == 2.0
    @test autodiff(Forward, unrooted_phi_loss, Duplicated(2.0, 1.0), Const(0))[1] == 2.0
    @test_throws Enzyme.Compiler.EnzymeRuntimeExceptionMI autodiff(Reverse, unrooted_phi_loss, Active, Active(2.0), Const(1))
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3924
# Julia 1.10's `sret_union` calling convention uses a stack `[N x i8]` alloca and
# selects between `addrspacecast` of that alloca and a boxed `addrspace(10)` union
# return, feeding an `addrspace(11)` phi in `_foldl_impl`. `nodecayed_phis!` must
# snapshot the read-only stack alloca into a real GC object rather than casting
# the raw stack pointer to `addrspace(10)` (which corrupts GC marking when cached
# onto a split-mode tape).
struct SretUnionElem3924
    val::Int
end
struct SretUnionIter3924
    n::Int
end
@noinline Base.iterate(it::SretUnionIter3924, state::Int = 1)::Union{Nothing, Tuple{SretUnionElem3924, Int}} =
    state <= it.n ? (SretUnionElem3924(state), state + 1) : nothing

@noinline sret_map_elem_3924(c::Float64, b::SretUnionElem3924) = c^b.val
@noinline sret_inner_sum_3924(it::SretUnionIter3924, c::Float64) = sum(sret_map_elem_3924(c, b) for b in it)

@noinline function sret_mul_into_gc_3924!(x::Vector{Float64}, s::Float64)
    x[1] *= s
    GC.gc(true)
    return nothing
end

function sret_outer_3924!(x::Vector{Float64}, it::SretUnionIter3924, c::Float64)
    s = sret_inner_sum_3924(it, c)
    sret_mul_into_gc_3924!(x, s)
    return nothing
end

struct SretUnionActiveElem3924
    w::Float64
    idx::Int
end
struct SretUnionActiveIter3924
    scale::Float64
    n::Int
end
@noinline Base.iterate(it::SretUnionActiveIter3924, state::Int = 1)::Union{Nothing, Tuple{SretUnionActiveElem3924, Int}} =
    state <= it.n ? (SretUnionActiveElem3924(it.scale * state^2, state), state + 1) : nothing

@noinline sret_use_active_3924(c::Float64, elem::SretUnionActiveElem3924) = (c^2) * (elem.w^3)
@noinline sret_active_sum_3924(scale::Float64, c::Float64, n::Int) =
    sum(sret_use_active_3924(c, e) for e in SretUnionActiveIter3924(scale, n))

function sret_outer_active_3924!(out::Vector{Float64}, scale::Float64, c::Float64, n::Int)
    s = sret_active_sum_3924(scale, c, n)
    sret_mul_into_gc_3924!(out, s)
    return nothing
end

@testset "sret_union stack alloca feeding addrspace(11) phi (#3924)" begin
    x = [3.0]
    dx = [1.0]
    c = 2.0
    res = autodiff(Reverse, sret_outer_3924!, Const, Duplicated(x, dx), Const(SretUnionIter3924(8)), Active(c))
    @test x[1] ≈ 3.0 * sum(c^k for k in 1:8)
    @test res[1][3] ≈ 3.0 * sum(k * c^(k - 1) for k in 1:8)

    out = [2.0]
    dout = [1.0]
    sc = 1.5
    c2 = 2.5
    res2 = autodiff(Reverse, sret_outer_active_3924!, Const, Duplicated(out, dout), Active(sc), Active(c2), Const(10))
    @test out[1] ≈ 2.0 * sum((c2^2) * ((sc * k^2)^3) for k in 1:10)
    @test res2[1][2] ≈ 2.0 * sum((c2^2) * 3 * (sc * k^2)^2 * k^2 for k in 1:10)
    @test res2[1][3] ≈ 2.0 * sum(2 * c2 * ((sc * k^2)^3) for k in 1:10)
end
