using Enzyme, Test

# Julia >= 1.12 emits the stack slot of an immutable struct (here the closure built in
# the loop) as an untyped `[N x i64]` alloca. Copying a struct with padding into it
# (the `Bool` of `ClosureFlag` is followed by 7 padding bytes) left Enzyme unable to
# type those words (EnzymeAD/Enzyme.jl#3757).
struct ClosureFlag
    static::Bool
end
struct ClosureWrapped{A}
    flag::ClosureFlag
    data::A
end

@noinline function closure_step!(w)
    d = w.data
    @inbounds for j in eachindex(d)
        d[j] = sin(d[j])
    end
    return nothing
end
@noinline closure_callit(f) = (f(); nothing)

function closure_cost(w, n)
    J = 0.0
    for _ in 1:n
        closure_callit(() -> closure_step!(w))
        J += sum(w.data)
    end
    return J
end

function closure_cost_ref(w, n)
    J = 0.0
    for _ in 1:n
        closure_step!(w)
        J += sum(w.data)
    end
    return J
end

@testset "Closure capturing a struct with padding, built in a loop" begin
    make() = ClosureWrapped(ClosureFlag(true), view(collect(1.0:4.0), 1:4))
    w = make()
    dw = make_zero(w)
    autodiff(set_runtime_activity(Reverse), closure_cost, Active, Duplicated(w, dw), Const(2))
    wref = make()
    dwref = make_zero(wref)
    autodiff(set_runtime_activity(Reverse), closure_cost_ref, Active, Duplicated(wref, dwref), Const(2))
    @test any(!iszero, dw.data)
    @test dw.data ≈ dwref.data
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3926
# `sret_union` calls allocate an untyped `[N x i8]` stack buffer marked with
# `"enzymejl_sret_union_bytes"`. When the payload struct contains padding bytes,
# Enzyme needs type/inactivity metadata on that alloca so static activity and
# `memcpy` type analysis succeed without requiring `set_runtime_activity`.
struct PaddedSretElem
    val::UInt8
end
struct PaddedSretIter
    n::Int
end
@noinline Base.iterate(it::PaddedSretIter, state::Int = 1)::Union{Nothing, Tuple{PaddedSretElem, Int}} =
    state <= it.n ? (PaddedSretElem(UInt8(state)), state + 1) : nothing

@noinline padded_sret_map(c::Float64, b::PaddedSretElem) = c^Int(b.val)

function padded_sret_loop(n::Int, c::Float64)
    s = 0.0
    for k in 1:n
        u = Base.iterate(PaddedSretIter(n), k)
        u === nothing && break
        s += padded_sret_map(c, u[1])
    end
    return s
end

struct PaddedSretAlt
    a::UInt8
    b::UInt16
end
@noinline function multi_sret_step(k::Int)::Union{Nothing, Tuple{PaddedSretElem, Int}, Tuple{PaddedSretAlt, Int}}
    if k == 1
        return (PaddedSretElem(0x02), 2)
    elseif k == 2
        return (PaddedSretAlt(0x01, 0x0003), 3)
    else
        return nothing
    end
end

function multi_sret_cost(c::Float64)
    s = 0.0
    for k in 1:3
        u = multi_sret_step(k)
        if u isa Tuple{PaddedSretElem, Int}
            s += padded_sret_map(c, u[1])
        elseif u isa Tuple{PaddedSretAlt, Int}
            s += c^Int(u[1].b)
        end
    end
    return s
end

struct PaddedActiveSretElem
    w::Float64
    tag::UInt8
end
struct PaddedActiveSretIter
    scale::Float64
    n::Int
end
@noinline Base.iterate(it::PaddedActiveSretIter, state::Int = 1)::Union{Nothing, Tuple{PaddedActiveSretElem, Int}} =
    state <= it.n ? (PaddedActiveSretElem(it.scale * state^2, UInt8(state)), state + 1) : nothing

@inline use_padded_active_sret(c::Float64, elem::PaddedActiveSretElem) = (c^2) * (elem.w^3)

function padded_active_sret_loop(scale::Float64, c::Float64, n::Int)
    s = 0.0
    for k in 1:n
        u = Base.iterate(PaddedActiveSretIter(scale, n), k)
        u === nothing && break
        s += use_padded_active_sret(c, u[1])
    end
    return s
end

padded_sret_foldl(it::PaddedSretIter, c::Float64) = sum(padded_sret_map(c, b) for b in it)
padded_active_sret_foldl(scale::Float64, c::Float64, n::Int) =
    sum(use_padded_active_sret(c, e) for e in PaddedActiveSretIter(scale, n))

@testset "sret_union stack alloca with padding (#3926)" begin
    c = 2.0
    res_loop = autodiff(Reverse, padded_sret_loop, Active, Const(4), Active(c))
    @test res_loop[1][2] ≈ sum(k * c^(k - 1) for k in 1:4)

    res_multi = autodiff(Reverse, multi_sret_cost, Active, Active(c))
    @test res_multi[1][1] ≈ 2 * c + 3 * c^2

    sc = 1.5
    c2 = 2.5
    res_act = autodiff(Reverse, padded_active_sret_loop, Active, Active(sc), Active(c2), Const(6))
    @test res_act[1][1] ≈ sum((c2^2) * 3 * (sc * k^2)^2 * k^2 for k in 1:6)
    @test res_act[1][2] ≈ sum(2 * c2 * ((sc * k^2)^3) for k in 1:6)

    @static if VERSION >= v"1.11-"
        res_foldl = autodiff(Reverse, padded_sret_foldl, Active, Const(PaddedSretIter(4)), Active(c))
        @test res_foldl[1][2] ≈ sum(k * c^(k - 1) for k in 1:4)

        res_act_foldl = autodiff(Reverse, padded_active_sret_foldl, Active, Active(sc), Active(c2), Const(6))
        @test res_act_foldl[1][1] ≈ sum((c2^2) * 3 * (sc * k^2)^2 * k^2 for k in 1:6)
        @test res_act_foldl[1][2] ≈ sum(2 * c2 * ((sc * k^2)^3) for k in 1:6)
    end
end
