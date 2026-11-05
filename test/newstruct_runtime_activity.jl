using Enzyme
using Test

struct ImmutPairNS
    x::Float64
    y::Float64
end

struct MixedPairNS
    x::Float64
    v::Vector{Float64}
end

mutable struct MutPairNS
    x::Float64
    v::Vector{Float64}
end

@noinline ns_id_any(@nospecialize(x)) = Base.inferencebarrier(x)
@eval ns_construct_new(T, a, b) = $(Expr(:new, :T, :a, :b))
@eval ns_construct_splatnew(T, fields) = $(Expr(:splatnew, :T, :fields))

# Forces @jl_new_structv via non-inferred Type
function make_newstructv_immut(T_ref::Base.RefValue{Any}, x::Float64, c::Float64)
    T = T_ref[]
    s = ns_construct_new(T, x, c)::ImmutPairNS
    return 2.0 * s.x + 3.0 * s.y
end

function make_newstructv_mixed(T_ref::Base.RefValue{Any}, x::Float64, v::Vector{Float64})
    T = T_ref[]
    s = ns_construct_new(T, x, v)::MixedPairNS
    return 2.0 * s.x + 5.0 * s.v[1]
end

function make_newstructv_mut!(T_ref::Base.RefValue{Any}, x::Float64, v::Vector{Float64})
    T = T_ref[]
    s = ns_construct_new(T, x, v)::MutPairNS
    s.x *= 4.0
    return s.x + 3.0 * s.v[1]
end

# Forces @jl_f_tuple via non-inferred element
function make_f_tuple_mixed(x::Float64, v::Vector{Float64}, active_x::Bool)
    ax = active_x ? ns_id_any(x) : ns_id_any(7.0)
    av = ns_id_any(v)
    t = (ax, av)
    return 2.0 * (t[1]::Float64) + 3.0 * (t[2]::Vector{Float64})[1]
end

# Forces @jl_new_structt via Expr(:splatnew)
function make_newstructt_immut(T_ref::Base.RefValue{Any}, x::Float64, c::Float64, active_tup::Bool)
    T = T_ref[]
    tup = active_tup ? ns_id_any((x, c)) : ns_id_any((10.0, c))
    s = ns_construct_splatnew(T, tup)::ImmutPairNS
    return 2.0 * s.x + 3.0 * s.y
end

function make_newstructt_mixed(T_ref::Base.RefValue{Any}, x::Float64, v::Vector{Float64})
    T = T_ref[]
    tup = ns_id_any((x, v))
    s = ns_construct_splatnew(T, tup)::MixedPairNS
    return 2.0 * s.x + 5.0 * s.v[1]
end

@testset "newstructv, f_tuple, and newstructt Forward & runtime activity" begin
    @testset "newstructv Forward & Reverse (immut, mixed, mut)" begin
        tref_immut = Ref{Any}(ImmutPairNS)
        tref_mixed = Ref{Any}(MixedPairNS)
        tref_mut = Ref{Any}(MutPairNS)

        # Immutable struct with Const second field
        @test autodiff(
            set_runtime_activity(Forward),
            make_newstructv_immut,
            Const(tref_immut),
            Duplicated(3.0, 1.0),
            Const(4.0),
        )[1] ≈ 2.0

        res_b = autodiff(
            set_runtime_activity(Forward),
            make_newstructv_immut,
            Const(tref_immut),
            BatchDuplicated(3.0, (1.0, 5.0)),
            Const(4.0),
        )[1]
        @test res_b[1] ≈ 2.0
        @test res_b[2] ≈ 10.0

        res_rev = autodiff(
            set_runtime_activity(Reverse),
            make_newstructv_immut,
            Active,
            Const(tref_immut),
            Active(3.0),
            Const(4.0),
        )
        @test res_rev[1][2] ≈ 2.0

        # MixedState struct with runtime-inactive vector Duplicated(v, v)
        v = [6.0]
        @test autodiff(
            set_runtime_activity(Forward),
            make_newstructv_mixed,
            Const(tref_mixed),
            Duplicated(3.0, 2.0),
            Duplicated(v, v),
        )[1] ≈ 4.0
        @test v == [6.0]

        res_mix_rev = autodiff(
            set_runtime_activity(Reverse),
            make_newstructv_mixed,
            Active,
            Const(tref_mixed),
            Active(3.0),
            Duplicated(v, v),
        )
        @test res_mix_rev[1][2] ≈ 2.0
        @test v == [6.0]

        # Mutable struct (DupState overall, even when containing Float64 + Vector)
        vm = [6.0]
        dvm = [0.0]
        res_mut_fwd = autodiff(
            set_runtime_activity(Forward),
            make_newstructv_mut!,
            Const(tref_mut),
            Duplicated(3.0, 2.0),
            Duplicated(vm, vm),
        )
        @test res_mut_fwd[1] ≈ 8.0
        @test vm == [6.0]

        res_mut_rev = autodiff(
            set_runtime_activity(Reverse),
            make_newstructv_mut!,
            Active,
            Const(tref_mut),
            Active(3.0),
            Duplicated(vm, dvm),
        )
        @test res_mut_rev[1][2] ≈ 4.0
        @test dvm[1] ≈ 3.0
    end

    @testset "f_tuple Forward & Reverse runtime activity" begin
        v = [4.0]
        dv = [2.0]
        # Active x + runtime-inactive v
        @test autodiff(
            set_runtime_activity(Forward),
            make_f_tuple_mixed,
            Duplicated(3.0, 5.0),
            Duplicated(v, v),
            Const(true),
        )[1] ≈ 10.0
        @test v == [4.0]

        # Runtime-inactive x + runtime-inactive v (all-inactive tuple!)
        @test autodiff(
            set_runtime_activity(Forward),
            make_f_tuple_mixed,
            Duplicated(3.0, 5.0),
            Duplicated(v, v),
            Const(false),
        )[1] ≈ 0.0
        @test v == [4.0]

        # Batched Forward width 2
        res_b = autodiff(
            set_runtime_activity(Forward),
            make_f_tuple_mixed,
            BatchDuplicated(3.0, (5.0, 1.0)),
            BatchDuplicated(v, (dv, v)),
            Const(true),
        )[1]
        @test res_b[1] ≈ 16.0
        @test res_b[2] ≈ 2.0

        # Reverse active vs all-inactive
        dv_r = [0.0]
        res_r1 = autodiff(
            set_runtime_activity(Reverse),
            make_f_tuple_mixed,
            Active,
            Active(3.0),
            Duplicated(v, dv_r),
            Const(true),
        )
        @test res_r1[1][1] ≈ 2.0
        @test dv_r[1] ≈ 3.0

        res_r2 = autodiff(
            set_runtime_activity(Reverse),
            make_f_tuple_mixed,
            Active,
            Active(3.0),
            Duplicated(v, v),
            Const(false),
        )
        @test res_r2[1][1] ≈ 0.0
        @test v == [4.0]
    end

    @testset "newstructt Forward, Reverse, and FoR runtime activity" begin
        tref_immut = Ref{Any}(ImmutPairNS)
        tref_mixed = Ref{Any}(MixedPairNS)

        # Forward width 1 & width 2 (active tuple vs runtime-inactive tuple)
        @test autodiff(
            set_runtime_activity(Forward),
            make_newstructt_immut,
            Const(tref_immut),
            Duplicated(3.0, 4.0),
            Const(5.0),
            Const(true),
        )[1] ≈ 8.0

        @test autodiff(
            set_runtime_activity(Forward),
            make_newstructt_immut,
            Const(tref_immut),
            Duplicated(3.0, 4.0),
            Const(5.0),
            Const(false),
        )[1] ≈ 0.0

        res_b = autodiff(
            set_runtime_activity(Forward),
            make_newstructt_immut,
            Const(tref_immut),
            BatchDuplicated(3.0, (4.0, 7.0)),
            Const(5.0),
            Const(true),
        )[1]
        @test res_b[1] ≈ 8.0
        @test res_b[2] ≈ 14.0

        # Reverse width 1 & width 2 (active tuple vs runtime-inactive tuple)
        res_r_act = autodiff(
            set_runtime_activity(Reverse),
            make_newstructt_immut,
            Active,
            Const(tref_immut),
            Active(3.0),
            Const(5.0),
            Const(true),
        )
        @test res_r_act[1][2] ≈ 2.0

        res_r_inact = autodiff(
            set_runtime_activity(Reverse),
            make_newstructt_immut,
            Active,
            Const(tref_immut),
            Active(3.0),
            Const(5.0),
            Const(false),
        )
        @test res_r_inact[1][2] ≈ 0.0

        # MixedState newstructt with runtime-inactive vector in Forward & Reverse
        v = [6.0]
        dv = [0.0]
        @test autodiff(
            set_runtime_activity(Forward),
            make_newstructt_mixed,
            Const(tref_mixed),
            Duplicated(3.0, 2.0),
            Duplicated(v, v),
        )[1] ≈ 4.0
        @test v == [6.0]

        res_mix_r = autodiff(
            set_runtime_activity(Reverse),
            make_newstructt_mixed,
            Active,
            Const(tref_mixed),
            Active(3.0),
            Duplicated(v, dv),
        )
        @test res_mix_r[1][2] ≈ 2.0
        @test dv[1] ≈ 5.0

        # Forward-over-Reverse through newstructt!
        function loss_newstructt_sq(tref::Base.RefValue{Any}, x::Float64, c::Float64)
            T = tref[]
            tup = ns_id_any((x * x, c))
            s = ns_construct_splatnew(T, tup)::ImmutPairNS
            return 3.0 * s.x + s.y
        end
        function grad_loss_newstructt(tref::Base.RefValue{Any}, x::Float64, c::Float64)
            return autodiff(
                set_runtime_activity(Reverse),
                loss_newstructt_sq,
                Active,
                Const(tref),
                Active(x),
                Const(c),
            )[1][2]
        end
        # loss = 3*x^2 + c => d/dx = 6*x => d^2/dx^2 = 6.0
        hvp = autodiff(
            set_runtime_activity(Forward),
            grad_loss_newstructt,
            Const(tref_immut),
            Duplicated(4.0, 1.0),
            Const(5.0),
        )[1]
        @test hvp ≈ 6.0
    end
end
