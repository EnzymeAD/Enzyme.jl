using Enzyme
using Test

mutable struct MutGetfieldBox
    a::Any
    b::Any
end

@noinline id_any(@nospecialize(x)) = Base.inferencebarrier(x)

function dyn_getfield_f64(s::MutGetfieldBox, sym::Symbol, x::Float64)
    v = getfield(id_any(s), sym)::Float64
    return v * x
end

function dyn_getfield_vec(s::MutGetfieldBox, sym::Symbol, x::Float64)
    v = getfield(id_any(s), sym)::Vector{Float64}
    return sum(v) * x
end

function dyn_nthfield_vec(s::MutGetfieldBox, idx::Int, x::Float64)
    v = getfield(id_any(s), idx)::Vector{Float64}
    return sum(v) * x
end

function dyn_nthfield_f64(s::MutGetfieldBox, idx::Int, x::Float64)
    v = getfield(id_any(s), idx)::Float64
    return v * x
end

@noinline _make_het_tup1(x::Vector{Float64}) = (2.0 .* x, Base.inferencebarrier(:a))
@noinline _make_het_tup2(x::Vector{Float64}) = (3.0 .* x, Base.inferencebarrier("b"))

function phi_nthfield_rtact(x::Vector{Float64}, flag::Bool, idx::Int)
    tup = flag ? _make_het_tup1(x) : _make_het_tup2(x)
    return sum(getfield(tup, idx)::Vector{Float64})
end

function const_box_getfield_rtact(c::Ref{Any}, sym::Symbol, x::Float64)
    s = c[]
    v = getfield(s, sym)::Vector{Float64}
    return sum(v) * x
end

function const_box_nthfield_rtact(c::Ref{Any}, idx::Int, x::Float64)
    t = c[]
    v = getfield(t, idx)::Vector{Float64}
    return sum(v) * x
end

@testset "Dynamic getfield and nthfield runtime activity" begin
    @testset "Symbolic getfield on mutable struct (active, runtime-inactive, and mixed batch)" begin
        s = MutGetfieldBox(3.0, [2.0, 4.0])

        # Runtime-inactive Duplicated(s, s): primal must NOT be mutated in Reverse
        res = autodiff(
            set_runtime_activity(Reverse),
            dyn_getfield_f64,
            Active,
            Duplicated(s, s),
            Const(:a),
            Active(5.0),
        )
        @test res[1][3] ≈ 3.0
        @test s.a == 3.0

        res_vec = autodiff(
            set_runtime_activity(Reverse),
            dyn_getfield_vec,
            Active,
            Duplicated(s, s),
            Const(:b),
            Active(5.0),
        )
        @test res_vec[1][3] ≈ 6.0
        @test s.b == [2.0, 4.0]

        # Batched Reverse on scalar field (exercises batched mutable setfield! in rt_jl_getfield_rev)
        ds1 = MutGetfieldBox(0.0, [0.0, 0.0])
        ds2 = MutGetfieldBox(0.0, [0.0, 0.0])
        autodiff(
            set_runtime_activity(Reverse),
            dyn_getfield_f64,
            Active,
            BatchDuplicated(s, (ds1, ds2)),
            Const(:a),
            Active(5.0),
        )
        @test ds1.a ≈ 5.0
        @test ds2.a ≈ 5.0

        # Batched Reverse when all lanes are runtime-inactive (s, s)
        autodiff(
            set_runtime_activity(Reverse),
            dyn_getfield_f64,
            Active,
            BatchDuplicated(s, (s, s)),
            Const(:a),
            Active(5.0),
        )
        @test s.a == 3.0

        autodiff(
            set_runtime_activity(Reverse),
            dyn_getfield_vec,
            Active,
            BatchDuplicated(s, (s, s)),
            Const(:b),
            Active(5.0),
        )
        @test s.b == [2.0, 4.0]

        # Batched Reverse with mixed active lane 1 and runtime-inactive lane 2
        ds1_mix = MutGetfieldBox(0.0, [0.0, 0.0])
        autodiff(
            set_runtime_activity(Reverse),
            dyn_getfield_f64,
            Active,
            BatchDuplicated(s, (ds1_mix, s)),
            Const(:a),
            Active(5.0),
        )
        @test s.a == 3.0
        @test ds1_mix.a ≈ 5.0

        # Forward mode with runtime-inactive and mixed batch lanes
        @test autodiff(
            set_runtime_activity(Forward),
            dyn_getfield_vec,
            Duplicated(s, s),
            Const(:b),
            Duplicated(5.0, 1.0),
        ) == (6.0,)

        ds_fwd = MutGetfieldBox(1.5, [1.0, 2.0])
        @test autodiff(
            set_runtime_activity(Forward),
            dyn_getfield_vec,
            BatchDuplicated(s, (ds_fwd, s)),
            Const(:b),
            BatchDuplicated(5.0, (0.0, 1.0)),
        ) == ((var"1" = 15.0, var"2" = 6.0),)
    end

    @testset "Indexed nthfield on runtime-inactive and mixed-batch container" begin
        s = MutGetfieldBox(3.0, [2.0, 4.0])

        @test autodiff(
            set_runtime_activity(Forward),
            dyn_nthfield_vec,
            Duplicated(s, s),
            Const(2),
            Duplicated(5.0, 1.0),
        ) == (6.0,)

        res = autodiff(
            set_runtime_activity(Reverse),
            dyn_nthfield_f64,
            Active,
            Duplicated(s, s),
            Const(1),
            Active(5.0),
        )
        @test res[1][3] ≈ 3.0
        @test s.a == 3.0

        autodiff(
            set_runtime_activity(Reverse),
            dyn_nthfield_f64,
            Active,
            BatchDuplicated(s, (s, s)),
            Const(1),
            Active(5.0),
        )
        @test s.a == 3.0

        ds1 = MutGetfieldBox(0.0, [0.0, 0.0])
        autodiff(
            set_runtime_activity(Reverse),
            dyn_nthfield_f64,
            Active,
            BatchDuplicated(s, (ds1, s)),
            Const(1),
            Active(5.0),
        )
        @test s.a == 3.0
        @test ds1.a ≈ 5.0
    end

    @testset "Const container under runtime activity (width 1 and 2)" begin
        c_sym = Ref{Any}(MutGetfieldBox(3.0, [2.0, 4.0]))
        @test autodiff(
            set_runtime_activity(Forward),
            const_box_getfield_rtact,
            Const(c_sym),
            Const(:b),
            Duplicated(5.0, 1.0),
        ) == (6.0,)
        @test autodiff(
            set_runtime_activity(Forward),
            const_box_getfield_rtact,
            Const(c_sym),
            Const(:b),
            BatchDuplicated(5.0, (1.0, 2.0)),
        ) == ((var"1" = 6.0, var"2" = 12.0),)
        @test autodiff(
            set_runtime_activity(Reverse),
            const_box_getfield_rtact,
            Active,
            Const(c_sym),
            Const(:b),
            Active(5.0),
        )[1][3] ≈ 6.0

        c_idx = Ref{Any}(([2.0, 4.0], :tag))
        @test autodiff(
            set_runtime_activity(Forward),
            const_box_nthfield_rtact,
            Const(c_idx),
            Const(1),
            Duplicated(5.0, 1.0),
        ) == (6.0,)
        @test autodiff(
            set_runtime_activity(Forward),
            const_box_nthfield_rtact,
            Const(c_idx),
            Const(1),
            BatchDuplicated(5.0, (1.0, 2.0)),
        ) == ((var"1" = 6.0, var"2" = 12.0),)
        @test autodiff(
            set_runtime_activity(Reverse),
            const_box_nthfield_rtact,
            Active,
            Const(c_idx),
            Const(1),
            Active(5.0),
        )[1][3] ≈ 6.0
    end

    @testset "PHI-node tuple nthfield under runtime activity" begin
        dx = zeros(2)
        autodiff(
            set_runtime_activity(Reverse),
            phi_nthfield_rtact,
            Active,
            Duplicated([1.5, -0.5], dx),
            Const(true),
            Const(1),
        )
        @test dx ≈ [2.0, 2.0]

        dx1 = zeros(2)
        dx2 = zeros(2)
        autodiff(
            set_runtime_activity(Reverse),
            phi_nthfield_rtact,
            Active,
            BatchDuplicated([1.5, -0.5], (dx1, dx2)),
            Const(false),
            Const(1),
        )
        @test dx1 ≈ [3.0, 3.0]
        @test dx2 ≈ [3.0, 3.0]
    end
end
