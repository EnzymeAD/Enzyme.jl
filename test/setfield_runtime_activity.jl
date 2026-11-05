using Enzyme
using Test

mutable struct MutSetfieldAny
    a::Any
    b::Any
end

mutable struct MutSetfieldFloat
    x::Float64
end

@noinline setfield_id_any(@nospecialize(x)) = Base.inferencebarrier(x)

function setfield_to_box!(s, sym::Symbol, x::Float64)
    any_s = setfield_id_any(s)
    Core.setfield!(any_s, sym, x)
    return nothing
end

function setfield_copy_any!(dst, src, sym::Symbol)
    any_dst = setfield_id_any(dst)
    any_src = setfield_id_any(src)
    v = Core.getfield(any_src, sym)
    Core.setfield!(any_dst, sym, v)
    return nothing
end

function setfield_used_return_const!(s, sym::Symbol, c::Float64, x::Float64)
    any_s = setfield_id_any(s)
    r = Core.setfield!(any_s, sym, c)::Float64
    return r * x
end

function setfield_overwrite_then_read!(s, sym::Symbol, x::Float64, c::Float64)
    any_s = setfield_id_any(s)
    Core.setfield!(any_s, sym, 2.0 * x)
    # Overwrite with a constant value: adjoint after this must not flow to x
    Core.setfield!(any_s, sym, c)
    return Core.getfield(any_s, sym)::Float64
end

@testset "setfield! runtime activity" begin
    @testset "Forward runtime-inactive struct is not clobbered by active value" begin
        s = MutSetfieldAny(3.0, [1.0, 2.0])
        autodiff(
            set_runtime_activity(Forward),
            setfield_to_box!,
            Duplicated(s, s),
            Const(:a),
            Duplicated(10.0, 7.0),
        )
        # Primal must hold primal stored value 10.0, NOT tangent 7.0
        @test s.a == 10.0

        # Active struct receives active tangent (including when val == dval == 5.0)
        s2 = MutSetfieldAny(3.0, nothing)
        ds2 = MutSetfieldAny(99.0, nothing)
        autodiff(
            set_runtime_activity(Forward),
            setfield_to_box!,
            Duplicated(s2, ds2),
            Const(:a),
            Duplicated(5.0, 5.0),
        )
        @test s2.a == 5.0
        @test ds2.a == 5.0

        # Batched Forward mixed active/runtime-inactive struct lanes
        sb = MutSetfieldAny(3.0, nothing)
        dsb1 = MutSetfieldAny(99.0, nothing)
        autodiff(
            set_runtime_activity(Forward),
            setfield_to_box!,
            BatchDuplicated(sb, (dsb1, sb)),
            Const(:a),
            BatchDuplicated(10.0, (4.0, 8.0)),
        )
        @test sb.a == 10.0
        @test dsb1.a == 4.0
    end

    @testset "Forward Const / runtime-inactive stored value zeroes active struct shadow" begin
        s = MutSetfieldAny(3.0, nothing)
        ds = MutSetfieldAny(42.0, nothing)
        autodiff(
            set_runtime_activity(Forward),
            setfield_to_box!,
            Duplicated(s, ds),
            Const(:a),
            Const(10.0),
        )
        @test s.a == 10.0
        @test ds.a == 0.0

        # Runtime-inactive value coming from runtime-inactive source struct
        src = MutSetfieldAny(5.0, nothing)
        dst = MutSetfieldAny(1.0, nothing)
        ddst = MutSetfieldAny(77.0, nothing)
        autodiff(
            set_runtime_activity(Forward),
            setfield_copy_any!,
            Duplicated(dst, ddst),
            Duplicated(src, src),
            Const(:a),
        )
        @test dst.a == 5.0
        @test ddst.a == 0.0
    end

    @testset "Forward & Reverse setfield! with Const stored value and used return" begin
        for mode in (set_runtime_activity(Forward), set_runtime_activity(Reverse))
            s = MutSetfieldAny(1.0, nothing)
            ds = MutSetfieldAny(0.0, nothing)
            if mode isa Enzyme.ForwardMode
                res = autodiff(
                    mode,
                    setfield_used_return_const!,
                    Duplicated(s, ds),
                    Const(:a),
                    Const(4.0),
                    Duplicated(3.0, 2.0),
                )
                @test res[1] ≈ 8.0
                @test s.a == 4.0
                @test ds.a == 0.0

                # Batched Forward width 2
                s_b = MutSetfieldAny(1.0, nothing)
                ds_b1 = MutSetfieldAny(0.0, nothing)
                ds_b2 = MutSetfieldAny(0.0, nothing)
                res_b = autodiff(
                    mode,
                    setfield_used_return_const!,
                    BatchDuplicated(s_b, (ds_b1, ds_b2)),
                    Const(:a),
                    Const(4.0),
                    BatchDuplicated(3.0, (2.0, 5.0)),
                )
                @test res_b[1][1] ≈ 8.0
                @test res_b[1][2] ≈ 20.0
            else
                res = autodiff(
                    mode,
                    setfield_used_return_const!,
                    Active,
                    Duplicated(s, ds),
                    Const(:a),
                    Const(4.0),
                    Active(3.0),
                )
                @test res[1][4] ≈ 4.0
                @test s.a == 4.0

                # Batched Reverse width 2
                s_b = MutSetfieldAny(1.0, nothing)
                ds_b1 = MutSetfieldAny(0.0, nothing)
                ds_b2 = MutSetfieldAny(0.0, nothing)
                res_b = autodiff(
                    mode,
                    setfield_used_return_const!,
                    Active,
                    BatchDuplicated(s_b, (ds_b1, ds_b2)),
                    Const(:a),
                    Const(4.0),
                    Active(3.0),
                )
                @test res_b[1][4][1] ≈ 4.0
                @test res_b[1][4][2] ≈ 4.0
                @test s_b.a == 4.0
            end
        end
    end

    @testset "Reverse runtime-inactive struct and runtime-inactive value" begin
        # Runtime-inactive destination struct Duplicated(s, s) with Active stored value
        s = MutSetfieldAny(3.0, nothing)
        res = autodiff(
            set_runtime_activity(Reverse),
            setfield_to_box!,
            Const,
            Duplicated(s, s),
            Const(:a),
            Active(10.0),
        )
        @test res[1][3] ≈ 0.0
        @test s.a == 10.0

        # Active destination struct seeded with d_out, active stored value
        s2 = MutSetfieldAny(3.0, nothing)
        ds2 = MutSetfieldAny(5.0, nothing)
        res2 = autodiff(
            set_runtime_activity(Reverse),
            setfield_to_box!,
            Const,
            Duplicated(s2, ds2),
            Const(:a),
            Active(10.0),
        )
        @test res2[1][3] ≈ 5.0
        @test s2.a == 10.0
        @test ds2.a == 0.0

        # Overwrite with Const clears shadow slot so earlier active store gets 0 adjoint
        s3 = MutSetfieldAny(1.0, nothing)
        ds3 = MutSetfieldAny(0.0, nothing)
        res3 = autodiff(
            set_runtime_activity(Reverse),
            setfield_overwrite_then_read!,
            Active,
            Duplicated(s3, ds3),
            Const(:a),
            Active(3.0),
            Const(9.0),
        )
        @test res3[1][3] ≈ 0.0
        @test s3.a == 9.0

        # Batched Reverse mixed active/runtime-inactive struct lanes (ds1, s)
        sb = MutSetfieldAny(3.0, nothing)
        dsb1 = MutSetfieldAny(6.0, nothing)
        res_b = autodiff(
            set_runtime_activity(Reverse),
            setfield_to_box!,
            Const,
            BatchDuplicated(sb, (dsb1, sb)),
            Const(:a),
            Active(10.0),
        )
        @test res_b[1][3][1] ≈ 6.0
        @test res_b[1][3][2] ≈ 0.0
        @test sb.a == 10.0
        @test dsb1.a == 0.0
    end
end
