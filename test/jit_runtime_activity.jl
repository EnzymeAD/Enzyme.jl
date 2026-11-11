using Enzyme
using Test

mutable struct JitMutBox
    v::Float64
end

struct JitMixedClosure
    scale::Float64
    buf::Vector{Float64}
end

(c::JitMixedClosure)(x::Float64) = c.scale * x + c.buf[1]
(c::JitMixedClosure)(x::Float64, y::Float64) = c.scale * x + c.buf[1] * y

dyn_passthrough_vec(v::Vector{Float64}) = v
dyn_alloc_vec(v::Vector{Float64}) = copy(v)
dyn_read_box(b::JitMutBox) = 3.0 * b.v
dyn_mutate_box!(b::JitMutBox, x::Float64) = (b.v += 2.0 * x; nothing)
dyn_scale_scalar(x::Float64) = 3.0 * x

function call_dyn_scalar_cond(fref::Base.RefValue{Any}, x::Float64, active_branch::Bool)
    arg = active_branch ? Base.inferencebarrier(x) : Base.inferencebarrier(4.0)
    return (fref[])(arg)::Float64
end

function call_dyn_passthrough(fref::Base.RefValue{Any}, v::Vector{Float64}, x::Float64)
    w = (fref[])(v)::Vector{Float64}
    return w[1] * x
end

function call_dyn_alloc_then_mutate(fref::Base.RefValue{Any}, v::Vector{Float64}, x::Float64)
    w = (fref[])(v)::Vector{Float64}
    w[1] += x
    return w[1]
end

function call_dyn_box(fref::Base.RefValue{Any}, b::JitMutBox)
    return (fref[])(b)::Float64
end

function call_dyn_mutate_box!(fref::Base.RefValue{Any}, b::JitMutBox, x::Float64)
    (fref[])(b, x)
    return nothing
end

function call_dyn_closure(cvec::Vector{Any}, x::Float64)
    return (cvec[1])(x)::Float64
end

function call_dyn_iterate_closure(cref::Base.RefValue{Any}, args::Vector{Float64})
    return (cref[])(args...)::Float64
end

@testset "Generic JIT & iterate runtime activity" begin
    @testset "All-inactive dynamic call preserves primal===shadow vs distinct fresh allocation" begin
        fref_pass = Ref{Any}(dyn_passthrough_vec)
        fref_alloc = Ref{Any}(dyn_alloc_vec)
        v = [5.0]

        # Passthrough of runtime-inactive vector: shadow === primal, so w[1] is runtime-inactive (dw[1] = 0)
        res_fwd = autodiff(
            set_runtime_activity(Forward),
            call_dyn_passthrough,
            Const(fref_pass),
            Duplicated(v, v),
            Duplicated(2.0, 1.0),
        )
        @test res_fwd[1] ≈ 5.0
        @test v[1] == 5.0

        res_fwd_b = autodiff(
            set_runtime_activity(Forward),
            call_dyn_passthrough,
            Const(fref_pass),
            BatchDuplicated(v, (v, v)),
            BatchDuplicated(2.0, (1.0, 3.0)),
        )
        @test res_fwd_b[1][1] ≈ 5.0
        @test res_fwd_b[1][2] ≈ 15.0

        res_rev = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_passthrough,
            Active,
            Const(fref_pass),
            Duplicated(v, v),
            Active(2.0),
        )
        @test res_rev[1][3] ≈ 5.0
        @test v[1] == 5.0

        res_rev_b = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_passthrough,
            Active,
            Const(fref_pass),
            BatchDuplicated(v, (v, v)),
            Active(2.0),
        )
        @test res_rev_b[1][3][1] ≈ 5.0
        @test res_rev_b[1][3][2] ≈ 5.0
        @test v[1] == 5.0

        # Fresh allocation from runtime-inactive input: distinct zeroed shadow must be preserved
        # so downstream active `w[1] += x` does NOT clobber primal or lose derivative wrt x!
        res_alloc_fwd = autodiff(
            set_runtime_activity(Forward),
            call_dyn_alloc_then_mutate,
            Const(fref_alloc),
            Duplicated(v, v),
            Duplicated(3.0, 7.0),
        )
        @test res_alloc_fwd[1] ≈ 7.0
        @test v[1] == 5.0

        res_alloc_fwd_b = autodiff(
            set_runtime_activity(Forward),
            call_dyn_alloc_then_mutate,
            Const(fref_alloc),
            BatchDuplicated(v, (v, v)),
            BatchDuplicated(3.0, (7.0, 11.0)),
        )
        @test res_alloc_fwd_b[1][1] ≈ 7.0
        @test res_alloc_fwd_b[1][2] ≈ 11.0

        res_alloc_rev = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_alloc_then_mutate,
            Active,
            Const(fref_alloc),
            Duplicated(v, v),
            Active(3.0),
        )
        @test res_alloc_rev[1][3] ≈ 1.0
        @test v[1] == 5.0

        res_alloc_rev_b = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_alloc_then_mutate,
            Active,
            Const(fref_alloc),
            BatchDuplicated(v, (v, v)),
            Active(3.0),
        )
        @test res_alloc_rev_b[1][3][1] ≈ 1.0
        @test res_alloc_rev_b[1][3][2] ≈ 1.0
        @test v[1] == 5.0
    end

    @testset "Batched dynamic call with inactive lane 1 and active lane 2 (any_ne)" begin
        # Lane 1 is runtime-inactive (b === db1), Lane 2 is active (db2 !== b)!
        fref_read = Ref{Any}(dyn_read_box)
        b = JitMutBox(4.0)
        db2 = JitMutBox(5.0)
        res_fwd = autodiff(
            set_runtime_activity(Forward),
            call_dyn_box,
            Const(fref_read),
            BatchDuplicated(b, (b, db2)),
        )
        @test res_fwd[1][1] ≈ 0.0
        @test res_fwd[1][2] ≈ 15.0
        @test b.v == 4.0

        # Runtime-inactive branch vs active branch on boxed scalar in Reverse (width 1 & 2)
        fref_sc = Ref{Any}(dyn_scale_scalar)
        res_sc_act = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_scalar_cond,
            Active,
            Const(fref_sc),
            Active(4.0),
            Const(true),
        )
        @test res_sc_act[1][2] ≈ 3.0

        res_sc_inact = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_scalar_cond,
            Active,
            Const(fref_sc),
            Active(4.0),
            Const(false),
        )
        @test res_sc_inact[1][2] ≈ 0.0

        # Reverse mode with both lanes runtime-inactive on mutable box
        fref_mut = Ref{Any}(dyn_mutate_box!)
        bm = JitMutBox(4.0)
        res_rev = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_mutate_box!,
            Const,
            Const(fref_mut),
            BatchDuplicated(bm, (bm, bm)),
            Active(3.0),
        )
        @test res_rev[1][3][1] ≈ 0.0
        @test res_rev[1][3][2] ≈ 0.0
        @test bm.v == 10.0
    end

    @testset "MixedState closure in dynamic generic & iterate calls" begin
        @noinline jit_id_any(@nospecialize(x)) = Base.inferencebarrier(x)

        function build_and_call_mixed_closure(scale::Float64, buf::Vector{Float64}, x::Float64)
            cl = jit_id_any(JitMixedClosure(scale, buf))
            return cl(x)::Float64
        end

        # Runtime-inactive MixedState closure via Vector{Any} passed as Duplicated(v, v)
        c_inact = JitMixedClosure(3.0, [5.0])
        v_inact = Any[c_inact]
        res_inact = autodiff(
            set_runtime_activity(Reverse),
            call_dyn_closure,
            Active,
            Duplicated(v_inact, v_inact),
            Active(2.0),
        )
        @test res_inact[1][2] ≈ 3.0
        @test c_inact.scale == 3.0
        @test c_inact.buf == [5.0]

        buf = [5.0]
        dbuf = [0.0]
        grads = autodiff(
            set_runtime_activity(Reverse),
            build_and_call_mixed_closure,
            Active,
            Active(3.0),
            Duplicated(buf, dbuf),
            Active(2.0),
        )
        # scale * x + buf[1] => d/d(scale) = x = 2.0, d/d(buf[1]) = 1.0, d/dx = scale = 3.0
        @test grads[1][1] ≈ 2.0
        @test dbuf[1] ≈ 1.0
        @test grads[1][3] ≈ 3.0

        # Batched width 2
        buf_b = [5.0]
        dbuf_b1 = [0.0]
        dbuf_b2 = [0.0]
        grads_b = autodiff(
            set_runtime_activity(Reverse),
            build_and_call_mixed_closure,
            Active,
            Active(3.0),
            BatchDuplicated(buf_b, (dbuf_b1, dbuf_b2)),
            Active(2.0),
        )
        @test grads_b[1][1][1] ≈ 2.0
        @test grads_b[1][1][2] ≈ 2.0
        @test dbuf_b1[1] ≈ 1.0
        @test dbuf_b2[1] ≈ 1.0
        @test grads_b[1][3][1] ≈ 3.0
        @test grads_b[1][3][2] ≈ 3.0

        # Dynamic iterate with MixedState closure (width 1 & width 2)
        function build_and_iterate_mixed_closure(scale::Float64, buf::Vector{Float64}, args::Vector{Float64})
            cl = jit_id_any(JitMixedClosure(scale, buf))
            return cl(args...)::Float64
        end

        buf2 = [5.0]
        dbuf2 = [0.0]
        vargs = [2.0, 4.0]
        dvargs = [0.0, 0.0]
        grads_it = autodiff(
            set_runtime_activity(Reverse),
            build_and_iterate_mixed_closure,
            Active,
            Active(3.0),
            Duplicated(buf2, dbuf2),
            Duplicated(vargs, dvargs),
        )
        # scale * args[1] + buf[1] * args[2] = 3*2 + 5*4 = 26
        @test grads_it[1][1] ≈ 2.0
        @test dbuf2[1] ≈ 4.0
        @test dvargs ≈ [3.0, 5.0]

        buf2_b = [5.0]
        dbuf2_b1 = [0.0]
        dbuf2_b2 = [0.0]
        vargs_b = [2.0, 4.0]
        dvargs_b1 = [0.0, 0.0]
        grads_it_b = autodiff(
            set_runtime_activity(Reverse),
            build_and_iterate_mixed_closure,
            Active,
            Active(3.0),
            BatchDuplicated(buf2_b, (dbuf2_b1, dbuf2_b2)),
            BatchDuplicated(vargs_b, (vargs_b, dvargs_b1)),
        )
        @test grads_it_b[1][1][1] ≈ 2.0
        @test grads_it_b[1][1][2] ≈ 2.0
        @test dbuf2_b1[1] ≈ 4.0
        @test dbuf2_b2[1] ≈ 4.0
        @test vargs_b == [2.0, 4.0]
        @test dvargs_b1 ≈ [3.0, 5.0]

        # All-inactive dynamic call with abstract Any return yielding Float64 (RefValue shadow in Reverse)
        @noinline dyn_any_id(@nospecialize(x)) = Base.inferencebarrier(x)
        function call_dyn_any_ret(fref::Base.RefValue{Any}, x::Float64)
            c = (fref[])(7.0)::Float64
            return c * x
        end
        fref_any = Ref{Any}(dyn_any_id)
        @test autodiff(
            set_runtime_activity(Reverse),
            call_dyn_any_ret,
            Active,
            Const(fref_any),
            Active(3.0),
        )[1][2] ≈ 7.0
    end
end
