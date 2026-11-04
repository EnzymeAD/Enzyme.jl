using Enzyme, Test

const LLVM = Enzyme.Compiler.LLVM

@noinline function tanh_work(x, k)
    w = Vector{Float64}(undef, k)
    s = 0.0
    for i in 1:k
        z = tanh(x[i])
        s += z * sqrt(1 + s^2)
        w[i] = s
    end
    return w[k] * s
end

function tanh_work_sum(x)
    s = 0.0
    for k in 1:length(x)
        s += tanh_work(x, k)
    end
    return s
end

function calls_julia_safepoint(f)
    for bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        if callee isa LLVM.Function && LLVM.name(callee) == "julia.safepoint"
            return true
        end
    end
    return false
end

function functions_with_safepoint(mod)
    fs = LLVM.Function[]
    for f in LLVM.functions(mod)
        isempty(LLVM.blocks(f)) && continue
        calls_julia_safepoint(f) && push!(fs, f)
    end
    return fs
end

@testset "Calls of functions with a GC safepoint stay GC safepoints" begin
    # Enzyme marks math functions such as `tanh` readnone, but their Julia bodies
    # hold a GC safepoint. GC lowering must not take calls to them as calls that
    # cannot collect, or a value live across the call (here the shadow of `w`) is
    # left unrooted and freed by a collection that another thread starts.
    Enzyme.Compiler.JuliaContext() do ctx
        _, mod = Enzyme.Compiler.reflect(tanh_work_sum, Active, Tuple{Duplicated{Vector{Float64}}}; second_stage = false)
        marked_pure = 0
        for f in functions_with_safepoint(mod)
            if Enzyme.Compiler.has_fn_attr(f, LLVM.StringAttribute("enzyme_math")) &&
                    Enzyme.Compiler.excludes_safepoint(collect(LLVM.function_attributes(f)))
                marked_pure += 1
            end
        end
        @test marked_pure > 0

        # Everything up to, but not including, GC lowering.
        Enzyme.Compiler.post_optimize!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        with_safepoint = functions_with_safepoint(mod)
        @test !isempty(with_safepoint)
        for f in with_safepoint
            @test !Enzyme.Compiler.excludes_safepoint(collect(LLVM.function_attributes(f)))
            for call in Enzyme.Compiler.calls_of(f)
                @test !Enzyme.Compiler.excludes_safepoint(Enzyme.Compiler.callsite_function_attributes(call))
            end
        end
    end
    x = [0.1 * sin(i) for i in 1:6]
    dx = zero(x)
    autodiff(Reverse, tanh_work_sum, Active, Duplicated(x, dx))
    h = 1.0e-6
    for i in eachindex(x)
        xp = copy(x)
        xp[i] += h
        xm = copy(x)
        xm[i] -= h
        @test dx[i] ≈ (tanh_work_sum(xp) - tanh_work_sum(xm)) / 2h rtol = 1.0e-6 atol = 1.0e-8
    end
end

# The tape of a call's derivative is a Julia object whose pointer fields are filled
# with `nothing` before the shadows allocated next are stored into them. Dead store
# elimination used to drop a `nothing` store that such a shadow later overwrote, so
# a collection triggered by allocating the shadow scanned an uninitialized field.
# Runtime activity through Base's `copyto!` from a view (`unalias`) crashed this way
# on Julia 1.12 and 1.13 within a few hundred thousand calls.
function tape_view_copy(x, s, idx)
    w = convert(Vector{Float64}, view(x, idx))
    t = 0.0
    @inbounds for i in eachindex(w)
        t += w[i] * log(s)
    end
    return t
end

function repeat_call(g, reps)
    acc = 0.0
    for _ in 1:reps
        acc += g()
    end
    return acc
end

@testset "Tape pointer fields are initialized before the next allocation" begin
    reps = 150_000
    for (n, m) in ((7, 4), (1000, 500), (1000, 4))
        x = collect(range(0.1, 2.0; length = n))
        idx = collect(1:(n ÷ m):n)[1:m]
        g = () -> autodiff(
            set_runtime_activity(Reverse), tape_view_copy, Active, Const(x), Active(2.5), Const(idx)
        )[1][2]
        @test repeat_call(g, reps) / reps ≈ sum(x[idx]) / 2.5
    end
end
