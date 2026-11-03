using Enzyme, Test
using Enzyme: EnzymeRules

# Derivative-free callees called natively instead of being emitted (Julia 1.10 and later).

struct NCStructure
    idx::Vector{Int}
    n::Int
end
@noinline nc_structure(n::Int) = NCStructure(collect(1:n), n)

function nc_loss(x::Vector{Float64})
    s = nc_structure(length(x))
    acc = 0.0
    for i in s.idx
        acc += x[i]^2
    end
    return acc
end

# `_sortperm` on integers takes and returns only boxed values, so Julia compiles it with
# the boxed `jl_fptr_args` ABI and not a specialized entry: it must stay emitted.
nc_sortperm(x) = (p = sortperm([3, 1, 2]); x * p[1])

# A read-only function whose callee calls an effect-free native callee. On Julia 1.10 and
# 1.11 that call goes through `jl_invoke`; unless it is marked read-only-or-throw, Enzyme
# no longer finds `nc_shared` read-only-or-throw and clones it, and on Julia 1.10 (typed
# pointers) that clone failed LLVM verification.
struct NCView
    parent::Vector{Float64}
    size::NTuple{5, Int}
    strides::NTuple{5, Int}
    offset::Int
end
@noinline function nc_strides(dims::NTuple{5, Int})
    s2 = dims[1]
    s3 = s2 * dims[2]
    s4 = s3 * dims[3]
    return (1, s2, s3, s4, s4 * dims[4])
end
@noinline function nc_view(x::Vector{Float64})
    dims = (length(x) ÷ 72, 3, 4, 3, 2)
    return NCView(x, dims, nc_strides(dims), 0)
end
@noinline function nc_shared(x::Vector{Float64}, p::NTuple{3, Int})
    t = nc_view(x)
    return t.size[p[1]] * t.strides[p[2]] == t.size[p[3]]
end
nc_branch(x, p) = nc_shared(x, p) ? sum(x) : 2 * sum(x)

# A callee with few boxed parameters and a boxed return: Julia gives it a specialized entry
# point all the same, so it is called natively.
@noinline nc_structure2(idx::Vector{Int}) = NCStructure(idx, length(idx))
function nc_loss2(x::Vector{Float64})
    s = nc_structure2(collect(1:length(x)))
    acc = 0.0
    for i in s.idx
        acc += x[i]^2
    end
    return acc
end

# Keyword calls: `@noinline` keeps the `Core.kwcall` wrapper out of line, so it is a
# candidate for a native call. Whether its function has a rule is asked of the signature
# without the wrapper, as every other decision point asks it.
@noinline function nc_kw(x::Int; iters::Int = 3)
    acc = 0
    for i in 1:iters
        acc += (x * i) % 7 + i ÷ 2
    end
    return acc
end
@noinline function nc_kw_ruled(x::Int; iters::Int = 3)
    acc = 0
    for i in 1:iters
        acc += (x * i) % 7 + i ÷ 2
    end
    return acc
end
function EnzymeRules.forward(config, ::Const{typeof(nc_kw_ruled)}, ::Type{<:Const}, x::Const{Int}; iters::Int = 3)
    return Const(nc_kw_ruled(x.val; iters))
end
nc_kw_user(x) = x * nc_kw(2; iters = 3)

kwcall_methodinstance(f, nt, args...) =
    Enzyme.my_methodinstance(nothing, typeof(Core.kwcall), Tuple{nt, typeof(f), args...})

# The job `onehot` emits for `llvmcall` outside any `EnzymeContext` is one of a host
# compilation whose parameters turn native callees off.
function primal_job(mi::Core.MethodInstance, params)
    GPUCompiler = Enzyme.Compiler.GPUCompiler
    config = GPUCompiler.CompilerConfig(Enzyme.Compiler.DefaultCompilerTarget(), params; kernel = false)
    return GPUCompiler.CompilerJob(mi, config)
end

@static if VERSION < v"1.12-" || v"1.12-beta3" <= VERSION
    @testset "Native derivative-free callees" begin
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x
        @test autodiff(Forward, nc_loss, Duplicated(x, ones(3)))[1] ≈ 12.0

        @test autodiff(Reverse, nc_sortperm, Active, Active(2.0))[1][1] ≈ 2.0
        @test autodiff(Forward, nc_sortperm, Duplicated(2.0, 1.0))[1] ≈ 2.0

        world = Base.get_world_counter()
        interp = Enzyme.Compiler.primal_interp_world(Forward, world)
        nt = NamedTuple{(:iters,), Tuple{Int}}
        mi_kw = kwcall_methodinstance(nc_kw, nt, Int)
        mi_kw_ruled = kwcall_methodinstance(nc_kw_ruled, nt, Int)
        @test Enzyme.Compiler.native_entry(interp, mi_kw) !== nothing
        # A keyword call of a function with a rule is never derivative-free. On Julia 1.12
        # and later it is called natively as one with a rule, which the handlers apply.
        @test Enzyme.Compiler.native_edge(interp, mi_kw_ruled) === nothing
        @static if VERSION >= v"1.12-"
            @test Enzyme.Compiler.native_entry(interp, mi_kw_ruled)[3] === :frule
        else
            @test Enzyme.Compiler.native_entry(interp, mi_kw_ruled) === nothing
        end
        @test autodiff(Forward, nc_kw_user, Duplicated(2.0, 1.0))[1] ≈ nc_kw(2; iters = 3)
        @test autodiff(Reverse, nc_kw_user, Active, Active(2.0))[1][1] ≈ nc_kw(2; iters = 3)

        mi2 = Enzyme.my_methodinstance(nothing, typeof(nc_structure2), Tuple{Vector{Int}})
        @test Enzyme.Compiler.native_entry(interp, mi2) !== nothing
        x = [1.0, 2.0, 3.0]
        dx = zero(x)
        autodiff(Reverse, nc_loss2, Active, Duplicated(x, dx))
        @test dx ≈ 2 .* x

        mi_zero = Enzyme.my_methodinstance(nothing, typeof(Enzyme.zerosetfn), Tuple{Vector{Float64}, Int})
        GPUCompiler = Enzyme.Compiler.GPUCompiler
        params = Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode)
        @test params.native_callees
        @test GPUCompiler.get_interpreter(primal_job(mi_zero, params)).native_callees
        params = Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode; native_callees = false)
        @test !GPUCompiler.get_interpreter(primal_job(mi_zero, params)).native_callees
        @test Enzyme.onehot([1.0, 2.0]) == ([1.0, 0.0], [0.0, 1.0])

        # Julia 1.10: Enzyme keeps its inference results in caches of its own, and a job
        # that does not call natively must not find a caller inferred by one that does.
        @static if VERSION < v"1.11.0-DEV.1552"
            CC = Core.Compiler
            mode = Enzyme.API.DEM_ReverseModeCombined
            native_cache = Enzyme.Compiler.enzyme_ci_cache(mode, true)
            cache = Enzyme.Compiler.enzyme_ci_cache(mode, false)
            @test native_cache !== cache
            mi_loss = Enzyme.my_methodinstance(nothing, typeof(nc_loss), Tuple{Vector{Float64}})
            mi_callee = Enzyme.my_methodinstance(nothing, typeof(nc_structure), Tuple{Int})
            @test CC.get(CC.WorldView(native_cache, world, world), mi_loss, nothing) !== nothing
            @test CC.get(CC.WorldView(native_cache, world, world), mi_callee, nothing) === nothing
            interp = Enzyme.Compiler.Interpreter.EnzymeInterpreter(cache, nothing, world, mode, true; native_callees = false)
            CC.typeinf_ext_toplevel(interp, mi_loss)
            @test CC.get(CC.code_cache(interp), mi_callee, nothing) !== nothing
        end

        # TODO: with code coverage `nc_shared` writes its counters, so Enzyme clones it
        # anyway, and on Julia 1.10 the clone fails verification until Enzyme_jll
        # carries EnzymeAD/Enzyme#3491.
        @static if VERSION >= v"1.11-" || Base.JLOptions().code_coverage == 0
            x = ones(144)
            for p in ((1, 3, 2), (2, 1, 2))
                dx = zero(x)
                autodiff(Reverse, nc_branch, Active, Duplicated(x, dx), Const(p))
                @test dx == fill(nc_shared(x, p) ? 1.0 : 2.0, 144)
            end
        end
    end
end

# Primals of functions with custom rules. Enzyme calls the rule when one applies. When none
# does and the call is constant (its function, arguments and return are all constant),
# Enzyme calls the primal as is; on Julia 1.12 and 1.13 natively instead of emitted. Any
# other call without a rule throws a `MethodError`.

using Enzyme: EnzymeRules

# Rules for active arguments only, as ChainRules imports make them: no rule applies when
# every argument is constant, and the return is constant.
nc_ruled(x) = x^2

function EnzymeRules.forward(config, func::Const{typeof(nc_ruled)}, ::Type{<:Union{Duplicated, DuplicatedNoNeed}}, x::Duplicated)
    d = 100 * x.dval
    return EnzymeRules.needs_primal(config) ? Duplicated(func.val(x.val), d) : d
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(nc_ruled)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(nc_ruled)}, dret::Active, tape, x::Active)
    return (100 * dret.val,)
end

# The first call goes through the rule, the second calls the primal.
nc_ruled_use(x, c) = nc_ruled(x) + x * nc_ruled(c)
# An active call only: the rule, whose derivative is 100.
nc_ruled_active(x) = 2 * nc_ruled(x)

# A rule for constant arguments only: it applies to a constant call, and Enzyme calls it
# rather than the primal. No rule applies to an active call, which throws.
nc_construle(x) = x^2
function EnzymeRules.forward(config, ::Const{typeof(nc_construle)}, ::Type{<:Const}, x::Const)
    return EnzymeRules.needs_primal(config) ? 100 * x.val : nothing
end
nc_construle_use(x, c) = x * nc_construle(c)
nc_construle_active(x) = 2 * nc_construle(x)

# A return with GC roots, returned through an sret buffer. Activity analysis gives the
# buffer a shadow although the argument is constant, so the return is not constant: the
# call needs a rule, and none applies. Its stub is still compiled (see
# `bind_ruled_stub!`).
struct NCRooted
    v::Vector{Float64}
    s::Float64
end
nc_rooted(c) = NCRooted([c, 2c], 3c)

function EnzymeRules.forward(config, ::Const{typeof(nc_rooted)}, ::Type, c::Duplicated)
    error("nc_rooted: the rule is not expected to run")
end
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(nc_rooted)}, ::Type, c::Active)
    error("nc_rooted: the rule is not expected to run")
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(nc_rooted)}, dret, tape, c::Active)
    error("nc_rooted: the rule is not expected to run")
end

nc_rooted_use(x, c) = (r = nc_rooted(c); x * r.s * r.v[2])

# A return with GC roots and inline data, from a function Julia does not inline, so that
# its stub, which returns through the sret and roots of its own parameters, stays a call.
# As for `nc_rooted`, the return is not constant.
@noinline nc_mixed(c) = (view(c, :), (1, 2, 3, 4))

function EnzymeRules.forward(config, ::Const{typeof(nc_mixed)}, ::Type, c::Duplicated)
    error("nc_mixed: the rule is not expected to run")
end
function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, ::Const{typeof(nc_mixed)}, ::Type, c::Duplicated)
    error("nc_mixed: the rule is not expected to run")
end
function EnzymeRules.reverse(config::EnzymeRules.RevConfig, ::Const{typeof(nc_mixed)}, dret, tape, c::Duplicated)
    error("nc_mixed: the rule is not expected to run")
end

nc_mixed_use(x, c) = (r = nc_mixed(c); x * r[1][2] * r[2][4])

# A keyword call of a function with a rule.
nc_kw(x; scale = 1.0) = scale * x^2
function EnzymeRules.forward(config, ::Const{typeof(nc_kw)}, ::Type, x::Duplicated; scale = 1.0)
    error("nc_kw: the rule is not expected to run")
end
nc_kw_use(x, c) = x * nc_kw(c; scale = 3.0)

# A reverse rule only: forward over reverse differentiates through the primal, which the
# inner reverse differentiation calls with constant arguments.
nc_rronly(x) = x^2

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(nc_rronly)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (x.val,))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(nc_rronly)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (2 * xv * dret.val,)
end

nc_rronly_use(a, c) = a * nc_rronly(c)
nc_rronly_grad(c) = autodiff_deferred(Reverse, Const(nc_rronly_use), Active, Active(1.0), Const(c))[1][1]

# Called natively, the primal is ordinary Julia code: `ignore_derivatives` is the
# identity, and `within_autodiff()` is false. Emitted, `EnzymeInterpreter` folds
# `within_autodiff()` to true, as in any code Enzyme compiles.
nc_within(x) = Enzyme.within_autodiff() ? 2x : 3x
function EnzymeRules.forward(config, ::Const{typeof(nc_within)}, ::Type, x::Duplicated)
    error("nc_within: the rule is not expected to run")
end
nc_within_use(x, c) = x * nc_within(c)

nc_ignore(x) = Enzyme.ignore_derivatives(x)^2
function EnzymeRules.forward(config, ::Const{typeof(nc_ignore)}, ::Type, x::Duplicated)
    error("nc_ignore: the rule is not expected to run")
end
nc_ignore_use(x, c) = x * nc_ignore(c)

@testset "Primals of functions with rules, in constant calls" begin
    # d/dx = 100 (the rule) + c^2 (the primal)
    @test autodiff(Forward, nc_ruled_use, Duplicated(3.0, 1.0), Const(2.0))[1] ≈ 104.0
    @test autodiff(ForwardWithPrimal, nc_ruled_use, Duplicated(3.0, 1.0), Const(2.0)) == (104.0, 21.0)
    @test autodiff(Reverse, nc_ruled_use, Active, Active(3.0), Const(2.0))[1][1] ≈ 104.0

    # d/dx = 3c^2
    @test autodiff(Forward, nc_kw_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ 12.0

    @test nc_rronly_grad(3.0) ≈ 9.0
    # d/dc c^2
    @test autodiff(Forward, nc_rronly_grad, Duplicated(3.0, 1.0))[1] ≈ 6.0

    # d/dx = 3c natively, 2c emitted
    native = v"1.12-beta3" <= VERSION < v"1.14-"
    @test autodiff(Forward, nc_within_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ (native ? 6.0 : 4.0)
end

# Emitted, the primal leaves `ignore_derivatives` to the Enzyme pipeline, which lowers it
# only in the code it differentiates.
@static if v"1.12-beta3" <= VERSION < v"1.14-"
    @testset "Natively called primals of functions with rules are ordinary Julia code" begin
        # d/dx = c^2
        @test autodiff(Forward, nc_ignore_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ 4.0
    end
end

@testset "Rules of functions with rules still apply" begin
    # An active call goes through the rule.
    @test autodiff(Forward, nc_ruled_active, Duplicated(3.0, 1.0))[1] ≈ 200.0
    @test autodiff(Reverse, nc_ruled_active, Active, Active(3.0))[1][1] ≈ 200.0

    # A rule for constant arguments applies to a constant call: d/dx = 100c
    @test autodiff(Forward, nc_construle_use, Duplicated(1.0, 1.0), Const(2.0))[1] ≈ 200.0
    # No rule applies to an active call.
    @test_throws MethodError autodiff(Forward, nc_construle_active, Duplicated(2.0, 1.0))

    # A call whose return is not constant needs a rule, although its arguments are.
    @test_throws MethodError autodiff(Forward, nc_rooted_use, Duplicated(1.5, 1.0), Const(2.0))
    # Before Julia 1.12 the rule handler rejects the mixed activity of the return first
    # (see `enzyme_custom_setup_ret`).
    rooted_rev_error = VERSION < v"1.12" ? Enzyme.Compiler.MixedReturnException : MethodError
    @test_throws rooted_rev_error autodiff(Reverse, nc_rooted_use, Active, Active(1.5), Const(2.0))
    c = [1.0, 2.0]
    @test_throws MethodError autodiff(ForwardWithPrimal, nc_mixed_use, Duplicated(1.5, 1.0), Const(c))
    @test_throws MethodError autodiff(ReverseWithPrimal, nc_mixed_use, Active, Active(1.5), Const(c))
end
