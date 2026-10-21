module RuleDeps

# Enzyme verifies the rule answers its caches depend on itself (src/rule_deps.jl), instead
# of giving native backedges to code for them. These tests pin down the cases native edges
# get wrong or do not reach: a lookup that missed in an overlay method table
# (JuliaLang/julia#63959), and inference results that decided a call has no rule.

using Enzyme
using Enzyme: EnzymeRules, API, FFIABI
using Enzyme.Compiler: EnzymeTarget, EnzymeCompilerParams, UnknownTapeType
using Test
import .EnzymeRules: forward, augmented_primal, reverse, needs_primal, AugmentedReturn

const C = Enzyme.Compiler
const GPUCompiler = C.GPUCompiler
using .GPUCompiler: CompilerJob, CompilerConfig

# A pretend back-end with an overlay method table of its own, like CUDA.jl's.
struct RDTarget <: GPUCompiler.AbstractCompilerTarget end
struct RDParams <: GPUCompiler.AbstractCompilerParams end
Base.Experimental.@MethodTable(rd_query_table)
Base.Experimental.@MethodTable(rd_table)
GPUCompiler.method_table(@nospecialize(job::CompilerJob{RDTarget, RDParams})) = rd_table

rd_query(x::Float64) = x * x

@testset "a rule query that missed in an overlay table sees a later overlay rule" begin
    sig = Tuple{typeof(rd_query), Float64}
    world = Base.get_world_counter()
    view = Core.Compiler.OverlayMethodTable(world, rd_query_table)
    @test !C.cached_has_frule(sig, world, view)

    # The answer records both lookups, the overlay table's first, each finding nothing.
    answer = C.rule_answer(EnzymeRules.forward, true, sig, world, view)
    @test !answer.value
    @test length(answer.lookups) == 2
    @test answer.lookups[1].table === rd_query_table
    @test answer.lookups[2].table === nothing
    @test all(l -> isempty(l.methods), answer.lookups)

    @eval Base.Experimental.@overlay rd_query_table forward(config, ::Const{typeof(rd_query)}, ::Type{<:Duplicated}, x::Duplicated) =
        Duplicated(x.val * x.val, 100.0 * x.dval)
    world2 = Base.get_world_counter()

    # Verifying the answer in the new world finds the overlay method and closes it ...
    @test !C.answer_valid!(answer, world2)
    @test answer.max_world < world2
    # ... while it still holds in the world it was made in.
    @test C.answer_valid!(answer, world)
    @test !C.cached_has_frule(sig, world, view)
    # The query in the new world sees the rule, through the overlay table only.
    @test C.cached_has_frule(sig, world2, Core.Compiler.OverlayMethodTable(world2, rd_query_table))
    @test !C.cached_has_frule(sig, world2, nothing)
end

rd_callee(x::Float64) = x * x
rd_caller(x::Float64) = rd_callee(x) + 1.0

function rd_job(@nospecialize(caller), world)
    mode = API.DEM_ForwardMode
    mi = C.my_methodinstance(Forward, typeof(caller), Tuple{Float64}, world)
    params = EnzymeCompilerParams(
        Tuple{Const{typeof(caller)}, Duplicated{Float64}}, mode, 1, Duplicated{Float64},
        true, true, (false, false), false, false, UnknownTapeType, FFIABI, false, false, false,
    )
    target = GPUCompiler.nest_target(EnzymeTarget(), RDTarget())
    params = GPUCompiler.nest_params(params, RDParams())
    return mi, CompilerJob(mi, CompilerConfig(target, params; kernel = false), world)
end

function enzyme_ci(job, mi, world)
    ci = ccall(:jl_rettype_inferred, Any, (Any, Any, UInt, UInt), C.enzyme_cache_owner(job), mi, world, world)
    return ci isa Core.CodeInstance ? ci : nothing
end

function callee_answers(ci)
    deps = C.rule_deps_of(ci)
    deps === nothing && return C.RuleAnswer[]
    return filter(a -> any(l -> occursin("rd_callee", string(l.sig)), a.lookups), deps.answers)
end

@static if C.HAS_RULE_DEPS
    @testset "an inference result that missed in an overlay table is verified against a later overlay rule" begin
        world = Base.get_world_counter()
        mi, job = rd_job(rd_caller, world)
        @test C.return_type(GPUCompiler.get_interpreter(job), mi) === Float64
        ci = enzyme_ci(job, mi, world)
        @test ci !== nothing
        # The result carries the answers it decided `rd_callee` has no rule with, and those
        # include the miss in the back-end's table.
        answers = callee_answers(ci)
        @test !isempty(answers)
        @test all(a -> !a.value, answers)
        @test any(a -> any(l -> l.table === rd_table && isempty(l.methods), a.lookups), answers)

        @eval Base.Experimental.@overlay rd_table forward(config, ::Const{typeof(rd_callee)}, ::Type{<:Duplicated}, x::Duplicated) =
            Duplicated(x.val * x.val, 100.0 * x.dval)
        world2 = Base.get_world_counter()

        # Julia's own edges do not see the overlay method (JuliaLang/julia#63959) ...
        @test ci.max_world == typemax(UInt)
        # ... Enzyme's verification does, and closes the instance.
        @test !C.ci_rule_valid!(ci, world2)
        @test ci.max_world < world2
        @test C.ci_rule_valid!(ci, world)

        # Inference in the new world makes a new instance, which found the rule.
        mi2, job2 = rd_job(rd_caller, world2)
        @test C.return_type(GPUCompiler.get_interpreter(job2), mi2) === Float64
        ci2 = enzyme_ci(job2, mi2, world2)
        @test ci2 !== nothing && ci2 !== ci
        @test any(a -> a.value, callee_answers(ci2))
    end
end

# Defining and deleting reverse rules recompiles the thunks that baked the answer in, also
# through a caller the callee was inferred (and would be inlined) into.
rd_rev(x) = x^2
call_rd_rev(x) = rd_rev(x) + 1.0

@testset "defining and deleting a reverse rule" begin
    @test autodiff(Reverse, rd_rev, Active(3.0))[1][1] ≈ 6.0
    @test autodiff(Reverse, call_rd_rev, Active(3.0))[1][1] ≈ 6.0

    @eval augmented_primal(config, ::Const{typeof(rd_rev)}, ::Type{<:Active}, x::Active) =
        AugmentedReturn(needs_primal(config) ? x.val^2 : nothing, nothing, nothing)
    @eval reverse(config, ::Const{typeof(rd_rev)}, dret::Active, tape, x::Active) = (100.0 * dret.val,)

    @test autodiff(Reverse, rd_rev, Active(3.0))[1][1] ≈ 100.0
    @test autodiff(Reverse, call_rd_rev, Active(3.0))[1][1] ≈ 100.0

    for m in methods(augmented_primal, Tuple{Any, Const{typeof(rd_rev)}, Vararg{Any}})
        Base.delete_method(m)
    end
    for m in methods(reverse, Tuple{Any, Const{typeof(rd_rev)}, Vararg{Any}})
        Base.delete_method(m)
    end

    @static if C.HAS_RULE_DEPS || VERSION >= v"1.13.1-DEV"
        @test autodiff(Reverse, rd_rev, Active(3.0))[1][1] ≈ 6.0
        @test autodiff(Reverse, call_rd_rev, Active(3.0))[1][1] ≈ 6.0
    end
end

# An `inactive` rule for a callee makes the derivative through it zero, and deleting it
# restores it; the callee goes through a caller inferred before the rule existed.
rd_ina(x) = 3.0 * x
call_rd_ina(x) = rd_ina(x) + x

@testset "defining and deleting an inactive rule" begin
    @test autodiff(Forward, call_rd_ina, Duplicated(2.0, 1.0))[1] ≈ 4.0
    @eval EnzymeRules.inactive(::typeof(rd_ina), args...) = nothing
    @test autodiff(Forward, call_rd_ina, Duplicated(2.0, 1.0))[1] ≈ 1.0
    for m in methods(EnzymeRules.inactive, Tuple{typeof(rd_ina), Vararg{Any}})
        Base.delete_method(m)
    end
    @static if C.HAS_RULE_DEPS || VERSION >= v"1.13.1-DEV"
        @test autodiff(Forward, call_rd_ina, Duplicated(2.0, 1.0))[1] ≈ 4.0
    end
end

# A compiled thunk is checked in the world it runs in, not only when it is compiled: code
# compiled before a rule existed, and not invalidated natively since, recompiles.
rd_late(x) = x^3
rd_late_fwd(x) = autodiff(Forward, rd_late, Duplicated(x, 1.0))[1]

@static if C.HAS_RULE_DEPS
    @testset "a thunk checks its rule answers where it runs" begin
        @test rd_late_fwd(2.0) ≈ 12.0
        C.reset_rule_stats!()
        @eval forward(config, ::Const{typeof(rd_late)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) = 7.0 * x.dval
        @eval forward(config, func::Const{typeof(rd_late)}, ::Type{<:Duplicated}, x::Duplicated) =
            Duplicated(func.val(x.val), 7.0 * x.dval)
        @test rd_late_fwd(2.0) ≈ 7.0
        # The same world again takes the recompiled thunk without verifying anything.
        stats = (C.RULE_STATS.guard_recompiled, C.RULE_STATS.verified)
        @test rd_late_fwd(2.0) ≈ 7.0
        @test (C.RULE_STATS.guard_recompiled, C.RULE_STATS.verified) == stats
    end
end

end # module
