using Enzyme, Test
using Enzyme.Compiler: EnzymeTarget, EnzymeCompilerParams, PrimalCompilerParams, UnknownTapeType
using Enzyme: API, FFIABI
const GPUCompiler = Enzyme.Compiler.GPUCompiler
using .GPUCompiler: CompilerJob, CompilerConfig

# A pretend backend that, like CUDA.jl, defines its method table only for its own job type.
struct MTTestTarget <: GPUCompiler.AbstractCompilerTarget end
struct MTTestParams <: GPUCompiler.AbstractCompilerParams end
Base.Experimental.@MethodTable(mt_test_table)
GPUCompiler.method_table(@nospecialize(job::CompilerJob{MTTestTarget, MTTestParams})) = mt_test_table

mt_test_fn(x::Float64) = x
Base.Experimental.@overlay mt_test_table mt_test_fn(x::Float64) = "overlaid"
# The overlay is picked up at call sites, so infer a caller. Use a separate caller per
# job: before Julia 1.11 Enzyme keeps one global code cache per mode, shared across method
# tables, so inferring the same method instance under both tables would reuse the first
# result.
mt_test_native_caller(x::Float64) = mt_test_fn(x)
mt_test_overlay_caller(x::Float64) = mt_test_fn(x)

function enzyme_job(caller, inner_target, inner_params, mode, world)
    mi = Enzyme.Compiler.my_methodinstance(Reverse, typeof(caller), Tuple{Float64}, world)
    params = EnzymeCompilerParams(
        Tuple{Const{typeof(caller)}, Active{Float64}}, mode, 1, Active{Float64},
        true, true, (false, false), false, false, UnknownTapeType, FFIABI, false, false, false,
    )
    target = GPUCompiler.nest_target(EnzymeTarget(), inner_target)
    params = GPUCompiler.nest_params(params, inner_params)
    return mi, CompilerJob(mi, CompilerConfig(target, params; kernel = false), world)
end

@testset "method_table forwards through EnzymeTarget" begin
    world = Base.get_world_counter()
    mode = API.DEM_ReverseModeCombined

    mi, native_job = enzyme_job(mt_test_native_caller, GPUCompiler.NativeCompilerTarget(), PrimalCompilerParams(mode), mode, world)
    @test GPUCompiler.method_table(native_job) === GPUCompiler.GLOBAL_METHOD_TABLE
    @test Enzyme.Compiler.return_type(GPUCompiler.get_interpreter(native_job), mi) === Float64

    mi, backend_job = enzyme_job(mt_test_overlay_caller, MTTestTarget(), MTTestParams(), mode, world)
    @test backend_job.config.target isa EnzymeTarget{MTTestTarget}
    @test backend_job.config.params isa EnzymeCompilerParams{MTTestParams}
    @test GPUCompiler.method_table(backend_job) === mt_test_table

    interp = GPUCompiler.get_interpreter(backend_job)
    @test Core.Compiler.method_table(interp).mt === mt_test_table
    # Inference under the Enzyme job now sees the backend's overlay, as the primal job does.
    @test Enzyme.Compiler.return_type(interp, mi) === String
    @static if VERSION >= v"1.11.0-DEV.1552"
        @test Enzyme.Compiler.enzyme_cache_owner(backend_job).method_tables === (mt_test_table,)
    end
end

# A pretend backend that, like OpenCL.jl, stacks a shared method table underneath its own,
# by defining `method_table_view`.
struct MTStackTarget <: GPUCompiler.AbstractCompilerTarget end
struct MTStackParams <: GPUCompiler.AbstractCompilerParams end
Base.Experimental.@MethodTable(mt_shared_table)
GPUCompiler.method_table(@nospecialize(job::CompilerJob{MTStackTarget, MTStackParams})) = mt_test_table
GPUCompiler.method_table_view(@nospecialize(job::CompilerJob{MTStackTarget, MTStackParams})) =
    GPUCompiler.StackedMethodTable(job.world, mt_test_table, mt_shared_table)

mt_shared_fn(x::Float64) = x
Base.Experimental.@overlay mt_shared_table mt_shared_fn(x::Float64) = "shared"
mt_test_stacked_caller(x::Float64) = (mt_test_fn(x), mt_shared_fn(x))

@testset "method_table_view forwards through EnzymeTarget" begin
    world = Base.get_world_counter()
    mode = API.DEM_ReverseModeCombined

    mi, job = enzyme_job(mt_test_stacked_caller, MTStackTarget(), MTStackParams(), mode, world)
    @test GPUCompiler.method_table_view(job) isa GPUCompiler.StackedMethodTable

    # Inference under the Enzyme job sees both the backend's and the shared overlay.
    interp = GPUCompiler.get_interpreter(job)
    @test Core.Compiler.method_table(interp) isa GPUCompiler.StackedMethodTable
    @test Enzyme.Compiler.return_type(interp, mi) === Tuple{String, String}

    @static if VERSION >= v"1.11.0-DEV.1552"
        # The cache token covers the whole stack, so it differs from that of a backend that
        # only uses the same main table.
        token = Enzyme.Compiler.enzyme_cache_owner(job)
        @test token.method_tables === (mt_test_table, mt_shared_table)
        @test Core.Compiler.cache_owner(interp) === token
        _, other_job = enzyme_job(mt_test_overlay_caller, MTTestTarget(), MTTestParams(), mode, world)
        @test token !== Enzyme.Compiler.enzyme_cache_owner(other_job)
    end
end

module Issue3962DummyRuntime
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

Base.Experimental.@MethodTable(issue3962_device_table)
struct Issue3962CUDAJobParams <: GPUCompiler.AbstractCompilerParams end
GPUCompiler.method_table(::CompilerJob{<:Any, Issue3962CUDAJobParams}) = issue3962_device_table
GPUCompiler.runtime_module(::CompilerJob{<:Any, Issue3962CUDAJobParams}) = Issue3962DummyRuntime

@noinline function issue3962_dev_only(x::Float64)
    x < 0.0 && error("Host fallback called instead of @overlay!")
    return x
end
Base.Experimental.@overlay issue3962_device_table @inline issue3962_dev_only(x::Float64) = 1.0 - x * x

@noinline issue3962_g(x::Float64) = issue3962_dev_only(x)

function Enzyme.EnzymeRules.forward(
        config::Enzyme.EnzymeRules.FwdConfig,
        ::Const{typeof(issue3962_g)},
        ::Type{<:Annotation},
        x::Duplicated{Float64},
    )
    y = issue3962_g(x.val)
    dy = -2.0 / sqrt(π) * exp(-x.val^2) * x.dval
    Enzyme.EnzymeRules.needs_primal(config) && Enzyme.EnzymeRules.needs_shadow(config) && return Duplicated(y, dy)
    Enzyme.EnzymeRules.needs_shadow(config) && return dy
    return y
end

function Enzyme.EnzymeRules.augmented_primal(
        config::Enzyme.EnzymeRules.RevConfig,
        ::Const{typeof(issue3962_g)},
        ::Type{<:Annotation},
        x::Active{Float64},
    )
    primal = Enzyme.EnzymeRules.needs_primal(config) ? issue3962_g(x.val) : nothing
    return Enzyme.EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(
        config::Enzyme.EnzymeRules.RevConfig,
        ::Const{typeof(issue3962_g)},
        dret::Active{Float64},
        tape,
        x::Active{Float64},
    )
    scale = issue3962_dev_only(x.val)
    return (scale * exp(-x.val^2) * dret.val,)
end

issue3962_f(x::Float64) = issue3962_g(x) + x
issue3962_h(x::Float64) = issue3962_dev_only(x) + x

function issue3962_kern_no_rule(out::Ptr{Float64}, x::Ptr{Float64})
    v = unsafe_load(x, 1)
    d = Enzyme.autodiff_deferred(Enzyme.Forward, Const(issue3962_h), Duplicated, Duplicated(v, 1.0))[1]
    unsafe_store!(out, d, 1)
    return nothing
end

function issue3962_kern_fwd_rule(out::Ptr{Float64}, x::Ptr{Float64})
    v = unsafe_load(x, 1)
    d = Enzyme.autodiff_deferred(Enzyme.Forward, Const(issue3962_f), Duplicated, Duplicated(v, 1.0))[1]
    unsafe_store!(out, d, 1)
    return nothing
end

function issue3962_kern_rev_rule(out::Ptr{Float64}, x::Ptr{Float64})
    v = unsafe_load(x, 1)
    d = Enzyme.autodiff_deferred(Enzyme.Reverse, Const(issue3962_f), Active, Active(v))[1][1]
    unsafe_store!(out, d, 1)
    return nothing
end

@testset "Custom rule in GPU CompilerJob nested_codegen! (#3962)" begin
    world = Base.get_world_counter()
    ptx_target = GPUCompiler.PTXCompilerTarget(; cap = v"7.5")
    params = Issue3962CUDAJobParams()
    sig = Tuple{Ptr{Float64}, Ptr{Float64}}

    # Verify nested_compiler_job preserves both target (e.g. PTXCompilerTarget) and params
    source_ptx = GPUCompiler.methodinstance(typeof(issue3962_kern_fwd_rule), sig)
    outer_ptx_job = CompilerJob(source_ptx, CompilerConfig(ptx_target, params; kernel = true), world)
    rule_mi = GPUCompiler.methodinstance(typeof(issue3962_g), Tuple{Float64})
    nested_ptx_job = Enzyme.Compiler.@with Enzyme.Compiler.ENZYME_CONTEXT => Enzyme.Compiler.EnzymeContext(world, outer_ptx_job) begin
        Enzyme.Compiler.nested_compiler_job(rule_mi, API.DEM_ForwardMode, world)
    end
    @test nested_ptx_job.config.target === ptx_target
    @test nested_ptx_job.config.params === params
    @test nested_ptx_job.config.kernel === false
    @test GPUCompiler.method_table(nested_ptx_job) === issue3962_device_table

    # Verify full nested_codegen! compilation through GPUCompiler.compile(:llvm, job)
    host_target = isdefined(GPUCompiler, :NativeCompilerTarget) ? GPUCompiler.NativeCompilerTarget() : GPUCompiler.DefaultCompilerTarget()
    for fn in (issue3962_kern_no_rule, issue3962_kern_fwd_rule, issue3962_kern_rev_rule)
        source = GPUCompiler.methodinstance(typeof(fn), sig)
        config = CompilerConfig(host_target, params; kernel = false)
        job = CompilerJob(source, config, world)
        ir_str = GPUCompiler.JuliaContext() do ctx
            ir, _ = GPUCompiler.compile(:llvm, job)
            string(ir)
        end
        @test !occursin("Host fallback called", ir_str)
    end
end
