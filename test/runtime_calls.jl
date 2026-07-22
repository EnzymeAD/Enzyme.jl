using Enzyme, Test, Logging

@testset "jl_typeof" begin
    # https://github.com/EnzymeAD/Enzyme.jl/issues/2405
    function foo(x)
        @ccall jl_typeof(Ref(x)::Ref{Float64})::Any
        x + 1
    end
    @test autodiff(Reverse, foo, Active(1.0))[1][1] == 1.0
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3284
cfunc_target(x) = x * x

# Heap allocation forces GC frame/rooting code in the callee, so it reads
# pgcstack: on 1.12+ a dispatch site handed a raw native specsig pointer takes
# that pgcstack out of the swiftself register, which GPUCompiler-emitted
# modules (gcstack_arg = false) do not set.
function cfunc_callback(val::Float64)::Float64
    arr = [val]
    return arr[1] + 1.0
end

@noinline function run_cfunction_3284(x::Float64)
    cfunc = @cfunction(cfunc_callback, Float64, (Float64,))
    return ccall(cfunc, Float64, (Float64,), x)
end

function Enzyme.EnzymeRules.augmented_primal(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(cfunc_target)},
        ::Type{<:Active},
        x::Active,
    )
    val = run_cfunction_3284(x.val)
    primal = Enzyme.EnzymeRules.needs_primal(config) ? val : nothing
    return Enzyme.EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(
        config::Enzyme.EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(cfunc_target)},
        ::Active,
        tape,
        x::Active,
    )
    return (1.0,)
end

@testset "cfunction from custom rule (#3284)" begin
    # native execution, so the runtime converter caches the native target
    @test run_cfunction_3284(2.0) == 3.0
    # the rule executes the GPUCompiler-compiled copy of run_cfunction_3284
    @test Enzyme.autodiff(Reverse, cfunc_target, Active(2.0))[1][1] == 1.0
end

@testset "jl_f_current_scope" begin
    function foo_logger(p)
        sol = Logging.with_logger(Logging.current_logger()) do
            p[1] * p[2]
        end
        return sol
    end
    g = Enzyme.gradient(set_runtime_activity(Reverse), foo_logger, [2.0, 3.0])[1]
    @test g ≈ [3.0, 2.0]
end

