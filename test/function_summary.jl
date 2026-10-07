using Enzyme, Test
using Enzyme: API
using Enzyme.Compiler: SummarizeFunctions, function_summaries, function_summary,
    has_unknown_effects, may_flow_to_arg, may_write_arg, may_write_global

@noinline function summary_callee!(y, x)
    @inbounds y[1] = x[1] * 2.0
    return nothing
end
function summary_caller(x)
    y = [0.0]
    summary_callee!(y, x)
    return @inbounds y[1]
end

@noinline function summary_off_callee!(y, x)
    @inbounds y[1] = x[1] * 3.0
    return nothing
end
function summary_off_caller(x)
    y = [0.0]
    summary_off_callee!(y, x)
    return @inbounds y[1]
end

ci_mi(ci::Core.CodeInstance) =
    isdefined(Core.Compiler, :get_ci_mi) ? Core.Compiler.get_ci_mi(ci) : ci.def

function cached_summary(@nospecialize(T))
    for (ci, s) in function_summaries()
        ci_mi(ci).specTypes == T && return ci, s
    end
    return nothing
end

@testset "Function summaries" begin
    if !API.has_function_summary()
        @info "libEnzyme has no function summaries, skipping"
        @test_skip false
    else
        # off by default
        dx = [0.0]
        autodiff(Reverse, summary_off_caller, Active, Duplicated([3.0], dx))
        @test dx == [3.0]
        @test cached_summary(Tuple{typeof(summary_off_callee!), Vector{Float64}, Vector{Float64}}) === nothing

        SummarizeFunctions[] = true
        try
            dx = [0.0]
            autodiff(Reverse, summary_caller, Active, Duplicated([3.0], dx))
            @test dx == [2.0]
        finally
            SummarizeFunctions[] = false
        end

        found = cached_summary(Tuple{typeof(summary_callee!), Vector{Float64}, Vector{Float64}})
        @test found !== nothing
        ci, s = found
        @test function_summary(ci) === s

        # y and x are the last two LLVM parameters
        y = s.nargs - 1
        x = s.nargs
        @test size(s.flow) == (s.nargs + 1, s.nargs + 2)
        @test may_write_arg(s, y)
        @test !may_write_arg(s, x)
        @test s.reads_arg[x]
        @test may_flow_to_arg(s, x, y)
        @test !may_flow_to_arg(s, y, x)
        @test !any(s.escapes_arg)
        @test !has_unknown_effects(s)
        @test !may_write_global(s)
    end
end
