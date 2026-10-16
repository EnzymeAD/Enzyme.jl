using Enzyme, Test
using Enzyme: API
using Enzyme.Compiler: SUMMARY_ON_CODE_INSTANCE, SummarizeFunctions, function_summary,
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

# The summary of a `CodeInstance` of `f` for the signature `T`, and that instance.
function cached_summary(f, @nospecialize(T))
    @static if SUMMARY_ON_CODE_INSTANCE
        for m in methods(f), mi in Base.specializations(m)
            mi.specTypes == T || continue
            ci = isdefined(mi, :cache) ? mi.cache : nothing
            while ci !== nothing
                s = function_summary(ci)
                s === nothing || return ci, s
                ci = isdefined(ci, :next) ? ci.next : nothing
            end
        end
    else
        for (ci, s) in Enzyme.Compiler.function_summaries()
            ci_mi(ci).specTypes == T && return ci, s
        end
    end
    return nothing
end

# A package whose precompilation differentiates a function, so that the `CodeInstance`s of
# its callee, with their summaries, are serialized into its package image.
const SUMMARY_PACKAGE = """
module SummaryPrecompile
using Enzyme

@noinline function callee!(y, x)
    @inbounds y[1] = x[1] * 2.0
    return nothing
end
function caller(x)
    y = [0.0]
    callee!(y, x)
    return @inbounds y[1]
end

Enzyme.autodiff(Reverse, caller, Active, Duplicated([3.0], [0.0]))
end
"""

const SUMMARY_LOAD = """
using SummaryPrecompile, Enzyme
f = SummaryPrecompile.callee!
T = Tuple{typeof(f), Vector{Float64}, Vector{Float64}}
found = false
for m in methods(f), mi in Base.specializations(m)
    mi.specTypes == T || continue
    ci = isdefined(mi, :cache) ? mi.cache : nothing
    while ci !== nothing
        s = Enzyme.Compiler.function_summary(ci)
        if s !== nothing
            global found = s.writes_arg[s.nargs - 1] && s.flow[s.nargs, s.nargs - 1]
        end
        ci = isdefined(ci, :next) ? ci.next : nothing
    end
end
print(Base.isprecompiled(Base.PkgId(Base.UUID("2a51e3b6-6e3b-4a4b-9d0c-5f3b6a1c2d4e"), "SummaryPrecompile")), " ", found)
"""

@testset "Function summaries" begin
    if !API.has_function_summary()
        @info "libEnzyme has no function summaries, skipping"
        @test_skip false
    else
        @test SummarizeFunctions[] == SUMMARY_ON_CODE_INSTANCE
        enabled = SummarizeFunctions[]

        SummarizeFunctions[] = false
        try
            dx = [0.0]
            autodiff(Reverse, summary_off_caller, Active, Duplicated([3.0], dx))
            @test dx == [3.0]
        finally
            SummarizeFunctions[] = enabled
        end
        @test cached_summary(summary_off_callee!, Tuple{typeof(summary_off_callee!), Vector{Float64}, Vector{Float64}}) === nothing

        SummarizeFunctions[] = true
        try
            dx = [0.0]
            autodiff(Reverse, summary_caller, Active, Duplicated([3.0], dx))
            @test dx == [2.0]
        finally
            SummarizeFunctions[] = enabled
        end

        found = cached_summary(summary_callee!, Tuple{typeof(summary_callee!), Vector{Float64}, Vector{Float64}})
        @test found !== nothing
        ci, s = found
        @test function_summary(ci) === s
        @static if SUMMARY_ON_CODE_INSTANCE
            @test ci.owner isa Enzyme.Compiler.EnzymeCacheToken
        end

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
        @test isempty(s.globals_write)

        @static if SUMMARY_ON_CODE_INSTANCE
            # the summaries survive serialization into a package image
            mktempdir() do dir
                pkg = joinpath(dir, "SummaryPrecompile")
                mkpath(joinpath(pkg, "src"))
                write(
                    joinpath(pkg, "Project.toml"),
                    """
                    name = "SummaryPrecompile"
                    uuid = "2a51e3b6-6e3b-4a4b-9d0c-5f3b6a1c2d4e"

                    [deps]
                    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
                    """
                )
                write(joinpath(pkg, "src", "SummaryPrecompile.jl"), SUMMARY_PACKAGE)
                # after the active environment, where its dependencies are found
                load_path = join([Base.LOAD_PATH..., dir], Sys.iswindows() ? ';' : ':')
                project = Base.active_project()
                cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $SUMMARY_LOAD`
                out = withenv("JULIA_LOAD_PATH" => load_path) do
                    read(cmd, String)
                end
                @test out == "true true"
            end
        end
    end
end
