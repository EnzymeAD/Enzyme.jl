# This file defines JET analysis to check Enzyme's auto-differentiability.
# In particularity, the analysis will look for:
# - any active `Union`-typed result
# - (unimplemented yet) dynamic dispatch
# - and more...?

# exports
# -------

export report_enzyme, @report_enzyme, test_enzyme, @test_enzyme

# analyzer
# --------

using JET.JETInterface

struct EnzymeAnalyzer <: AbstractAnalyzer
    state::AnalyzerState
    cache_key::UInt
end
function EnzymeAnalyzer(; jetconfigs...)
    return EnzymeAnalyzer(
        AnalyzerState(; jetconfigs...),
        __ANALYZER_CACHE_KEY[] += 1,
        )
end
const __ANALYZER_CACHE_KEY = Ref{UInt}(0)

JETInterface.AnalyzerState(analyzer::EnzymeAnalyzer) = analyzer.state
function JETInterface.AbstractAnalyzer(analyzer::EnzymeAnalyzer, state::AnalyzerState)
    return EnzymeAnalyzer(
        state,
        analyzer.cache_key,
        )
end
JETInterface.ReportPass(analyzer::EnzymeAnalyzer) = EnzymeAnalysisPass()
JETInterface.get_cache_key(analyzer::EnzymeAnalyzer) = analyzer.cache_key

function Core.Compiler.finish(frame::Core.Compiler.InferenceState, analyzer::EnzymeAnalyzer)
    ReportPass(analyzer)(UnionResultReport, analyzer, frame)
    return Base.@invoke Core.Compiler.finish(frame::Core.Compiler.InferenceState, analyzer::AbstractAnalyzer)
end

struct EnzymeAnalysisPass <: ReportPass end

@reportdef struct UnionResultReport <: InferenceErrorReport
    @nospecialize(tt) # ::Type
    @nospecialize(rt) # ::Type
end
JETInterface.get_msg(::Type{UnionResultReport}, @nospecialize(args...)) = "potentially active Union result detected"
function JETInterface.print_error_report(io, report::UnionResultReport)
    Base.@invoke JETInterface.print_error_report(io, report::InferenceErrorReport)
    printstyled(io, "::", report.rt; color = :cyan)
end

# check if this return value is used in the caller and its type is inferred as Union:
# XXX `Core.Compiler.call_result_unused` is a very inaccurate model of Enzyme's activity analysis,
# and so this active Union return check might be very incomplete
function (::EnzymeAnalysisPass)(::Type{UnionResultReport}, analyzer::EnzymeAnalyzer, frame::Core.Compiler.InferenceState)
    parent = frame.parent
    if isa(parent, Core.Compiler.InferenceState) && !(Core.Compiler.call_result_unused(parent))
        rt = frame.bestguess
        if isa(rt, Union)
            add_new_report!(analyzer, frame.result, UnionResultReport(frame.linfo, frame.linfo.specTypes, rt))
            return true
        end
    end
    return false
end

# entry
# -----

import JET: report_call, test_call, get_reports

analyze_autodiff_call(entry, f, ::Type{<:Annotation}, args...) = (@nospecialize; _analyze_autodiff_call(entry, f, args...))
analyze_autodiff_call(entry, f, args...) = (@nospecialize; _analyze_autodiff_call(entry, f, args...))
function _analyze_autodiff_call(entry, f, args...)
    @nospecialize f args
    args′ = annotate(args...)
    tt = getargtypes(args′)
    return entry(f, tt; analyzer=EnzymeAnalyzer)
end

function apply_autodiff_args(f, @nospecialize(ex))
    if Meta.isexpr(ex, :do)
        dof = esc(ex.args[2])
        autodiff′, args... = map(esc, ex.args[1].args)
        return quote
            if $autodiff′ !== autodiff
                throw(ArgumentError("@$($f) expects `autodiff(...)` call expression"))
            end
            $f($dof, $(args...))
        end
    elseif !(Meta.isexpr(ex, :call) && length(ex.args) ≥ 1)
        throw(ArgumentError("@$f expects `autodiff(...)` call expression"))
    end
    autodiff′, args... = map(esc, ex.args)
    return quote
        if $autodiff′ !== autodiff
            throw(ArgumentError("@$($f) expects `autodiff(...)` call expression"))
        end
        $f($(args...))
    end
end

"""
    report_enzyme(args...) -> result::JETCallResult

Analyzes potential problems for Enzyme to auto-differentiate `args`.
`args` should be valid arguments to [`autodiff`](@ref) function, i.e.
the call `autodiff(args...)` should meet the `autodiff` interface.

In particularity, `report_enzyme` detects if there is any potentially active `Union`-typed
result, which confuses Enzymes's code generation.
If such `Union`-typed result is unused anywhere, `report_enzyme` doesn't report it as an issue,
since Enzyme can auto-differentiate it without problem.

Note that this analysis is _not_ complete in terms of covering Enzyme's auto-differentiability --
`report_enzyme` models Enzyme's activity analysis very inaccurately, meaning there may be
some code that Enzyme differentiates without any problem while `report_enzyme` raises an issue.

```julia
julia> union_result(cond, x) = cond ? x : 0
union_result (generic function with 1 method)

julia> report_enzyme(Active, true, Active(1.0)) do cond, x
           union_result(cond, x) * x
       end
═════ 1 possible error found ═════
┌ @ none:2 Main.union_result(cond, x)
│┌ @ none:1 union_result(::Bool, ::Float64)
││ potentially active Union result detected: union_result(::Bool, ::Float64)::Union{Float64, Int64}
│└──────────

julia> report_enzyme(Active, true, Active(1.0)) do cond, x
           union_result(cond, x) # inactive Union-typed result
           x * x
       end
No errors detected

julia> union_result(cond, x) = cond ? x : zero(x) # fix the Union-typed result
union_result (generic function with 1 method)

julia> report_enzyme(Active, true, Active(1.0)) do cond, x
           union_result(cond, x) * x
       end
No errors detected
```
"""
report_enzyme(args...) = (@nospecialize; analyze_autodiff_call(report_call, args...))

"""
    @report_enzyme autodiff(...)

Takes valid [`autodiff`](@ref) call expression and analyzes potential problems for Enzyme to
auto-differentiate it.

See also [`report_enzyme`](@ref).

```julia
julia> union_result(cond, x) = cond ? x : 0
union_result (generic function with 1 method)

julia> @report_enzyme autodiff(Active, true, Active(1.0)) do cond, x
           union_result(cond, x) * x
       end
═════ 1 possible error found ═════
┌ @ none:2 Main.union_result(cond, x)
│┌ @ none:1 union_result(::Bool, ::Float64)
││ potentially active Union result detected: union_result(::Bool, ::Float64)::Union{Float64, Int64}
│└──────────
end
```
"""
macro report_enzyme(ex) apply_autodiff_args(report_enzyme, ex) end

# TODO support test configurations?

"""
    test_enzyme(args...) -> JETCallResult

Tests `args` can be safely auto-differentiated by Enzyme.jl
`args` should be valid arguments to [`autodiff`](@ref) function, i.e.
the call `autodiff(args...)` should meet the `autodiff` interface.

See also [`@test_enzyme`](@ref), [`report_enzyme`](@ref).
"""
test_enzyme(args...) = (@nospecialize; analyze_autodiff_call(test_call, args...))

"""
    @test_enzyme autodiff(...)

Tests the given [`autodiff`](@ref) call can be safely auto-differentiated by Enzyme.
Returns a `Pass` result if it is, a `Fail` result if if contains any potential problems,
or an `Error` result if this macro encounters an unexpected error.
When the test `Fail`s, abstract call stack to each problem location will also be printed
to `stdout`.

See also [`report_enzyme`](@ref).
"""
macro test_enzyme(ex) apply_autodiff_args(test_enzyme, ex) end
