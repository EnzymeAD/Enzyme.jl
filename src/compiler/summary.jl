"""
    FunctionSummary

What the LLVM function emitted for a `CodeInstance` does to floating-point data, computed
from its body alone. Calls stay symbolic: what a callee does with the memory it is given is
in the callee's own summary, not in this one.

The shape follows ActivityAnalysis.jl's `ActivityDescriptor`, with indices of the LLVM
function's parameters (not of the Julia arguments; ghost arguments have none, and the
`swiftself` task pointer, `sret` and return roots have one each). For `n = nargs`:

- `flow[s, t]`: data from source `s` may reach sink `t`. Sources `1:n` are the
  parameters (their value, or memory reachable from them) and `n + 1` any global; sinks
  `1:n` are the memory reachable from each parameter, `n + 1` the return value and
  `n + 2` any global.
- `points_to[s, t]`: memory reachable from source `s` may become reachable from sink `t`
  (a pointer to it is stored there, or returned). Same shape as `flow`.
- `reads_arg[i]`, `writes_arg[i]`: floating-point data may be read from or written to the
  memory reachable from parameter `i`; `writes_arg_any[i]`: data of any type may be.
- `escapes_arg[i]`: memory reachable from parameter `i` may become reachable from
  elsewhere (a row of `points_to`).
- `globals_read`, `globals_write`, `globals_write_any`: names of the LLVM globals read or
  written; `"*"` stands for one that cannot be named (Julia objects addressed by constant)
  and for memory a callee returned.
- `flags`: `API.SUMMARY_*` bits, e.g. `API.SUMMARY_UNKNOWN` when the function has effects
  the summary cannot describe.
"""
struct FunctionSummary
    nargs::Int
    reads_arg::BitVector
    writes_arg::BitVector
    writes_arg_any::BitVector
    escapes_arg::BitVector
    flow::BitMatrix
    points_to::BitMatrix
    globals_read::Vector{String}
    globals_write::Vector{String}
    globals_write_any::Vector{String}
    flags::UInt32
end

function summary_globals(ref::API.EnzymeFunctionSummaryRef, kind::API.CSummaryGlobals)
    n = API.EnzymeFunctionSummaryNumGlobals(ref, kind)
    names = Vector{String}(undef, n)
    for i in 1:n
        names[i] = Base.unsafe_string(API.EnzymeFunctionSummaryGlobal(ref, kind, i - 1))
    end
    return names
end

# The C API lays out matrices row-major with one row per source.
function summary_matrix(bytes::Vector{UInt8}, nsources::Int, nsinks::Int)
    m = falses(nsources, nsinks)
    for s in 1:nsources, t in 1:nsinks
        m[s, t] = bytes[(s - 1) * nsinks + t] != 0
    end
    return m
end

"""
    FunctionSummary(f::LLVM.Function)

Summarize `f`, which must have a body. Requires `API.has_function_summary()`.
"""
function FunctionSummary(f::LLVM.Function)
    ref = API.EnzymeComputeFunctionSummary(f)
    try
        n = Int(API.EnzymeFunctionSummaryNumArgs(ref))
        reads = falses(n)
        writes = falses(n)
        writes_any = falses(n)
        escapes = falses(n)
        for i in 1:n
            bits = API.EnzymeFunctionSummaryArgEffects(ref, i - 1)
            reads[i] = bits & API.SUMMARY_ARG_READ_FP != 0
            writes[i] = bits & API.SUMMARY_ARG_WRITE_FP != 0
            writes_any[i] = bits & API.SUMMARY_ARG_WRITE_ANY != 0
            escapes[i] = bits & API.SUMMARY_ARG_ESCAPE != 0
        end
        bytes = Vector{UInt8}(undef, (n + 1) * (n + 2))
        API.EnzymeFunctionSummaryFlow(ref, bytes)
        flow = summary_matrix(bytes, n + 1, n + 2)
        API.EnzymeFunctionSummaryPointsTo(ref, bytes)
        points_to = summary_matrix(bytes, n + 1, n + 2)
        return FunctionSummary(
            n,
            reads,
            writes,
            writes_any,
            escapes,
            flow,
            points_to,
            summary_globals(ref, API.SUMMARY_GLOBALS_READ_FP),
            summary_globals(ref, API.SUMMARY_GLOBALS_WRITE_FP),
            summary_globals(ref, API.SUMMARY_GLOBALS_WRITE_ANY),
            API.EnzymeFunctionSummaryFlags(ref),
        )
    finally
        API.EnzymeFreeFunctionSummary(ref)
    end
end

# Queries named as ActivityAnalysis.jl's for its `ActivityDescriptor`.
may_flow_to_arg(s::FunctionSummary, i::Int, j::Int) = s.flow[i, j]
may_flow_to_return(s::FunctionSummary, i::Int) = s.flow[i, s.nargs + 1]
may_flow_to_global(s::FunctionSummary, i::Int) = s.flow[i, s.nargs + 2]
may_alias_arg(s::FunctionSummary, k::Int, j::Int) = s.points_to[k, j]
may_alias_return(s::FunctionSummary, k::Int) = s.points_to[k, s.nargs + 1]
may_write_arg(s::FunctionSummary, j::Int) = s.writes_arg[j]
may_write_global(s::FunctionSummary) =
    !isempty(s.globals_write) || s.flags & API.SUMMARY_UNKNOWN_WRITE != 0
has_unknown_effects(s::FunctionSummary) = s.flags & API.SUMMARY_UNKNOWN != 0

"""
    SummarizeFunctions[]

Compute a [`FunctionSummary`](@ref) for each `CodeInstance` emitted for differentiation and
cache it (see [`function_summary`](@ref)). Off by default.
"""
const SummarizeFunctions = Ref(false)

const FUNCTION_SUMMARIES = IdDict{Core.CodeInstance, FunctionSummary}()
const FUNCTION_SUMMARIES_LOCK = ReentrantLock()

"""
    function_summary(ci::Core.CodeInstance)

The cached [`FunctionSummary`](@ref) of `ci`, or `nothing`.
"""
function function_summary(ci::Core.CodeInstance)
    return @lock FUNCTION_SUMMARIES_LOCK get(FUNCTION_SUMMARIES, ci, nothing)
end

"""
    function_summaries()

A snapshot of the cache: the `CodeInstance`s summarized so far and their summaries.
"""
function function_summaries()
    return @lock FUNCTION_SUMMARIES_LOCK collect(FUNCTION_SUMMARIES)
end

is_summarized(ci::Core.CodeInstance) =
    @lock FUNCTION_SUMMARIES_LOCK haskey(FUNCTION_SUMMARIES, ci)

"""
    summarize_code_instances!(mod::LLVM.Module, compiled)

Summarize the function emitted into `mod` for each `CodeInstance` of `compiled` (GPUCompiler's
`meta.compiled`) that has none cached yet, if [`SummarizeFunctions`](@ref) is set and the
loaded libEnzyme can. Does not change `mod`.
"""
function summarize_code_instances!(mod::LLVM.Module, compiled)
    SummarizeFunctions[] || return nothing
    API.has_function_summary() || return nothing
    fns = functions(mod)
    for (_, k) in compiled
        haskey(k, :ci) || continue
        ci = k.ci
        ci isa Core.CodeInstance || continue
        is_summarized(ci) && continue
        specfunc = k.specfunc
        specfunc === nothing && continue
        fname = GPUCompiler.safe_name(specfunc)
        haskey(fns, fname) || continue
        f = fns[fname]
        isempty(blocks(f)) && continue
        summary = FunctionSummary(f)
        @lock FUNCTION_SUMMARIES_LOCK get!(FUNCTION_SUMMARIES, ci, summary)
    end
    return nothing
end
