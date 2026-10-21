"""
    FunctionSummary

A summary of information needed for Enzyme analyses. This includes activity information
(`flow`), alias information (`points_to`), effects information (`reads_arg`, `writes_arg`,
`writes_arg_any`, `escapes_arg`, `globals_read`, `globals_write`, `globals_write_any`)

Using the indices of the LLVM function's parameters (not of the Julia arguments; ghost
arguments have none, and the `swiftself` task pointer, `sret` and return roots have one
each). For `n = nargs`:

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
    globals_read::Set{String}
    globals_write::Set{String}
    globals_write_any::Set{String}
    flags::UInt32
end

function summary_globals(ref::API.EnzymeFunctionSummaryRef, kind::API.CSummaryGlobals)
    n = Int(API.EnzymeFunctionSummaryNumGlobals(ref, kind))
    names = Set{String}()
    sizehint!(names, n)
    for i in 1:n
        push!(names, Base.unsafe_string(API.EnzymeFunctionSummaryGlobal(ref, kind, i - 1)))
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

# Where summaries live: on GPUCompiler 2.x (Julia 1.11+), in the results CompilerCaching
# attaches to each of Enzyme's `CodeInstance`s, so they are serialized into package images
# along with them; otherwise in a cache for this session.
const SUMMARY_ON_CODE_INSTANCE = HAS_GPUCOMPILER_2 && VERSION >= v"1.11.0-DEV.1552"

"""
    SummarizeFunctions[]

Compute a [`FunctionSummary`](@ref) for each `CodeInstance` emitted for differentiation and
cache it (see [`function_summary`](@ref)). On by default with GPUCompiler 2.x, where the
summary is stored on the `CodeInstance`; off by default otherwise.
"""
const SummarizeFunctions = Ref(SUMMARY_ON_CODE_INSTANCE)

const FUNCTION_SUMMARIES_LOCK = ReentrantLock()

@static if SUMMARY_ON_CODE_INSTANCE
    @doc """
        FunctionSummaryResults

    The results CompilerCaching attaches to one of Enzyme's `CodeInstance`s: its
    [`FunctionSummary`](@ref), once computed.
    """
    mutable struct FunctionSummaryResults
        summary::Union{Nothing, FunctionSummary}
        FunctionSummaryResults() = new(nothing)
    end

    # Results are only attached to the `CodeInstance`s Enzyme's interpreter owns.
    is_enzyme_code_instance(ci::Core.CodeInstance) = ci.owner isa EnzymeCacheToken

    @doc """
        function_summary(ci::Core.CodeInstance)

    The [`FunctionSummary`](@ref) stored on `ci`, or `nothing`.
    """
    function function_summary(ci::Core.CodeInstance)
        is_enzyme_code_instance(ci) || return nothing
        res = GPUCompiler.CompilerCaching.results(FunctionSummaryResults, ci)
        return @lock FUNCTION_SUMMARIES_LOCK res.summary
    end

    needs_summary(ci::Core.CodeInstance) =
        is_enzyme_code_instance(ci) && function_summary(ci) === nothing

    function store_summary!(ci::Core.CodeInstance, summary::FunctionSummary)
        res = GPUCompiler.CompilerCaching.results(FunctionSummaryResults, ci)
        @lock FUNCTION_SUMMARIES_LOCK begin
            res.summary === nothing && (res.summary = summary)
        end
        return nothing
    end
else
    const FUNCTION_SUMMARIES = IdDict{Core.CodeInstance, FunctionSummary}()

    @doc """
        function_summary(ci::Core.CodeInstance)

    The cached [`FunctionSummary`](@ref) of `ci`, or `nothing`.
    """
    function function_summary(ci::Core.CodeInstance)
        return @lock FUNCTION_SUMMARIES_LOCK get(FUNCTION_SUMMARIES, ci, nothing)
    end

    @doc """
        function_summaries()

    A snapshot of the cache: the `CodeInstance`s summarized so far and their summaries.
    """
    function function_summaries()
        return @lock FUNCTION_SUMMARIES_LOCK collect(FUNCTION_SUMMARIES)
    end

    needs_summary(ci::Core.CodeInstance) =
        !(@lock FUNCTION_SUMMARIES_LOCK haskey(FUNCTION_SUMMARIES, ci))

    function store_summary!(ci::Core.CodeInstance, summary::FunctionSummary)
        @lock FUNCTION_SUMMARIES_LOCK get!(FUNCTION_SUMMARIES, ci, summary)
        return nothing
    end
end

"""
    summarize_code_instances!(mod::LLVM.Module, compiled)

Summarize the function emitted into `mod` for each `CodeInstance` of `compiled` (GPUCompiler's
`meta.compiled`) that has no summary yet, if [`SummarizeFunctions`](@ref) is set and the
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
        needs_summary(ci) || continue
        specfunc = k.specfunc
        specfunc === nothing && continue
        fname = GPUCompiler.safe_name(specfunc)
        haskey(fns, fname) || continue
        f = fns[fname]
        isempty(blocks(f)) && continue
        store_summary!(ci, FunctionSummary(f))
    end
    return nothing
end
