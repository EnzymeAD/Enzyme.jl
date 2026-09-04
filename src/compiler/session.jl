# Session-scoped compilation state.
#
# Everything here is valid only within the current Julia process: JIT-linked `CompileResult`s
# hold the links of thunk entry points into Enzyme's ORC JIT, and `tapes` memoizes tape types
# per compiler job. None of it may survive into a package
# image, so `reset_session!` runs from `__init__` and at the end of the precompile workload.
# Module-level caches that describe the current process belong here rather than in their own
# `const Dict`s.
# The result of compiling one differentiation job: the entry points (thunk handles for
# `FFIABI`, inline modules for `InlineABI`), the tape type and the invalidation edges.
struct CompileResult{AT, PT}
    adjoint::AT
    primal::PT
    TapeType::Type
    edges::Vector{Any}
end

# The session-portable product of differentiating one job: the post-optimized module as
# bitcode, the entry symbols, the tape type, the invalidation edges, and (for nested
# differentiation) the pre-optimization module string. Linking it (`link_artifact!`) needs
# no compiler, only the JIT, so a fresh session can start a precompiled thunk from this
# alone. `nothing` for the `InlineABI` mode, which carries its module in the thunk value.
mutable struct ThunkArtifact
    const bitcode::Vector{UInt8}
    const adjoint_name::String
    const primal_name::Union{Nothing, String}
    const tape_type::Type
    const edges::Vector{Any}
    const prepost::String
    # The Julia values the bitcode refers to symbolically (name => object), re-registered
    # when the artifact is linked in another session (see `link_artifact!`).
    const manifest::Vector{Pair{String, Any}}
    # Whether the artifact is valid in another process: the manifest is reconstructible and
    # no address of this session was baked into the bitcode.
    const persistable::Bool
end

# The compilation results attached to a primal-partition `CodeInstance`: one artifact per
# codegen configuration that reused that inference (keyed by `config_key`). On 1.11+ this
# rides with the `CodeInstance`, including into a package image; a zero-arg constructor is
# required for `CompilerCaching.results`.
mutable struct EnzymeResults
    const entries::Vector{Pair{UInt, ThunkArtifact}}
    EnzymeResults() = new(Pair{UInt, ThunkArtifact}[])
end

# Counts calls to `emit` (a run of enzyme-core); tests assert a reloaded thunk does not
# bump it.
const EMIT_COUNT = Ref(0)

# A compiled thunk entry point as linked into this session's JIT.
struct LinkedThunk
    ptr::Ptr{Cvoid}
    epoch::UInt64
    # The symbol of the entry point and the module it was linked from (before
    # post-optimization), which nested differentiation splices into the outer module.
    name::String
    modstr::String
end

# What a thunk object holds in place of a function pointer: what it takes to compile (or
# find) the entry point in any session, and the link of this session once it is known. A
# thunk object is embedded as a constant in the generated `thunk` method's code, so it can
# be loaded from a package image; `thunk_pointer` then compiles and links on first use.
mutable struct ThunkHandle
    const mi::Core.MethodInstance
    const config::GPUCompiler.CompilerConfig
    const which::Symbol   # :adjoint or :primal
    @atomic linked::Union{Nothing, LinkedThunk}
end

ThunkHandle(mi::Core.MethodInstance, config::GPUCompiler.CompilerConfig, which::Symbol) =
    ThunkHandle(mi, config, which, nothing)

struct ThunkCache
    thunks::Dict{UInt, CompileResult}
    # Links made while generating a package image; they must not be stored on the handles,
    # which are serialized with the image.
    session_links::IdDict{ThunkHandle, LinkedThunk}
    tapes::Dict{UInt, Type}
    lock::ReentrantLock
end

function ThunkCache()
    return ThunkCache(
        Dict{UInt, CompileResult}(),
        IdDict{ThunkHandle, LinkedThunk}(),
        Dict{UInt, Type}(),
        ReentrantLock(),
    )
end

const THUNK_CACHE = ThunkCache()

# Identifies the current process to state that caches pointers: a value stamped with a
# different epoch was produced in another session (or before `reset_session!`) and must be
# re-linked. Always odd, so a zero-initialized field never matches.
const SESSION_EPOCH = Ref{UInt64}(0)

"""
    reset_session!()

Drop every process-local compilation result and start a new session epoch. Called from
`__init__` and after the precompile workload so that no JIT pointer reaches a package image.
"""
function reset_session!()
    lock(THUNK_CACHE.lock)
    try
        empty!(THUNK_CACHE.thunks)
        empty!(THUNK_CACHE.session_links)
        empty!(THUNK_CACHE.tapes)
    finally
        unlock(THUNK_CACHE.lock)
    end
    SESSION_EPOCH[] = hash(time_ns(), UInt64(getpid())) | 0x01
    return nothing
end
