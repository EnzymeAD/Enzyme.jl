# What a compiled thunk is held by, and the state of this session that links it.
#
# A thunk object holds its entry points by what it takes to find or compile them in any
# session: an entry-point `CodeInstance` on Julia 1.12+, a `ThunkHandle` on 1.10 and 1.11.
# The address the JIT gave the entry is state of this process only; it lives in the tables
# below (and, on 1.12+, in the instance's `specptr`), which `clear_caches!` drops.

# The result of compiling one differentiation job: the entry points (entry instances or
# thunk handles for `FFIABI`, inline modules for `InlineABI`), the tape type and the
# invalidation edges.
struct CompileResult{AT, PT}
    adjoint::AT
    primal::PT
    TapeType::Type
    edges::Vector{Any}
end

# What `CompilerCaching` attaches to a thunk's entry-point `CodeInstance` (Julia 1.12+). It
# rides with the instance, including into a package image; a zero-arg constructor is
# required for `CompilerCaching.results`. Nothing is attached yet: the compiled module holds
# addresses of the session that made it, so it cannot be reused by another.
mutable struct EnzymeResults
    EnzymeResults() = new()
end

# Whether a thunk's entry point lives in a `CodeInstance` (see `thunk_entry_ci!`).
const HAS_CI_THUNKS = VERSION >= v"1.12-"

# Counts calls to `_thunk` from `cached_compilation` (a run of enzyme-core); tests assert
# that a thunk linked again does not bump it needlessly.
const EMIT_COUNT = Ref(0)

# Identifies the current session to the links below: a link stamped with another epoch was
# made before `clear_caches!` and must be made again.
const SESSION_EPOCH = Ref{UInt64}(0)

# A compiled thunk entry point as linked into this session's JIT.
struct LinkedThunk
    ptr::Ptr{Cvoid}
    # On Julia 1.12+ the entry point lives in this `CodeInstance`'s `specptr` and `ptr` is a
    # copy of it; `nothing` on the versions that hold thunks by a `ThunkHandle`.
    ci::Union{Nothing, Core.CodeInstance}
    epoch::UInt64
end

# What a thunk object holds in place of a function pointer on Julia 1.10 and 1.11 (on 1.12+
# it holds its entry-point `CodeInstance`): what it takes to compile (or find) the entry
# point in any session, and the link of this session once it is known. A thunk object is
# embedded as a constant in the generated `thunk` method's code, so it can be loaded from a
# package image; `thunk_pointer` then compiles and links on first use.
mutable struct ThunkHandle
    const mi::Core.MethodInstance
    const config::GPUCompiler.CompilerConfig
    const which::Symbol   # :adjoint or :primal
    @atomic linked::Union{Nothing, LinkedThunk}
end

ThunkHandle(mi::Core.MethodInstance, config::GPUCompiler.CompilerConfig, which::Symbol) =
    ThunkHandle(mi, config, which, nothing)

# Compiled and linked thunks, by their entry-point `CodeInstance` (both instances of a thunk
# key the same result) on Julia 1.12+, and by `hash(job)` otherwise and for `InlineABI`.
const cache = IdDict{Any, CompileResult}()

# The link behind each entry-point `CodeInstance` published in this session (Julia 1.12+).
const thunk_links = IdDict{Core.CodeInstance, LinkedThunk}()

# Links made while generating a package image; they must not be stored on the handles,
# which are serialized with the image (Julia 1.10 and 1.11).
const session_links = IdDict{ThunkHandle, LinkedThunk}()

# Guards the three tables above and `autodiff_cache`.
const cache_lock = ReentrantLock()

"""
    drop_thunk_links!()

Forget every thunk this session linked: the compilation results, the links, and the IR kept
for nested differentiation. The entry points published in entry instances are withdrawn, so
the next use of a thunk links it again, and the links made before now no longer count. The
code stays in the JIT, along with the objects it roots; linking the same thunk again reuses
it. Part of `clear_caches!`.
"""
function drop_thunk_links!()
    lock(cache_lock)
    try
        @static if HAS_CI_THUNKS
            for ci in keys(thunk_links)
                withdraw_thunk_entry!(ci)
            end
        end
        empty!(cache)
        empty!(thunk_links)
        empty!(session_links)
        empty!(autodiff_cache)
        SESSION_EPOCH[] += 1
    finally
        unlock(cache_lock)
    end
    return nothing
end
