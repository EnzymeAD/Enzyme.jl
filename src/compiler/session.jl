# Session-scoped compilation state.
#
# Everything here is valid only within the current Julia process: JIT-linked `CompileResult`s
# hold function pointers into Enzyme's ORC JIT, `by_ptr` maps those pointers back to the module
# they were compiled from (used by `check_ir` to differentiate through a previously compiled
# thunk), and `tapes` memoizes tape types per compiler job. None of it may survive into a package
# image, so `reset_session!` runs from `__init__` and at the end of the precompile workload.
# Module-level caches that describe the current process belong here rather than in their own
# `const Dict`s.
struct ThunkCache
    thunks::Dict{UInt, CompileResult}
    by_ptr::Dict{Ptr{Cvoid}, Tuple{String, String}}
    tapes::Dict{UInt, Type}
    lock::ReentrantLock
end

function ThunkCache()
    return ThunkCache(
        Dict{UInt, CompileResult}(),
        Dict{Ptr{Cvoid}, Tuple{String, String}}(),
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
        empty!(THUNK_CACHE.by_ptr)
        empty!(THUNK_CACHE.tapes)
    finally
        unlock(THUNK_CACHE.lock)
    end
    SESSION_EPOCH[] = hash(time_ns(), UInt64(getpid())) | 0x01
    return nothing
end
