
function reinsert_gcmarker_pass!(fn::LLVM.Function)
    # the pgcstack of a cfunction thunk is only valid after
    # its adopt-thread check, so it must not be hoisted to the entry.
    if is_cfunc_wrapper(fn)
        return false
    end
    reinsert_gcmarker!(fn)
    unique_gcmarker!(fn)
    return true
end
