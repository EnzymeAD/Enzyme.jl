
function reinsert_gcmarker_pass!(fn::LLVM.Function)
    reinsert_gcmarker!(fn)
    unique_gcmarker!(fn)
    return true
end
