using Random

function EnzymeRules.inactive(::typeof(Base.CoreLogging.logmsg_code), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.shouldlog), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.current_logger), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.current_logger_for_env), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.fixup_stdlib_path), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.handle_message), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.CoreLogging.logging_error), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.to_tuple_type), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.println), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.print), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.show), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.flush), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.string), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.repr), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.print_to_string), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.Threads.threadid), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.Threads.nthreads), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.eps), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.nextfloat), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.prevfloat), args...)
    return nothing
end
function EnzymeRules.inactive(::Type{Base.Val}, args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Core.kwfunc), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.rand), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.rand!), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.randn), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.default_rng), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Random.seed!), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.thisind), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.nextind), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(LLVM.ThreadSafeContext))
    nothing
end
function EnzymeRules.inactive(::typeof(LLVM.context), ::LLVM.ThreadSafeContext)
    nothing
end
function EnzymeRules.inactive(::typeof(LLVM.activate), ::LLVM.Context)
    nothing
end
function EnzymeRules.inactive(::typeof(LLVM.dispose), ::LLVM.ThreadSafeContext)
    nothing
end
function EnzymeRules.inactive(::typeof(LLVM.deactivate), ::LLVM.Context)
    nothing
end