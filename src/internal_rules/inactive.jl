@static if VERSION >= v"1.12"
function EnzymeRules.inactive(::typeof(Base.CoreLogging.handle_message_nothrow), args...)
    return nothing
end
end
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
function EnzymeRules.inactive(::typeof(Base.CoreLogging.handle_message), args...; kwargs...)
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
function EnzymeRules.inactive(::typeof(Base.thisind), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.nextind), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Core.Compiler.return_type), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.Broadcast.combine_eltypes), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.typejoin), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.size), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.hash), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(
    ::typeof(Base.setindex!),
    ::IdDict{K,V},
    ::K,
    ::V,
) where {K,V<:Integer}
    return nothing
end

function EnzymeRules.inactive_noinl(::typeof(Base.hasproperty), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.startswith), ::AbstractString, args...)
    return nothing
end

Enzyme.EnzymeRules.inactive_noinl(::typeof(Core._compute_sparams), args...) = nothing
function EnzymeRules.inactive(::typeof(Base.time_ns), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.time), args...)
    return nothing
end
import Printf as _EnzymePrintf
function EnzymeRules.inactive(::typeof(_EnzymePrintf.format), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(Base.GC.enable), args...)
    return nothing
end

@inline EnzymeRules.inactive_type(v::Type{Nothing}) = true
@inline EnzymeRules.inactive_type(v::Type{Union{}}) = true
@inline EnzymeRules.inactive_type(v::Type{Char}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Integer} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:DataType} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:Module} = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:AbstractString} = true
@inline EnzymeRules.inactive_type(v::Type{Core.MethodMatch}) = true
@inline EnzymeRules.inactive_type(v::Type{Core.Compiler.WorldRange}) = true
@inline EnzymeRules.inactive_type(v::Type{Core.MethodInstance}) = true
@inline EnzymeRules.inactive_type(v::Type{T}) where {T<:IO} = true

