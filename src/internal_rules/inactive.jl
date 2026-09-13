@static if VERSION >= v"1.12"
function EnzymeRules.inactive(::typeof(Base.CoreLogging.handle_message_nothrow), args...)
    return nothing
end
function EnzymeRules.inactive(::typeof(LinearAlgebra.norm_recursive_check), args...)
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

#=
An atomic over an integer never carries a derivative, so the read-modify-write
has nothing to differentiate. Saying so explicitly also keeps Enzyme away from
how these lower: since Julia 1.13 `Threads.atomic_add!` and friends are
`(@atomic :acquire_release x.value + v).first` rather than an `llvmcall`, which
emits the variadic `julia.atomicmodify` intrinsic. Enzyme's C++ side treats that
declaration as an ordinary subfunction to recurse into and aborts, because a
variadic callee's `arg_size()` does not match the call's operand count:

    Assertion `nowrite_shadows.size() == todiff->arg_size()' failed.

Marking the wrappers inactive blocks the inlining that would otherwise bury the
intrinsic in a caller, so Enzyme never sees it. GPUArrays reaches this through
`DataRef` refcounting: `copy(::DataRef)` retains, which bumps a
`Threads.Atomic{Int}`.

Atomics over floating-point values are deliberately not listed: those do
accumulate derivatives and need real handling, not an inactive marker.
=#
for op in (
        :atomic_add!, :atomic_sub!, :atomic_and!, :atomic_or!,
        :atomic_xor!, :atomic_nand!, :atomic_max!, :atomic_min!,
    )
    @eval function EnzymeRules.inactive(
            ::typeof(Base.Threads.$op), ::Base.Threads.Atomic{<:Integer}, ::Integer
        )
        return nothing
    end
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
function EnzymeRules.inactive_noinl(::typeof(Base.ht_keyindex), args...)
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
function EnzymeRules.inactive_noinl(::typeof(Base.time_ns), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.time), args...)
    return nothing
end
import Printf as _EnzymePrintf
function EnzymeRules.inactive_noinl(::typeof(_EnzymePrintf.format), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.GC.enable), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.mightalias), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base._parentsmatch), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.dataids), args...)
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.signature_type), args...) 
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.methods), args...) 
    return nothing
end
function EnzymeRules.inactive_noinl(::typeof(Base.fieldnames), args...) 
    return nothing
end

# Querying a lock never mutates it and never carries derivative information.
# `lock`, `unlock` and `trylock` are not inactive: the reverse sweep of a
# locked region must itself hold the lock, so they get proper rules in
# core.jl that run `unlock` as the adjoint of `lock` and vice versa.
function EnzymeRules.inactive_noinl(::typeof(Base.islocked), ::Base.AbstractLock)
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

