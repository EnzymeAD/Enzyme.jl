# Base specializes foreach on Tuples, but NamedTuples take its generic loop.
# That loop joins boxed scalar and array fields before AD's type analysis.
# Visit each fixed field with its own type, preserving field order,
# shortest-iterator termination, and the original field values. Do not call
# foreach on converted operands: that could select a user Tuple-only overload.
@inline function foreach_records(f, records::Tuple)
    any(isempty, records) && return nothing
    f(map(first, records)...)
    return foreach_records(f, map(Base.tail, records))
end

@inline function override_foreach(f, records...)
    return foreach_records(f, map(Tuple, records))
end

const FOREACH_SINGLE_METHOD = which(Base.foreach, Tuple{Any, Any})
const FOREACH_MULTIPLE_METHOD = which(Base.foreach, Tuple{Any, Any, Any})

foreach_rewrite(interp::EnzymeInterpreter, f, argtypes, sv::AbsIntState) = nothing

function foreach_rewrite(interp::EnzymeInterpreter, f::typeof(Base.foreach), argtypes, sv::AbsIntState)
    length(argtypes) >= 3 || return nothing
    sig = Core.Compiler.argtypes_to_type(argtypes)
    Base.isdispatchtuple(sig) || return nothing
    named = false
    for T in sig.parameters[3:end]
        T <: Union{Tuple, NamedTuple} && isconcretetype(T) || return nothing
        named |= T <: NamedTuple
    end
    named || return nothing
    method = length(argtypes) == 3 ? FOREACH_SINGLE_METHOD : FOREACH_MULTIPLE_METHOD
    return checked_method_rewrite(interp, f, sig, sv, (method, override_foreach))
end
