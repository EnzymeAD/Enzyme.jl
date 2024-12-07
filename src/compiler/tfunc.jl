import EnzymeCore: Annotation
import EnzymeCore.EnzymeRules: FwdConfig, RevConfig, forward, augmented_primal, inactive, _annotate_tt

function add_backedge!(caller::Core.MethodInstance, callee::Core.MethodInstance, @nospecialize(sig))
    ccall(:jl_method_instance_add_backedge, Cvoid, (Any, Any, Any), callee, sig, caller)
    return nothing
end

function add_mt_backedge!(caller::Core.MethodInstance, mt::Core.MethodTable, @nospecialize(sig))
    ccall(:jl_method_table_add_backedge, Cvoid, (Any, Any, Any), mt, sig, caller)
    return nothing
end


function has_frule_from_sig(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(TT), caller::MethodInstance)::Bool
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:FwdConfig,<:Annotation{ft},Type{<:Annotation},tt...}
    return isapplicable(interp, forward, TT, caller)
end

function has_rrule_from_sig(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(TT), caller::MethodInstance)::Bool
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:RevConfig,<:Annotation{ft},Type{<:Annotation},tt...}
    return isapplicable(interp, augmented_primal, TT, caller)
end


function is_inactive_from_sig(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(TT), caller::MethodInstance)
    return isapplicable(interp, inactive, TT, caller)
end

# `hasmethod` is a precise match using `Core.Compiler.findsup`,
# but here we want the broader query using `Core.Compiler.findall`.
# Also add appropriate backedges to the caller `MethodInstance` if given.
function isapplicable(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(f), @nospecialize(TT), caller::MethodInstance)::Bool
    tt = Base.to_tuple_type(TT)
    sig = Base.signature_type(f, tt)
    mt = ccall(:jl_method_table_for, Any, (Any,), sig)
    mt isa Core.MethodTable || return false
    result = Core.Compiler.findall(sig, Core.Compiler.method_table(interp); limit=-1)
    (result === nothing || result === missing) && return false
    @static if isdefined(Core.Compiler, :MethodMatchResult)
        (; matches) = result
    else
        matches = result
    end
    # also need an edge to the method table in case something gets
    # added that did not intersect with any existing method
    fullmatch = Core.Compiler._any(match::Core.MethodMatch -> match.fully_covers, matches)
    if !fullmatch
        add_mt_backedge!(caller, mt, sig)
    end
    if Core.Compiler.isempty(matches)
        return false
    else
        for i = 1:Core.Compiler.length(matches)
            match = Core.Compiler.getindex(matches, i)::Core.MethodMatch
            edge = Core.Compiler.specialize_method(match)::Core.MethodInstance
            add_backedge!(caller, edge, sig)
        end
        return true
    end
end