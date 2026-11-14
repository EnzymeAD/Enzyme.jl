import EnzymeCore: Annotation
import EnzymeCore.EnzymeRules: FwdConfig, RevConfig, forward, augmented_primal, inactive, _annotate_tt

function has_frule_from_sig(
        @nospecialize(interp::Core.Compiler.AbstractInterpreter),
        @nospecialize(TT::Type), sv::Core.Compiler.AbsIntState, partialedge::Bool = true
    )::Bool
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:FwdConfig, <:Annotation{ft}, Type{<:Annotation}, tt...}
    fwd_sig = Tuple{typeof(EnzymeRules.forward), <:EnzymeRules.FwdConfig, <:Enzyme.EnzymeCore.Annotation, Type{<:Enzyme.EnzymeCore.Annotation}, Vararg{Enzyme.EnzymeCore.Annotation}}
    return isapplicable(interp, forward, TT, sv, fwd_sig)
end

function has_rrule_from_sig(
        @nospecialize(interp::Core.Compiler.AbstractInterpreter),
        @nospecialize(TT::Type), sv::Core.Compiler.AbsIntState, partialedge::Bool = true
    )::Bool
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:RevConfig, <:Annotation{ft}, Type{<:Annotation}, tt...}
    rev_sig = Tuple{typeof(EnzymeRules.augmented_primal), <:EnzymeRules.RevConfig, <:Enzyme.EnzymeCore.Annotation, Type{<:Enzyme.EnzymeCore.Annotation}, Vararg{Enzyme.EnzymeCore.Annotation}}
    return isapplicable(interp, augmented_primal, TT, sv, rev_sig)
end


function is_inactive_from_sig(
        @nospecialize(interp::Core.Compiler.AbstractInterpreter),
        @nospecialize(TT::Type), sv::Core.Compiler.AbsIntState
    )
    return isapplicable(interp, inactive, TT, sv, nothing)
end

# `hasmethod` is a precise match using `Core.Compiler.findsup`,
# but here we want the broader query using `Core.Compiler.findall`.
# Also add appropriate backedges to the caller `MethodInstance` if given.
function isapplicable(
        @nospecialize(interp::Core.Compiler.AbstractInterpreter),
        @nospecialize(f), @nospecialize(TT::Type), sv::Core.Compiler.AbsIntState, @nospecialize(partialsig::Union{Type, Nothing})
    )::Bool
    tt = Base.to_tuple_type(TT)
    sig = Base.signature_type(f, tt)
    mt = ccall(:jl_method_table_for, Any, (Any,), sig)
    mt isa Core.MethodTable || return false
    result = Core.Compiler.findall(sig, Core.Compiler.method_table(interp); limit = -1)
    (result === nothing || result === missing) && return false
    @static if isdefined(Core.Compiler, :MethodMatchResult)
        (; matches) = result
    else
        matches = result
    end
    # also need an edge to the method table in case something gets
    # added that did not intersect with any existing method
    # fullmatch = Core.Compiler._any(match::Core.MethodMatch -> match.fully_covers, matches)
    # if !fullmatch
    if true
        if partialsig === nothing
            Core.Compiler.add_mt_backedge!(sv, mt, sig)
        else
            pmt = ccall(:jl_method_table_for, Any, (Any,), partialsig)
            Core.Compiler.add_mt_backedge!(sv, pmt, partialsig)
        end
    end
    if Core.Compiler.isempty(matches)
        return false
    else
        #for i = 1:Core.Compiler.length(matches)
        #    match = Core.Compiler.getindex(matches, i)::Core.MethodMatch
        #    edge = Core.Compiler.specialize_method(match)::Core.MethodInstance
        #    Core.Compiler.add_backedge!(sv, edge)
        #end
        return true
    end
end
