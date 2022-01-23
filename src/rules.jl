module EnzymeRules

import ..Enzyme
import Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, Annotation

function forward end

"""
	augmented_primal(::typeof(f), args...)

Return the primal computation value and a tape
"""
function augmented_primal end

"""
Takes gradient of derivative, activity annotation, and tape
"""
function reverse end

import Core.Compiler: argtypes_to_type
function has_frule(@nospecialize(TT), world=Base.get_world_counter())
    atype = Tuple{typeof(EnzymeRules.forward), Type{TT}, Type, Vector{Type}}

    if VERSION < v"1.8.0-"
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), atype, world)
    else
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, Any, UInt), atype, nothing, world)
    end

    return res !== nothing
end

function has_rrule(@nospecialize(TT), world=Base.get_world_counter())
    atype = Tuple{typeof(EnzymeRules.reverse), Type{TT}, Type, Vector{Type}, Bool, Bool, UInt64, Vector{Bool}}
    
    if VERSION < v"1.8.0-"
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), atype, world)
    else
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, Any, UInt), atype, nothing, world)
    end

    return res !== nothing
end

end
