module EnzymeRules

"""
	augmented_primal(::typeof(f), args...)

Return the primal computation value and a tape
"""
function augmented_primal end

"""
    forward

Calculate the forward derivative
"""
function forward end

"""
Takes gradient of derivative, activity annotation, and tape
"""
function reverse end

import Core.Compiler: argtypes_to_type
function has_frule(@nospecialize(TT), world=Base.get_world_counter())
    atype = Tuple{typeof(forward), Type{TT}, Type, Vector{Type}}

    if VERSION < v"1.8.0-"
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), atype, world)
    else
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, Any, UInt), atype, nothing, world)
    end

    return res !== nothing
end

function has_rrule(@nospecialize(TT), world=Base.get_world_counter())
    atype = Tuple{typeof(reverse), Type{TT}, Type, Vector{Type}, Bool, Bool, UInt64, Vector{Bool}}
    
    if VERSION < v"1.8.0-"
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), atype, world)
    else
        res = ccall(:jl_gf_invoke_lookup, Any, (Any, Any, UInt), atype, nothing, world)
    end

    return res !== nothing
end

function issupported()
    @static if VERSION < v"1.7.0"
        return false
    else
        return true
    end
end

end # EnzymeRules
