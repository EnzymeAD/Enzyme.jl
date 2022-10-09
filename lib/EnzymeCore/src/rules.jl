module EnzymeRules

import EnzymeCore: Annotation

"""
	augmented_primal(::typeof(f), args...)

Return the primal computation value and a tape
"""
function augmented_primal end

"""
    forward(func::Annotation{typeof(f)}, RT::Type{<:Annotation}, args::Annotation...)

Calculate the forward derivative. The first argument `func` is the callable
for which the rule applies to. Either wrapped in a [`Const`](@ref)), or
a [`Duplicated`](@ref) if it is a closure.
The second argument is the return type annotation, and all other arguments are
the annotated function arguments.
"""
function forward end

struct Config{NeedsPrimal, NeedsShadow, Width, Overwritten} end
const ConfigWidth{Width} = Config{<:Any,<:Any, Width}

needs_primal(::Config{NeedsPrimal}) where NeedsPrimal = NeedsPrimal
needs_shadow(::Config{<:Any, NeedsShadow}) where NeedsShadow = NeedsShadow
width(::Config{<:Any, <:Any, Width}) where Width = Width
overwritten(::Config{<:Any, <:Any, <:Any, Overwritten}) where Overwritten = Overwritten

"""
	augmented_primal(::Config, func::Annotation{typeof(f)}, RT::Type{<:Annotation}, args::Annotation...)

Return the primal computation value and a tape
"""
function augmented_primal end

"""
Takes gradient of derivative, activity annotation, and tape
"""
function reverse end

_annotate(T::DataType) = TypeVar(gensym(), Annotation{T})
_annotate(::Type{T}) where T = TypeVar(gensym(), Annotation{T})
_annotate(VA::Core.TypeofVararg) = Core.TypeofVararg(_annotate(VA.T), VA.N)

function has_frule_from_sig(@nospecialize(TT); world=Base.get_world_counter())
    TT = Base.unwrap_unionall(TT)
    ft = TT.parameters[1]
    tt = map(_annotate, TT.parameters[2:end])
    TT = Tuple{<:Annotation{ft}, Type{<:Annotation}, tt...}
    isapplicable(forward, TT; world)
end

function has_frule(@nospecialize(f); world=Base.get_world_counter())
    TT = Tuple{<:Annotation{Core.Typeof(f)}, Type{<:Annotation}, Vararg{<:Annotation}}
    isapplicable(forward, TT; world)
end

# Do we need this one?
function has_frule(@nospecialize(f), @nospecialize(TT::Type{<:Tuple}); world=Base.get_world_counter())
    TT = Base.unwrap_unionall(TT)
    TT = Tuple{<:Annotation{Core.Typeof(f)}, Type{<:Annotation}, TT.parameters...}
    isapplicable(forward, TT; world)
end

# Do we need this one?
function has_frule(@nospecialize(f), @nospecialize(RT::Type); world=Base.get_world_counter())
    TT = Tuple{<:Annotation{Core.Typeof(f)}, Type{RT}, Vararg{<:Annotation}}
    isapplicable(forward, TT; world)
end

# Do we need this one?
function has_frule(@nospecialize(f), @nospecialize(RT::Type), @nospecialize(TT::Type{<:Tuple}); world=Base.get_world_counter())
    TT = Base.unwrap_unionall(TT)
    TT = Tuple{<:Annotation{Core.Typeof(f)}, Type{RT}, TT.parameters...}
    isapplicable(forward, TT; world)
end

# Base.hasmethod is a precise match we want the broader query.
function isapplicable(@nospecialize(f), @nospecialize(TT); world=Base.get_world_counter())
    tt = Base.to_tuple_type(TT)
    sig = Base.signature_type(f, tt)
    return !isempty(Base._methods_by_ftype(sig, -1, world)) # TODO cheaper way of querying?
end

function has_rrule(@nospecialize(TT), world=Base.get_world_counter())
    return false
end

function issupported()
    @static if VERSION < v"1.7.0"
        return false
    else
        return true
    end
end

end # EnzymeRules
