module EnzymeRules

import EnzymeCore: Annotation
export Config, ConfigWidth
export needs_primal, needs_shadow, width, overwritten

import Base: unwrapva, isvarargtype, unwrap_unionall, rewrap_unionall

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

Must return a tuple of length 2.
The first-value is primal value and the second is the tape. If no tape is
required return `(val, nothing)`.
"""
function augmented_primal end

"""
    reverse(::Config, func::Annotation{typeof(f)}, dret::Annotation, tape, args::Annotation...)

Takes gradient of derivative, activity annotation, and tape
"""
function reverse end

function _annotate(@nospecialize(T))
    if isvarargtype(T)
        VA = T
        T = _annotate(Core.Compiler.unwrapva(VA))
        if isdefined(VA, :N)
            return Vararg{T, VA.N}
        else
            return Vararg{T}
        end
    else
        return TypeVar(gensym(), Annotation{T})
    end
end

function _annotate_tt(@nospecialize(TT0))
    TT = Base.unwrap_unionall(TT0)
    ft = TT.parameters[1]
    tt = map(T->_annotate(Base.rewrap_unionall(T, TT0)), TT.parameters[2:end])
    return ft, tt
end

function has_frule_from_sig(@nospecialize(TT); world=Base.get_world_counter())
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:Annotation{ft}, Type{<:Annotation}, tt...}
    isapplicable(forward, TT; world)
end

function has_rrule_from_sig(@nospecialize(TT); world=Base.get_world_counter())
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:Config, <:Annotation{ft}, <:Annotation, <:Any, tt...}
    isapplicable(reverse, TT; world)
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

function issupported()
    @static if VERSION < v"1.7.0"
        return false
    else
        return true
    end
end

end # EnzymeRules
