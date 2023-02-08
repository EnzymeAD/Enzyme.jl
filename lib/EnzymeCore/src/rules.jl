module EnzymeRules

import EnzymeCore: Annotation, Const, Duplicated
export Config, ConfigWidth, AugmentedReturn
export needs_primal, needs_shadow, width, overwritten
export primal_type, shadow_type, tape_type

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

@inline needs_primal(::Config{NeedsPrimal}) where NeedsPrimal = NeedsPrimal
@inline needs_shadow(::Config{<:Any, NeedsShadow}) where NeedsShadow = NeedsShadow
@inline width(::Config{<:Any, <:Any, Width}) where Width = Width
@inline overwritten(::Config{<:Any, <:Any, <:Any, Overwritten}) where Overwritten = Overwritten

struct AugmentedReturn{PrimalType,ShadowType,TapeType}
    primal::PrimalType
    shadow::ShadowType
    tape::TapeType
end
@inline primal_type(::Type{AugmentedReturn{PrimalType,ShadowType,TapeType}}) where {PrimalType,ShadowType,TapeType} = PrimalType
@inline shadow_type(::Type{AugmentedReturn{PrimalType,ShadowType,TapeType}}) where {PrimalType,ShadowType,TapeType} = ShadowType
@inline tape_type(::Type{AugmentedReturn{PrimalType,ShadowType,TapeType}}) where {PrimalType,ShadowType,TapeType} = TapeType
"""
    augmented_primal(::Config, func::Annotation{typeof(f)}, RT::Type{<:Annotation}, args::Annotation...)

Must return an AugmentedReturn type.
* The primal must be the same type of the original return if needs_primal(config), otherwise nothing.
* The shadow must be nothing if needs_shadow(config) is false. If width is 1, the shadow should be the same
  type of the original return. If the width is greater than 1, the shadow should be NTuple{original return, width}.
* The tape can be any type (including Nothing) and is preserved for the reverse call.
"""
function augmented_primal end

"""
    reverse(::Config, func::Annotation{typeof(f)}, dret::Active, tape, args::Annotation...)
    reverse(::Config, func::Annotation{typeof(f)}, ::Type{<:Annotation), tape, args::Annotation...)

Takes gradient of derivative, activity annotation, and tape. If there is an active return dret is passed
as Active{T} with the active return val. Otherwise dret is passed as Type{Duplicated{T}}, etc.
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
    TT = Tuple{<:Config, <:Annotation{ft}, Type{<:Annotation}, tt...}
    isapplicable(augmented_primal, TT; world)
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

"""
    inactive(func::typeof(f), args::...)

Mark a particular function as always being inactive in both its return result and the function call itself.
"""
function inactive end

# Base.hasmethod is a precise match we want the broader query.
function is_inactive_from_sig(@nospecialize(TT); world=Base.get_world_counter())
    return isapplicable(inactive, TT; world)
end

end # EnzymeRules
