# recursively apply f to all fields of x for which f is implemented; all other fields are
# left unchanged
function map_fields_recursive(f, x::T...) where {T}
    fields = map(ConstructionBase.getfields, x)
    all(isempty, fields) && return first(x)
    new_fields = map(fields...) do xi...
        return map_fields_recursive(f, xi...)
    end
    return _construct(T, new_fields...)
end
function map_fields_recursive(f, x::T...) where {T<:Union{Array,Tuple,NamedTuple}}
    map(x...) do xi...
        map_fields_recursive(f, xi...)
    end
end
map_fields_recursive(f, x::T...) where {T<:AbstractFloat} = f(x...)
map_fields_recursive(f, x::Array{<:Number}...) = f(x...)

rand_tangent(x) = rand_tangent(Random.default_rng(), x)
function rand_tangent(rng, x)
    v, from_vec = to_vec(x)
    T = eltype(v)
    # make numbers prettier sometimes when errors are printed.
    v_new = rand(rng, -9:T(0.01):9, length(v))
    return from_vec(v_new)
end

auto_activity(arg) = auto_activity(Random.default_rng(), arg)
function auto_activity(rng, arg::Tuple)
    if length(arg) == 2 && arg[2] isa Type && arg[2] <: Annotation
        return _build_activity(rng, arg...)
    end
    return Const(arg)
end
auto_activity(rng, activity::Annotation) = activity
auto_activity(rng, activity) = Const(activity)

_build_activity(rng, primal, ::Type{<:Const}) = Const(primal)
_build_activity(rng, primal, ::Type{<:Active}) = Active(primal)
function _build_activity(rng, primal, ::Type{<:Duplicated})
    return Duplicated(primal, rand_tangent(rng, primal))
end
function _build_activity(rng, primal, ::Type{<:BatchDuplicated})
    return BatchDuplicated(primal, ntuple(_ -> rand_tangent(rng, primal), 2))
end
function _build_activity(rng, primal, T::Type{<:Annotation})
    throw(ArgumentError("Unsupported activity type: $T"))
end

# below code is adapted from https://github.com/JuliaDiff/FiniteDifferences.jl/blob/99ad77f05bdf6c023b249025dbb8edc746d52b4f/src/to_vec.jl
# MIT Expat License
# Copyright (c) 2018 Invenia Technical Computing

# get around the constructors and make the type directly
# Note this is moderately evil accessing julia's internals
@generated function _force_construct(T, args...)
    return Expr(:splatnew, :T, :args)
end

function _construct(T, args...)
    try
        return ConstructionBase.constructorof(T)(args...)
    catch MethodError
        return _force_construct(T, args...)
    end
end
