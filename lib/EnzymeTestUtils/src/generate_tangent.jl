# recursively apply f to all fields of x for which f is implemented; all other fields are
# left unchanged
function map_fields_recursive(f, x::T...) where {T}
    fields = map(ConstructionBase.getfields, x)
    all(isempty, fields) && return first(x)
    new_fields = map(fields...) do xi...
        map_fields_recursive(f, xi...)
    end
    return ConstructionBase.constructorof(T)(new_fields...)
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

function zero_tangent(x)
    v, from_vec = to_vec(x)
    return from_vec(zero(v))
end

function auto_activity(arg::Tuple)
    if length(arg) == 2 && arg[2] isa Type && arg[2] <: Annotation
        return _build_activity(arg...)
    end
    return Const(arg)
end
auto_activity(activity::Annotation) = activity
auto_activity(activity) = Const(activity)

_build_activity(primal, ::Type{<:Const}) = Const(primal)
_build_activity(primal, ::Type{<:Active}) = Active(primal)
_build_activity(primal, ::Type{<:Duplicated}) = Duplicated(primal, rand_tangent(primal))
function _build_activity(primal, ::Type{<:BatchDuplicated})
    return BatchDuplicated(primal, ntuple(_ -> rand_tangent(primal), 2))
end
function _build_activity(primal, T::Type{<:Annotation})
    throw(ArgumentError("Unsupported activity type: $T"))
end
