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
rand_tangent(rng, x) = map_fields_recursive(Base.Fix1(rand_tangent, rng), x)
# make numbers prettier sometimes when errors are printed.
rand_tangent(rng, ::T) where {T<:AbstractFloat} = rand(rng, -9:T(0.01):9)
rand_tangent(rng, x::T) where {T<:Array{<:Number}} = rand_tangent.(rng, x)

zero_tangent(x) = map_fields_recursive(zero_tangent, x)
zero_tangent(::T) where {T<:AbstractFloat} = zero(T)
zero_tangent(x::T) where {T<:Array{<:Number}} = zero_tangent.(x)

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
