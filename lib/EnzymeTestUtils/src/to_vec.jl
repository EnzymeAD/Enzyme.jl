# alternative to FiniteDifferences.to_vec to use Enzyme's semantics for arrays instead of
# ChainRules': Enzyme treats tangents of AbstractArrays the same as tangents of any other
# struct (i.e. with a container of the same type as the original), while ChainRules
# represents the tangent with an array of some type that is tangent to the subspace defined
# by the original array type.

# base case: we've unwrapped to a number, so we break the recursion
to_vec(x::AbstractFloat) = ([x], only)

# types: they and their fields aren't differentiable, so we bypass them
to_vec(x::Type) = (Float32[], _ -> x)

# below code is adapted from https://github.com/JuliaDiff/FiniteDifferences.jl/blob/99ad77f05bdf6c023b249025dbb8edc746d52b4f/src/to_vec.jl
# MIT Expat License
# Copyright (c) 2018 Invenia Technical Computing

# structs: recursively call to_vec on each field
function to_vec(x::T) where {T}
    fields = fieldnames(T)
    isempty(fields) && return (Float32[], _ -> x)
    x_vecs_and_from_vecs = map(to_vec ∘ Base.Fix1(getfield, x), fields)
    x_vecs, from_vecs = first.(x_vecs_and_from_vecs), last.(x_vecs_and_from_vecs)
    x_vec, from_vec = to_vec(x_vecs)
    function struct_from_vec(x_vec_new::Vector{<:AbstractFloat})
        x_vecs_new = from_vec(x_vec_new)
        fields_new = map((f, v) -> f(v), from_vecs, x_vecs_new)
        return ConstructionBase.constructorof(T)(fields_new...)
    end
    return x_vec, struct_from_vec
end

# basic containers: recursively call to_vec on each element
function to_vec(x::Union{DenseVector,Tuple})
    x_vecs_and_from_vecs = map(to_vec, x)
    x_vecs, from_vecs = first.(x_vecs_and_from_vecs), last.(x_vecs_and_from_vecs)
    subvec_lengths = map(length, x_vecs)
    subvec_ends = cumsum(subvec_lengths)
    subvec_inds = map(subvec_lengths, subvec_ends) do l, e
        return (e - l + 1):e
    end
    function DenseVector_Tuple_from_vec(x_vec_new::Vector{<:AbstractFloat})
        x_new = map(from_vecs, subvec_inds) do from_vec, inds
            return from_vec(x_vec_new[inds])
        end
        return oftype(x, x_new)
    end
    x_vec = reduce(vcat, x_vecs; init=Float32[])
    return x_vec, DenseVector_Tuple_from_vec
end
function to_vec(x::DenseArray)
    x_vec, from_vec = to_vec(vec(x))
    function DenseArray_from_vec(x_vec_new::Vector{<:AbstractFloat})
        x_new = reshape(from_vec(x_vec_new), size(x))
        return oftype(x, x_new)
    end
    return x_vec, DenseArray_from_vec
end
function to_vec(x::NamedTuple)
    x_tuple = values(x)
    x_vec, from_vec = to_vec(x_tuple)
    function NamedTuple_from_vec(x_vec_new::Vector{<:AbstractFloat})
        x_tuple_new = from_vec(x_vec_new)
        return typeof(x)(x_tuple_new)
    end
    return x_vec, NamedTuple_from_vec
end
