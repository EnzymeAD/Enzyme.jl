"""
    are_activities_compatible(Tret, activities...) -> Bool

Return `true` if return activity type `Tret` and activity types `activities` are compatible.
"""
function are_activities_compatible(Tret, activities...)
    return _all_or_no_batch(Tret, activities...)
end

#=
    _all_or_no_batch(activities...) -> Bool

Returns `true` if `activities` are compatible in terms of batched activities.

When a test set loops over many activities, some of which may be `BatchedDuplicated` or
`BatchedDuplicatedNoNeed`, this is useful for skipping those combinations that are
incompatible and will raise errors.
=#
function _all_or_no_batch(activities...)
    no_batch = !_any_batch(activities...)
    all_batch_or_const = all(activities) do T
        T <: Union{BatchDuplicated,BatchDuplicatedNoNeed,Const}
    end
    return all_batch_or_const || no_batch
end

function _any_batch(activities...)
    return any(activities) do T
        T <: Union{BatchDuplicated,BatchDuplicatedNoNeed}
    end
end

_batch_size(::Type{BatchDuplicated{T,N}}) where {T,N} = N
_batch_size(::Type{<:Annotation}) = nothing
function _batch_size(activities...)
    sizes = filter(!isnothing, map(_batch_size, activities))
    isempty(sizes) && return nothing
    @assert all(==(sizes[1]), sizes)
    return sizes[1]
end
