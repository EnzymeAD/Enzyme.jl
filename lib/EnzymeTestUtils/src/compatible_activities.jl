"""
    are_activities_compatible(Tret, activities...) -> Bool

Return `true` if return activity type `Tret` and activity types `activities` are compatible.
"""
function are_activities_compatible(Tret, activities...)
    any_batch = Tret <: BatchDuplicatedNoNeed || _any_batch_duplicated(Tret, activities...)
    any_batch || return true
    (Tret <: DuplicatedNoNeed || _any_duplicated(Tret, activities...)) && return false
    return true
end

_any_are_type(S::Type, activities...) = any(Base.Fix2(<:, S), activities)

_any_batch_duplicated(activities...) = _any_are_type(BatchDuplicated, activities...)

_any_duplicated(activities...) = _any_are_type(Duplicated, activities...)

_batch_size(::Type{BatchDuplicated{T, N}}) where {T, N} = N
_batch_size(::Type{<:Annotation}) = nothing
function _batch_size(activities...)
    sizes = filter(!isnothing, map(_batch_size, activities))
    isempty(sizes) && return 1
    @assert all(==(sizes[1]), sizes)
    return sizes[1]
end
