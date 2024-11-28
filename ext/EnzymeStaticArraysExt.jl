module EnzymeStaticArraysExt

using StaticArrays
using Enzyme

@inline function Base.convert(::Type{SArray}, tpa::Enzyme.TupleArray{T,S,L,N}) where {T,S,L,N}
    SArray{Tuple{S...},T,N,L}(tpa.data)
end
@inline Base.convert(::Type{StaticArray}, tpa::Enzyme.TupleArray) = convert(SArray, tpa)

@inline function Enzyme.tupstack(rows::Tuple{Vararg{T}}, outshape::Tuple{Vararg{Int}}, inshape::Tuple{Vararg{Int}}) where {T<:StaticArrays.SArray}
    reshape(reduce(hcat, map(vec, rows)), Size(outshape..., inshape...))
end

@inline Enzyme.specialize_output(output, input::StaticArray) = convert(SArray, output)

@inline function Enzyme.onehot(x::StaticArrays.SArray{S, T, N, L}) where {S, T, N, L}
    ntuple(Val(L)) do i
        Base.@_inline_meta
        StaticArrays.SArray{S, T, N, L}(Enzyme.onehot(NTuple{L, T})[i])
    end
end

@inline function Enzyme.onehot(x::StaticArrays.SArray{S, T, N, L}, start::Int, endl::Int) where {S, T, N, L}
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        StaticArrays.SArray{S, T, N, L}(
        ntuple(Val(L)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end)
    end
end

@inline function Enzyme.EnzymeCore.make_zero(
    prev::FT
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:SArray{S,T}}
    return Base.zero(prev)::FT
end
@inline function Enzyme.EnzymeCore.make_zero(
    prev::FT
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T}}
    return Base.zero(prev)::FT
end

@inline function Enzyme.EnzymeCore.make_zero(
    ::Type{FT}, seen::IdDict, prev::FT, ::Val{copy_if_inactive} = Val(false)
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:SArray{S,T},copy_if_inactive}
    return Base.zero(prev)::FT
end
@inline function Enzyme.EnzymeCore.make_zero(
    ::Type{FT}, seen::IdDict, prev::FT, ::Val{copy_if_inactive} = Val(false)
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T},copy_if_inactive}
    if haskey(seen, prev)
        return seen[prev]
    end
    new = Base.zero(prev)::FT
    seen[prev] = new
    return new
end

@inline function Enzyme.EnzymeCore.make_zero!(
    prev::FT, seen
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T}}
    if !isnothing(seen)
        if prev in seen
            return nothing
        end
        push!(seen, prev)
    end
    fill!(prev, zero(T))
    return nothing
end
@inline function Enzyme.EnzymeCore.make_zero!(
    prev::FT
) where {S,T<:Union{AbstractFloat,Complex{<:AbstractFloat}},FT<:MArray{S,T}}
    Enzyme.EnzymeCore.make_zero!(prev, nothing)
    return nothing
end

end
