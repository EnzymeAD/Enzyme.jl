module EnzymeStaticArraysExt

using StaticArrays
using Enzyme

@inline function Base.convert(::Type{SArray}, tpa::Enzyme.TupleArray{T,S,L,N}) where {T,S,L,N}
    SArray{Tuple{S...},T,N,L}(tpa.data)
end
@inline Base.convert(::Type{StaticArray}, tpa::Enzyme.TupleArray) = convert(SArray, tpa)

@inline function Enzyme.tupstack(rows::(NTuple{N, <:StaticArrays.SArray} where N), inshape, outshape)
    reshape(reduce(hcat, map(vec, rows)), Size(inshape..., outshape...))
end

@inline function Enzyme.onehot(x::StaticArrays.SArray{S, T, N, L}) where {S, T, N, L}
    ntuple(Val(L)) do i
        Base.@_inline_meta
        StaticArrays.SArray{S, T, N, L}(Enzyme.onehot(NTuple{L, T})[i])
    end
end

@inline function Enzyme.onehot(x::StaticArrays.SArray{S, T, N, L}, start, endl) where {S, T, N, L}
    ntuple(Val(endl-start+1)) do i
        Base.@_inline_meta
        StaticArrays.SArray{S, T, N, L}(
        ntuple(Val(L)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end)
    end
end

@inline function Enzyme.EnzymeCore.make_zero(x::FT)::FT where {FT<:SArray}
    return Base.zero(x)
end
@inline function Enzyme.EnzymeCore.make_zero(x::FT)::FT where {FT<:MArray}
    return Base.zero(x)
end

end
