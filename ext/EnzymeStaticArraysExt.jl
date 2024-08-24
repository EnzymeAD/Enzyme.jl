module EnzymeStaticArraysExt

using StaticArrays
using Enzyme

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

Enzyme.gradient_output(df, x::StaticArray) = similar_type(x)(df)

_jacsize(::Size{s1}, ::Size{s2}) where {s1,s2} = Size(s1..., s2...)

Enzyme.jacsize(df1::StaticArray, x::StaticArray) = _jacsize(Size(df1), Size(x))

end
