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


#NOTE: the following fix spurious allocations but at least one of these methods is API breaking
#Enzyme.gradient_output_forward(df, df1::Number, x::StaticArray) = similar_type(x)(df)

#_gradsize(::Size{s1}, ::Size{s2}) where {s1,s2} = Size(s1..., s2...)
#Enzyme.gradient_output_size(df1::StaticArray, x::StaticArray) = _gradsize(Size(df1), Size(x))

end
