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
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end)
    end
end

@inline Enzyme._gradient_output(res, x::StaticArray) = similar_type(x)(res)

@inline _combine_size(::Size{s1}, ::Size{s2}) where {s1,s2} = Size{(s1..., s2...)}()

@inline Enzyme._jacobian_output(cols, col1::Number, x::StaticArray) = similar_type(x)(cols)

@inline function Enzyme._jacobian_output(cols, cols1::StaticArray, x::StaticArray)
    reshape(reduce(hcat, cols), _combine_size(Size(cols1), Size(x)))
end

end
