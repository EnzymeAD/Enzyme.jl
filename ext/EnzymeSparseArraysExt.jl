module EnzymeSparseArraysExt

using SparseArrays
using Enzyme

Enzyme.strip_types(x::SparseVector{<:Enzyme.GoodNum}) = x
Enzyme.strip_types(x::SparseMatrixCSC{<:Enzyme.GoodNum}) = x

end
