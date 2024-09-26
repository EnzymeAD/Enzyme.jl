module EnzymeSparseArraysExt

using SparseArrays
using Enzyme

@inline Enzyme.Compiler.is_arrayorvararg_ty(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = true
@inline Enzyme.Compiler.ptreltype(::Type{SparseArrays.CHOLMOD.Dense{T}}) where T = T

end
