module EnzymeStaticArraysExt

using StaticArrays
using Enzyme

Enzyme.strip_types(x::StaticArrays.SArray{<:Any, <:Enzyme.GoodNum}) = x
Enzyme.strip_types(x::StaticArrays.MArray{<:Any, <:Enzyme.GoodNum}) = x

end
