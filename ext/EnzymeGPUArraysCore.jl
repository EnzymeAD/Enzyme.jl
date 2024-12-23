module EnzymeGPUArraysCoreExt

using GPUArraysCore
using Enzyme

@inline function Enzyme.onehot(x::AbstractGPUArray)
    onehot_internal(zerosetfn, x, 0, length(x))
end

@inline function Enzyme.onehot(x::AbstractGPUArray, start::Int, endl::Int)
    onehot_internal(zerosetfn, x, start-1, endl-start+1)
end

function Enzyme.zerosetfn(x::AbstractGPUArray, i::Int)
    res = zero(x)
    @allowscalar @inbounds res[i] = 1
    return res
end

function Enzyme.zerosetfn!(x::AbstractGPUArray, i::Int, val)
    @allowscalar @inbounds x[i] = += val
    return
end


end # module
