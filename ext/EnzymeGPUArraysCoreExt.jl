module EnzymeGPUArraysCoreExt

using GPUArraysCore
using Enzyme

function Enzyme.zerosetfn(x::AbstractGPUArray, i::Int)
    res = zero(x)
    @allowscalar @inbounds res[i] = 1
    return res
end

function Enzyme.zerosetfn!(x::AbstractGPUArray, i::Int, val)
    @allowscalar @inbounds x[i] += val
    return
end

@inline function Enzyme.onehot(x::AbstractGPUArray)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, 0, length(x))
    N = length(x)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i] = 1
        return res
    end
end

@inline function onehot(x::AbstractArray, start::Int, endl::Int)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, start-1, endl-start+1)
    return ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i + start - 1] = 1
        return res
    end
end

end # module
