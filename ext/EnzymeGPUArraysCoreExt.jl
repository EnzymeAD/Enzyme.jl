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

@inline function Enzyme.onehot(x::AbstractGPUArray; stacked::Union{Val{true}, Val{false}} = Val(false))
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, 0, length(x))
    N = length(x)
    ret = ntuple(Val(N)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i] = 1
        return res
    end
    stacked isa Val{false} && return ret
    return stack(ret)
end

@inline function Enzyme.onehot(x::AbstractGPUArray, start::Int, endl::Int; stacked::Union{Val{true}, Val{false}} = Val(false))
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, start-1, endl-start+1)
    ret = ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i + start - 1] = 1
        return res
    end
    stacked isa Val{false} && return ret
    return stack(ret)
end

end # module
