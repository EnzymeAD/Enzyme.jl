module EnzymeTestUtilsGPUArraysCoreExt

using GPUArraysCore
using EnzymeTestUtils
using Enzyme

function EnzymeTestUtils.acopyto!(dst, src::AbstractGPUArray)
    temp = Array{eltype(src)}(undef, size(src))
    Base.copyto!(temp, src)
    EnzymeTestUtils.acopyto!(dst, temp)
end

# basic containers: loop over defined elements, recursively converting them to vectors
function EnzymeTestUtils.to_vec(x::AbstractGPUArray{<:EnzymeTestUtils.ElementType}, seen_vecs::EnzymeTestUtils.AliasDict)
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const(Core.Typeof(x))
    if has_seen || is_const
        x_vec = Float32[]
    else
        x_vec = reshape(x, length(x))
        seen_vecs[x] = x_vec
    end
    sz = size(x)
    function FastGPUArray_from_vec(x_vec_new::AbstractVector{<:EnzymeTestUtils.ElementType}, seen_xs::EnzymeTestUtils.AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Arrays must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return reshape(seen_xs[x], size(x))
        is_const && return x
        x_new = reshape(x_vec_new, sz)
        if Core.Typeof(x_new) != Core.Typeof(x)
            x_new = Core.Typeof(x)(x_new)
        end
        seen_xs[x] = x_new
        return x_new
    end
    return x_vec, FastGPUArray_from_vec
end

# basic containers: loop over defined elements, recursively converting them to vectors
function to_vec(x::AbstractGPUArray{<:Complex{<:EnzymeTestUtils.ElementType}}, seen_vecs::EnzymeTestUtils.AliasDict)
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const(Core.Typeof(x))
    if has_seen || is_const
        x_vec = Float32[]
    else
        y = reshape(x, length(x))
        x_vec = vcat(real.(y), imag.(y))
        seen_vecs[x] = x_vec
    end
    sz = size(x)
    function ComplexGPUArray_from_vec(x_vec_new::AbstractVector{<:EnzymeTestUtils.ElementType}, seen_xs::EnzymeTestUtils.AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Arrays must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return reshape(seen_xs[x], size(x))
        is_const && return x
	x_new = Array{eltype(x)}(undef, sz)
        @inbounds @simd for i in 1:length(x)
            x_new[i] = eltype(x)(x_vec_new[i], x_vec_new[i + length(x)])
        end
	x_new = Core.Typeof(x)(x_new)
        seen_xs[x] = x_new
        return x_new
    end
    return x_vec, ComplexGPUArray_from_vec
end

# basic containers: loop over defined elements, recursively converting them to vectors
function to_vec(x::AbstractGPUArray, seen_vecs::EnzymeTestUtils.AliasDict)
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const(Core.Typeof(x))
    if has_seen || is_const
        x_vec = Float32[]
    else
        x_vecs = nothing
        from_vecs = []
        subvec_inds = UnitRange{Int}[]
        l = 0
        for i in eachindex(x)
            isassigned(x, i) || continue
            xi_vec, xi_from_vec = to_vec(x[i], seen_vecs)
            push!(subvec_inds, (l + 1):(l + length(xi_vec)))
            push!(from_vecs, xi_from_vec)
            x_vecs = EnzymeTestUtils.append_or_merge(x_vecs, xi_vec)
            l += length(xi_vec)
        end

        if x_vecs === nothing
            x_vecs = (Float32[], true)
        end
        x_vec = x_vecs[1]
        seen_vecs[x] = x_vec
    end
    function GPUArray_from_vec(x_vec_new::AbstractVector{<:EnzymeTestUtils.ElementType}, seen_xs::EnzymeTestUtils.AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Arrays must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return reshape(seen_xs[x], size(x))
        is_const && return x
	x_new = Array{eltype(x_vew_new)}(undef, size(x))
        k = 1
        for i in eachindex(x)
            isassigned(x, i) || continue
            xi = from_vecs[k](@view(x_vec_new[subvec_inds[k]]), seen_xs)
            x_new[i] = xi
            k += 1
        end
	x_new = Core.Typeof(x)(x_new)
        seen_xs[x] = x_new
        return x_new
    end
    return x_vec, GPUArray_from_vec
end

end # module
