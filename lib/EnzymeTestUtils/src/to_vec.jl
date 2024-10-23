# Like an IdDict, but also handles cases where 2 arrays share the same memory due to
# reshaping
struct AliasDict{K,V} <: AbstractDict{K,V}
    id_dict::IdDict{K,V}
    dataids_dict::IdDict{Tuple{UInt,Vararg{UInt}},V}
end
AliasDict() = AliasDict(IdDict(), IdDict{Tuple{UInt,Vararg{UInt}},Any}())

function Base.haskey(d::AliasDict, key)
    haskey(d.id_dict, key) && return true
    key isa Array && haskey(d.dataids_dict, Base.dataids(key)) && return true
    return false
end

Base.getindex(d::AliasDict, key) = d.id_dict[key]
function Base.getindex(d::AliasDict, key::Array)
    haskey(d.id_dict, key) && return d.id_dict[key]
    dataids = Base.dataids(key)
    return d.dataids_dict[dataids]
end

function Base.setindex!(d::AliasDict, val, key)
    d.id_dict[key] = val
    if key isa Array
        dataids = Base.dataids(key)
        d.dataids_dict[dataids] = val
    end
    return d
end

# alternative to FiniteDifferences.to_vec to use Enzyme's semantics for arrays instead of
# ChainRules': Enzyme treats tangents of AbstractArrays the same as tangents of any other
# struct (i.e. with a container of the same type as the original), while ChainRules
# represents the tangent with an array of some type that is tangent to the subspace defined
# by the original array type.
# We take special care that floats that occupy the same memory in the argument only appear
# once in the vector, and that the reconstructed object shares the same memory pattern

function to_vec(x)
    x_vec, from_vec_inner = to_vec(x, AliasDict())
    from_vec(x_vec::Vector{<:AbstractFloat}) = from_vec_inner(x_vec, AliasDict())
    return x_vec, from_vec
end

# base case: we've unwrapped to a number, so we break the recursion
function to_vec(x::AbstractFloat, seen_vecs::AliasDict)
    AbstractFloat_from_vec(v::Vector{<:AbstractFloat}, _) = oftype(x, only(v))
    return [x], AbstractFloat_from_vec
end

# basic containers: loop over defined elements, recursively converting them to vectors
function to_vec(x::RT, seen_vecs::AliasDict) where {RT<:Array}
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const_nongen(RT, nothing)
    if has_seen || is_const
        x_vec = Float32[]
    else
        x_vecs = Vector{<:AbstractFloat}[]
        from_vecs = []
        subvec_inds = UnitRange{Int}[]
        l = 0
        for i in eachindex(x)
            isassigned(x, i) || continue
            xi_vec, xi_from_vec = to_vec(x[i], seen_vecs)
            push!(x_vecs, xi_vec)
            push!(from_vecs, xi_from_vec)
            push!(subvec_inds, (l + 1):(l + length(xi_vec)))
            l += length(xi_vec)
        end
        x_vec = reduce(vcat, x_vecs; init=Float32[])
        seen_vecs[x] = x_vec
    end
    function Array_from_vec(x_vec_new::Vector{<:AbstractFloat}, seen_xs::AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Arrays must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return reshape(seen_xs[x], size(x))
        is_const && return x
        x_new = typeof(x)(undef, size(x))
        k = 1
        for i in eachindex(x)
            isassigned(x, i) || continue
            xi = from_vecs[k](x_vec_new[subvec_inds[k]], seen_xs)
            x_new[i] = xi
            k += 1
        end
        seen_xs[x] = x_new
        return x_new
    end
    return x_vec, Array_from_vec
end

@static if VERSION < v"1.11-"
else
# basic containers: loop over defined elements, recursively converting them to vectors
function to_vec(x::RT, seen_vecs::AliasDict) where {RT<:GenericMemory}
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const_nongen(RT, nothing)
    if has_seen || is_const
        x_vec = Float32[]
    else
        x_vecs = Vector{<:AbstractFloat}[]
        from_vecs = []
        subvec_inds = UnitRange{Int}[]
        l = 0
        for i in eachindex(x)
            isassigned(x, i) || continue
            xi_vec, xi_from_vec = to_vec(x[i], seen_vecs)
            push!(x_vecs, xi_vec)
            push!(from_vecs, xi_from_vec)
            push!(subvec_inds, (l + 1):(l + length(xi_vec)))
            l += length(xi_vec)
        end
        x_vec = reduce(vcat, x_vecs; init=Float32[])
        seen_vecs[x] = x_vec
    end
    function Memory_from_vec(x_vec_new::Vector{<:AbstractFloat}, seen_xs::AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Arrays must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return reshape(seen_xs[x], size(x))
        is_const && return x
        x_new = typeof(x)(undef, size(x))
        k = 1
        for i in eachindex(x)
            isassigned(x, i) || continue
            xi = from_vecs[k](x_vec_new[subvec_inds[k]], seen_xs)
            x_new[i] = xi
            k += 1
        end
        seen_xs[x] = x_new
        return x_new
    end
    return x_vec, Memory_from_vec
end
end

function to_vec(x::Tuple, seen_vecs::AliasDict)
    x_vec, from_vec = to_vec(collect(x), seen_vecs)
    function Tuple_from_vec(x_vec_new::Vector{<:AbstractFloat}, seen_xs::AliasDict)
        return typeof(x)(Tuple(from_vec(x_vec_new, seen_xs)))
    end
    return x_vec, Tuple_from_vec
end
function to_vec(x::NamedTuple, seen_vecs::AliasDict)
    x_vec, from_vec = to_vec(values(x), seen_vecs)
    function NamedTuple_from_vec(x_vec_new::Vector{<:AbstractFloat}, seen_xs::AliasDict)
        return NamedTuple{keys(x)}(from_vec(x_vec_new, seen_xs))
    end
    return x_vec, NamedTuple_from_vec
end

# fallback: for any other struct, loop over fields, recursively converting them to vectors
function to_vec(x::RT, seen_vecs::AliasDict) where {RT}
    has_seen = haskey(seen_vecs, x)
    is_const = Enzyme.Compiler.guaranteed_const_nongen(RT, nothing)
    if has_seen || is_const
        x_vec = Float32[]
    else
        @assert !Base.isabstracttype(RT)
        @assert Base.isconcretetype(RT)
        nf = fieldcount(RT)
        flds = Vector{Any}(undef, nf)
        for i in 1:nf
            if isdefined(x, i)
                flds[i] = xi = getfield(x, i)
            elseif !ismutable(x)
                nf = i - 1 # rest of tail must be undefined values
                break
            end
        end
        x_vec, fields_from_vec = to_vec(flds, seen_vecs)
        if ismutable(x)
            seen_vecs[x] = x_vec
        end
    end
    function Struct_from_vec(x_vec_new::Vector{<:AbstractFloat}, seen_xs::AliasDict)
        if xor(has_seen, haskey(seen_xs, x))
            throw(ErrorException("Objects must be reconstructed in the same order as they are vectorized."))
        end
        has_seen && return seen_xs[x]
        (is_const || nf == 0) && return x
        flds_new = fields_from_vec(x_vec_new, seen_xs)
        if ismutable(x)
            x_new = ccall(:jl_new_struct_uninit, Any, (Any,), RT)
            for i in 1:nf
                if isdefined(x, i)
                    xi = flds_new[i]
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x_new, i - 1, xi)
                end
            end
        else
            x_new = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds_new, nf)
        end
        if ismutable(x)
            seen_xs[x] = x_new
        end
        return x_new
    end
    return x_vec, Struct_from_vec
end
