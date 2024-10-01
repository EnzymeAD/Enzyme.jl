"""
    recursive_map(f, xs::T...) where {T}
    recursive_map(
        ::Type{T},
        f,
        seen::IdDict,
        xs::NTuple{N,T},
        ::Val{copy_if_inactive}=Val(false),
        isleaftype=Returns(false),
    ) where {T,N,copy_if_inactive}

Recursively map `f` over the differentiable contents of `N` objects `xs = (x1, x2, ..., xN)`
of arbitrary but identical type and layout. Returns a new object `y` of the same type and
layout such that each differentiable leaf value `yi` in `y` equals
`yi = f(x1i, x2i, ..., xNi)` where `x1i, x2i, ..., xNi` are the corresponding leaf values in
the `xs`.

For each subtree in the `xs` that can be proven by type to only contain non-differentiable
values, `f` is not invoked and the corresponding subtree `yi` in `y` is taken from the first
element of `xs` such that `yi == x1i`. If `copy_if_inactive == false`, this is done by
sharing, `yi = x1i`; if `copy_if_inactive == true`, it is done by copying,
`yi = deepcopy(x1i)`.

Each element in `xs` is assumed to have the same structure as `x1 = first(xs)`, including
which fields, if any, reference the same memory or are undefined. This structure will be
mirrored in the return value `y`. If this assumption does not hold, errors or incorrect
results may occur.

A function `isleaftype` can be provided to customize which types are considered leafs:
values of type `T` such that `isleaftype(T) == true` are not recursed into, but instead
passed to `f`. Non-array types with `fieldcount(T) == 0`, including built-in floats and
other primitve types, are always considered leaf types as they cannot be recursed into.
"""
recursive_map(f::F, xs::T...) where {F,T} = recursive_map(T, f, IdDict(), xs)::T

@inline function recursive_map(
    ::Type{T},
    f::F,
    seen::IdDict,
    xs::NTuple{N,T},
    ::Val{copy_if_inactive}=Val(false),
    isleaftype::L=Returns(false),
) where {T,F,N,L,copy_if_inactive}
    x1 = first(xs)
    if guaranteed_const_nongen(T, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(x1, seen)::T : first(xs)
    elseif haskey(seen, xs)
        return seen[x1]::T
    end
    y = if isleaftype(T)
        f(xs...)::T
    else
        _recursive_map(T, f, seen, xs, Val(copy_if_inactive), isleaftype)::T
    end
    seen[x1] = y
    return y
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, args...
) where {RT,F,N}
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
 
    @inline function newyi(i)
        xis = ntuple(j -> getfield(xs[j], i), N)
        ST = Core.Typeof(first(xis))
        return recursive_map(ST, f, seen, xis, args...)
    end
   
    nf = fieldcount(RT)
    x1 = first(xs)
    if ismutabletype(RT)
        if all(i -> isdefined(x1, i), 1:nf)
            # fast path when all fields are set
            return splatnew(RT, ntuple(newyi, Val(nf)))
        else
            y = ccall(:jl_new_struct_uninit, Any, (Any,), RT)
            for i in 1:nf
                if isdefined(x1, i)
                    yi = newyi(i)
                    if Base.isconst(RT, i)
                        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, yi)
                    else
                        setfield!(y, i, yi)
                    end
                end
            end
            return y
        end
    elseif nf == 0
        return f(xs...)::RT
    elseif isdefined(x1, nf)
        # fast path when all fields are set
        return splatnew(RT, ntuple(newyi, Val(nf)))
    else
        flds = Vector{Any}(undef, nf)
        nset = nf
        for i in 1:nf
            if isdefined(x1, i)
                @inbounds flds[i] = newyi(i)
            else
                nset = i - 1  # rest of tail must be undefined values
                break
            end
        end
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nset)
    end
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, args...
) where {RT<:Array,F,N}
    y = RT(undef, size(first(xs)))
    x1 = first(xs)
    for I in eachindex(y, xs...)
        @inbounds if isassigned(x1, I)
            xIs = ntuple(j -> xs[j][I], N)
            ST = Core.Typeof(first(xIs))
            y[I] = recursive_map(ST, f, seen, xIs, args...)
        end
    end
    return y
end

"""
    recursive_map!!(f, y::T, xs::T...) where {T}
    recursive_map!!(
        f, y::T seen::Base.IdSet, xs::NTuple{N,T}, isleaftype=Returns(false)
    ) where {T,N}

Recursively update `y` such that each differentiable leaf value `yi` in `y` is updated to
equal `f(x1i, x2i, ..., xNi)`, where `x1i, x2i, ..., xNi` are the corresponding leaf values
in the `xs`. Each subtree in `y` that can be proven by type to only contain
non-differentiable values is left unchanged.

Each element in `xs` is assumed to have the same structure as `y`, including which fields,
if any, reference the same memory or are undefined. If this assumption does not hold, errors
or incorrect results may occur.

If every differentiable value in `y` is contained in a mutable object (i.e., `y` has
inferred activity state Duplicated), this function performs a fully in-place update and
returns `y`. If every differentiable value is held in immutable storage (i.e., `y`
has inferred activity state Active), this function is equivalent to `recursive_map` and `y`
is not used. If differentiable values within `y` are contained in a mix of mutable and
immutable locations (i.e., `y` has inferred activity state MixedDuplicated), mutable memory
is updated in-place and reused, but the returned value will be a newly constructed object.
In all cases, the returned value has the same type and structure as `y`.

A function `isleaftype` can be provided to customize which types are considered leafs:
values of type `T` such that `isleaftype(T) == true` are not recursed into, but instead
passed to `f`. Non-array types with `fieldcount(T) == 0`, including built-in floats and
other primitve types, are always considered leaf types as they cannot be recursed into.

The function `f` should follow the semantics of `recursive_map!!`. That is,

* For leaves `yi` where all differentiable values are in immutable storage, i.e., with
inferred activity state Active, `f` should have a corresponding out-of-place method
`newyi = f(x1i, x2i, ..., xNi)`. This includes the default leaf types.

* For leaves `yi` with differentiable values contained in mutable storage, i.e., with
inferred activity state Duplicated or MixedDuplicated, the function `f` should have a
corresponding method `newyi = f(yi, x1i, x2i, ..., xNi)` that updates the mutable parts of
`y` in-place and reuses them in the returned value `newyi`. If the inferred activity state
is Duplicated this implies `newyi === yi`.
"""
function recursive_map!!(f::F, y::T, xs::T...) where {F,T}
    return recursive_map!!(f, y, Base.IdSet(), xs)::T
end

@inline function recursive_map!!(
    f::F, y::T, seen::Base.IdSet, xs::NTuple{N,T}, isleaftype::L=Returns(false)
) where {F,T,N,L}
    activitystate = active_reg_inner(T, (), nothing, Val(false))
    if (activitystate == AnyState) || (y in seen)  # guaranteed const or already handled dup
        return y
    elseif activitystate == DupState
        push!(seen, y)
        if isleaftype(T)
            return f(y, xs...)::T
        else
            return _recursive_map_dup!(f, y, seen, xs, isleaftype)::T
        end
    else
        if isleaftype(T)
            return (activitystate == ActiveState) ? f(xs...)::T : f(y, xs...)::T
        else
            return _recursive_map_active_or_mixed!!(f, y, seen, xs, isleaftype)::T
        end
    end
end

@inline function _recursive_map_dup!(
    f::F, y::T, seen, xs::NTuple{N,T}, isleaftype
) where {F,T,N}
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    for i = 1:nf
        if isdefined(y, i)
            yi = getfield(y, i)
            xis = ntuple(j -> getfield(xs[j], i), N)
            newyi = recursive_map!!(f, yi, seen, xis, isleaftype)
            if newyi !== yi
                if Base.isconst(T, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, newyi)
                else
                    setfield!(y, i, newyi)
                end
            end
        end
    end
    return y
end

@inline function _recursive_map_dup!(
    f::F, y::Array{T,M}, seen, xs::NTuple{N,Array{T,M}}, isleaftype
) where {F,T,M,N}
    for I in eachindex(y, xs...)
        @inbounds if isassigned(y, I)
            yvalue = y[I]
            xvalues = ntuple(j -> xs[j][I], N)
            newyvalue = recursive_map!!(f, yvalue, seen, xvalues, isleaftype)
            if newyvalue !== yvalue
                y[I] = newyvalue
            end
        end
    end
    return y
end

@inline function _recursive_map_active_or_mixed!!(
    f::F, y::T, seen, xs::NTuple{N,T}, isleaftype
) where {F,N,T}
    @assert !ismutabletype(T)
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)

    @inline function newyi(i)
        yi = getfield(y, i)
        xis = ntuple(j -> getfield(xs[j], i), N)
        return recursive_map!!(f, yi, seen, xis, isleaftype)
    end

    nf = fieldcount(T)
    if nf == 0
        return f(xs...)::T
    elseif isdefined(y, nf)
        # fast path when all fields are set
        return splatnew(T, ntuple(newyi, Val(nf)))
    else
        flds = Vector{Any}(undef, nf)
        nset = nf
        for i = 1:nf
            if isdefined(y, i)
                @inbounds flds[i] = newyi(i)
            else
                nset = i - 1  # rest of tail must be undefined values
                break
            end
        end
        return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nset)
    end
end

"""
    recursive_map!(f, y::T, xs::T...) where {T}
    recursive_map!(
        f, y::T seen::Base.IdSet, xs::NTuple{N,T}, isleaftype=Returns(false)
    ) where {T,N}

Recursively update `y` in-place such that each differentiable leaf value `yi` in `y` is
updated to equal `f(x1i, x2i, ..., xNi)`, where `x1i, x2i, ..., xNi` are the corresponding
leaf values in the `xs`. This works like `recursive_map!!`, except it requires `y` that can
be fully updated in-place, and always returns `nothing`. See the docstring for
`recursive_map!!` for further details.
"""
function recursive_map!(f::F, y::T, xs::T...) where {F,T}
    return recursive_map!(f, y, Base.IdSet(), xs)::Nothing
end

function recursive_map!(f, y::T, seen, xs::NTuple{N,T}, isleaftype) where {N,T}
    if active_reg_inner(T, (), nothing, Val(true)) == ActiveState  # justActive
        msg = (
            "recursive_map! called on objects that may "
            * "contain immutable differentiable values"
        )
        throw(ArgumentError(msg))
    end
    recursive_map!!(f, y, seen, xs, isleaftype)
    return nothing
end

const _RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

_zero_f(p) = EnzymeCore.make_zero(p)

function _zero_f(pout::T, pin::T) where {T}
    @assert pout === pin
    EnzymeCore.make_zero!(pout)
    return pout
end

_zero_isleaftype(::Type{<:Union{_RealOrComplexFloat,Array{<:_RealOrComplexFloat}}}) = true
_zero_isleaftype(::Any) = false

@inline function EnzymeCore.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
) where {RT,copy_if_inactive}
    return recursive_map(
        RT, _zero_f, seen, (prev,), Val(copy_if_inactive), _zero_isleaftype
    )::RT
end

@inline function EnzymeCore.make_zero(prev::FT) where {FT<:_RealOrComplexFloat}
    return Base.zero(prev)::FT
end

@inline function EnzymeCore.make_zero(prev::Array{FT,N}) where {FT<:_RealOrComplexFloat,N}
    # convert because Base.zero may return different eltype when FT is not concrete
    return convert(Array{FT,N}, Base.zero(prev))::Array{FT,N}
end

@inline function EnzymeCore.make_zero!(prev, seen::Base.IdSet=Base.IdSet())
    return recursive_map!(_zero_f, prev, seen, (prev,), _zero_isleaftype)::Nothing
end

@inline function EnzymeCore.make_zero!(prev::Array{T,N}) where {T<:_RealOrComplexFloat,N}
    fill!(prev, zero(T))
    return nothing
end
