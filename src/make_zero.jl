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

Recursively map `f` over the active contents of `N` objects `xs = (x1, x2, ..., xN)` of
arbitrary but identical type and layout. Returns a new object `y` of the same type and
layout such that each actively typed leaf value `yi` in `y` equals
`yi = f(x1i, x2i, ..., xNi)` where `x1i, x2i, ..., xNi` are the corresponding leaf values in
the `xs`.

For each subtree in the `xs` that can be proven by type to only contain inactively typed
values, `f` is not invoked and the corresponding subtree `yi` in `y` is taken from the first
element of `xs` such that `yi == x1i`. If `copy_if_inactive == false`, this is done by
sharing, `yi = x1i`; if `copy_if_inactive == true`, it is done by copying,
`yi = deepcopy(x1i)`.

The first element of `xs` is also used as the source of truth for aliasing, that is, which
values, if any, within the objects are references to the same memory. This structure is
reproduced in the return value `y`.

A function `isleaftype` can be provided to customize which types are considered leafs:
values of type `T` such that `isleaftype(T) == true` are passed to `f` rather than recursed
into. Non-array types with `fieldcount(T) == 0`, including all primitve types such as
built-in floats, are always considered leaf types as they cannot be recursed into.
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
    if ismutabletype(RT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), RT)
        for i in 1:nf
            if all(x -> isdefined(x, i), xs)
                yi = newyi(i)
                if Base.isconst(RT, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, yi)
                else
                    setfield!(y, i, yi)
                end
            end
        end
        return y
    elseif nf == 0
        return f(xs...)::RT
    elseif all(x -> isdefined(x, nf), xs)
        # fast path when all fields are set
        return splatnew(RT, ntuple(newyi, Val(nf)))
    else
        flds = Vector{Any}(undef, nf)
        nset = nf
        for i in 1:nf
            if all(x -> isdefined(x, i), xs)
                flds[i] = newyi(i)
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
    for I in eachindex(xs...)
        if all(x -> isassigned(x, I), xs)
            xIs = ntuple(j -> xs[j][I], N)
            ST = Core.Typeof(first(xIs))
            @inbounds y[I] = recursive_map(ST, f, seen, xIs, args...)
        end
    end
    return y
end

"""
    recursive_map!(f, y::T, xs::T...) where {T}
    recursive_map!(
        f, y::T seen::Base.IdSet, xs::NTuple{N,T}, isleaftype=Returns(false)
    ) where {T,N}

Recursively update `y` in-place such that each actively typed leaf value `yi` in `y` is
updated to equal `yi = f(x1i, x2i, ..., xNi)`, where `x1i, x2i, ..., xNi` are the
corresponding leaf values in the `xs`. Each subtree in `y` that can be proven by type to
only contain inactively typed values is left unchanged.

Every actively typed value in `y` must be contained in a mutable object, but `y` itself need
not be mutable. For example, `y = (1, [(2.0,)])` is a valid input; `y` itself is not
mutable, so the inactively typed integer `1` cannot be changed, but the actively typed value
`2.0` is nested within an array. In other words, the inferred activity state for `y` must be
either duplicated or constant (in the latter case, `y` will be left unchanged).

A function `isleaftype` can be provided to customize which types are considered leafs:
values of type `T` such that `isleaftype(T) == true` are updated directly by `f` rather than
recursed into. Non-array types with `fieldcount(T) == 0`, including all primitve types such
as built-in floats, are always considered leaf types as they cannot be recursed into.

If a custom leaf `yi` has active values contained only in mutable objects, i.e., would be a
valid input to `recursive_map!` (in other words, has inferred activity state Duplicated),
the function `f` must have a corresponding method `f(yi, x1i, x2i, ..., xNi)` that mutates
`y` in-place and returns `nothing`, like `recursive_map!` would.

If a custom leaf `yi` has active values contained in a mix of mutable and immutable
locations, i.e., would be a valid input to `recursive_map_immutable!` (in other words, has
inferred activity state MixedDuplicated), the function `f` must have a corresponding method
`newyi = f(yi, x1i, x2i, ..., xNi)` that updates the mutable parts of `yi` in-place and
wraps them in a return value `newyi` with updated values in immutable locations, like
`recursive_map_immutable!` would.

The default leaf types always have inferred activity state Active and only require the
out-of-place method `yi = f(x1i, x2i, ..., xNi)`, like `recursive_map`.
"""
function recursive_map!(f::F, y::T, xs::T...) where {F,T}
    return recursive_map!(f, y, Base.IdSet(), xs)::Nothing
end

@inline function recursive_map!(
    f::F, y::T, seen::Base.IdSet, xs::NTuple{N,T}, isleaftype::L=Returns(false)
) where {F,T,N,L}
    activitystate = active_reg_inner(T, (), nothing, Val(false))
    if (activitystate == AnyState) || (y in seen)  # guaranteed const or already handled
        return nothing
    elseif activitystate == DupState
        push!(seen, y)
        if isleaftype(T)
            return f(y, xs...)::Nothing
        else
            return _recursive_map!(f, y, seen, xs, isleaftype)::Nothing
        end
    else
        error(
            "recursive_map! only accepts types with "
            * "inferred activity state DupState or AnyState"
        )
    end
end

@inline function _recursive_map!(
    f::F, y::T, seen, xs::NTuple{N,T}, isleaftype
) where {F,T,N}
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    for i = 1:nf
        if isdefined(y, i) && all(x -> isdefined(x, i), xs)
            yi = getfield(y, i)
            xis = ntuple(j -> getfield(xs[j], i), N)
            SBT = Core.Typeof(yi)
            if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
                newyi = recursive_map_immutable!(f, yi, seen, xis, isleaftype)
                if Base.isconst(T, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, newyi)
                else
                    setfield!(y, i, newyi)
                end
            else
                recursive_map!(f, yi, seen, xis, isleaftype)
            end
        end
    end
    return nothing
end

@inline function _recursive_map!(
    f::F, y::Array{T,M}, seen, xs::NTuple{N,Array{T,M}}, isleaftype
) where {F,T,M,N}
    for I in eachindex(y, xs...)
        if isassigned(y, I) && all(x -> isassigned(x, I), xs)
            yvalue = y[I]
            xvalues = ntuple(j -> xs[j][I], N)
            SBT = Core.Typeof(yvalue)
            if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
                @inbounds y[I] = recursive_map_immutable!(
                    f, yvalue, seen, xvalues, isleaftype
                )
            else
                recursive_map!(f, yvalue, seen, xvalues, isleaftype)
            end
        end
    end
    return nothing
end

"""
    recursive_map_immutable!(f, y::T, xs::T...) where {T}
    recursive_map_immutable!(
        f, y::T seen::Base.IdSet, xs::NTuple{N,T}, isleaftype=Returns(false)
    ) where {T,N}

Recursively update the mutable parts of `y` in-place and wrap new immutable objects around
them, such that each actively typed leaf value `newyi` in the return value `newy` equals
`newyi = f(x1i, x2i, ..., xNi)`, where `x1i, x2i, ..., xNi` are the corresponding leaf
values in the `xs`. Each subtree in `y` that can be proven by type to only contain
inactively typed values is included in `newy` unchanged.

If every actively typed value in `y` is contained in a mutable object (i.e., `y` has
inferred activity state Duplicated), this function is equivalent to `recursive_map!`, except
it also returns `y`. If every actively typed value is in an immutable location (i.e., `y`
has inferred activity state Active), it is equivalent to `recursive_map` and `y` is not
used. If active values within `y` are contained in a mix of mutable and immutable locations
(i.e., `y` has inferred activity state MixedDuplicated), the returned value will be equal to
what `recursive_map` would return, but with mutable storage within `y` reused.

A function `isleaftype` can be provided to customize which types are considered leafs:
values of type `T` such that `isleaftype(T) == true` are updated directly by `f` rather than
recursed into. Non-array types with `fieldcount(T) == 0`, including all primitve types such
as built-in floats, are always considered leaf types as they cannot be recursed into.

If a custom leaf `yi` has all active values contained in mutable objects, i.e., would be a
valid input to `recursive_map!` (in other words, has inferred activity state Duplicated),
the function `f` must have a corresponding method `f(yi, x1i, x2i, ..., xNi)` that mutates
`y` in-place and returns `nothing`, like `recursive_map!` would.

If a custom leaf `yi` has active values contained in a mix of mutable and immutable
locations, i.e., would be a valid input to `recursive_map_immutable!` (in other words, has
inferred activity state MixedDuplicated), the function `f` must have a corresponding method
`newyi = f(yi, x1i, x2i, ..., xNi)` that updates the mutated parts of `yi` in-place and
reuses them in a newly constructed return value `newyi` with updated values in immutable
locations of active type, like `recursive_map_immutable!` would.

The default leaf types always have inferred activity state Active and only require the
out-of-place method `yi = f(x1i, x2i, ..., xNi)`, like `recursive_map`.
"""
function recursive_map_immutable!(f::F, y::T, xs::T...) where {F,T}
    return recursive_map_immutable!(f, y, Base.IdSet(), xs)::T
end

@inline function recursive_map_immutable!(
    f::F, y::T, seen::Base.IdSet, xs::NTuple{N,T}, isleaftype::L=Returns(false)
) where {F,N,T,L}
    activitystate = active_reg_inner(T, (), nothing, Val(false))
    if activitystate == AnyState  # guaranteed const
        return y
    elseif activitystate == DupState
        recursive_map!(f, y, seen, xs, isleaftype)::Nothing
        return y
    else
        if isleaftype(T)
            return (activitystate == ActiveState) ? f(xs...)::T : f(y, xs...)::T
        else
            return _recursive_map_immutable!(f, y, seen, xs, isleaftype)::T
        end
    end
end

@inline function _recursive_map_immutable!(
    f::F, y::T, seen, xs::NTuple{N,T}, isleaftype
) where {F,N,T}
    @assert !ismutabletype(T)
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)

    @inline function newyi(i)
        yi = getfield(y, i)
        xis = ntuple(j -> getfield(xs[j], i), N)
        ST = Core.Typeof(first(xis))
        if active_reg_inner(ST, (), nothing, Val(true)) == ActiveState #=justActive=#
            return recursive_map_immutable!(f, yi, seen, xis, isleaftype)
        else
            recursive_map!(f, yi, seen, xis, isleaftype)
            return yi
        end
    end

    nf = fieldcount(T)
    if nf == 0
        newy = f(xs...)::T
    elseif isdefined(y, nf) && all(x -> isdefined(x, nf), xs)
        # fast path when all fields are set
        newy = splatnew(T, ntuple(newyi, Val(nf)))
    else
        flds = Vector{Any}(undef, nf)
        nset = nf
        for i = 1:nf
            if isdefined(y, i) && all(x -> isdefined(x, i), xs)
                flds[i] = newyi(i)
            else
                nset = i - 1 # rest of tail must be undefined values
                break
            end
        end
        newy = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nset)
    end
    return newy
end

const _RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

@inline function EnzymeCore.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
) where {RT,copy_if_inactive}
    isleaftype(_) = false
    isleaftype(::Type{<:Union{_RealOrComplexFloat,Array{<:_RealOrComplexFloat}}}) = true
    f(p) = EnzymeCore.make_zero(p)
    return recursive_map(RT, f, seen, (prev,), Val(copy_if_inactive), isleaftype)::RT
end

@inline function EnzymeCore.make_zero(prev::FT) where {FT<:_RealOrComplexFloat}
    return Base.zero(prev)::FT
end

@inline function EnzymeCore.make_zero(prev::Array{FT,N}) where {FT<:_RealOrComplexFloat,N}
    # convert because Base.zero may return different eltype when FT is not concrete
    return convert(Array{FT,N}, Base.zero(prev))::Array{FT,N}
end

@inline function EnzymeCore.make_zero!(prev, seen::Base.IdSet=Base.IdSet())
    isleaftype(_) = false
    isleaftype(::Type{<:Union{_RealOrComplexFloat,Array{<:_RealOrComplexFloat}}}) = true
    f(p) = make_zero(p)
    function f(pout::T, pin::T) where {T}
        @assert pout === pin
        EnzymeCore.make_zero!(pout)
        return nothing
    end
    return recursive_map!(f, prev, seen, (prev,), isleaftype)::Nothing
end

@inline function EnzymeCore.make_zero!(prev::Array{T,N}) where {T<:_RealOrComplexFloat,N}
    fill!(prev, zero(T))
    return nothing
end
