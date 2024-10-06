
@inline function EnzymeCore.make_zero(x::FT)::FT where {FT<:AbstractFloat}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(x::Complex{FT})::Complex{FT} where {FT<:AbstractFloat}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(
    x::Array{FT,N},
)::Array{FT,N} where {FT<:AbstractFloat,N}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(
    x::Array{Complex{FT},N},
)::Array{Complex{FT},N} where {FT<:AbstractFloat,N}
    return Base.zero(x)
end

@inline function EnzymeCore.make_zero(
    ::Type{Array{FT,N}},
    seen::IdDict,
    prev::Array{FT,N},
    ::Val{copy_if_inactive} = Val(false),
)::Array{FT,N} where {copy_if_inactive,FT<:AbstractFloat,N}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = Base.zero(prev)
    seen[prev] = newa
    return newa
end
@inline function EnzymeCore.make_zero(
    ::Type{Array{Complex{FT},N}},
    seen::IdDict,
    prev::Array{Complex{FT},N},
    ::Val{copy_if_inactive} = Val(false),
)::Array{Complex{FT},N} where {copy_if_inactive,FT<:AbstractFloat,N}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = Base.zero(prev)
    seen[prev] = newa
    return newa
end

@inline function EnzymeCore.make_zero(
    ::Type{RT},
    seen::IdDict,
    prev::RT,
    ::Val{copy_if_inactive} = Val(false),
)::RT where {copy_if_inactive,RT<:AbstractFloat}
    return RT(0)
end

@inline function EnzymeCore.make_zero(
    ::Type{Complex{RT}},
    seen::IdDict,
    prev::Complex{RT},
    ::Val{copy_if_inactive} = Val(false),
)::Complex{RT} where {copy_if_inactive,RT<:AbstractFloat}
    return RT(0)
end

@inline function EnzymeCore.make_zero(
    ::Type{RT},
    seen::IdDict,
    prev::RT,
    ::Val{copy_if_inactive} = Val(false),
)::RT where {copy_if_inactive,RT<:Array}
    if haskey(seen, prev)
        return seen[prev]
    end
    if guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    newa = RT(undef, size(prev))
    seen[prev] = newa
    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            innerty = Core.Typeof(pv)
            @inbounds newa[I] =
                EnzymeCore.make_zero(innerty, seen, pv, Val(copy_if_inactive))
        end
    end
    return newa
end

@inline function EnzymeCore.make_zero(
    ::Type{RT},
    seen::IdDict,
    prev::RT,
    ::Val{copy_if_inactive} = Val(false),
)::RT where {copy_if_inactive,RT<:Tuple}
    return ntuple(length(prev)) do i
        Base.@_inline_meta
        EnzymeCore.make_zero(RT.parameters[i], seen, prev[i], Val(copy_if_inactive))
    end
end

@inline function EnzymeCore.make_zero(
    ::Type{NamedTuple{A,RT}},
    seen::IdDict,
    prev::NamedTuple{A,RT},
    ::Val{copy_if_inactive} = Val(false),
)::NamedTuple{A,RT} where {copy_if_inactive,A,RT}
    return NamedTuple{A,RT}(EnzymeCore.make_zero(RT, seen, RT(prev), Val(copy_if_inactive)))
end

@inline function EnzymeCore.make_zero(
    ::Type{Core.Box},
    seen::IdDict,
    prev::Core.Box,
    ::Val{copy_if_inactive} = Val(false),
) where {copy_if_inactive}
    if haskey(seen, prev)
        return seen[prev]
    end
    prev2 = prev.contents
    res = Core.Box()
    seen[prev] = res
    res.contents = Base.Ref(
        EnzymeCore.make_zero(Core.Typeof(prev2), seen, prev2, Val(copy_if_inactive)),
    )
    return res
end

@inline function EnzymeCore.make_zero(
    ::Type{RT},
    seen::IdDict,
    prev::RT,
    ::Val{copy_if_inactive} = Val(false),
)::RT where {copy_if_inactive,RT}
    if guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(prev, seen) : prev
    end
    if haskey(seen, prev)
        return seen[prev]
    end
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    if ismutable(prev)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), RT)::RT
        seen[prev] = y
        for i = 1:nf
            if isdefined(prev, i)
                xi = getfield(prev, i)
                T = Core.Typeof(xi)
                xi = EnzymeCore.make_zero(T, seen, xi, Val(copy_if_inactive))
                if Base.isconst(RT, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i-1, xi)
                else
                    setfield!(y, i, xi)
                end
            end
        end
        return y
    end

    if nf == 0
        return prev
    end

    flds = Vector{Any}(undef, nf)
    for i = 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            xi = EnzymeCore.make_zero(Core.Typeof(xi), seen, xi, Val(copy_if_inactive))
            flds[i] = xi
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nf)
    seen[prev] = y
    return y
end

function make_zero_immutable!(prev::T, seen::S)::T where {T<:AbstractFloat,S}
    zero(T)
end

function make_zero_immutable!(
    prev::Complex{T},
    seen::S,
)::Complex{T} where {T<:AbstractFloat,S}
    zero(T)
end

function make_zero_immutable!(prev::T, seen::S)::T where {T<:Tuple,S}
    ntuple(Val(length(T.parameters))) do i
        Base.@_inline_meta
        make_zero_immutable!(prev[i], seen)
    end
end

function make_zero_immutable!(prev::NamedTuple{a,b}, seen::S)::NamedTuple{a,b} where {a,b,S}
    NamedTuple{a,b}(ntuple(Val(length(T.parameters))) do i
        Base.@_inline_meta
        make_zero_immutable!(prev[a[i]], seen)
    end)
end


function make_zero_immutable!(prev::T, seen::S)::T where {T,S}
    if guaranteed_const_nongen(T, nothing)
        return prev
    end
    @assert !ismutable(prev)

    RT = Core.Typeof(prev)
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)

    flds = Vector{Any}(undef, nf)
    for i = 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            ST = Core.Typeof(xi)
            flds[i] = if active_reg_inner(ST, (), nothing, Val(true)) == ActiveState #=justActive=#
                make_zero_immutable!(xi, seen)
            else
                EnzymeCore.make_zero!(xi, seen)
                xi
            end
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nf)::T
end

@inline function EnzymeCore.make_zero!(
    prev::Base.RefValue{T},
    seen::ST,
)::Nothing where {T<:AbstractFloat,ST}
    T[] = zero(T)
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Base.RefValue{Complex{T}},
    seen::ST,
)::Nothing where {T<:AbstractFloat,ST}
    T[] = zero(Complex{T})
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Array{T,N},
    seen::ST,
)::Nothing where {T<:AbstractFloat,N,ST}
    fill!(prev, zero(T))
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Array{Complex{T},N},
    seen::ST,
)::Nothing where {T<:AbstractFloat,N,ST}
    fill!(prev, zero(Complex{T}))
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Base.RefValue{T},
)::Nothing where {T<:AbstractFloat}
    EnzymeCore.make_zero!(prev, nothing)
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Base.RefValue{Complex{T}},
)::Nothing where {T<:AbstractFloat}
    EnzymeCore.make_zero!(prev, nothing)
    nothing
end

@inline function EnzymeCore.make_zero!(prev::Array{T,N})::Nothing where {T<:AbstractFloat,N}
    EnzymeCore.make_zero!(prev, nothing)
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Array{Complex{T},N},
)::Nothing where {T<:AbstractFloat,N}
    EnzymeCore.make_zero!(prev, nothing)
    nothing
end

@inline function EnzymeCore.make_zero!(prev::Array{T,N}, seen::ST)::Nothing where {T,N,ST}
    if guaranteed_const_nongen(T, nothing)
        return
    end
    if in(seen, prev)
        return
    end
    push!(seen, prev)

    for I in eachindex(prev)
        if isassigned(prev, I)
            pv = prev[I]
            SBT = Core.Typeof(pv)
            if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
                @inbounds prev[I] = make_zero_immutable!(pv, seen)
                nothing
            else
                EnzymeCore.make_zero!(pv, seen)
                nothing
            end
        end
    end
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Base.RefValue{T},
    seen::ST,
)::Nothing where {T,ST}
    if guaranteed_const_nongen(T, nothing)
        return
    end
    if in(seen, prev)
        return
    end
    push!(seen, prev)

    pv = prev[]
    SBT = Core.Typeof(pv)
    if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
        prev[] = make_zero_immutable!(pv, seen)
        nothing
    else
        EnzymeCore.make_zero!(pv, seen)
        nothing
    end
    nothing
end

@inline function EnzymeCore.make_zero!(prev::Core.Box, seen::ST)::Nothing where {ST}
    pv = prev.contents
    T = Core.Typeof(pv)
    if guaranteed_const_nongen(T, nothing)
        return
    end
    if in(seen, prev)
        return
    end
    push!(seen, prev)
    SBT = Core.Typeof(pv)
    if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
        prev.contents = EnzymeCore.make_zero_immutable!(pv, seen)
        nothing
    else
        EnzymeCore.make_zero!(pv, seen)
        nothing
    end
    nothing
end

@inline function EnzymeCore.make_zero!(
    prev::T,
    seen::S = Base.IdSet{Any}(),
)::Nothing where {T,S}
    if guaranteed_const_nongen(T, nothing)
        return
    end
    if in(prev, seen)
        return
    end
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)


    if nf == 0
        return
    end

    push!(seen, prev)

    for i = 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            SBT = Core.Typeof(xi)
            if guaranteed_const_nongen(SBT, nothing)
                continue
            end
            if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
                setfield!(prev, i, make_zero_immutable!(xi, seen))
                nothing
            else
                EnzymeCore.make_zero!(xi, seen)
                nothing
            end
        end
    end
    return
end
