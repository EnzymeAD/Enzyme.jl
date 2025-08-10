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

@static if VERSION < v"1.11-"
else
@inline function EnzymeCore.make_zero(
    x::GenericMemory{kind, FT},
)::GenericMemory{kind, FT} where {FT<:AbstractFloat,kind}
    return Base.zero(x)
end
@inline function EnzymeCore.make_zero(
    x::GenericMemory{kind, Complex{FT}},
)::GenericMemory{kind, Complex{FT}} where {FT<:AbstractFloat,kind}
    return Base.zero(x)
end
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

@static if VERSION < v"1.11-"
else
@inline function EnzymeCore.make_zero(
    ::Type{GenericMemory{kind, FT}},
    seen::IdDict,
    prev::GenericMemory{kind, FT},
    ::Val{copy_if_inactive} = Val(false),
)::GenericMemory{kind, FT} where {copy_if_inactive,FT<:AbstractFloat,kind}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = Base.zero(prev)
    seen[prev] = newa
    return newa
end
@inline function EnzymeCore.make_zero(
    ::Type{GenericMemory{kind, Complex{FT}}},
    seen::IdDict,
    prev::GenericMemory{kind, Complex{FT}},
    ::Val{copy_if_inactive} = Val(false),
)::GenericMemory{kind, Complex{FT}} where {copy_if_inactive,FT<:AbstractFloat,kind}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = Base.zero(prev)
    seen[prev] = newa
    return newa
end
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
    return Complex{RT}(0)
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

@static if VERSION < v"1.11-"
else
@inline function EnzymeCore.make_zero(
    ::Type{RT},
    seen::IdDict,
    prev::RT,
    ::Val{copy_if_inactive} = Val(false),
)::RT where {copy_if_inactive,RT<:GenericMemory}
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
    prevtup = RT(prev)
    TT = Core.Typeof(prevtup)  # RT can be abstract
    return NamedTuple{A,RT}(EnzymeCore.make_zero(TT, seen, prevtup, Val(copy_if_inactive)))
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
    res.contents = EnzymeCore.make_zero(Core.Typeof(prev2), seen, prev2, Val(copy_if_inactive))
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
    return zero(T)
end

function make_zero_immutable!(
    prev::Complex{T},
    seen::S,
)::Complex{T} where {T<:AbstractFloat,S}
    return zero(Complex{T})
end

function make_zero_immutable!(prev::T, seen::S)::T where {T<:Tuple,S}
    if guaranteed_const_nongen(T, nothing)
        return prev  # unreachable from make_zero!
    end
    ntuple(Val(length(T.parameters))) do i
        Base.@_inline_meta
        p = prev[i]
        SBT = Core.Typeof(p)
        if guaranteed_const_nongen(SBT, nothing)
            p  # covered by several tests even if not shown in coverage
        elseif !ismutabletype(SBT)
            make_zero_immutable!(p, seen)
        else
            EnzymeCore.make_zero!(p, seen)
            p
        end
    end
end

function make_zero_immutable!(prev::NamedTuple{a,b}, seen::S)::NamedTuple{a,b} where {a,b,S}
    if guaranteed_const_nongen(NamedTuple{a,b}, nothing)
        return prev  # unreachable from make_zero!
    end
    NamedTuple{a,b}(ntuple(Val(length(b.parameters))) do i
        Base.@_inline_meta
        p = prev[a[i]]
        SBT = Core.Typeof(p)
        if guaranteed_const_nongen(SBT, nothing)
            p  # covered by several tests even if not shown in coverage
        elseif !ismutabletype(SBT)
            make_zero_immutable!(p, seen)
        else
            EnzymeCore.make_zero!(p, seen)
            p
        end
    end)
end


function make_zero_immutable!(prev::T, seen::S)::T where {T,S}
    if guaranteed_const_nongen(T, nothing)
        return prev  # unreachable from make_zero!
    end
    @assert !ismutabletype(T)
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    flds = Vector{Any}(undef, nf)
    for i = 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            ST = Core.Typeof(xi)
            flds[i] = if guaranteed_const_nongen(ST, nothing)
                xi
            elseif !ismutabletype(ST)
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
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, flds, nf)::T
end

macro register_make_zero_inplace(sym)
    quote
        @inline function $sym(
            prev::Base.RefValue{T},
            seen::ST,
        )::Nothing where {T<:AbstractFloat,ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            prev[] = zero(T)
            return nothing
        end

        @inline function $sym(
            prev::Base.RefValue{Complex{T}},
            seen::ST,
        )::Nothing where {T<:AbstractFloat,ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            prev[] = zero(Complex{T})
            return nothing
        end
                @inline function $sym(
            prev::Array{T,N},
            seen::ST,
        )::Nothing where {T<:AbstractFloat,N,ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(T))
            return nothing
        end

        @inline function $sym(
            prev::Array{Complex{T},N},
            seen::ST,
        )::Nothing where {T<:AbstractFloat,N,ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(Complex{T}))
            return nothing
        end

        @static if VERSION < v"1.11-"
        else
        @inline function $sym(
            prev::GenericMemory{kind, T},
            seen::ST,
        )::Nothing where {T<:AbstractFloat,kind,ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(T))
            return nothing
        end

        @inline function $sym(
            prev::GenericMemory{kind, Complex{T}},
            seen::ST,
        )::Nothing where {T<:AbstractFloat,kind,ST}
            if !isnothing(seen)
                if prev in seen
                    return nothing
                end
                push!(seen, prev)
            end
            fill!(prev, zero(Complex{T}))
            return nothing
        end
        end

        @inline function $sym(
            prev::Base.RefValue{T},
        )::Nothing where {T<:AbstractFloat}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(
            prev::Base.RefValue{Complex{T}},
        )::Nothing where {T<:AbstractFloat}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(prev::Array{T,N})::Nothing where {T<:AbstractFloat,N}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(
            prev::Array{Complex{T},N},
        )::Nothing where {T<:AbstractFloat,N}
            $sym(prev, nothing)
            return nothing
        end

        @static if VERSION < v"1.11-"
        else
        @inline function $sym(
            prev::GenericMemory{kind, T}
        )::Nothing where {T<:AbstractFloat,kind}
            $sym(prev, nothing)
            return nothing
        end

        @inline function $sym(
            prev::GenericMemory{kind, Complex{T}}
        )::Nothing where {T<:AbstractFloat,kind}
            $sym(prev, nothing)
            return nothing
        end
        end

        @inline function $sym(prev::Array{T,N}, seen::ST)::Nothing where {T,N,ST}
            if guaranteed_const_nongen(T, nothing)
                return nothing
            end
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            for I in eachindex(prev)
                if isassigned(prev, I)
                    pv = prev[I]
                    SBT = Core.Typeof(pv)
                    if guaranteed_const_nongen(SBT, nothing)
                        continue
                    elseif !ismutabletype(SBT)
                        @inbounds prev[I] = make_zero_immutable!(pv, seen)
                    else
                        $sym(pv, seen)
                    end
                end
            end
            return nothing
        end

        @static if VERSION < v"1.11-"
        else
        @inline function $sym(prev::GenericMemory{kind, T}, seen::ST)::Nothing where {T,kind,ST}
            if guaranteed_const_nongen(T, nothing)
                return nothing
            end
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            for I in eachindex(prev)
                if isassigned(prev, I)
                    pv = prev[I]
                    SBT = Core.Typeof(pv)
                    if guaranteed_const_nongen(SBT, nothing)
                        continue
                    elseif !ismutabletype(SBT)
                        @inbounds prev[I] = make_zero_immutable!(pv, seen)
                    else
                        $sym(pv, seen)
                    end
                end
            end
            return nothing
        end
        end

        @inline function $sym(
            prev::Base.RefValue{T},
            seen::ST,
        )::Nothing where {T,ST}
            if guaranteed_const_nongen(T, nothing)
                return nothing
            end
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            pv = prev[]
            SBT = Core.Typeof(pv)
            if guaranteed_const_nongen(SBT, nothing)
                return nothing
            elseif !ismutabletype(SBT)
                prev[] = make_zero_immutable!(pv, seen)
            else
                $sym(pv, seen)
            end
            return nothing
        end

        @inline function $sym(prev::Core.Box, seen::ST)::Nothing where {ST}
            if prev in seen
                return nothing
            end
            push!(seen, prev)
            pv = prev.contents
            SBT = Core.Typeof(pv)
            if guaranteed_const_nongen(SBT, nothing)
                return nothing
            elseif !ismutabletype(SBT)
                prev.contents = make_zero_immutable!(pv, seen)
            else
                $sym(pv, seen)
            end
            return nothing
        end

        @inline $sym(prev) = $sym(prev, Base.IdSet())
    end
end

@register_make_zero_inplace(Enzyme.make_zero!)
@register_make_zero_inplace(Enzyme.remake_zero!)

@inline function EnzymeCore.make_zero!(prev::T, seen::S)::Nothing where {T,S}
    if guaranteed_const_nongen(T, nothing)
        return nothing
    end
    if prev in seen
        return nothing
    end
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    push!(seen, prev)
    for i = 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            SBT = Core.Typeof(xi)
            activitystate = active_reg_inner(SBT, (), nothing)
            if activitystate == AnyState  # guaranteed_const
                continue
            elseif ismutabletype(T) && !ismutabletype(SBT)
                yi = make_zero_immutable!(xi, seen)
                if Base.isconst(T, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), prev, i-1, yi)
                else
                    setfield!(prev, i, yi)
                end
            elseif activitystate == DupState
                EnzymeCore.make_zero!(xi, seen)
            else
                msg = "cannot set $xi to zero in-place, as it contains differentiable values in immutable positions\nIf the argument is known to have all immutable positions already zero (e.g. was the result of Enzyme.make_zero), Enzyme.remake_zero! will skip this error check."
                throw(ArgumentError(msg))
            end
        end
    end
    return nothing
end

@inline function EnzymeCore.remake_zero!(prev::T, seen::S)::Nothing where {T,S}
    if guaranteed_const_nongen(T, nothing)
        return nothing
    end
    if prev in seen
        return nothing
    end
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    push!(seen, prev)
    for i = 1:nf
        if isdefined(prev, i)
            xi = getfield(prev, i)
            SBT = Core.Typeof(xi)
            activitystate = active_reg_inner(SBT, (), nothing)
            if activitystate == AnyState  # guaranteed_const
                continue
            elseif ismutabletype(T) && !ismutabletype(SBT)
                yi = make_zero_immutable!(xi, seen)
                if Base.isconst(T, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), prev, i-1, yi)
                else
                    setfield!(prev, i, yi)
                end
            elseif activitystate == DupState
                EnzymeCore.make_zero!(xi, seen)
            elseif activitystate == MixedState
                EnzymeCore.remake_zero!(xi, seen)
            end
        end
    end
    return nothing
end
