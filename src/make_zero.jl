const _RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

@inline function EnzymeCore.make_zero(prev::FT) where {FT<:_RealOrComplexFloat}
    return Base.zero(prev)::FT
end

@inline function EnzymeCore.make_zero(
    ::Type{FT},
    @nospecialize(seen::IdDict),
    prev::FT,
    @nospecialize(_::Val{copy_if_inactive}=Val(false)),
) where {FT<:_RealOrComplexFloat,copy_if_inactive}
    return EnzymeCore.make_zero(prev)::FT
end

@inline function EnzymeCore.make_zero(prev::Array{FT,N}) where {FT<:_RealOrComplexFloat,N}
    # convert because Base.zero may return different eltype when FT is not concrete
    return convert(Array{FT,N}, Base.zero(prev))::Array{FT,N}
end

@inline function EnzymeCore.make_zero(
    ::Type{Array{FT,N}},
    seen::IdDict,
    prev::Array{FT,N},
    @nospecialize(_::Val{copy_if_inactive}=Val(false)),
) where {FT<:_RealOrComplexFloat,N,copy_if_inactive}
    if haskey(seen, prev)
        return seen[prev]::Array{FT,N}
    end
    newa = EnzymeCore.make_zero(prev)
    seen[prev] = newa
    return newa::Array{FT,N}
end

@inline function EnzymeCore.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, ::Val{copy_if_inactive}=Val(false)
) where {RT,copy_if_inactive}
    isleaftype(_) = false
    isleaftype(::Type{<:Union{_RealOrComplexFloat,Array{<:_RealOrComplexFloat}}}) = true
    f(p) = EnzymeCore.make_zero(Core.Typeof(p), seen, p, Val(copy_if_inactive))
    return recursive_map(RT, f, seen, (prev,), Val(copy_if_inactive), isleaftype)::RT
end

recursive_map(f::F, xs::T...) where {F,T} = recursive_map(T, f, IdDict(), xs)::T

@inline function recursive_map(
    ::Type{RT},
    f::F,
    seen::IdDict,
    xs::NTuple{N,RT},
    ::Val{copy_if_inactive}=Val(false),
    isleaftype::L=Returns(false),
) where {RT,F,N,L,copy_if_inactive}
    if guaranteed_const_nongen(RT, nothing)
        return copy_if_inactive ? Base.deepcopy_internal(first(xs), seen) : first(xs)
    elseif isleaftype(RT)
        return f(xs...)::RT
    end
    return _recursive_map(RT, f, seen, xs, Val(copy_if_inactive), isleaftype)::RT
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, args...
) where {RT<:Array,F,N}
    if haskey(seen, xs)
        return seen[xs]::RT
    end
    y = RT(undef, size(first(xs)))
    seen[xs] = y
    for I in eachindex(xs...)
        if all(x -> isassigned(x, I), xs)
            xIs = ntuple(j -> xs[j][I], N)
            ST = Core.Typeof(first(xIs))
            @inbounds y[I] = recursive_map(ST, f, seen, xIs, args...)
        end
    end
    return y
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, args...
) where {M,RT<:NTuple{M,Any},F,N}
    return ntuple(M) do i
        Base.@_inline_meta
        recursive_map(RT.parameters[i], f, seen, ntuple(j -> xs[j][i], N), args...)
    end
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, args...
) where {T,RT<:NamedTuple{<:Any,T},F,N}
    yT = recursive_map(T, f, seen, ntuple(j -> T(xs[j]), N), args...)
    return RT(yT)
end

@inline function _recursive_map(
    ::Type{Core.Box}, f::F, seen::IdDict, xs::NTuple{N,Core.Box}, args...
) where {F,N}
    if haskey(seen, xs)
        return seen[xs]::Core.Box
    end
    xcontents = ntuple(j -> xs[j].contents, N)
    ST = Core.Typeof(first(xcontents))
    res = Core.Box()
    seen[xs] = res
    res.contents = Base.Ref(recursive_map(ST, f, seen, xcontents, args...))
    return res
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, args...
) where {RT,F,N}
    if haskey(seen, xs)
        return seen[xs]::RT
    end
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
    elseif nf == 0
        y = f(xs...)::RT
    elseif all(x -> isdefined(x, nf), xs)
        # fast path when all fields are set
        y = splatnew(RT, ntuple(newyi, Val(nf)))
    else
        flds = Vector{Any}(undef, nf)
        nset = nf
        for i in 1:nf
            if all(x -> isdefined(x, i), xs)
                flds[i] = newyi(i)
            else
                nset = i - 1 # rest of tail must be undefined values
                break
            end
        end
        y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nset)
    end
    seen[xs] = y
    return y
end

@inline function recursive_map_immutable!(f::F, y::T, xs::T...) where {F,T}
    return recursive_map_immutable!(f, y, Base.IdSet(), xs)::T
end

@inline function recursive_map_immutable!(
    f::F, y::T, seen::Base.IdSet, xs::NTuple{N,T}, isleaftype::L=Returns(false)
) where {F,N,T,L}
    if guaranteed_const_nongen(T, nothing)
        return y
    elseif isleaftype(T)
        # If there exist T such that isleaftype(T) and T does not have mutable content that
        # is not guaranteed const, then f must have a corresponding non-mutating method:
        return f(xs...)::T
    end
    return _recursive_map_immutable!(f, y, seen, xs, isleaftype)::T
end

@inline function _recursive_map_immutable!(
    f::F, y::T, seen, xs::NTuple{N,T}, isleaftype
) where {F,M,T<:NTuple{M,Any},N}
    return ntuple(M) do i
        Base.@_inline_meta
        recursive_map_immutable!(f, y[i], seen, ntuple(j -> xs[j][i], N), isleaftype)
    end
end

@inline function _recursive_map_immutable!(
    f::F, y::NT, seen, xs::NTuple{N,NT}, isleaftype
) where {F,T,NT<:NamedTuple{<:Any,T},N}
    newTy = recursive_map_immutable!(f, T(y), seen, ntuple(j -> T(xs[j]), N), isleaftype)
    return NT(newTy)
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
        return if active_reg_inner(ST, (), nothing, Val(true)) == ActiveState #=justActive=#
            recursive_map_immutable!(f, yi, seen, xis, isleaftype)
        else
            recursive_map!(f, yi, seen, xis, isleaftype)
            yi
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

@inline function EnzymeCore.make_zero!(prev::Base.RefValue{T}) where {T<:_RealOrComplexFloat}
    prev[] = zero(T)
    return nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Base.RefValue{T}, seen::Base.IdSet,
) where {T<:_RealOrComplexFloat}
    if prev in seen
        return nothing
    end
    push!(seen, prev)
    EnzymeCore.make_zero!(prev)
    return nothing
end

@inline function EnzymeCore.make_zero!(prev::Array{T,N}) where {T<:_RealOrComplexFloat,N}
    fill!(prev, zero(T))
    return nothing
end

@inline function EnzymeCore.make_zero!(
    prev::Array{T,N}, seen::Base.IdSet,
) where {T<:_RealOrComplexFloat,N}
    if prev in seen
        return nothing
    end
    push!(seen, prev)
    EnzymeCore.make_zero!(prev)
    return nothing
end

@inline function EnzymeCore.make_zero!(prev, seen::Base.IdSet=Base.IdSet())
    LeafType = Union{
        _RealOrComplexFloat,
        Base.RefValue{<:_RealOrComplexFloat},
        Array{<:_RealOrComplexFloat},
    }
    isleaftype(_) = false
    isleaftype(::Type{<:LeafType}) = true
    f(p) = make_zero(p)
    function f(pout::T, pin::T) where {T}
        @assert pout === pin
        EnzymeCore.make_zero!(pout, seen)
        return nothing
    end
    return recursive_map!(f, prev, seen, (prev,), isleaftype)::Nothing
end

@inline function recursive_map!(f::F, y::T, xs::T...) where {F,T}
    return recursive_map!(f, y, Base.IdSet(), xs)::Nothing
end

@inline function recursive_map!(
    f::F, y::T, seen::Base.IdSet, xs::NTuple{N,T}, isleaftype::L=Returns(false)
) where {F,T,N,L}
    if guaranteed_const_nongen(T, nothing)
        return nothing
    elseif isleaftype(T)
        # If there exist T such that isleaftype(T) and T has mutable content that is not
        # guaranteed const, including mutables nested inside immutables like Tuple{Vector},
        # then f must have a corresponding mutating method:
        f(y, xs...)
        return nothing
    end
    return _recursive_map!(f, y, seen, xs, isleaftype)::Nothing
end

@inline function _recursive_map!(
    f::F, y::Array{T,M}, seen, xs::NTuple{N,Array{T,M}}, isleaftype
) where {F,T,M,N}
    if y in seen
        return nothing
    end
    push!(seen, y)
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

@inline function _recursive_map!(
    f::F, y::Base.RefValue{T}, seen, xs::NTuple{N,Base.RefValue{T}}, isleaftype
) where {F,T,N}
    if y in seen
        return nothing
    end
    push!(seen, y)
    yvalue = y[]
    xvalues = ntuple(j -> xs[j][], N)
    SBT = Core.Typeof(yvalue)
    if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
        y[] = recursive_map_immutable!(f, yvalue, seen, xvalues, isleaftype)
    else
        recursive_map!(f, yvalue, seen, xvalues, isleaftype)
    end
    return nothing
end

@inline function _recursive_map!(
    f::F, y::Core.Box, seen, xs::NTuple{N,Core.Box}, isleaftype
) where {F,N}
    if y in seen
        return nothing
    end
    push!(seen, y)
    ycontents = y.contents
    xcontents = ntuple(j -> xs[j].contents, N)
    SBT = Core.Typeof(ycontents)
    if active_reg_inner(SBT, (), nothing, Val(true)) == ActiveState #=justActive=#
        y.contents = recursive_map_immutable!(f, ycontents, seen, xcontents, isleaftype)
    else
        recursive_map!(f, ycontents, seen, xcontents, isleaftype)
    end
    return nothing
end

@inline function _recursive_map!(
    f::F, y::T, seen, xs::NTuple{N,T}, isleaftype
) where {F,T,N}
    if y in seen
        return nothing
    end
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)
    if nf == 0
        return nothing
    end
    push!(seen, y)
    for i = 1:nf
        if isdefined(y, i) && all(x -> isdefined(x, i), xs)
            yi = getfield(y, i)
            xis = ntuple(j -> getfield(xs[j], i), N)
            SBT = Core.Typeof(yi)
            activitystate = active_reg_inner(SBT, (), nothing, Val(false))
            if activitystate == AnyState
                continue
            elseif activitystate == DupState
                recursive_map!(f, yi, seen, xis, isleaftype)
            else
                yi = recursive_map_immutable!(f, yi, seen, xis, isleaftype)
                if Base.isconst(T, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, yi)
                else
                    setfield!(y, i, yi)
                end
            end
        end
    end
    return nothing
end
