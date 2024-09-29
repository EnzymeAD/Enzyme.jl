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
