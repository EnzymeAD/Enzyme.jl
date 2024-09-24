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
    return Complex{RT}(0)
end

@inline function EnzymeCore.make_zero(
    ::Type{RT}, seen::IdDict, prev::RT, copyval::Val{copy_if_inactive}=Val(false)
) where {copy_if_inactive,RT}
    function f(p)
        T = Core.Typeof(p)
        if guaranteed_const_nongen(T, nothing)
            return (copy_if_inactive ? Base.deepcopy_internal(p, seen) : p)::T
        end
        return EnzymeCore.make_zero(T, seen, p, copyval)::T
    end
    function isleaftype(::Type{T}) where {T}
        baseTs = Union{
            AbstractFloat,
            Complex{<:AbstractFloat},
            Array{<:AbstractFloat},
            Array{<:Complex{<:AbstractFloat}},
        }
        return (T <: baseTs) || guaranteed_const_nongen(T, nothing)
    end
    return recursive_map(RT, f, seen, (prev,), isleaftype)::RT
end

recursive_map(f::F, x::T...) where {F,T} = recursive_map(T, f, IdDict(), x)::T

@inline function recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, isleaftype::L=Returns(false)
) where {RT,F,N,L}
    if isleaftype(RT)
        return f(xs...)::RT
    end
    return _recursive_map(RT, f, seen, xs, isleaftype)::RT
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, isleaftype
) where {RT<:Array,F,N}
    if haskey(seen, xs)
        return seen[xs]
    end
    x1 = first(xs)
    s = size(x1)
    @assert all(x -> (size(x) == s), xs[2:end])
    y = RT(undef, s)
    seen[xs] = y
    for i in eachindex(x1)
        if all(x -> isassigned(x, i), xs)
            xis = ntuple(j -> xs[j][i], N)
            T = Core.Typeof(first(xis))
            @inbounds y[i] = recursive_map(T, f, seen, xis, isleaftype)
        end
    end
    return y
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, isleaftype
) where {M,RT<:NTuple{M,Any},F,N}
    return ntuple(M) do i
        Base.@_inline_meta
        xis = ntuple(j -> xs[j][i], N)
        recursive_map(RT.parameters[i], f, seen, xis, isleaftype)
    end
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, isleaftype
) where {T,RT<:NamedTuple{<:Any,T},F,N}
    y = recursive_map(T, f, seen, ntuple(i -> T(xs[i]), N), isleaftype)
    return RT(y)
end

@inline function _recursive_map(
    ::Type{Core.Box}, f::F, seen::IdDict, xs::NTuple{N,Core.Box}, isleaftype
) where {F,N}
    if haskey(seen, xs)
        return seen[xs]
    end
    xcontents = ntuple(i -> xs[i].contents, N)
    T = Core.Typeof(first(xcontents))
    res = Core.Box()
    seen[xs] = res
    res.contents = Base.Ref(recursive_map(T, f, seen, xcontents, isleaftype))
    return res
end

@inline function _recursive_map(
    ::Type{RT}, f::F, seen::IdDict, xs::NTuple{N,RT}, isleaftype
) where {RT,F,N}
    if haskey(seen, xs)
        return seen[xs]
    end
    @assert !Base.isabstracttype(RT)
    @assert Base.isconcretetype(RT)
    nf = fieldcount(RT)
    
    if ismutabletype(RT)
        y = ccall(:jl_new_struct_uninit, Any, (Any,), RT)
        seen[xs] = y
        for i in 1:nf
            if all(x -> isdefined(x, i), xs)
                xis = ntuple(j -> getfield(xs[j], i), N)
                T = Core.Typeof(first(xis))
                yi = recursive_map(T, f, seen, xis, isleaftype)
                if Base.isconst(RT, i)
                    ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), y, i - 1, yi)
                else
                    setfield!(y, i, yi)
                end
            end
        end
        return y
    end
    
    if nf == 0
        y = f(xs...)
        seen[xs] = y
        return y
    end

    flds = Vector{Any}(undef, nf)
    for i in 1:nf
        if isdefined(xs, i)
            xis = ntuple(j -> getfield(xs[j], i), N)
            T = Core.Typeof(first(xis))
            yi = recursive_map(T, f, seen, xis, isleaftype)
            flds[i] = yi
        else
            nf = i - 1 # rest of tail must be undefined values
            break
        end
    end
    y = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), RT, flds, nf)
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
