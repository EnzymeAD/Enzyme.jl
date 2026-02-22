# Recursively return x + f(y), where y is active, otherwise x

@inline function recursive_add(
        x::T,
        y::T,
        f::F = identity,
        forcelhs::F2 = guaranteed_const,
    ) where {T, F, F2}
    if forcelhs(T)
        return x
    end
    return splatnew(
        T, ntuple(Val(fieldcount(T))) do i
            Base.@_inline_meta
            prev = getfield(x, i)
            next = getfield(y, i)
            recursive_add(prev, next, f, forcelhs)
        end
    )
end

@inline function recursive_add(
        x::T,
        y::T,
        f::F = identity,
        forcelhs::F2 = guaranteed_const,
    ) where {T <: AbstractFloat, F, F2}
    if forcelhs(T)
        return x
    end
    return x + f(y)
end

@inline function recursive_add(
        x::T,
        y::T,
        f::F = identity,
        forcelhs::F2 = guaranteed_const,
    ) where {T <: Complex, F, F2}
    if forcelhs(T)
        return x
    end
    return x + f(y)
end

@inline mutable_register(::Type{T}) where {T <: Integer} = true
@inline mutable_register(::Type{T}) where {T <: AbstractFloat} = false
@inline mutable_register(::Type{Complex{T}}) where {T <: AbstractFloat} = false
@inline mutable_register(::Type{T}) where {T <: Tuple} = false
@inline mutable_register(::Type{T}) where {T <: NamedTuple} = false
@inline mutable_register(::Type{Core.Box}) = true
@inline mutable_register(::Type{T}) where {T <: Array} = true
@inline mutable_register(::Type{T}) where {T} = ismutabletype(T)

# Recursively In-place accumulate(aka +=). E.g. generalization of x .+= f(y)
@inline function recursive_accumulate(x::Array{T}, y::Array{T}, f::F = identity) where {T, F}
    return if !mutable_register(T)
        for I in eachindex(x)
            prev = x[I]
            @inbounds x[I] = recursive_add(x[I], (@inbounds y[I]), f, mutable_register)
        end
    end
end


# Recursively In-place accumulate(aka +=). E.g. generalization of x .+= f(y)
@inline function recursive_accumulate(x::Core.Box, y::Core.Box, f::F = identity) where {F}
    return recursive_accumulate(x.contents, y.contents, seen, f)
end

@inline function recursive_accumulate(x::T, y::T, f::F = identity) where {T, F}
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    nf = fieldcount(T)

    for i in 1:nf
        if isdefined(x, i)
            xi = getfield(x, i)
            ST = Core.Typeof(xi)
            if !mutable_register(ST)
                @assert ismutable(x)
                yi = getfield(y, i)
                nexti = recursive_add(xi, yi, f, mutable_register)
                setfield!(x, i, nexti)
            end
        end
    end
    return
end
