# Recursively return x + f(y), where y is active, otherwise x

@inline function recursive_add(
    x::T,
    y::T,
    f::F = identity,
    forcelhs::F2 = guaranteed_const,
) where {T,F,F2}
    if forcelhs(T)
        return x
    end
    splatnew(T, ntuple(Val(fieldcount(T))) do i
        Base.@_inline_meta
        prev = getfield(x, i)
        next = getfield(y, i)
        recursive_add(prev, next, f, forcelhs)
    end)
end

@inline function recursive_add(
    x::T,
    y::T,
    f::F = identity,
    forcelhs::F2 = guaranteed_const,
) where {T<:AbstractFloat,F,F2}
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
) where {T<:Complex,F,F2}
    if forcelhs(T)
        return x
    end
    return x + f(y)
end

@inline mutable_register(::Type{T}) where {T<:Integer} = true
@inline mutable_register(::Type{T}) where {T<:AbstractFloat} = false
@inline mutable_register(::Type{Complex{T}}) where {T<:AbstractFloat} = false
@inline mutable_register(::Type{T}) where {T<:Tuple} = false
@inline mutable_register(::Type{T}) where {T<:NamedTuple} = false
@inline mutable_register(::Type{Core.Box}) = true
@inline mutable_register(::Type{T}) where {T<:Array} = true
@inline mutable_register(::Type{T}) where {T} = ismutabletype(T)

@inline function atomicrmw_add!(ptr::Ptr{Float64}, val::Float64)
    Base.llvmcall(
        "atomicrmw fadd double* %0, double %1 monotonic\nret void",
        Cvoid,
        Tuple{Ptr{Float64}, Float64},
        ptr,
        val
    )
end

@inline function atomicrmw_add!(ptr::Ptr{Float32}, val::Float32)
    Base.llvmcall(
        "atomicrmw fadd float* %0, float %1 monotonic\nret void",
        Cvoid,
        Tuple{Ptr{Float32}, Float32},
        ptr,
        val
    )
end


@generated function atomic_accumulate!(x::Ptr{T}, y::Ptr{T}, f::F = identity) where {T,F}
    if !Base.isabstracttype(T) && Base.isconcretetype(T)
        if T === Float64
            return quote
                atomicrmw_add!(Ptr{Float64}(x), Float64(f(unsafe_load(y))))
                return nothing
            end
        elseif T === Float32
            return quote
                atomicrmw_add!(Ptr{Float32}(x), Float32(f(unsafe_load(y))))
                return nothing
            end
        elseif T <: AbstractFloat
            return quote
                if guaranteed_const($T)
                    return nothing
                else
                    error("Atomic accumulation not supported for type $T")
                end
            end
        else
            nf = fieldcount(T)
            desc = Base.DataTypeFieldDesc(T)
            exprs = []
            for i = 1:nf
                ST = fieldtype(T, i)
                if !desc[i].isptr
                    push!(exprs, quote
                        px = Ptr{$ST}(x + fieldoffset($T, $i))
                        py = Ptr{$ST}(y + fieldoffset($T, $i))
                        atomic_accumulate!(px, py, f)
                    end)
                else
                    push!(exprs, quote
                        if guaranteed_const($ST)
                            # Do nothing
                        else
                            error("Atomic accumulation not supported for boxed type $T")
                        end
                    end)
                end
            end
            return quote
                if guaranteed_const($T)
                    return nothing
                end
                $(exprs...)
                return nothing
            end
        end
    else
        return quote
            if guaranteed_const($T)
                return nothing
            else
                error("Atomic accumulation not supported for type $T")
            end
        end
    end
end

# Recursively In-place accumulate(aka +=). E.g. generalization of x .+= f(y)
@inline function recursive_accumulate(x::Array{T}, y::Array{T}, ::Val{atomic} = Val(false), f::F = identity) where {T,atomic,F}
    if !mutable_register(T)
        GC.@preserve x y begin
            for I in eachindex(x)
                if atomic
                    atomic_accumulate!(pointer(x, I), pointer(y, I), f)
                else
                    @inbounds x[I] = recursive_add(x[I], (@inbounds y[I]), f, mutable_register)
                end
            end
        end
    end
end

@inline function recursive_accumulate(x::Core.Box, y::Core.Box, ::Val{atomic} = Val(false), f::F = identity) where {atomic,F}
    recursive_accumulate(x.contents, y.contents, Val(atomic), f)
end

@generated function recursive_accumulate(x::T, y::T, ::Val{atomic} = Val(false), f::F = identity) where {T,atomic,F}
    @assert !Base.isabstracttype(T)
    @assert Base.isconcretetype(T)
    if !ismutabletype(T)
        return :(nothing)
    end
    if atomic
        return quote
            GC.@preserve x y begin
                px = Ptr{T}(pointer_from_objref(x))
                py = Ptr{T}(pointer_from_objref(y))
                atomic_accumulate!(px, py, f)
            end
            return nothing
        end
    else
        nf = fieldcount(T)
        exprs = []
        for i = 1:nf
            ST = fieldtype(T, i)
            if !mutable_register(ST)
                push!(exprs, quote
                    if isdefined(x, $i)
                        xi = getfield(x, $i)
                        yi = getfield(y, $i)
                        nexti = recursive_add(xi, yi, f, mutable_register)
                        setfield!(x, $i, nexti)
                    end
                end)
            else
                push!(exprs, quote
                    if isdefined(x, $i) && isdefined(y, $i)
                        xi = getfield(x, $i)
                        yi = getfield(y, $i)
                        recursive_accumulate(xi, yi, Val(false), f)
                    end
                end)
            end
        end
        return quote
            $(exprs...)
            return nothing
        end
    end
end

@generated function recursive_accumulate(x::Base.RefValue{T}, y::Base.RefValue{T}, ::Val{atomic} = Val(false), f::F = identity) where {T,atomic,F}
    if ismutabletype(T)
        return quote
            recursive_accumulate(x[], y[], Val($atomic), f)
            return nothing
        end
    else
        return quote
            if $atomic
                GC.@preserve x y begin
                    px = Ptr{T}(pointer_from_objref(x))
                    py = Ptr{T}(pointer_from_objref(y))
                    atomic_accumulate!(px, py, f)
                end
            else
                x[] = recursive_add(x[], y[], f, mutable_register)
            end
            return nothing
        end
    end
end
