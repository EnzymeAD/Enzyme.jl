module EnzymeCUDAExt

using CUDA
using Enzyme
using Enzyme: EnzymeRules

function _zero!(ptr::Ptr{T}, off::Integer, n::Integer) where {T <: AbstractFloat}
    n == 0 && return nothing
    Base.Libc.memset(ptr + off * sizeof(T), 0, n * sizeof(T))
    return nothing
end
function _zero!(ptr::CuPtr{T}, off::Integer, n::Integer) where {T <: AbstractFloat}
    n == 0 && return nothing
    bytes = reinterpret(CuPtr{UInt8}, ptr + off * sizeof(T))
    CUDA.memset(bytes, UInt8(0), n * sizeof(T))
    return nothing
end

#= Element-wise `dst .+= src`, moving across host/device as needed.

 The device path could instead call cuBLAS via `axpy!(one(eltype(dst)), src, dst)`,
 but broadcast is faster in the size regime these shadows live in:

eltype    n           broadcast(us)   axpy(us)        speedup(bcast/axpy)
Float32   1           7.896           13.734          0.57
Float32   16          8.115           13.123          0.62
Float32   256         8.668           13.615          0.64
Float32   4096        8.096           13.318          0.61
Float32   65536       8.483           13.476          0.63
Float32   1048576     8.266           13.597          0.61
Float32   16777216    96.727          95.242          1.02
Float64   1           9.326           13.458          0.69
Float64   16          9.092           13.423          0.68
Float64   256         9.014           14.109          0.64
Float64   4096        8.579           13.794          0.62
Float64   65536       8.256           14.423          0.57
Float64   1048576     9.157           13.786          0.66
Float64   16777216    256.104         257.62          0.99
=#
_accumulate!(dst::CuArray, src::CuArray) = (dst .+= src; nothing)
_accumulate!(dst::Array, src::Array) = (dst .+= src; nothing)
_accumulate!(dst::CuArray, src::Array) = _accumulate!(dst, CuArray(src))
_accumulate!(dst::Array, src::CuArray) = _accumulate!(dst, Array(src))

function _accumulate!(
        acc::Union{Ptr, CuPtr}, aoff::Integer,
        val::Union{Ptr, CuPtr}, voff::Integer, n::Integer,
    )
    dst = acc isa CuPtr ? unsafe_wrap(CuArray, acc + aoff, n; own = false) :
        unsafe_wrap(Array, acc + aoff, n; own = false)
    src = val isa CuPtr ? unsafe_wrap(CuArray, val + voff, n; own = false) :
        unsafe_wrap(Array, val + voff, n; own = false)
    _accumulate!(dst, src)
    return nothing
end

function _accumulate_and_zero!(src, soff, dest, doff, n::Integer)
    n == 0 && return nothing
    _accumulate!(src, soff, dest, doff, n)
    _zero!(dest, doff, n)
    return nothing
end

@inline function _shadow(x, config, batch)
    return EnzymeRules.width(config) == 1 ? x.dval : x.dval[batch]
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(pointer)},
        ::Type{RT},
        array::Annotation{<:StridedCuArray},
        index::Const;
        kwargs...,
    ) where {RT}
    primal = if EnzymeRules.needs_primal(config)
        func.val(array.val, index.val; kwargs...)
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config) && !(array isa Const)
        if EnzymeRules.width(config) == 1
            func.val(array.dval, index.val; kwargs...)
        else
            ntuple(Val(EnzymeRules.width(config))) do batch
                func.val(array.dval[batch], index.val; kwargs...)
            end
        end
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(pointer)},
        ::Type{RT},
        tape,
        array::Annotation{<:StridedCuArray},
        index::Const;
        kwargs...,
    ) where {RT}
    return (nothing, nothing)
end

const COPY_DIRECTIONS = (
    (Ptr, CuPtr),
    (CuPtr, Ptr),
    (CuPtr, CuPtr),
)

for (DstPtr, SrcPtr) in COPY_DIRECTIONS
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:$DstPtr},
                src::Annotation{<:$SrcPtr},
                n::Const;
                kwargs...,
            ) where {RT}
            func.val(dest.val, src.val, n.val; kwargs...)
            primal = EnzymeRules.needs_primal(config) ? dest.val : nothing
            shadow = if !(RT <: Const) && EnzymeRules.needs_shadow(config) &&
                    !(dest isa Const)
                dest.dval
            else
                nothing
            end
            return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                tape,
                dest::Annotation{<:$DstPtr},
                src::Annotation{<:$SrcPtr},
                n::Const;
                kwargs...,
            ) where {RT}
            if !(dest isa Const)
                for batch in 1:EnzymeRules.width(config)
                    ddest = _shadow(dest, config, batch)
                    if src isa Const
                        _zero!(ddest, 0, n.val)
                    else
                        _accumulate_and_zero!(
                            _shadow(src, config, batch), 0, ddest, 0, n.val
                        )
                    end
                end
            end
            return (nothing, nothing, nothing)
        end
    end
end

end # module
