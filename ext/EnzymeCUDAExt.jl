module EnzymeCUDAExt

using CUDA
using Enzyme
using Enzyme: EnzymeRules

@inline _asarray(ptr::Ptr{T}, n::Integer) where {T} =
    unsafe_wrap(Array, ptr, n; own = false)
@inline _asarray(ptr::CuPtr{T}, n::Integer) where {T} =
    unsafe_wrap(CuArray, ptr, n; own = false)

@inline _stage(ptr::Ptr, offset::Integer, n::Integer) =
    _asarray(ptr + offset, n)
@inline _stage(ptr::CuPtr, offset::Integer, n::Integer) =
    _asarray(ptr + offset, n)
function _stage(ptr::CuArrayPtr{T}, offset::Integer, n::Integer) where {T}
    buffer = CuArray{T}(undef, n)
    Base.unsafe_copyto!(pointer(buffer), ptr, offset, n)
    return buffer
end

@inline _commit!(::Union{Ptr, CuPtr}, offset, buffer, n) = nothing
function _commit!(ptr::CuArrayPtr, offset, buffer, n)
    Base.unsafe_copyto!(ptr, offset, pointer(buffer), n)
    return nothing
end

function _accumulate_and_zero!(src, soff, dest, doff, n::Integer)
    n == 0 && return nothing
    typeof(src) === typeof(dest) && src == dest && soff == doff && return nothing

    dsrc = _stage(src, soff, n)
    ddest = _stage(dest, doff, n)
    if dsrc isa Array && ddest isa Array || dsrc isa CuArray && ddest isa CuArray
        dsrc .+= ddest
    else
        tmp = similar(dsrc)
        copyto!(tmp, ddest)
        dsrc .+= tmp
    end
    fill!(ddest, zero(eltype(ddest)))
    _commit!(src, soff, dsrc, n)
    _commit!(dest, doff, ddest, n)
    return nothing
end

function _zero!(dest, doff, n)
    ddest = _stage(dest, doff, n)
    fill!(ddest, zero(eltype(ddest)))
    _commit!(dest, doff, ddest, n)
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
    (CuArrayPtr, Ptr),
    (CuArrayPtr, CuPtr),
    (Ptr, CuArrayPtr),
    (CuPtr, CuArrayPtr),
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

for SrcPtr in (Ptr, CuPtr)
    @eval begin
        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:CuArrayPtr},
                doff::Const,
                src::Annotation{<:$SrcPtr},
                n::Const;
                kwargs...,
            ) where {RT}
            func.val(dest.val, doff.val, src.val, n.val; kwargs...)
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
                dest::Annotation{<:CuArrayPtr},
                doff::Const,
                src::Annotation{<:$SrcPtr},
                n::Const;
                kwargs...,
            ) where {RT}
            if !(dest isa Const)
                for batch in 1:EnzymeRules.width(config)
                    ddest = _shadow(dest, config, batch)
                    if src isa Const
                        _zero!(ddest, doff.val, n.val)
                    else
                        _accumulate_and_zero!(
                            _shadow(src, config, batch), 0,
                            ddest, doff.val, n.val,
                        )
                    end
                end
            end
            return (nothing, nothing, nothing, nothing)
        end

        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:$SrcPtr},
                src::Annotation{<:CuArrayPtr},
                soff::Const,
                n::Const;
                kwargs...,
            ) where {RT}
            func.val(dest.val, src.val, soff.val, n.val; kwargs...)
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
                dest::Annotation{<:$SrcPtr},
                src::Annotation{<:CuArrayPtr},
                soff::Const,
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
                            _shadow(src, config, batch), soff.val,
                            ddest, 0, n.val,
                        )
                    end
                end
            end
            return (nothing, nothing, nothing, nothing)
        end
    end
end

end # module
