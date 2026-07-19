module EnzymeGPUArraysCoreExt

using GPUArraysCore
using Enzyme
using Enzyme: EnzymeRules

function Enzyme.zerosetfn(x::AbstractGPUArray, i::Int)
    res = zero(x)
    @allowscalar @inbounds res[i] = 1
    return res
end

function Enzyme.zerosetfn!(x::AbstractGPUArray, i::Int, val)
    @allowscalar @inbounds x[i] += val
    return
end

@inline function Enzyme.onehot(x::AbstractGPUArray)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, 0, length(x))
    N = length(x)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i] = 1
        return res
    end
end

@inline function onehot(x::AbstractArray, start::Int, endl::Int)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, start-1, endl-start+1)
    ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i + start - 1] = 1
        return res
    end
end


# host <-> device copies (`copyto!` / `unsafe_copyto!`)

@inline _isdev(::AbstractGPUArray) = true
@inline _isdev(::Any) = false

@inline function _add_copy!(dst, doff::Integer, src, soff::Integer, n::Integer)
    n == 0 && return nothing
    if _isdev(dst) == _isdev(src)
        @views dst[doff:(doff + n - 1)] .+= src[soff:(soff + n - 1)]
    else
        tmp = similar(dst, n)                       # same side as dst
        copyto!(tmp, 1, src, soff, n)               
        @views dst[doff:(doff + n - 1)] .+= tmp
    end
    return nothing
end

@inline function _zero_copy!(arr, off::Integer, n::Integer)
    n == 0 && return nothing
    @views arr[off:(off + n - 1)] .= 0
    return nothing
end

const _COPY_DIRS = (
    (:(AbstractGPUArray), :(Array)),           # host -> device
    (:(Array), :(AbstractGPUArray)),           # device -> host
    (:(AbstractGPUArray), :(AbstractGPUArray)), # device -> device
)

for (DT, ST) in _COPY_DIRS
    @eval begin
        function EnzymeRules.forward(
                config::EnzymeRules.FwdConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:$DT},
                doffs::Const,
                src::Annotation{<:$ST},
                soffs::Const,
                n::Const,
            ) where {RT}
            do_, so_, n_ = doffs.val, soffs.val, n.val
            func.val(dest.val, do_, src.val, so_, n_)
            if !(dest isa Const)
                for b in 1:EnzymeRules.width(config)
                    ddest = EnzymeRules.width(config) == 1 ? dest.dval : dest.dval[b]
                    if !(src isa Const)
                        dsrc = EnzymeRules.width(config) == 1 ? src.dval : src.dval[b]
                        func.val(ddest, do_, dsrc, so_, n_)
                    else
                        _zero_copy!(ddest, do_, n_)
                    end
                end
            end
            if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
                return EnzymeRules.width(config) == 1 ?
                    Duplicated(dest.val, dest.dval) : BatchDuplicated(dest.val, dest.dval)
            elseif EnzymeRules.needs_shadow(config)
                return dest.dval
            elseif EnzymeRules.needs_primal(config)
                return dest.val
            else
                return nothing
            end
        end

        function EnzymeRules.augmented_primal(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                dest::Annotation{<:$DT},
                doffs::Const,
                src::Annotation{<:$ST},
                soffs::Const,
                n::Const,
            ) where {RT}
            func.val(dest.val, doffs.val, src.val, soffs.val, n.val)
            primal = EnzymeRules.needs_primal(config) ? dest.val : nothing
            shadow = (EnzymeRules.needs_shadow(config) && !(dest isa Const)) ? dest.dval : nothing
            return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
        end

        function EnzymeRules.reverse(
                config::EnzymeRules.RevConfig,
                func::Const{typeof(Base.unsafe_copyto!)},
                ::Type{RT},
                tape,
                dest::Annotation{<:$DT},
                doffs::Const,
                src::Annotation{<:$ST},
                soffs::Const,
                n::Const,
            ) where {RT}
            do_, so_, n_ = doffs.val, soffs.val, n.val
            if !(dest isa Const)
                for b in 1:EnzymeRules.width(config)
                    ddest = EnzymeRules.width(config) == 1 ? dest.dval : dest.dval[b]
                    if !(src isa Const)
                        dsrc = EnzymeRules.width(config) == 1 ? src.dval : src.dval[b]
                        _add_copy!(dsrc, so_, ddest, do_, n_)   
                    end
                    _zero_copy!(ddest, do_, n_)                  # copy overwrote dest
                end
            end
            return (nothing, nothing, nothing, nothing, nothing)
        end
    end
end

end # module
