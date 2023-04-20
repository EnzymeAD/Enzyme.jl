module BLASRules

using ..Enzyme
using LinearAlgebra.BLAS

const ConstOrDuplicated{T} = Union{Const{T},Duplicated{T}}

_safe_similar(x::AbstractArray, n::Integer) = similar(x, n)
_safe_similar(x::Ptr, n::Integer) = Array{eltype(x)}(undef, n)

function _strided_tape(n::Integer, x::Union{AbstractArray,Ptr}, incx::Integer)
    xtape = _safe_similar(x, n)
    BLAS.blascopy!(n, x, incx, xtape, 1)
    return xtape
end

_tape_stride(xtape::AbstractArray) = stride(xtape, 1)

function _maybe_primal_shadow(config, func, args)
    needs_primal = EnzymeRules.needs_primal(config)
    needs_shadow = EnzymeRules.needs_shadow(config)
    if needs_primal || needs_shadow
        r = func(args...)
    else
        r = nothing
    end
    primal = needs_primal ? r : nothing
    shadow = needs_shadow ? zero(r) : nothing
    return primal, shadow
end

for (fname, Ttype, trans) in (
    (:dot, :(BLAS.BlasReal), :identity),
    (:dotu, :(BLAS.BlasComplex), :identity),
    (:dotc, :(BLAS.BlasComplex), :conj),
)
    @eval begin
        function EnzymeRules.forward(
            func::Const{typeof(BLAS.$fname)},
            RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
            n::Const{<:Integer},
            X::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:$Ttype}
            RT <: Const && return nothing
            dval = if !(X isa Const) && !(Y isa Const)
                func.val(n.val, X.dval, incx.val, Y.val, incy.val) +
                func.val(n.val, X.val, incx.val, Y.dval, incy.val)
            elseif !(X isa Const)
                func.val(n.val, X.dval, incx.val, Y.val, incy.val)
            elseif !(Y isa Const)
                func.val(n.val, X.val, incx.val, Y.dval, incy.val)
            else
                nothing
            end
            if RT <: DuplicatedNoNeed
                return dval
            else
                val = func.val(n.val, X.val, incx.val, Y.val, incy.val)
                return Duplicated(val, dval)
            end
        end

        function EnzymeRules.augmented_primal(
            config::EnzymeRules.ConfigWidth{1},
            func::Const{typeof(BLAS.$fname)},
            ::Type{<:Union{Active,Duplicated}},
            n::Const{<:Integer},
            X::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:$Ttype}
            primal, shadow = _maybe_primal_shadow(
                config, func.val, (n.val, X.val, incx.val, Y.val, incy.val)
            )

            # build tape
            _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
            Xtape = Xow ? _strided_tape(n.val, X.val, incx.val) : nothing
            Ytape = Yow ? _strided_tape(n.val, Y.val, incy.val) : nothing
            tape = (Xtape, Ytape)

            return EnzymeRules.AugmentedReturn(primal, shadow, tape)
        end

        function EnzymeRules.reverse(
            config::EnzymeRules.ConfigWidth{1},
            ::Const{typeof(BLAS.$fname)},
            dret::Active,
            tape,
            n::Const{<:Integer},
            X::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:$Ttype}
            # restore from tape
            _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
            (Xtape, Ytape) = tape
            (Xval, incxval) = Xow ? (Xtape, _tape_stride(Xtape)) : (X.val, incx.val)
            (Yval, incyval) = Yow ? (Ytape, _tape_stride(Ytape)) : (Y.val, incy.val)

            X isa Const ||
                BLAS.axpy!(n.val, $(trans)(dret.val), Yval, incyval, X.dval, incx.val)
            Y isa Const || BLAS.axpy!(n.val, dret.val, Xval, incxval, Y.dval, incy.val)

            return (nothing, nothing, nothing, nothing, nothing)
        end
    end
end

end  # module
