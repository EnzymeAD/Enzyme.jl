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

for (fname, Ttype) in ((:dot, :BlasReal), (:dotu, :BlasComplex), (:dotc, :BlasComplex))
    @eval begin
        function EnzymeRules.forward(
            func::Const{typeof(BLAS.$fname)},
            RT::Type{<:Union{Const,DuplicatedNoNeed,Duplicated}},
            n::Const{<:Integer},
            X::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:BLAS.$Ttype}
            RT <: Const && return func.val(n.val, X.val, incx.val, Y.val, incy.val)

            dval = if !(X isa Const) && !(Y isa Const)
                func.val(n.val, X.dval, incx.val, Y.val, incy.val) +
                func.val(n.val, X.val, incx.val, Y.dval, incy.val)
            elseif !(X isa Const)
                func.val(n.val, X.dval, incx.val, Y.val, incy.val)
            elseif !(Y isa Const)
                func.val(n.val, X.val, incx.val, Y.dval, incy.val)
            else
                zero(T)
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
            RT::Type{<:Union{Const,Active}},
            n::Const{<:Integer},
            X::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:BLAS.$Ttype}
            primal, shadow = _maybe_primal_shadow(
                config, func.val, (n.val, X.val, incx.val, Y.val, incy.val)
            )

            # build tape
            if !(RT <: Const)
                _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
                if Xow || BLAS.$fname === BLAS.dotu
                    Xtape = _strided_tape(n.val, X.val, incx.val)
                else
                    Xtape = nothing
                end
                if Yow || BLAS.$fname === BLAS.dotu
                    Ytape = _strided_tape(n.val, Y.val, incy.val)
                else
                    Ytape = nothing
                end
                if BLAS.$fname === BLAS.dotu
                    conj!(Xtape)
                    conj!(Ytape)
                end
                tape = (Xtape, Ytape)
            else
                tape = nothing
            end

            return EnzymeRules.AugmentedReturn(primal, shadow, tape)
        end

        function EnzymeRules.reverse(
            config::EnzymeRules.ConfigWidth{1},
            ::Const{typeof(BLAS.$fname)},
            dret::Union{Active,Type{<:Const}},
            tape,
            n::Const{<:Integer},
            X::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:BLAS.$Ttype}
            ret = (nothing, nothing, nothing, nothing, nothing)
            dret isa Type{<:Const} && return ret

            # restore from tape
            (Xtape, Ytape) = tape
            (Xval, incxval) = Xtape === nothing ? (X.val, incx.val) : (Xtape, 1)
            (Yval, incyval) = Ytape === nothing ? (Y.val, incy.val) : (Ytape, 1)

            X isa Const || BLAS.axpy!(n.val, dret.val, Yval, incyval, X.dval, incx.val)
            Y isa Const || BLAS.axpy!(n.val, dret.val, Xval, incxval, Y.dval, incy.val)

            return ret
        end
    end
end

end  # module
