module BLASRules

using ..Enzyme
using LinearAlgebra.BLAS

const ConstOrDuplicated{T} = Union{Const{T},Duplicated{T}}

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
            r = func.val(n.val, X.val, incx.val, Y.val, incy.val)

            primal = EnzymeRules.needs_primal(config) ? r : nothing
            shadow = EnzymeRules.needs_shadow(config) ? zero(r) : nothing

            _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
            # copy only the elements we need to the tape
            if Xow
                Xtape = X.val isa Ptr ? Array{T}(undef, n.val) : similar(X.val, n.val)
                BLAS.blascopy!(n.val, X.val, incx.val, Xtape, 1)
            else
                Xtape = nothing
            end
            if Yow
                Ytape = Y.val isa Ptr ? Array{T}(undef, n.val) : similar(Y.val, n.val)
                BLAS.blascopy!(n.val, Y.val, incy.val, Ytape, 1)
            else
                Ytape = nothing
            end
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
            _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
            (Xtape, Ytape) = tape
            (Xval, incxval) = Xow ? (Xtape, 1) : (X.val, incx.val)
            (Yval, incyval) = Yow ? (Ytape, 1) : (Y.val, incy.val)

            if !(X isa Const)
                BLAS.axpy!(n.val, $(trans)(dret.val), Yval, incyval, X.dval, incx.val)
            end
            if !(Y isa Const)
                BLAS.axpy!(n.val, dret.val, Xval, incxval, Y.dval, incy.val)
            end

            return (nothing, nothing, nothing, nothing, nothing)
        end
    end
end

end  # module
