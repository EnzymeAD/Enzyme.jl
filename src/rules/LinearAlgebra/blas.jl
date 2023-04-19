const ConstOrDuplicated{T} = Union{Const{T}, Duplicated{T}}

for (fname, Ttype, dret_X_trans) in ((:dot, :(BLAS.BlasReal), :identity),
                                     (:dotu, :(BLAS.BlasComplex), :identity),
                                     (:dotc, :(BLAS.BlasComplex), :conj))
    @eval begin
        function EnzymeRules.forward(func::Const{typeof(BLAS.$fname)},
                                     RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
                                     n::Const{<:Integer},
                                     X::Union{ConstOrDuplicated{<:AbstractArray{T}}},
                                     incx::Const{<:Integer},
                                     Y::Union{ConstOrDuplicated{<:AbstractArray{T}}},
                                     incy::Const{<:Integer}) where {T <: $Ttype}
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

        function EnzymeRules.augmented_primal(config::EnzymeRules.ConfigWidth{1},
                                              func::Const{typeof(BLAS.$fname)},
                                              ::Type{<:Union{Active, Duplicated}},
                                              n::Const{<:Integer},
                                              X::Union{ConstOrDuplicated{<:AbstractArray{T}
                                                                         }},
                                              incx::Const{<:Integer},
                                              Y::Union{ConstOrDuplicated{<:AbstractArray{T}
                                                                         }},
                                              incy::Const{<:Integer}) where {T <: $Ttype}
            r = func.val(n.val, X.val, incx.val, Y.val, incy.val)
            if EnzymeRules.needs_primal(config)
                primal = r
            else
                primal = nothing
            end
            if EnzymeRules.needs_shadow(config)
                shadow = zero(r)
            else
                shadow = nothing
            end
            _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
            # copy only the elements we need
            Xtape = Xow ? BLAS.blascopy!(n.val, X.val, incx.val, similar(X.val, n.val), 1) :
                    nothing
            Ytape = Yow ? BLAS.blascopy!(n.val, Y.val, incy.val, similar(Y.val, n.val), 1) :
                    nothing
            tape = (Xtape, Ytape)

            return EnzymeRules.AugmentedReturn(primal, shadow, tape)
        end

        function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1},
                                     ::Const{typeof(BLAS.$fname)},
                                     dret::Active,
                                     tape,
                                     n::Const{<:Integer},
                                     X::Union{ConstOrDuplicated{<:AbstractArray{T}}},
                                     incx::Const{<:Integer},
                                     Y::Union{ConstOrDuplicated{<:AbstractArray{T}}},
                                     incy::Const{<:Integer}) where {T <: $Ttype}
            _, _, Xow, _, Yow = EnzymeRules.overwritten(config)
            (Xtape, Ytape) = tape
            (Xval, incxval) = Xow ? (Xtape, 1) : (X.val, incx.val)
            (Yval, incyval) = Yow ? (Ytape, 1) : (Y.val, incy.val)
            X isa Const ||
                BLAS.axpy!(n.val, $(dret_X_trans)(dret.val), Yval, incyval, X.dval,
                           incx.val)
            Y isa Const || BLAS.axpy!(n.val, dret.val, Xval, incxval, Y.dval, incy.val)
            return (nothing, nothing, nothing, nothing, nothing)
        end
    end
end
