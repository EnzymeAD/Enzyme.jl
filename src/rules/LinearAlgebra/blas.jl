module BLASRules

using ..Enzyme
using LinearAlgebra.BLAS

const ConstOrDuplicated{T} = Union{Const{T},Duplicated{T}}
const ConstOrBatchDuplicated{T} = Union{ConstOrDuplicated{T},BatchDuplicated{T}}

_safe_similar(x::AbstractArray, n::Integer) = similar(x, n)
_safe_similar(x::Ptr, n::Integer) = Array{eltype(x)}(undef, n)

function _strided_tape(n::Integer, x::Union{AbstractArray,Ptr}, incx::Integer)
    xtape = _safe_similar(x, n)
    BLAS.blascopy!(n, x, incx, xtape, 1)
    return xtape
end

function _strided_view(n::Integer, x::AbstractArray, incx::Integer)
    ind = range(1; step=incx, length=n)
    return view(x, ind)
end
function _strided_view(n::Integer, x::Ptr, incx::Integer)
    ind = range(1; step=incx, length=n)
    dim = abs(last(ind) - first(ind) + 1)
    y = Base.unsafe_wrap(Array, x, dim)
    return view(y, ind)
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

_map_tuple(f, xs::Tuple...) = map(f, xs...)
_map_tuple(f, xs...) = f(xs...)

for (fname, Ttype) in ((:dot, :BlasReal), (:dotu, :BlasComplex), (:dotc, :BlasComplex))
    @eval begin
        function EnzymeRules.forward(
            func::Const{typeof(BLAS.$fname)},
            RT::Type{
                <:Union{
                    Const,DuplicatedNoNeed,Duplicated,BatchDuplicatedNoNeed,BatchDuplicated
                },
            },
            n::Const{<:Integer},
            X::ConstOrBatchDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incx::Const{<:Integer},
            Y::ConstOrBatchDuplicated{<:Union{Ptr{T},AbstractArray{T}}},
            incy::Const{<:Integer},
        ) where {T<:BLAS.$Ttype}
            RT <: Const && return func.val(n.val, X.val, incx.val, Y.val, incy.val)

            dval = if !(X isa Const) && !(Y isa Const)
                _map_tuple(X.dval, Y.dval) do dX, dY
                    func.val(n.val, dX, incx.val, Y.val, incy.val) +
                    func.val(n.val, X.val, incx.val, dY, incy.val)
                end
            elseif !(X isa Const)
                _map_tuple(dX -> func.val(n.val, dX, incx.val, Y.val, incy.val), X.dval)
            elseif !(Y isa Const)
                _map_tuple(dY -> func.val(n.val, X.val, incx.val, dY, incy.val), Y.dval)
            else
                zero(T)
            end

            if RT <: Union{DuplicatedNoNeed,BatchDuplicatedNoNeed}
                return dval
            else
                val = func.val(n.val, X.val, incx.val, Y.val, incy.val)
                return RT(val, dval)
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
                Xtape = if Xow && !(Y isa Const)
                    (_strided_tape(n.val, X.val, incx.val), 1)
                else
                    (X.val, incx.val)
                end
                Ytape = if Yow && !(X isa Const)
                    (_strided_tape(n.val, Y.val, incy.val), 1)
                else
                    (Y.val, incy.val)
                end
                tape = (Xtape..., Ytape...)
            else
                tape = nothing
            end

            return EnzymeRules.AugmentedReturn(primal, shadow, tape)
        end

        function EnzymeRules.reverse(
            config::EnzymeRules.ConfigWidth{1},
            fun::Const{typeof(BLAS.$fname)},
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
            Xval, incxval, Yval, incyval = tape

            if fun.val === BLAS.dotu
                if !(X isa Const)
                    dXview = _strided_view(n.val, X.dval, incx.val)
                    dXview .+= conj.(_strided_view(n.val, Yval, incyval)) .* dret.val
                end
                if !(Y isa Const)
                    dYview = _strided_view(n.val, Y.dval, incy.val)
                    dYview .+= conj.(_strided_view(n.val, Xval, incxval)) .* dret.val
                end
            else
                X isa Const || BLAS.axpy!(n.val, dret.val, Yval, incyval, X.dval, incx.val)
                Y isa Const || BLAS.axpy!(n.val, dret.val, Xval, incxval, Y.dval, incy.val)
            end

            return ret
        end
    end
end

end  # module
