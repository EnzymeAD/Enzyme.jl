_hypotforward(x::Const) = zero(x.val)
_hypotforward(x) = real(conj(x.val) * x.dval)
_hypotforward(x::Const, i) = zero(x.val)
_hypotforward(x, i) = real(conj(x.val) * x.dval[i])

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.hypot)},
        RT,
        x::Annotation,
        y::Annotation,
        z::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    if EnzymeRules.needs_shadow(config)
        h = func.val(x.val, y.val, z.val, map(x -> x.val, xs)...)
        n = iszero(h) ? one(h) : h
        if EnzymeRules.width(config) == 1
            dh = (
                _hypotforward(x) +
                    _hypotforward(y) +
                    _hypotforward(z) +
                    sum(_hypotforward, xs, init = zero(real(x.val)))
            ) / n
            if EnzymeRules.needs_primal(config)
                return Duplicated(h, dh)
            else
                return dh
            end
        else
            dh = ntuple(
                i -> (
                    _hypotforward(x, i) +
                        _hypotforward(y, i) +
                        _hypotforward(z, i) +
                        sum(x -> _hypotforward(x, i), xs; init = zero(real(x.val)))
                ) / n,
                Val(EnzymeRules.width(config)),
            )
            if EnzymeRules.needs_primal(config)
                return BatchDuplicated(h, dh)
            else
                return dh
            end
        end
    elseif EnzymeRules.needs_primal(config)
        return func.val(x.val, y.val, z.val, map(x -> x.val, xs)...)
    else
        return nothing
    end
end

_hypotreverse(x::Const, ::Val{W}, dret::Const, h) where {W} = nothing
_hypotreverse(x::Const, ::Val{W}, dret, h) where {W} = nothing
function _hypotreverse(x, w::Val{W}, dret::Const, h) where {W}
    if W == 1
        return zero(x.val)
    else
        return ntuple(Returns(zero(x.val)), w)
    end
end
function _hypotreverse(x, w::Val{W}, dret, h) where {W}
    if W == 1
        return x.val * dret.val / h
    else
        return ntuple(i -> x.val * dret.val[i] / h, w)
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.hypot)},
        ::Type,
        x::Annotation,
        y::Annotation,
        z::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    h = hypot(x.val, y.val, z.val, map(x -> x.val, xs)...)
    primal = needs_primal(config) ? h : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.hypot)},
        dret,
        tape,
        x::Annotation,
        y::Annotation,
        z::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    h = hypot(x.val, y.val, z.val, map(x -> x.val, xs)...)
    n = iszero(h) ? one(h) : h
    w = Val(EnzymeRules.width(config))
    dx = _hypotreverse(x, w, dret, n)
    dy = _hypotreverse(y, w, dret, n)
    dz = _hypotreverse(z, w, dret, n)
    dxs = map(x -> _hypotreverse(x, w, dret, n), xs)
    return (dx, dy, dz, dxs...)
end

