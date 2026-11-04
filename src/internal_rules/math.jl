_hypotvalue(x) = x.val

# Change the evaluation order when x * dx leaves the normal range. Always
# normalizing first can instead lose a tiny x / h that a large seed rescales.
function _hypotmul(x, dx, h)
    product = x * dx
    if isfinite(product) && abs(product) >= floatmin(real(product))
        return product / h
    else
        return (x / h) * dx
    end
end

_hypotforward(x::Const, h, i) = zero(real(x.val))
function _hypotforward(x, h, i)
    dx = i === nothing ? x.dval : x.dval[i]
    return real(_hypotmul(conj(x.val), dx, h))
end

struct HypotForward{A, H}
    args::A
    h::H
end
(f::HypotForward)(i) = sum(_hypotforward.(f.args, f.h, i))

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(Base.hypot)},
        RT,
        x::Annotation,
        y::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    args = (x, y, xs...)
    if EnzymeRules.needs_shadow(config)
        h = func.val(map(_hypotvalue, args)...)
        n = iszero(h) ? one(h) : h
        tangent = HypotForward(args, n)
        if EnzymeRules.width(config) == 1
            dh = tangent(nothing)
            if EnzymeRules.needs_primal(config)
                return Duplicated(h, dh)
            else
                return dh
            end
        else
            dh = ntuple(tangent, Val(EnzymeRules.width(config)))
            if EnzymeRules.needs_primal(config)
                return BatchDuplicated(h, dh)
            else
                return dh
            end
        end
    elseif EnzymeRules.needs_primal(config)
        return func.val(map(_hypotvalue, args)...)
    else
        return nothing
    end
end

_hypotreverse(x::Const, ::Val{W}, ::Type{<:Const}, h) where {W} = nothing
_hypotreverse(x::Const, ::Val{W}, dret, h) where {W} = nothing
function _hypotreverse(x, w::Val{W}, ::Type{<:Const}, h) where {W}
    if W == 1
        return zero(x.val)
    else
        return ntuple(Returns(zero(x.val)), w)
    end
end
function _hypotreverse(x, w::Val{W}, dret, h) where {W}
    if W == 1
        return _hypotreverse_seed((x.val, h), dret)
    else
        return map(Base.Fix1(_hypotreverse_seed, (x.val, h)), dret)
    end
end

function _hypotreverse_seed(x_h, dret)
    x, h = x_h
    return oftype(x, _hypotmul(x, dret.val, h))
end

struct HypotReverse{W, D, H}
    width::Val{W}
    dret::D
    h::H
end
(r::HypotReverse)(x) = _hypotreverse(x, r.width, r.dret, r.h)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(Base.hypot)},
        ::Type,
        x::Annotation,
        y::Annotation,
        xs::Vararg{Annotation, N}
    ) where {N}
    h = hypot(map(_hypotvalue, (x, y, xs...))...)
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
        xs::Vararg{Annotation, N}
    ) where {N}
    args = (x, y, xs...)
    h = hypot(map(_hypotvalue, args)...)
    n = iszero(h) ? one(h) : h
    w = Val(EnzymeRules.width(config))
    return map(HypotReverse(w, dret, n), args)
end
