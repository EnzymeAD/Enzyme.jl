module EnzymeGPUArraysExt

using GPUArrays
using Enzyme
using Enzyme.EnzymeCore.EnzymeRules:
    EnzymeRules,
    Annotation,
    needs_primal,
    needs_shadow,
    width

#=
Rules for the two reduction entry points GPUArrays owns.

`sum`/`mapreduce` over a GPU array land in `GPUArrays._mapreduce`, which
allocates the output and calls `GPUArrays.mapreducedim!` to fill it. Both are
backend-agnostic: the bodies below only need `fill!`, broadcast, and a call back
into the same function on the shadows, so one rule per entry point covers every
`AbstractGPUArray` backend. Without them Enzyme descends into the backend
reduction kernel, which it cannot always differentiate.

Only `identity`/`add_sum` is handled; other `f`/`op` pairs still go through the
kernel.
=#

# Batch element `i` of a shadow: bare at width 1, an N-tuple when batched.
@inline _bget(x, ::Val{1}, ::Int) = x
@inline _bget(x, ::Val{N}, i::Int) where {N} = x[i]

# Unwraps the return cotangent. At width > 1 `dres` is a tuple of `Active`s, not
# one `Active` holding a tuple.
@inline _dret(dres, w::Val, i::Int) = _bget(dres, w, i).val

# True when an operand takes no cotangent: `Const`, or a runtime-activity shadow
# that aliases the primal. Writing to the latter would clobber the primal.
@inline _isconst(_, ::Const) = true
@inline _isconst(config, x::Annotation) =
    EnzymeRules.runtime_activity(config) && x.dval === x.val

# The `init` for a reduction over shadows. `init` is a constant offset of the
# primal, so its derivative is zero.
@inline _dinit(::Nothing) = nothing
@inline _dinit(init) = zero(init)

# The tangent of `_mapreduce` for one batch element.
@inline function _mapreduce_shadow(config, ofn, f, op, A, dA, dims, init)
    res = ofn.val(f.val, op.val, dA; dims, init = _dinit(init))
    # A shadow that aliases its primal holds primal values, not a tangent.
    return _isconst(config, A) ? zero(res) : res
end

#=
`mapreducedim!(identity, add_sum, R, A; init)` writes `sum(A)` over the reduced
dims into `R`. `init` has no default here on purpose: GPUArrays only passes it
from `_mapreduce`, where `R` is a fresh buffer that `init` initializes. A call
that omits it accumulates onto `R`'s existing contents instead, and the reverse
rule's unconditional zeroing of `dR` would be wrong for that case, so leave
those calls to the kernel.

The derivative is the same reduction on the shadows: the map is `identity` and
the reduction is a sum, so `dR = sum(dA)` and `dA += dR`.
=#
function EnzymeRules.forward(
        config,
        ofn::Const{typeof(GPUArrays.mapreducedim!)},
        ::Type{RT},
        f::Const{typeof(Base.identity)},
        op::Const{typeof(Base.add_sum)},
        R::Annotation{<:AnyGPUArray{T}},
        A::Annotation;
        init,
    ) where {RT, T}
    if !(R isa DuplicatedNoNeed || R isa BatchDuplicatedNoNeed)
        ofn.val(f.val, op.val, R.val, A.val; init)
    end

    if !_isconst(config, R)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dR = _bget(R.dval, Val(N), i)
            # A `Const` input contributes nothing, and `init` means `R`'s prior
            # contents don't either, so the output shadow is zero.
            if _isconst(config, A)
                Base.fill!(dR, zero(T))
            else
                ofn.val(
                    f.val, op.val, dR, _bget(A.dval, Val(N), i);
                    init = _dinit(init),
                )
            end
            nothing
        end
    end

    return if needs_primal(config) && needs_shadow(config)
        R
    elseif needs_shadow(config)
        R.dval
    elseif needs_primal(config)
        R.val
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
        config,
        ofn::Const{typeof(GPUArrays.mapreducedim!)},
        ::Type{RT},
        f::Const{typeof(Base.identity)},
        op::Const{typeof(Base.add_sum)},
        R::Annotation{<:AnyGPUArray{T}},
        A::Annotation;
        init,
    ) where {RT, T}
    ofn.val(f.val, op.val, R.val, A.val; init)

    primal = needs_primal(config) ? R.val : nothing
    shadow = needs_shadow(config) ? R.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config,
        ofn::Const{typeof(GPUArrays.mapreducedim!)},
        ::Type{RT},
        tape,
        f::Const{typeof(Base.identity)},
        op::Const{typeof(Base.add_sum)},
        R::Annotation{<:AnyGPUArray{T}},
        A::Annotation;
        init,
    ) where {RT, T}
    if !_isconst(config, R)
        a_const = _isconst(config, A)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dR = _bget(R.dval, Val(N), i)
            # Each entry of `A` feeds exactly one entry of `R`, so the cotangent
            # broadcasts back along the reduced dims.
            if !a_const
                _bget(A.dval, Val(N), i) .+= dR
            end
            # `init` overwrites `R`, so its cotangent is consumed here even
            # when `A` is constant.
            Base.fill!(dR, zero(T))
            nothing
        end
    end

    return (nothing, nothing, nothing, nothing)
end

#=
`_mapreduce(identity, add_sum, A; dims, init)` allocates its own output, so
there is nothing to seed: the primal is `sum(A)` and the derivative is `sum(dA)`.
=#
function EnzymeRules.forward(
        config,
        ofn::Const{typeof(GPUArrays._mapreduce)},
        ::Type{RT},
        f::Const{typeof(Base.identity)},
        op::Const{typeof(Base.add_sum)},
        A::Annotation{<:AnyGPUArray{T}};
        dims::D,
        init,
    ) where {RT, T, D}
    N = width(config)
    return if needs_primal(config) && needs_shadow(config)
        primal = ofn.val(f.val, op.val, A.val; dims, init)
        if N == 1
            Duplicated(
                primal, _mapreduce_shadow(config, ofn, f, op, A, A.dval, dims, init),
            )
        else
            BatchDuplicated(
                primal,
                ntuple(Val(N)) do i
                    Base.@_inline_meta
                    _mapreduce_shadow(config, ofn, f, op, A, A.dval[i], dims, init)
                end
            )
        end
    elseif needs_shadow(config)
        if N == 1
            _mapreduce_shadow(config, ofn, f, op, A, A.dval, dims, init)
        else
            ntuple(Val(N)) do i
                Base.@_inline_meta
                _mapreduce_shadow(config, ofn, f, op, A, A.dval[i], dims, init)
            end
        end
    elseif needs_primal(config)
        ofn.val(f.val, op.val, A.val; dims, init)
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
        config,
        ofn::Const{typeof(GPUArrays._mapreduce)},
        ::Type{Active{RT}},
        f::Const{typeof(Base.identity)},
        op::Const{typeof(Base.add_sum)},
        A::Annotation{<:AnyGPUArray{T}};
        dims::D,
        init,
    ) where {RT, T, D}
    primal = needs_primal(config) ? ofn.val(f.val, op.val, A.val; dims, init) : nothing
    shadow = needs_shadow(config) ? A.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

# `dres` is untyped on purpose: at width 1 it is an `Active`, at width > 1 a
# tuple of them, and a `Const` when the result is unused.
function EnzymeRules.reverse(
        config,
        ofn::Const{typeof(GPUArrays._mapreduce)},
        dres,
        tape,
        f::Const{typeof(Base.identity)},
        op::Const{typeof(Base.add_sum)},
        A::Annotation{<:AnyGPUArray{T}};
        dims::D,
        init,
    ) where {T, D}
    if !_isconst(config, A) && !(dres isa Const)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            # `Ref` so that the cotangent is broadcast as a scalar; for a
            # non-scalar element type like `SVector` it would otherwise
            # broadcast over its own axes.
            _bget(A.dval, Val(N), i) .+= Ref(_dret(dres, Val(N), i))
            nothing
        end
    end

    return (nothing, nothing, nothing)
end

end # module
