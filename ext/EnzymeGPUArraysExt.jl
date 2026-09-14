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

    if !(R isa Const)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dR = _bget(R.dval, Val(N), i)
            # A `Const` input contributes nothing, and `init` means `R`'s prior
            # contents don't either, so the output shadow is zero.
            if A isa Const
                Base.fill!(dR, zero(T))
            else
                ofn.val(f.val, op.val, dR, _bget(A.dval, Val(N), i); init)
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
    if !(A isa Const) && !(R isa Const)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dR = _bget(R.dval, Val(N), i)
            # Each entry of `A` feeds exactly one entry of `R`, so the cotangent
            # broadcasts back along the reduced dims.
            _bget(A.dval, Val(N), i) .+= dR
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
            Duplicated(primal, ofn.val(f.val, op.val, A.dval; dims, init))
        else
            BatchDuplicated(
                primal,
                ntuple(Val(N)) do i
                    Base.@_inline_meta
                    ofn.val(f.val, op.val, A.dval[i]; dims, init)
                end
            )
        end
    elseif needs_shadow(config)
        if N == 1
            ofn.val(f.val, op.val, A.dval; dims, init)
        else
            ntuple(Val(N)) do i
                Base.@_inline_meta
                ofn.val(f.val, op.val, A.dval[i]; dims, init)
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
    if !(A isa Const) && !(dres isa Const)
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
