module EnzymeGPUArraysCoreExt

using GPUArraysCore
using Enzyme
using LinearAlgebra: LinearAlgebra, dot
using Enzyme.EnzymeCore: EnzymeCore
using Enzyme.EnzymeCore.EnzymeRules:
    EnzymeRules,
    RevConfig,
    Annotation,
    augmented_rule_return_type,
    needs_primal,
    needs_shadow,
    overwritten,
    width

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

@inline _bget(x, ::Val{1}, ::Int) = x
@inline _bget(x, ::Val{N}, i::Int) where {N} = x[i]

# Unwraps the return cotangent. At width > 1 `dret` is a tuple of `Active`s, not
# one `Active` holding a tuple.
@inline _dret(dret, w::Val, i::Int) = _bget(dret, w, i).val

# True when an operand takes no cotangent: `Const`, or a runtime-activity shadow
# that aliases the primal. Writing to the latter would clobber the primal.
@inline _isconst(_, ::Const) = true
@inline _isconst(config, x::Annotation) =
    EnzymeRules.runtime_activity(config) && x.dval === x.val

# One value per batch element.
@inline _bsplat(::Val{1}, v) = v
@inline _bsplat(::Val{N}, v) where {N} = ntuple(Returns(v), Val(N))

#=
Shadow of the returned output argument. Enzyme asks for one whenever the caller
gave the call an active return activity, which it does independently of the
output's own activity: a dynamically dispatched `generic_matmatmul!` gets
`Duplicated` guessed from its return type even with a `Const` C. A constant has
no shadow of its own, and Enzyme's representation of one is the primal itself,
which is what `_isconst` above recognizes.
=#
@inline _outshadow(_, out::Annotation) = out.dval
@inline _outshadow(config, out::Const) = _bsplat(Val(width(config)), out.val)

#=
Augmented return for the rules here: output argument plus a tape. Enzyme checks
the primal/shadow types, so get them from `augmented_rule_return_type`. Use the
2-arg form; the 3-arg one is `@generated` on the tape type, and a tape holding an
`A·B` product doesn't always infer.
=#
@inline function _augreturn(config, ::Type{RT}, out::Annotation, tape) where {RT}
    return augmented_rule_return_type(config, RT)(
        needs_primal(config) ? out.val : nothing,
        needs_shadow(config) ? _outshadow(config, out) : nothing,
        tape,
    )
end

# `conj(⟨dCᵢ, M⟩)` as a `T`: batch element `i`'s share of dα or dβ. A struct
# instead of a closure so `ntuple` gets something concrete.
struct _DotCotangent{T, W, D, M}
    dvals::D
    mat::M
end
@inline _DotCotangent{T}(::Val{W}, dvals::D, mat::M) where {T, W, D, M} =
    _DotCotangent{T, W, D, M}(dvals, mat)
@inline (f::_DotCotangent{T, W})(i::Int) where {T, W} =
    Enzyme._project(T, conj(dot(_bget(f.dvals, Val(W), i), f.mat)))::T

# Cotangent of an `Active` scalar: bare at width 1, an N-tuple when batched.
# Dispatch on `Val{N}` rather than an `N == 1` branch, which infers as `Any`.
# Returning the wrong type here is a hard error.
@inline _dscalar(w::Val{1}, ::Type{T}, dvals, M) where {T} =
    _DotCotangent{T}(w, dvals, M)(1)
@inline _dscalar(w::Val{N}, ::Type{T}, dvals, M) where {N, T} =
    ntuple(_DotCotangent{T}(w, dvals, M), Val(N))

# Same shape, zero value.
@inline _dzero(::Val{1}, ::Type{T}) where {T} = zero(T)
@inline _dzero(::Val{N}, ::Type{T}) where {N, T} = ntuple(Returns(zero(T)), Val(N))

# Materialized `conj`. No-op for real eltypes.
@inline _conj(X) = eltype(X) <: Real ? X : conj.(X)

@static if VERSION < v"1.12.0-DEV"
    @inline _gemm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, LinearAlgebra.MulAddMul(α, β))
else
    @inline _gemm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, α, β)
end

# `mul!` and `*` unwrap each operand and pass a char saying how to read the bare
# array: 'N' plain, 'T' transposed, 'C' adjoint, 'S'/'H' Symmetric/Hermitian.
@inline _plain(t::AbstractChar) = (k = uppercase(t); k == 'N' || k == 'T' || k == 'C')
@inline _opflags(t::AbstractChar) =
    (k = uppercase(t); k == 'N' ? (false, false) : k == 'T' ? (true, false) : (true, true))
@inline _adjop(f::Tuple{Bool, Bool}) = (!f[1], !f[2])
@inline _transop(f::Tuple{Bool, Bool}) = (!f[1], f[2])
@inline _opchar(f::Tuple{Bool, Bool}) = f[1] ? (f[2] ? 'C' : 'T') : 'N'
@inline _opdata(X, f::Tuple{Bool, Bool}) = (!f[1] && f[2]) ? _conj(X) : X

#=
Symmetric/Hermitian operands show up as 'S'/'H'. Only one triangle is stored and
each entry feeds two positions, so their cotangent needs a projection instead of
a matmul. Reject them rather than return the full matrix's cotangent. Triangular
operands go to `generic_trimatmul!`/`generic_mattrimul!` and never get here. Both
are follow-up work.
=#
@inline function _assert_rectangular(tA::AbstractChar, tB::AbstractChar)
    if !(_plain(tA) && _plain(tB))
        throw(
            ArgumentError(
                "Enzyme: reverse mode over a Symmetric/Hermitian GPU array operand is " *
                    "not supported yet (got wrapper chars '$(Char(tA))' and '$(Char(tB))')"
            )
        )
    end
    return nothing
end

#=
`dX += factor · op(L, fL) · op(R, fR)`, projected back through `tX`, the op on the
operand being written to. Both matmul cotangents have this shape (`dC·op(B)'` for
the left operand, `op(A)'·dC` for the right), so both go through
`generic_matmatmul!` with β = 1, which accumulates in place with no temporary and
takes `factor` as α. Undoing `tX` rewrites the product instead of transposing the
result, so the operands swap.
=#
function _pullback!(dX, tX::AbstractChar, L, fL, R, fR, factor)
    kind = uppercase(tX)
    if kind == 'N'
        _gemm!(dX, _opchar(fL), _opchar(fR), _opdata(L, fL), _opdata(R, fR), factor, true)
    elseif kind == 'T'
        gL, gR = _transop(fL), _transop(fR)
        _gemm!(dX, _opchar(gR), _opchar(gL), _opdata(R, gR), _opdata(L, gL), factor, true)
    else
        gL, gR = _adjop(fL), _adjop(fR)
        _gemm!(dX, _opchar(gR), _opchar(gL), _opdata(R, gR), _opdata(L, gL), factor, true)
    end
    return nothing
end

#=
    generic_matmatmul!(C, tA, tB, A, B, α, β)    C = α·op(A)·op(B) + β·C₀
    generic_matvecmul!(y, tA, A, x, α, β)        y = α·op(A)·x + β·y₀

Every rectangular GPU matmul goes through these, from `*` or 3-/5-arg `mul!`, on
CUBLAS or on the GPUArrays fallback kernels. One rule each covers plain,
transposed and adjointed operands in any combination:

    dA += conj(α)·dC·op(B)'          projected back through op(A)
    dB += conj(α)·op(A)'·dC          projected back through op(B)
    dC := conj(β)·dC                 cotangent w.r.t. C₀
    dα  = conj(⟨dC, op(A)·op(B)⟩)
    dβ  = conj(⟨dC, C₀⟩)

The first two are matmuls, so they go back through this same function with β = 1
instead of accumulating by hand. See `_pullback!`.

3-arg `mul!` arrives as α = true, β = false, which zeroes `dC`. That's right:
3-arg `mul!` overwrites `C`.

α and β are separate arguments on Julia ≥1.12 and one `MulAddMul` before that, so
the rules below are version-gated wrappers over two shared helpers.
=#

@inline function _matmul_caches(
        config, C::Annotation, tA::AbstractChar, tB::AbstractChar, A::Annotation, B::Annotation,
        ovw_A::Bool, ovw_B::Bool, ::Val{α_active}, ::Val{β_active},
    ) where {α_active, β_active}
    _assert_rectangular(tA, tB)
    # dβ needs C₀, dα the product, dA needs B, dB needs A. A `Const` C makes the
    # reverse pass return early and read none of them, and `cache_prod` costs an
    # extra matmul, so only cache what will actually be read.
    c_const = _isconst(config, C)
    cache_C = (β_active && !c_const) ? copy(C.val) : nothing
    cache_A = (ovw_A && !_isconst(config, B) && !c_const) ? copy(A.val) : nothing
    cache_B = (ovw_B && !_isconst(config, A) && !c_const) ? copy(B.val) : nothing
    cache_prod = (α_active && !c_const) ?
        LinearAlgebra.wrap(A.val, tA) * LinearAlgebra.wrap(B.val, tB) : nothing
    return (cache_C, cache_A, cache_B, cache_prod)
end

@inline function _matmul_reverse!(
        config, C::Annotation, tA::AbstractChar, tB::AbstractChar, A::Annotation, B::Annotation,
        αval::Number, βval::Number, tape, ::Val{α_active}, ::Val{β_active},
    ) where {α_active, β_active}
    N = width(config)
    if _isconst(config, C)
        # No shadow on C means nothing to pull back.
        dα = α_active ? _dzero(Val(N), typeof(αval)) : nothing
        dβ = β_active ? _dzero(Val(N), typeof(βval)) : nothing
        return (dα, dβ)
    end

    cache_C, cache_A, cache_B, cache_prod = tape
    Cval = cache_C !== nothing ? cache_C : C.val
    Aval = cache_A !== nothing ? cache_A : A.val
    Bval = cache_B !== nothing ? cache_B : B.val

    dα = α_active ? _dscalar(Val(N), typeof(αval), C.dval, cache_prod) : nothing
    dβ = β_active ? _dscalar(Val(N), typeof(βval), C.dval, Cval) : nothing

    αc = conj(αval)
    βc = conj(βval)
    fA = _opflags(tA)
    fB = _opflags(tB)
    a_const = _isconst(config, A)
    b_const = _isconst(config, B)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        dC = _bget(C.dval, Val(N), i)
        if !a_const
            _pullback!(_bget(A.dval, Val(N), i), tA, dC, (false, false), Bval, _adjop(fB), αc)
        end
        if !b_const
            _pullback!(_bget(B.dval, Val(N), i), tB, Aval, _adjop(fA), dC, (false, false), αc)
        end
        # Skip the scale at β = 1; it would be a no-op kernel.
        isone(βc) || (dC .*= βc)
        nothing
    end

    return (dα, dβ)
end

@static if VERSION < v"1.12.0-DEV"

    # Here α and β are one `MulAddMul`, so an active α or β is a single active
    # struct and its cotangent comes back as that struct.

    # `i -> MAM(dα[i], dβ[i])` without the closure.
    struct _PackMulAddMul{MAM, A, B}
        dα::A
        dβ::B
    end
    @inline _PackMulAddMul{MAM}(dα::A, dβ::B) where {MAM, A, B} =
        _PackMulAddMul{MAM, A, B}(dα, dβ)
    @inline (f::_PackMulAddMul{MAM})(i::Int) where {MAM} = MAM(f.dα[i], f.dβ[i])

    @inline _dmuladdmul(::Val{1}, ::Type{MAM}, dα, dβ) where {MAM} = MAM(dα, dβ)
    @inline function _dmuladdmul(::Val{N}, ::Type{MAM}, dα, dβ) where {N, MAM}
        return ntuple(_PackMulAddMul{MAM}(dα, dβ), Val(N))
    end

    function EnzymeRules.augmented_primal(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matmatmul!)},
            ::Type{RT},
            C::Annotation{<:AbstractGPUVecOrMat},
            tA::Const{<:AbstractChar},
            tB::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUVecOrMat},
            B::Annotation{<:AbstractGPUVecOrMat},
            add::Annotation{<:LinearAlgebra.MulAddMul},
        ) where {RT}
        active = Val(!(add isa Const))
        tape = _matmul_caches(
            config, C, tA.val, tB.val, A, B,
            overwritten(config)[5], overwritten(config)[6], active, active,
        )

        func.val(C.val, tA.val, tB.val, A.val, B.val, add.val)

        return _augreturn(config, RT, C, tape)
    end

    function EnzymeRules.reverse(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matmatmul!)},
            ::Type{RT},
            tape,
            C::Annotation{<:AbstractGPUVecOrMat},
            tA::Const{<:AbstractChar},
            tB::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUVecOrMat},
            B::Annotation{<:AbstractGPUVecOrMat},
            add::Annotation{<:LinearAlgebra.MulAddMul},
        ) where {RT}
        active = Val(!(add isa Const))
        dα, dβ = _matmul_reverse!(
            config, C, tA.val, tB.val, A, B,
            add.val.alpha, add.val.beta, tape, active, active,
        )
        dadd = (add isa Const) ? nothing :
            _dmuladdmul(Val(width(config)), typeof(add.val), dα, dβ)
        return (nothing, nothing, nothing, nothing, nothing, dadd)
    end

    function EnzymeRules.augmented_primal(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matvecmul!)},
            ::Type{RT},
            y::Annotation{<:AbstractGPUVector},
            tA::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUMatrix},
            x::Annotation{<:AbstractGPUVector},
            add::Annotation{<:LinearAlgebra.MulAddMul},
        ) where {RT}
        active = Val(!(add isa Const))
        tape = _matmul_caches(
            config, y, tA.val, 'N', A, x,
            overwritten(config)[4], overwritten(config)[5], active, active,
        )

        func.val(y.val, tA.val, A.val, x.val, add.val)

        return _augreturn(config, RT, y, tape)
    end

    function EnzymeRules.reverse(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matvecmul!)},
            ::Type{RT},
            tape,
            y::Annotation{<:AbstractGPUVector},
            tA::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUMatrix},
            x::Annotation{<:AbstractGPUVector},
            add::Annotation{<:LinearAlgebra.MulAddMul},
        ) where {RT}
        active = Val(!(add isa Const))
        dα, dβ = _matmul_reverse!(
            config, y, tA.val, 'N', A, x,
            add.val.alpha, add.val.beta, tape, active, active,
        )
        dadd = (add isa Const) ? nothing :
            _dmuladdmul(Val(width(config)), typeof(add.val), dα, dβ)
        return (nothing, nothing, nothing, nothing, dadd)
    end

else

    function EnzymeRules.augmented_primal(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matmatmul!)},
            ::Type{RT},
            C::Annotation{<:AbstractGPUVecOrMat},
            tA::Const{<:AbstractChar},
            tB::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUVecOrMat},
            B::Annotation{<:AbstractGPUVecOrMat},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
        ) where {RT}
        tape = _matmul_caches(
            config, C, tA.val, tB.val, A, B,
            overwritten(config)[5], overwritten(config)[6],
            Val(!(α isa Const)), Val(!(β isa Const)),
        )

        func.val(C.val, tA.val, tB.val, A.val, B.val, α.val, β.val)

        return _augreturn(config, RT, C, tape)
    end

    function EnzymeRules.reverse(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matmatmul!)},
            ::Type{RT},
            tape,
            C::Annotation{<:AbstractGPUVecOrMat},
            tA::Const{<:AbstractChar},
            tB::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUVecOrMat},
            B::Annotation{<:AbstractGPUVecOrMat},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
        ) where {RT}
        dα, dβ = _matmul_reverse!(
            config, C, tA.val, tB.val, A, B, α.val, β.val, tape,
            Val(!(α isa Const)), Val(!(β isa Const)),
        )
        return (nothing, nothing, nothing, nothing, nothing, dα, dβ)
    end

    function EnzymeRules.augmented_primal(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matvecmul!)},
            ::Type{RT},
            y::Annotation{<:AbstractGPUVector},
            tA::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUMatrix},
            x::Annotation{<:AbstractGPUVector},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
        ) where {RT}
        tape = _matmul_caches(
            config, y, tA.val, 'N', A, x,
            overwritten(config)[4], overwritten(config)[5],
            Val(!(α isa Const)), Val(!(β isa Const)),
        )

        func.val(y.val, tA.val, A.val, x.val, α.val, β.val)

        return _augreturn(config, RT, y, tape)
    end

    function EnzymeRules.reverse(
            config::RevConfig,
            func::Const{typeof(LinearAlgebra.generic_matvecmul!)},
            ::Type{RT},
            tape,
            y::Annotation{<:AbstractGPUVector},
            tA::Const{<:AbstractChar},
            A::Annotation{<:AbstractGPUMatrix},
            x::Annotation{<:AbstractGPUVector},
            α::Annotation{<:Number},
            β::Annotation{<:Number},
        ) where {RT}
        dα, dβ = _matmul_reverse!(
            config, y, tA.val, 'N', A, x, α.val, β.val, tape,
            Val(!(α isa Const)), Val(!(β isa Const)),
        )
        return (nothing, nothing, nothing, nothing, dα, dβ)
    end

end

# dot(a, b) = Σ conj(aᵢ)·bᵢ, so `da += conj(dr)·b` and `db += dr·a`. The asymmetry
# only shows up for complex eltypes.

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(dot)},
        ::Type{RT},
        a::Annotation{<:AbstractGPUArray},
        b::Annotation{<:AbstractGPUArray},
    ) where {RT}
    cache_a = (overwritten(config)[2] && !_isconst(config, b)) ? copy(a.val) : nothing
    cache_b = (overwritten(config)[3] && !_isconst(config, a)) ? copy(b.val) : nothing
    tape = (cache_a, cache_b)
    # A scalar return is `Active`, so there's no shadow to return.
    return augmented_rule_return_type(config, RT)(
        needs_primal(config) ? dot(a.val, b.val) : nothing, nothing, tape,
    )
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(dot)},
        dret,
        tape,
        a::Annotation{<:AbstractGPUArray},
        b::Annotation{<:AbstractGPUArray},
    )
    if !(dret isa Const)
        cache_a, cache_b = tape
        av = cache_a !== nothing ? cache_a : a.val
        bv = cache_b !== nothing ? cache_b : b.val
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dr = _dret(dret, Val(N), i)
            #= axpy! is slower. measured on 5090
            n	        broadcastmed (µs)	axpy! med (µs)	ratio
            1 000	    15.90	            23.34	        1.47
            100 000	    18.76	            22.78	        1.21
            1 000 000	19.30	            23.38	        1.21
            10 000 000	163.05	            162.54	        1.00
            100 000 000	1534	            1537	        1.00
            =#
            if !_isconst(config, a)
                # `a` is conjugated in the primal, so dr comes back conjugated.
                _bget(a.dval, Val(N), i) .+= conj(dr) .* bv
            end
            if !_isconst(config, b)
                _bget(b.dval, Val(N), i) .+= dr .* av
            end
            nothing
        end
    end
    return (nothing, nothing)
end

end # module
