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
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i] = 1
        return res
    end
end

@inline function onehot(x::AbstractArray, start::Int, endl::Int)
    # Enzyme.onehot_internal(Enzyme.zerosetfn, x, start-1, endl-start+1)
    return ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = zero(x)
        @allowscalar @inbounds res[i + start - 1] = 1
        return res
    end
end

#=
`make_zero`/`make_zero!` are defined element-wise, which would require scalar
indexing on a GPU array, so zero the array in bulk instead. That is only
equivalent to the element-wise definition when every part of the element type is
a float, which covers `Float32`, `ComplexF64`, `SVector{3,Float64}` etc. For
other element types the element-wise definition is run on a host copy instead.
=#
@inline _float_only(::Type{FT}) where {FT <: AbstractFloat} = true
@inline _float_only(::Type{Complex{FT}}) where {FT <: AbstractFloat} = true
@inline _float_only(::Type{FT}) where {FT} =
    isbitstype(FT) && fieldcount(FT) > 0 && all(_float_only, fieldtypes(FT))
@inline _bulk_zeroable(::Type{FT}) where {FT} =
    _float_only(FT) && hasmethod(Base.zero, Tuple{Type{FT}})

# An element type that is not `_bulk_zeroable` can still have float content that
# has to be zeroed, e.g. any struct mixing floats with flags or indices.
@inline function _make_zero_via_host(
        prev::AT, ::Val{copy_if_inactive}
    ) where {copy_if_inactive, FT, AT <: AbstractGPUArray{FT}}
    # Like `Array`: an inactive element type has nothing to zero, so return the
    # input (or a copy) and skip the host round-trip.
    if Enzyme.Compiler.guaranteed_const(FT)
        return copy_if_inactive ? copy(prev)::AT : prev
    end
    zeroed = map(Array(prev)) do x
        EnzymeCore.make_zero(Core.Typeof(x), IdDict(), x, Val(copy_if_inactive))
    end
    newa = similar(prev)
    copyto!(newa, zeroed)
    return newa::AT
end

@inline function EnzymeCore.make_zero(x::AbstractGPUArray{FT}) where {FT}
    if !_bulk_zeroable(FT)
        return _make_zero_via_host(x, Val(false))
    end
    return Base.zero(x)
end

@inline function EnzymeCore.make_zero(
        ::Type{AT},
        seen::IdDict,
        prev::AT,
        ::Val{copy_if_inactive} = Val(false),
    )::AT where {copy_if_inactive, FT, AT <: AbstractGPUArray{FT}}
    if haskey(seen, prev)
        return seen[prev]
    end
    newa = _bulk_zeroable(FT) ? Base.zero(prev) :
        _make_zero_via_host(prev, Val(copy_if_inactive))
    seen[prev] = newa
    return newa
end

@inline function EnzymeCore.make_zero!(
        prev::AbstractGPUArray{FT}, seen::ST
    )::Nothing where {FT, ST}
    if !isnothing(seen)
        if prev in seen
            return nothing
        end
        push!(seen, prev)
    end
    if _bulk_zeroable(FT)
        fill!(prev, zero(FT))
    elseif !Enzyme.Compiler.guaranteed_const(FT)
        copyto!(prev, map(EnzymeCore.make_zero, Array(prev)))
    end
    return nothing
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

#=
Augmented return for the rules here: output argument plus a tape. Enzyme checks
the primal/shadow types, so get them from `augmented_rule_return_type`. Use the
2-arg form; the 3-arg one is `@generated` on the tape type, and a tape holding an
`A·B` product doesn't always infer.
=#
@inline function _augreturn(config, ::Type{RT}, out::Annotation, tape) where {RT}
    return augmented_rule_return_type(config, RT)(
        needs_primal(config) ? out.val : nothing,
        needs_shadow(config) ? out.dval : nothing,
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

#=
The storage-level products these rules attach to, and that the reverse pass
calls back into. Julia below 1.13 routes GPU arrays through the non-public
`generic_matmatmul!`/`generic_matvecmul!`; JuliaLang/LinearAlgebra.jl#1671
superseded both with `mul!` methods taking the same leading wrapper chars.
GPUArrays overloads whichever pair its Julia version uses and stops forwarding
the old names at the same boundary (see the `VERSION < v"1.13.0-rc4"` blocks in
its src/host/linalg.jl), so the cutoff here has to be the same one: rules left
on `generic_matmatmul!` would silently stop firing and Enzyme would descend into
the backend kernels instead.
=#
@static if VERSION < v"1.13.0-rc4"
    const _matmat_entry = LinearAlgebra.generic_matmatmul!
    const _matvec_entry = LinearAlgebra.generic_matvecmul!
else
    const _matmat_entry = LinearAlgebra.mul!
    const _matvec_entry = LinearAlgebra.mul!
end

@static if VERSION < v"1.12.0-DEV"
    @inline _gemm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        _matmat_entry(C, tA, tB, A, B, LinearAlgebra.MulAddMul(α, β))
else
    @inline _gemm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        _matmat_entry(C, tA, tB, A, B, α, β)
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
the left operand, `op(A)'·dC` for the right), so both go through `_gemm!` with
β = 1, which accumulates in place with no temporary and takes `factor` as α.
Undoing `tX` rewrites the product instead of transposing the result, so the
operands swap.
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
    _matmat_entry(C, tA, tB, A, B, α, β)    C = α·op(A)·op(B) + β·C₀
    _matvec_entry(y, tA, A, x, α, β)        y = α·op(A)·x + β·y₀

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
    return ntuple(Val(N)) do i
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
            func::Const{typeof(_matmat_entry)},
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
            func::Const{typeof(_matmat_entry)},
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
            func::Const{typeof(_matvec_entry)},
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
            func::Const{typeof(_matvec_entry)},
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
            func::Const{typeof(_matmat_entry)},
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
            func::Const{typeof(_matmat_entry)},
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
            func::Const{typeof(_matvec_entry)},
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
            func::Const{typeof(_matvec_entry)},
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

#=
`fill!(A, x)` writes one scalar into every entry, so the derivative is `dA .= dx`
going forward and `dx += sum(dA)` coming back, with `dA` zeroed once read: the
call overwrites `A`, so whatever cotangent `A` carried in is consumed here.

Without a rule Enzyme descends into the backend fill, which on CUDA is a `memset`
it cannot differentiate. An integer element type carries no cotangent; its shadow
is only cleared so it can't feed junk into a later read.
=#
const _FillEltype = Union{AbstractFloat, Complex{<:AbstractFloat}, Integer}

# Whether an element type takes a cotangent at all.
@inline _fill_active(::Type{<:Union{AbstractFloat, Complex{<:AbstractFloat}}}) = true
@inline _fill_active(::Type) = false

# `sum(dAᵢ)` as a `T`: batch element `i`'s share of dx. A struct instead of a
# closure so `ntuple` gets something concrete.
struct _FillCotangent{T, W, D}
    dvals::D
end
@inline _FillCotangent{T}(::Val{W}, dvals::D) where {T, W, D} =
    _FillCotangent{T, W, D}(dvals)
@inline (f::_FillCotangent{T, W})(i::Int) where {T, W} =
    T(Enzyme._project(T, sum(_bget(f.dvals, Val(W), i))))::T

# Same shape rules as `_dscalar`: bare at width 1, an N-tuple when batched.
@inline _dfill(w::Val{1}, ::Type{T}, dvals) where {T} = _FillCotangent{T}(w, dvals)(1)
@inline _dfill(w::Val{N}, ::Type{T}, dvals) where {N, T} =
    ntuple(_FillCotangent{T}(w, dvals), Val(N))

# Writes `val` into every shadow of `A`. Used to seed the shadow in forward mode
# and to clear it in reverse.
@inline function _fill_shadows!(config, A::Annotation, val)
    N = width(config)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        fill!(_bget(A.dval, Val(N), i), val)
        nothing
    end
    return nothing
end

function EnzymeRules.forward(
        config,
        ofn::Const{typeof(Base.fill!)},
        ::Type{RT},
        A::Annotation{<:AbstractGPUArray{T}},
        x::Annotation,
    ) where {RT, T <: _FillEltype}
    if !(A isa DuplicatedNoNeed || A isa BatchDuplicatedNoNeed)
        ofn.val(A.val, x.val)
    end

    if !_isconst(config, A)
        if x isa Const
            _fill_shadows!(config, A, zero(T))
        else
            N = width(config)
            ntuple(Val(N)) do i
                Base.@_inline_meta
                fill!(_bget(A.dval, Val(N), i), _bget(x.dval, Val(N), i))
                nothing
            end
        end
    end

    return if needs_primal(config) && needs_shadow(config)
        A
    elseif needs_shadow(config)
        A.dval
    elseif needs_primal(config)
        A.val
    else
        nothing
    end
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        ofn::Const{typeof(Base.fill!)},
        ::Type{RT},
        A::Annotation{<:AbstractGPUArray{T}},
        x::Annotation,
    ) where {RT, T <: _FillEltype}
    ofn.val(A.val, x.val)

    # An inactive element type gets no reverse pass to clear its shadow.
    if !_fill_active(T) && !_isconst(config, A)
        _fill_shadows!(config, A, zero(T))
    end

    return EnzymeRules.AugmentedReturn(
        needs_primal(config) ? A.val : nothing,
        needs_shadow(config) ? A.dval : nothing,
        nothing,
    )
end

function EnzymeRules.reverse(
        config::RevConfig,
        ofn::Const{typeof(Base.fill!)},
        ::Type{RT},
        tape,
        A::Annotation{<:AbstractGPUArray{T}},
        x::Annotation{T2},
    ) where {RT, T <: _FillEltype, T2}
    a_const = _isconst(config, A)

    dx = if !(x isa Active)
        nothing
    elseif a_const || !_fill_active(T)
        _dzero(Val(width(config)), T2)
    else
        _dfill(Val(width(config)), T2, A.dval)
    end

    if _fill_active(T) && !a_const
        _fill_shadows!(config, A, zero(T))
    end

    return (nothing, dx)
end

end # module
