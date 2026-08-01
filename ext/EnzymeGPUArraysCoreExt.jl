module EnzymeGPUArraysCoreExt

using GPUArraysCore
using Enzyme
using LinearAlgebra: LinearAlgebra, dot, Symmetric, Hermitian, UpperTriangular,
    LowerTriangular, UnitUpperTriangular, UnitLowerTriangular, triu, tril, diag, diagind
using Enzyme.EnzymeCore: EnzymeCore
using Enzyme.EnzymeCore.EnzymeRules:
    EnzymeRules,
    RevConfig,
    Annotation,
    AugmentedReturn,
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

#=
The cotangent of an `Active` return. Batching splits this differently from a
shadow: it is one `Active` at width 1 but an N-tuple *of* `Active`s above it,
not an `Active` holding a tuple, so the unwrap has to happen per element.
=#
@inline _dret(dret, w::Val, i::Int) = _bget(dret, w, i).val

#=
An operand contributes no cotangent when it is `Const`, and also when runtime
activity hands us a `Duplicated` whose shadow is the primal itself -- writing to
that would corrupt the value the augmented pass just computed.
=#
@inline _isconst(_, ::Const) = true
@inline _isconst(config, x::Annotation) =
    EnzymeRules.runtime_activity(config) && x.dval === x.val

# Every rule here returns its output argument and stashes a tape.
@inline _augreturn(config, out::Annotation, tape) = AugmentedReturn(
    needs_primal(config) ? out.val : nothing,
    needs_shadow(config) ? out.dval : nothing,
    tape,
)

#=
Cotangent of an `Active` scalar argument: `f(i)`'s raw (possibly complex)
contribution projected onto `T`, bare at width 1 and an N-tuple when batched.
Dispatch on `Val{N}` rather than branching on `N == 1`, which infers as `Any` --
a reverse rule that misses the argument's exact type is a hard error.
=#
@inline _dscalar(::Val{1}, ::Type{T}, f::F) where {T, F} = Enzyme._project(T, f(1))::T
@inline function _dscalar(::Val{N}, ::Type{T}, f::F) where {N, T, F}
    return ntuple(i -> Enzyme._project(T, f(i))::T, Val(N))
end

#=
Before anything reaches the GPU, `mul!` and `*` strip the wrapper off each
operand and pass a char for how to read the bare array: 'N' plain, 'T'
transposed, 'C' adjoint, 'S' Symmetric, 'H' Hermitian, the case naming the
stored triangle ('S' upper, 's' lower). Julia ≥1.11 sends a `WrapperChar`, which
holds the case in its own field, so go through `Char` to get the triangle back.
=#
@inline _stored_upper(t::AbstractChar) = isuppercase(Char(t))

# `conj`, materialized; a no-op for real eltypes.
@inline _conj(X) = eltype(X) <: Real ? X : conj.(X)

#=
`adjoint(wrap(X, t))`, leaving at most one lazy wrapper on the array. The
backends peel off a single layer, so a nested `adjoint(transpose(X))` (or
`adjoint(Symmetric(X))` when complex) matches no method and scalar-indexes.
=#
@inline function _wrap_adjoint(X, t::AbstractChar)
    kind = uppercase(t)
    return if kind == 'N'
        adjoint(X)
    elseif kind == 'T'
        _conj(X)
    elseif kind == 'C'
        X
    elseif kind == 'S'
        Symmetric(_conj(X), _stored_upper(t) ? :U : :L)
    else
        Hermitian(X, _stored_upper(t) ? :U : :L)
    end
end

@static if VERSION < v"1.12.0-DEV"
    @inline _gemm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, LinearAlgebra.MulAddMul(α, β))
else
    @inline _gemm!(C, tA::AbstractChar, tB::AbstractChar, A, B, α::Number, β::Number) =
        LinearAlgebra.generic_matmatmul!(C, tA, tB, A, B, α, β)
end

#=
The ops a char can name -- identity, transpose, conj, adjoint -- form a group
under composition, so write one as the pair of flags (transposed, conjugated)
and the pullbacks stop needing a lookup table: `adjoint` flips both flags,
`transpose` flips the first, and applying either to a product also swaps its two
operands. `wrap`'s chars cover three of the four combinations; the fourth, a
bare conj, has to be materialized (`_opdata`), and only for complex eltypes.
=#
@inline _plain(t::AbstractChar) = (k = uppercase(t); k == 'N' || k == 'T' || k == 'C')
@inline _opflags(t::AbstractChar) =
    (k = uppercase(t); k == 'N' ? (false, false) : k == 'T' ? (true, false) : (true, true))
@inline _adjop(f::Tuple{Bool, Bool}) = (!f[1], !f[2])
@inline _transop(f::Tuple{Bool, Bool}) = (!f[1], f[2])
@inline _opchar(f::Tuple{Bool, Bool}) = f[1] ? (f[2] ? 'C' : 'T') : 'N'
@inline _opdata(X, f::Tuple{Bool, Bool}) = (!f[1] && f[2]) ? _conj(X) : X

#=
Accumulate `G` into the shadow of an operand the primal read as `wrap(X, t)`,
`G` being the (already scaled) cotangent of that wrapped operand. 'T'/'C' only
transpose it. 'S'/'H' need a projection: one triangle is stored and each of its
entries feeds two positions of the wrapped matrix, so (i,j) collects
G[i,j] + G[j,i] and the diagonal collects G[i,i]. Hermitian keeps only the real
part there, its diagonal being real by construction.

`G` has to arrive scaled by conj(α) rather than scaled here, because the
Hermitian projection is not linear in that factor: it conjugates one of the two
terms it sums, and takes the real part of the diagonal.
=#
function _accumulate_operand!(dX, t::AbstractChar, G)
    kind = uppercase(t)
    if kind == 'N'
        dX .+= G
    elseif kind == 'T'
        dX .+= transpose(G)
    elseif kind == 'C'
        dX .+= adjoint(G)
    else
        H = kind == 'S' ? G .+ transpose(G) : G .+ adjoint(G)
        dX .+= _stored_upper(t) ? triu(H, 1) : tril(H, -1)
        dg = view(dX, diagind(dX))
        dg .+= kind == 'S' ? diag(G) : real.(diag(G))
    end
    return nothing
end

# Scale a cotangent we just allocated, in place, skipping the no-op α = 1 case.
@inline function _scale!(G, factor)
    isone(factor) || (G .*= factor)
    return G
end

#=
`dX += factor · op(L, fL) · op(R, fR)`, projected back through the operand's own
op `tX`. Both cotangents of a matmul have that shape -- `dC·op(B)'` for the left
operand, `op(A)'·dC` for the right -- so both go back through
`generic_matmatmul!` with β = 1, which is the accumulation: no temporary, no
elementwise pass. Undoing `tX` rewrites the product rather than transposing its
result, which is where the operand swap comes from.

Symmetric/Hermitian can't play: `adjoint(Symmetric(B))` is not a gemm operand,
and the cotangent has to be projected onto a triangle regardless, so those form
the product and hand it to `_accumulate_operand!`.
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

Where every non-triangular GPU matmul ends up, whether it came from `*` or from
3-/5-arg `mul!`, and whether the backend takes it on to CUBLAS or to the
GPUArrays fallback kernels. One rule per function therefore covers every wrapper
combination:

    dA += conj(α)·dC·op(B)'          projected back through op(A)
    dB += conj(α)·op(A)'·dC          projected back through op(B)
    dC := conj(β)·dC                 cotangent w.r.t. C₀
    dα  = conj(⟨dC, op(A)·op(B)⟩)
    dβ  = conj(⟨dC, C₀⟩)

The first two are matmuls with a transposition, so they go back through this
same function with β = 1 and nothing is accumulated by hand -- see `_pullback!`.

3-arg `mul!` lands here with α = true, β = false, so `dC` gets zeroed, as it
must: that form overwrites `C` instead of accumulating into it.

α and β are separate arguments on Julia ≥1.12 and packed into a `MulAddMul`
before that, so the rules are version-gated wrappers over the two helpers below.
=#

@inline function _matmul_caches(
        config, C::Annotation, tA::AbstractChar, tB::AbstractChar, A::Annotation, B::Annotation,
        ovw_A::Bool, ovw_B::Bool, ::Val{α_active}, ::Val{β_active},
    ) where {α_active, β_active}
    #=
    dβ needs C₀, dα the product, dB needs A and dA needs B. A `Const` C means the
    reverse pass returns early and reads none of them, and `cache_prod` costs a
    second matmul, so gate everything on the cotangent actually being wanted.
    =#
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
        # Without C's shadow there is nothing to pull back from.
        dα = α_active ? _dscalar(Val(N), typeof(αval), Returns(zero(αval))) : nothing
        dβ = β_active ? _dscalar(Val(N), typeof(βval), Returns(zero(βval))) : nothing
        return (dα, dβ)
    end

    cache_C, cache_A, cache_B, cache_prod = tape
    Cval = cache_C !== nothing ? cache_C : C.val
    Aval = cache_A !== nothing ? cache_A : A.val
    Bval = cache_B !== nothing ? cache_B : B.val

    dα = α_active ?
        _dscalar(Val(N), typeof(αval), i -> conj(dot(_bget(C.dval, Val(N), i), cache_prod))) :
        nothing
    dβ = β_active ?
        _dscalar(Val(N), typeof(βval), i -> conj(dot(_bget(C.dval, Val(N), i), Cval))) :
        nothing

    αc = conj(αval)
    βc = conj(βval)
    fA = _opflags(tA)
    fB = _opflags(tB)
    a_const = _isconst(config, A)
    b_const = _isconst(config, B)
    plain = _plain(tA) && _plain(tB)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        dC = _bget(C.dval, Val(N), i)
        if !a_const
            dA = _bget(A.dval, Val(N), i)
            plain ? _pullback!(dA, tA, dC, (false, false), Bval, _adjop(fB), αc) :
                _accumulate_operand!(dA, tA, _scale!(dC * _wrap_adjoint(Bval, tB), αc))
        end
        if !b_const
            dB = _bget(B.dval, Val(N), i)
            plain ? _pullback!(dB, tB, Aval, _adjop(fA), dC, (false, false), αc) :
                _accumulate_operand!(dB, tB, _scale!(_wrap_adjoint(Aval, tA) * dC, αc))
        end
        # β = 1 is the accumulate-into-C case, where this would be a no-op kernel.
        isone(βc) || (dC .*= βc)
        nothing
    end

    return (dα, dβ)
end

@static if VERSION < v"1.12.0-DEV"

    #=
    With α and β packed into one `MulAddMul`, an active α or β arrives as a
    single active struct, so its cotangent has to come back as that struct type.
    =#
    @inline _dmuladdmul(::Val{1}, ::Type{MAM}, dα, dβ) where {MAM} = MAM(dα, dβ)
    @inline function _dmuladdmul(::Val{N}, ::Type{MAM}, dα, dβ) where {N, MAM}
        return ntuple(i -> MAM(dα[i], dβ[i]), Val(N))
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

        return _augreturn(config, C, tape)
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

        return _augreturn(config, y, tape)
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

        return _augreturn(config, C, tape)
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

        return _augreturn(config, y, tape)
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

#=
    generic_trimatmul!(C, uploc, isunitc, tfun, A, B)    C = tfun(T(A))·B
    generic_mattrimul!(C, uploc, isunitc, tfun, A, B)    C = A·tfun(T(B))

Triangular operands never reach `generic_matmatmul!`; LinearAlgebra sends them
here instead, again bare plus a description of how to read them: `uploc`
('U'/'L') names the stored triangle, `isunitc` is 'U' when the diagonal is
structurally 1, and `tfun` is identity/transpose/adjoint. The operand is
`tfun(T_uploc(A))` -- `transpose(UpperTriangular(X))` arrives as uploc 'U' with
`tfun` `transpose`. There is no α/β here, `C` is always overwritten, so this is
where `C`'s cotangent gets consumed. With `M` the operand as the primal reads it:

    dA += tfun(tril/triu(dC·B'))
    dB += M'·dC
    dC := 0

`M'·dC` goes through `mul!` into a zeroed buffer, not `*`. The GPUArrays kernels
for these two end with `C[i,j] += …` rather than `=`, so with the uninitialized
array `*` allocates they mix junk into the result; zeroing first makes that
accumulation harmless, and backends that do overwrite (CUBLAS trmm) don't care.
=#

#=
`adjoint(tfun(T_uploc(A)))`, again with a single lazy wrapper (see
`_wrap_adjoint`). Only `identity` leaves an adjoint behind, and with it the
flipped triangle; `transpose` and `adjoint` cancel against the outer one, the
former down to a conjugation of the data.
=#
@inline function _triangular_adjoint(A, uploc::AbstractChar, isunitc::AbstractChar, tfun::F) where {F}
    P = tfun === identity ? adjoint(A) : (tfun === transpose ? _conj(A) : A)
    upper = tfun === identity ? uploc != 'U' : uploc == 'U'
    return if upper
        isunitc == 'U' ? UnitUpperTriangular(P) : UpperTriangular(P)
    else
        isunitc == 'U' ? UnitLowerTriangular(P) : LowerTriangular(P)
    end
end

#=
Project `G` back onto A's stored triangle, skipping a structurally-fixed unit
diagonal. `tfun` swaps the triangles, so it can be undone after the projection
rather than before -- and it has to be, because `triu`/`tril` of a lazy transpose
drops into a scalar-indexing loop while broadcasting over one is fine.
=#
@inline function _accumulate_triangular!(
        dA, uploc::AbstractChar, isunitc::AbstractChar, tfun::F, G,
    ) where {F}
    k = isunitc == 'U' ? 1 : 0
    upper = tfun === identity ? uploc == 'U' : uploc != 'U'
    dA .+= tfun(upper ? triu(G, k) : tril(G, -k))
    return nothing
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(LinearAlgebra.generic_trimatmul!)},
        ::Type{RT},
        C::Annotation{<:AbstractGPUVecOrMat},
        uploc::Const{<:AbstractChar},
        isunitc::Const{<:AbstractChar},
        tfun::Const,
        A::Annotation{<:AbstractGPUMatrix},
        B::Annotation{<:AbstractGPUVecOrMat},
    ) where {RT}
    # `lmul!(T, B)` arrives as C === B, so the primal overwrites the B that dA needs.
    aliased = C.val === B.val
    cache_A = (overwritten(config)[6] && !_isconst(config, B) && !_isconst(config, C)) ? copy(A.val) : nothing
    cache_B = ((aliased || overwritten(config)[7]) && !_isconst(config, A) && !_isconst(config, C)) ?
        copy(B.val) : nothing

    func.val(C.val, uploc.val, isunitc.val, tfun.val, A.val, B.val)

    return _augreturn(config, C, (cache_A, cache_B, aliased))
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(LinearAlgebra.generic_trimatmul!)},
        ::Type{RT},
        tape,
        C::Annotation{<:AbstractGPUVecOrMat},
        uploc::Const{<:AbstractChar},
        isunitc::Const{<:AbstractChar},
        tfun::Const,
        A::Annotation{<:AbstractGPUMatrix},
        B::Annotation{<:AbstractGPUVecOrMat},
    ) where {RT}
    if !_isconst(config, C)
        cache_A, cache_B, aliased = tape
        Aval = cache_A !== nothing ? cache_A : A.val
        Bval = cache_B !== nothing ? cache_B : B.val
        M′ = _isconst(config, B) ? nothing :
            _triangular_adjoint(Aval, uploc.val, isunitc.val, tfun.val)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dC = _bget(C.dval, Val(N), i)
            # dA first: when the shadows alias, dB below overwrites dC.
            if !_isconst(config, A)
                _accumulate_triangular!(
                    _bget(A.dval, Val(N), i), uploc.val, isunitc.val, tfun.val,
                    dC * adjoint(Bval),
                )
            end
            if !_isconst(config, B)
                dB = _bget(B.dval, Val(N), i)
                G = LinearAlgebra.mul!(zero(dC), M′, dC)
                # Replace when C === B: dB is dC, and C's cotangent is spent here.
                aliased ? (dB .= G) : (dB .+= G)
            end
            aliased || fill!(dC, zero(eltype(dC)))
            nothing
        end
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(LinearAlgebra.generic_mattrimul!)},
        ::Type{RT},
        C::Annotation{<:AbstractGPUMatrix},
        uploc::Const{<:AbstractChar},
        isunitc::Const{<:AbstractChar},
        tfun::Const,
        A::Annotation{<:AbstractGPUMatrix},
        B::Annotation{<:AbstractGPUMatrix},
    ) where {RT}
    # `rmul!(A, T)` lands here as C === A.
    aliased = C.val === A.val
    cache_A = ((aliased || overwritten(config)[6]) && !_isconst(config, B) && !_isconst(config, C)) ?
        copy(A.val) : nothing
    cache_B = (overwritten(config)[7] && !_isconst(config, A) && !_isconst(config, C)) ? copy(B.val) : nothing

    func.val(C.val, uploc.val, isunitc.val, tfun.val, A.val, B.val)

    return _augreturn(config, C, (cache_A, cache_B, aliased))
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(LinearAlgebra.generic_mattrimul!)},
        ::Type{RT},
        tape,
        C::Annotation{<:AbstractGPUMatrix},
        uploc::Const{<:AbstractChar},
        isunitc::Const{<:AbstractChar},
        tfun::Const,
        A::Annotation{<:AbstractGPUMatrix},
        B::Annotation{<:AbstractGPUMatrix},
    ) where {RT}
    if !_isconst(config, C)
        cache_A, cache_B, aliased = tape
        Aval = cache_A !== nothing ? cache_A : A.val
        Bval = cache_B !== nothing ? cache_B : B.val
        M′ = _isconst(config, A) ? nothing :
            _triangular_adjoint(Bval, uploc.val, isunitc.val, tfun.val)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dC = _bget(C.dval, Val(N), i)
            # dB first: when the shadows alias, dA below overwrites dC.
            if !_isconst(config, B)
                _accumulate_triangular!(
                    _bget(B.dval, Val(N), i), uploc.val, isunitc.val, tfun.val,
                    adjoint(Aval) * dC,
                )
            end
            if !_isconst(config, A)
                dA = _bget(A.dval, Val(N), i)
                G = LinearAlgebra.mul!(zero(dC), dC, M′)
                # Replace when C === A: dA is dC, and C's cotangent is spent here.
                aliased ? (dA .= G) : (dA .+= G)
            end
            aliased || fill!(dC, zero(eltype(dC)))
            nothing
        end
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

#=
dot(a, b) = Σ conj(aᵢ)·bᵢ, so `da += conj(dr)·b` and `db += dr·a`. The
asymmetry only shows up for complex eltypes; test/ext/jlarrays.jl pins both
directions against Enzyme's own rule-free CPU path.
=#

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(dot)},
        ::Type,
        a::Annotation{<:AbstractGPUArray},
        b::Annotation{<:AbstractGPUArray},
    )
    primal = needs_primal(config) ? dot(a.val, b.val) : nothing
    cache_a = (overwritten(config)[2] && !_isconst(config, b)) ? copy(a.val) : nothing
    cache_b = (overwritten(config)[3] && !_isconst(config, a)) ? copy(b.val) : nothing
    return AugmentedReturn(primal, nothing, (cache_a, cache_b))
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
            if !_isconst(config, a)
                # `a` entered conjugated, so its cotangent picks up conj(dr).
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

# sum(x): the return cotangent broadcasts onto every element, `dx += dr`.

function EnzymeRules.augmented_primal(
        config::RevConfig,
        func::Const{typeof(sum)},
        ::Type,
        x::Annotation{<:AbstractGPUArray},
    )
    primal = needs_primal(config) ? sum(x.val) : nothing
    return AugmentedReturn(primal, nothing, nothing)
end

function EnzymeRules.reverse(
        config::RevConfig,
        func::Const{typeof(sum)},
        dret,
        tape,
        x::Annotation{<:AbstractGPUArray},
    )
    if !(dret isa Const) && !_isconst(config, x)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            _bget(x.dval, Val(N), i) .+= _dret(dret, Val(N), i)
            nothing
        end
    end
    return (nothing,)
end

end # module
