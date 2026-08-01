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

# Project an accumulated cotangent onto the (possibly real) parameter type.
_project(::Type{<:Real}, x) = real(x)
_project(::Type, x) = x

#=
Cotangent of an `Active` scalar argument: `f(i)`'s raw (possibly complex)
contribution projected onto `T`, bare at width 1 and an N-tuple when batched.
Dispatch on `Val{N}` rather than branching on `N == 1`, which infers as `Any` --
a reverse rule that misses the argument's exact type is a hard error.
=#
@inline _dscalar(::Val{1}, ::Type{T}, f::F) where {T, F} = _project(T, f(1))::T
@inline function _dscalar(::Val{N}, ::Type{T}, f::F) where {N, T, F}
    return ntuple(i -> _project(T, f(i))::T, Val(N))
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

@inline _plain(t::AbstractChar) = (k = uppercase(t); k == 'N' || k == 'T' || k == 'C')

#=
Accumulate `factor .* G` into the shadow of an operand the primal read as
`wrap(X, t)`, `G` being the cotangent of that wrapped operand. 'T'/'C' only
transpose it. 'S'/'H' need a projection: one triangle is stored and each of its
entries feeds two positions of the wrapped matrix, so (i,j) collects
G[i,j] + G[j,i] and the diagonal collects G[i,i]. Hermitian keeps only the real
part there, its diagonal being real by construction.
=#
function _accumulate_operand!(dX, t::AbstractChar, G, factor)
    kind = uppercase(t)
    if kind == 'N'
        dX .+= factor .* G
    elseif kind == 'T'
        dX .+= factor .* transpose(G)
    elseif kind == 'C'
        dX .+= factor .* adjoint(G)
    else
        H = kind == 'S' ? G .+ transpose(G) : G .+ adjoint(G)
        dX .+= factor .* (_stored_upper(t) ? triu(H, 1) : tril(H, -1))
        dg = view(dX, diagind(dX))
        dg .+= factor .* (kind == 'S' ? diag(G) : real.(diag(G)))
    end
    return nothing
end

#=
`dA += factor·dC·op(B)'`, projected back through op(A).

Both halves of that are themselves a matmul with a transposition, so as long as
'N'/'T'/'C' describe both operands the whole thing is one more trip through
`generic_matmatmul!` with β = 1: the accumulation is the rule's own β argument,
no temporary and no elementwise pass. Which chars come out of rearranging
`(dC·op(B)')` into a single gemm:

    tA = 'N'    factor·dC·op(B)'          op(B)' is 'C'/conj/'N' for tB N/T/C
    tA = 'T'    factor·conj(op(B))·dCᵀ    conj(op(B)) is conj/'C'/'T'
    tA = 'C'    factor·op(B)·dC'          op(B) is tB itself

The two `conj` slots are the only ones no char covers, and they cost a
materialized conjugate for complex eltypes only. Symmetric/Hermitian can't play:
`adjoint(Symmetric(B))` is not a gemm operand and the cotangent needs projecting
onto a triangle either way, so those fall back to forming `G` and accumulating.
=#
function _pullback_left!(dA, tA::AbstractChar, tB::AbstractChar, dC, B, factor)
    if !(_plain(tA) && _plain(tB))
        _accumulate_operand!(dA, tA, dC * _wrap_adjoint(B, tB), factor)
    elseif uppercase(tA) == 'N'
        kb = uppercase(tB)
        if kb == 'N'
            _gemm!(dA, 'N', 'C', dC, B, factor, true)
        elseif kb == 'C'
            _gemm!(dA, 'N', 'N', dC, B, factor, true)
        else
            _gemm!(dA, 'N', 'N', dC, _conj(B), factor, true)
        end
    elseif uppercase(tA) == 'T'
        kb = uppercase(tB)
        if kb == 'N'
            _gemm!(dA, 'N', 'T', _conj(B), dC, factor, true)
        elseif kb == 'T'
            _gemm!(dA, 'C', 'T', B, dC, factor, true)
        else
            _gemm!(dA, 'T', 'T', B, dC, factor, true)
        end
    else
        _gemm!(dA, tB, 'C', B, dC, factor, true)
    end
    return nothing
end

# `dB += factor·op(A)'·dC`, projected back through op(B): the mirror of
# `_pullback_left!`, with the same fallback for Symmetric/Hermitian.
function _pullback_right!(dB, tA::AbstractChar, tB::AbstractChar, A, dC, factor)
    if !(_plain(tA) && _plain(tB))
        _accumulate_operand!(dB, tB, _wrap_adjoint(A, tA) * dC, factor)
    elseif uppercase(tB) == 'N'
        ka = uppercase(tA)
        if ka == 'N'
            _gemm!(dB, 'C', 'N', A, dC, factor, true)
        elseif ka == 'C'
            _gemm!(dB, 'N', 'N', A, dC, factor, true)
        else
            _gemm!(dB, 'N', 'N', _conj(A), dC, factor, true)
        end
    elseif uppercase(tB) == 'T'
        ka = uppercase(tA)
        if ka == 'N'
            _gemm!(dB, 'T', 'N', dC, _conj(A), factor, true)
        elseif ka == 'T'
            _gemm!(dB, 'T', 'C', dC, A, factor, true)
        else
            _gemm!(dB, 'T', 'T', dC, A, factor, true)
        end
    else
        _gemm!(dB, 'C', tA, dC, A, factor, true)
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
same function with β = 1 and nothing is accumulated by hand -- see
`_pullback_left!`.

3-arg `mul!` lands here with α = true, β = false, so `dC` gets zeroed, as it
must: that form overwrites `C` instead of accumulating into it.

α and β are separate arguments on Julia ≥1.12 and packed into a `MulAddMul`
before that, so the rules are version-gated wrappers over the two helpers below.
=#

@inline function _matmul_caches(
        C::Annotation, tA::AbstractChar, tB::AbstractChar, A::Annotation, B::Annotation,
        ovw_A::Bool, ovw_B::Bool, ::Val{α_active}, ::Val{β_active},
    ) where {α_active, β_active}
    #=
    dβ needs C₀, dB needs A and dA needs B; copy each only when the reverse pass
    reads it and something has overwritten it by then.
    =#
    cache_C = β_active ? copy(C.val) : nothing
    cache_A = (ovw_A && !(B isa Const) && !(C isa Const)) ? copy(A.val) : nothing
    cache_B = (ovw_B && !(A isa Const) && !(C isa Const)) ? copy(B.val) : nothing
    cache_prod = α_active ? LinearAlgebra.wrap(A.val, tA) * LinearAlgebra.wrap(B.val, tB) : nothing
    return (cache_C, cache_A, cache_B, cache_prod)
end

@inline function _matmul_reverse!(
        config, C::Annotation, tA::AbstractChar, tB::AbstractChar, A::Annotation, B::Annotation,
        αval::Number, βval::Number, tape, ::Val{α_active}, ::Val{β_active},
    ) where {α_active, β_active}
    N = width(config)
    if C isa Const
        # Without C's shadow there is nothing to pull back from.
        dα = α_active ? _dscalar(Val(N), typeof(αval), _ -> zero(αval)) : nothing
        dβ = β_active ? _dscalar(Val(N), typeof(βval), _ -> zero(βval)) : nothing
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
    ntuple(Val(N)) do i
        Base.@_inline_meta
        dC = _bget(C.dval, Val(N), i)
        if !(A isa Const)
            _pullback_left!(_bget(A.dval, Val(N), i), tA, tB, dC, Bval, αc)
        end
        if !(B isa Const)
            _pullback_right!(_bget(B.dval, Val(N), i), tA, tB, Aval, dC, αc)
        end
        dC .*= βc
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
            C, tA.val, tB.val, A, B,
            overwritten(config)[5], overwritten(config)[6], active, active,
        )

        func.val(C.val, tA.val, tB.val, A.val, B.val, add.val)

        primal = needs_primal(config) ? C.val : nothing
        shadow = needs_shadow(config) ? C.dval : nothing
        return AugmentedReturn(primal, shadow, tape)
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
            y, tA.val, 'N', A, x,
            overwritten(config)[4], overwritten(config)[5], active, active,
        )

        func.val(y.val, tA.val, A.val, x.val, add.val)

        primal = needs_primal(config) ? y.val : nothing
        shadow = needs_shadow(config) ? y.dval : nothing
        return AugmentedReturn(primal, shadow, tape)
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
            C, tA.val, tB.val, A, B,
            overwritten(config)[5], overwritten(config)[6],
            Val(!(α isa Const)), Val(!(β isa Const)),
        )

        func.val(C.val, tA.val, tB.val, A.val, B.val, α.val, β.val)

        primal = needs_primal(config) ? C.val : nothing
        shadow = needs_shadow(config) ? C.dval : nothing
        return AugmentedReturn(primal, shadow, tape)
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
            y, tA.val, 'N', A, x,
            overwritten(config)[4], overwritten(config)[5],
            Val(!(α isa Const)), Val(!(β isa Const)),
        )

        func.val(y.val, tA.val, A.val, x.val, α.val, β.val)

        primal = needs_primal(config) ? y.val : nothing
        shadow = needs_shadow(config) ? y.dval : nothing
        return AugmentedReturn(primal, shadow, tape)
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
    cache_A = (overwritten(config)[6] && !(B isa Const) && !(C isa Const)) ? copy(A.val) : nothing
    cache_B = ((aliased || overwritten(config)[7]) && !(A isa Const) && !(C isa Const)) ?
        copy(B.val) : nothing

    func.val(C.val, uploc.val, isunitc.val, tfun.val, A.val, B.val)

    primal = needs_primal(config) ? C.val : nothing
    shadow = needs_shadow(config) ? C.dval : nothing
    return AugmentedReturn(primal, shadow, (cache_A, cache_B, aliased))
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
    if !(C isa Const)
        cache_A, cache_B, aliased = tape
        Aval = cache_A !== nothing ? cache_A : A.val
        Bval = cache_B !== nothing ? cache_B : B.val
        M′ = (B isa Const) ? nothing :
            _triangular_adjoint(Aval, uploc.val, isunitc.val, tfun.val)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dC = _bget(C.dval, Val(N), i)
            # dA first: when the shadows alias, dB below overwrites dC.
            if !(A isa Const)
                _accumulate_triangular!(
                    _bget(A.dval, Val(N), i), uploc.val, isunitc.val, tfun.val,
                    dC * adjoint(Bval),
                )
            end
            if !(B isa Const)
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
    cache_A = ((aliased || overwritten(config)[6]) && !(B isa Const) && !(C isa Const)) ?
        copy(A.val) : nothing
    cache_B = (overwritten(config)[7] && !(A isa Const) && !(C isa Const)) ? copy(B.val) : nothing

    func.val(C.val, uploc.val, isunitc.val, tfun.val, A.val, B.val)

    primal = needs_primal(config) ? C.val : nothing
    shadow = needs_shadow(config) ? C.dval : nothing
    return AugmentedReturn(primal, shadow, (cache_A, cache_B, aliased))
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
    if !(C isa Const)
        cache_A, cache_B, aliased = tape
        Aval = cache_A !== nothing ? cache_A : A.val
        Bval = cache_B !== nothing ? cache_B : B.val
        M′ = (A isa Const) ? nothing :
            _triangular_adjoint(Bval, uploc.val, isunitc.val, tfun.val)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            dC = _bget(C.dval, Val(N), i)
            # dB first: when the shadows alias, dA below overwrites dC.
            if !(B isa Const)
                _accumulate_triangular!(
                    _bget(B.dval, Val(N), i), uploc.val, isunitc.val, tfun.val,
                    adjoint(Aval) * dC,
                )
            end
            if !(A isa Const)
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
    cache_a = (overwritten(config)[2] && !(b isa Const)) ? copy(a.val) : nothing
    cache_b = (overwritten(config)[3] && !(a isa Const)) ? copy(b.val) : nothing
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
            dr = _bget(dret.val, Val(N), i)
            if !(a isa Const)
                # `a` entered conjugated, so its cotangent picks up conj(dr).
                _bget(a.dval, Val(N), i) .+= conj(dr) .* bv
            end
            if !(b isa Const)
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
    if !(dret isa Const) && !(x isa Const)
        N = width(config)
        ntuple(Val(N)) do i
            Base.@_inline_meta
            _bget(x.dval, Val(N), i) .+= _bget(dret.val, Val(N), i)
            nothing
        end
    end
    return (nothing,)
end

end # module
