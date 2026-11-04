using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using LinearAlgebra
using Random
using Test

# Products with the Q of a QR, LQ, QL, RQ, RZ or Hessenberg factorization, which LAPACK's
# gemqrt and orm* routines compute in place for BLAS matrices, with Q held constant.

householder_mul_out(f, side, trans, A, τ, C) = f(side, trans, A, τ, copy(C))
householder_mul_sum(f, side, trans, A, τ, C) = sum(householder_mul_out(f, side, trans, A, τ, C))
ormhr_out(side, trans, ilo, ihi, A, τ, C) = LAPACK.ormhr!(side, trans, ilo, ihi, A, τ, copy(C))
ormtr_out(side, uplo, trans, A, τ, C) = LAPACK.ormtr!(side, uplo, trans, A, τ, copy(C))

# geqrt! with block size nb. It leaves the entries of T that gemqrt! does not read (below the
# diagonal of each block) uninitialized, where a NaN would fail the check that a rule leaves
# T unchanged, so zero them.
function geqrt_defined!(A, nb)
    V, T = LAPACK.geqrt!(A, nb)
    k = size(T, 2)
    for j0 in 1:nb:k, j in j0:min(j0 + nb - 1, k), i in (j - j0 + 2):nb
        T[i, j] = 0
    end
    return V, T
end

# Each routine with the two arrays that store a 5×5 Q, from the matching factorization.
function householder_factors(T, rng)
    tall = rand(rng, T, 5, 3)
    wide = rand(rng, T, 3, 5)
    return (
        (LAPACK.gemqrt!, geqrt_defined!(copy(tall), 2)),
        (LAPACK.ormqr!, LAPACK.geqrf!(copy(tall))),
        (LAPACK.ormlq!, LAPACK.gelqf!(copy(wide))),
        (LAPACK.ormql!, LAPACK.geqlf!(copy(tall))),
        (LAPACK.ormrq!, LAPACK.gerqf!(copy(wide))),
        (LAPACK.ormrz!, LAPACK.tzrzf!(triu(wide))),
    )
end

const ACTIVITIES = (Const, Duplicated, BatchDuplicated)

# op(Q) for side 'L' and 'R', and every trans LAPACK accepts for the element type.
function householder_cases(T, rng)
    transes = T <: Real ? ('N', 'T') : ('N', 'C')
    cases = []
    for side in ('L', 'R'), trans in transes
        sizes = side == 'L' ? ((5,), (5, 2)) : ((2, 5),)
        for sz in sizes
            push!(cases, (side, trans, rand(rng, T, sz...)))
        end
    end
    return cases
end

@testset "Householder products ($T)" for T in (Float64, ComplexF64)
    rng = Xoshiro(3)
    cases = householder_cases(T, rng)
    @testset "$f" for (f, (A, τ)) in householder_factors(T, rng)
        for (side, trans, C) in cases
            # ormrz! is defined for a matrix C only.
            f === LAPACK.ormrz! && C isa AbstractVector && continue
            args = ((f, Const), (side, Const), (trans, Const), (A, Const), (τ, Const))
            @testset "side=$side trans=$trans C of size $(size(C)), $Tret, C $TC" for Tret in ACTIVITIES,
                    TC in ACTIVITIES
                are_activities_compatible(Tret, TC) || continue
                test_forward(householder_mul_out, Tret, args..., (C, TC))
                test_reverse(householder_mul_out, Tret, args..., (C, TC))
            end
        end
    end
    @testset "ormtr!, uplo=$uplo" for uplo in ('U', 'L')
        S = rand(rng, T, 5, 5)
        H = S + S'
        A, τ = LAPACK.hetrd!(uplo, H)
        for (side, trans, C) in cases
            args = ((side, Const), (uplo, Const), (trans, Const), (A, Const), (τ, Const))
            for Tret in ACTIVITIES, TC in ACTIVITIES
                are_activities_compatible(Tret, TC) || continue
                test_forward(ormtr_out, Tret, args..., (C, TC))
                test_reverse(ormtr_out, Tret, args..., (C, TC))
            end
        end
    end
    @testset "ormhr!, ilo=$ilo ihi=$ihi" for (ilo, ihi) in ((1, 5), (2, 4))
        A, τ = LAPACK.gehrd!(ilo, ihi, rand(rng, T, 5, 5))
        for (side, trans, C) in cases
            args = ((side, Const), (trans, Const), (ilo, Const), (ihi, Const), (A, Const), (τ, Const))
            for Tret in ACTIVITIES, TC in ACTIVITIES
                are_activities_compatible(Tret, TC) || continue
                test_forward(ormhr_out, Tret, args..., (C, TC))
                test_reverse(ormhr_out, Tret, args..., (C, TC))
            end
        end
    end
end

# The full square Q, as a matrix.
full_q(Q, n) = lmul!(Q, Matrix{eltype(Q)}(I, n, n))

# X annotated as T, with the shadow init(X) when T is Duplicated.
annotate(X, T, init) = T === Const ? Const(X) : Duplicated(X, init(X))
ones_like(X) = fill!(similar(X), 1)

# The rules have no derivative with respect to the arrays that store Q. In Forward mode a
# zero tangent of them contributes nothing, so only a nonzero one is refused there.
@testset "Householder products refuse active factors" begin
    A, τ = geqrt_defined!(rand(Xoshiro(4), 5, 3), 2)
    C = rand(Xoshiro(5), 5)
    v = rand(Xoshiro(6), 5)
    flags = (Const(LAPACK.gemqrt!), Const('L'), Const('N'))
    tangent = autodiff(
        Forward, householder_mul_out, Duplicated, flags..., Const(A), Const(τ), Duplicated(C, v)
    )[1]
    for (TA, Tτ) in ((Duplicated, Const), (Const, Duplicated), (Duplicated, Duplicated))
        a, t = annotate(A, TA, zero), annotate(τ, Tτ, zero)
        @test_throws ArgumentError autodiff(
            Reverse, householder_mul_sum, Active, flags..., a, t, Duplicated(C, zero(C))
        )
        @test autodiff(Forward, householder_mul_out, Duplicated, flags..., a, t, Duplicated(C, v))[1] ≈
            tangent
        a, t = annotate(A, TA, ones_like), annotate(τ, Tτ, ones_like)
        @test_throws ArgumentError autodiff(
            Forward, householder_mul_out, Duplicated, flags..., a, t, Duplicated(C, v)
        )
    end
    # A shadow that is the factor itself carries no derivative: Enzyme passes one for data
    # that is inactive at run time.
    dC = zero(C)
    autodiff(
        Reverse, householder_mul_sum, Active, flags..., Duplicated(A, A), Duplicated(τ, τ),
        Duplicated(C, dC)
    )
    @test dC ≈ full_q(LinearAlgebra.QRCompactWYQ(A, τ), 5)' * ones(5)
end

# Through `*` with the Q of each factorization. With M the full square Q, the gradient of
# w ⋅ (Q x) is Mᵀ w, and so on.
q_dot(x, Q, w) = dot(w, Q * x)
q_adjoint_dot(x, Q, w) = dot(w, Q' * x)
q_row_dot(x, Q, w) = (x' * Q) * w
q_matrix_dot(X, Q, W) = dot(W, Q * X)
q_matrix_right_dot(X, Q, W) = dot(W, X * Q)
q_sum_exp(x, Q) = sum(exp, Q * x)
q_snag(x, Q) = sum(abs2, Q * x)

function q_grad(f, x, Q)
    dx = zero(x)
    autodiff_deferred(Reverse, Const(f), Active, Duplicated(x, dx), Const(Q))
    return dx
end

q_directional(f, x, v, Q) = autodiff_deferred(Forward, Const(f), Duplicated, Duplicated(x, v), Const(Q))[1]

function q_types(rng)
    tall = randn(rng, 6, 3)
    square = randn(rng, 6, 6)
    return (
        "qr" => qr(tall).Q,
        "qr, two blocks" => LinearAlgebra.QRCompactWYQ(geqrt_defined!(copy(tall), 2)...),
        "qr, ColumnNorm" => qr(tall, ColumnNorm()).Q,
        "lq" => lq(copy(tall')).Q,
        "hessenberg" => hessenberg(square).Q,
        "hessenberg, Hermitian U" => hessenberg(Hermitian(square, :U)).Q,
        "hessenberg, Hermitian L" => hessenberg(Hermitian(square, :L)).Q,
    )
end

# Static Forward over Reverse through the product needs runtime activity for some Q types:
# the inner Reverse pass keeps the rule's constant factors on its tape, as it does for any
# custom rule given a constant array, and the outer Forward pass then raises
# EnzymeRuntimeActivityError. Which types depends on the Julia version, not on bounds
# checking (measured on 1.10.12 and 1.12.7; other versions are not pinned).
function static_forward_over_reverse_broken(name)
    if VERSION < v"1.11"
        return name == "lq" || startswith(name, "hessenberg")
    elseif v"1.12" <= VERSION < v"1.13"
        return name == "lq" || startswith(name, "qr")
    else
        return nothing
    end
end

@testset "x -> Q * x with constant Q: $name" for (name, Q) in q_types(Xoshiro(6))
    rng = Xoshiro(7)
    M = full_q(Q, 6)
    x = randn(rng, 6)
    w = randn(rng, 6)
    X = randn(rng, 6, 2)
    W = randn(rng, 6, 2)
    Xr = randn(rng, 2, 6)
    Wr = randn(rng, 2, 6)
    for mode in (Reverse, set_runtime_activity(Reverse))
        @test Enzyme.gradient(mode, q_dot, x, Const(Q), Const(w))[1] ≈ M' * w
        @test Enzyme.gradient(mode, q_adjoint_dot, x, Const(Q), Const(w))[1] ≈ M * w
        @test Enzyme.gradient(mode, q_row_dot, x, Const(Q), Const(w))[1] ≈ M * w
        @test Enzyme.gradient(mode, q_matrix_dot, X, Const(Q), Const(W))[1] ≈ M' * W
        @test Enzyme.gradient(mode, q_matrix_right_dot, Xr, Const(Q), Const(Wr))[1] ≈ Wr * M'
        @test Enzyme.gradient(mode, q_snag, x, Const(Q))[1] ≈ 2x
    end
    v = randn(rng, 6)
    @test autodiff(Forward, q_dot, Duplicated(x, v), Const(Q), Const(w))[1] ≈ dot(w, M * v)
    @test autodiff(Forward, q_adjoint_dot, Duplicated(x, v), Const(Q), Const(w))[1] ≈
        dot(w, M' * v)
    @test autodiff(Forward, q_row_dot, Duplicated(x, v), Const(Q), Const(w))[1] ≈ v' * M * w
    # Hessian-vector products: the Hessian of sum(exp, M x) is Mᵀ Diagonal(exp.(M x)) M.
    hv_ref = M' * (exp.(M * x) .* (M * v))
    q_hvp(mode) = autodiff(mode, q_grad, Duplicated, Const(q_sum_exp), Duplicated(x, v), Const(Q))[1]
    @test q_hvp(set_runtime_activity(Forward)) ≈ hv_ref
    broken = static_forward_over_reverse_broken(name)
    if broken === false
        @test q_hvp(Forward) ≈ hv_ref
    elseif broken === true
        @test_broken q_hvp(Forward) ≈ hv_ref
    end
    for mode in (Reverse, set_runtime_activity(Reverse))
        hv = Enzyme.gradient(mode, q_directional, Const(q_sum_exp), x, Const(v), Const(Q))[2]
        @test hv ≈ hv_ref
    end
end
