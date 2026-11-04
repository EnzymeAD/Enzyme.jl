using Enzyme
using EnzymeTestUtils
using FiniteDifferences
using LinearAlgebra
using Test

# Forward mode and forward over reverse through triangular solves, which LAPACK's trtrs
# computes for strided BLAS matrices.

trtrs_out(uplo, trans, diag, A, B) = LAPACK.trtrs!(uplo, trans, diag, A, copy(B))

function trtrs_matrix(T, n)
    A = rand(T, n, n)
    A[diagind(A)] .+= n
    return A
end

const ACTIVITIES = (Const, Duplicated, BatchDuplicated)

@testset "trtrs! rules ($T, B of size $sz)" for T in (Float64, ComplexF64), sz in ((3,), (3, 2))
    A = trtrs_matrix(T, 3)
    B = rand(T, sz...)
    @testset "uplo=$uplo trans=$trans diag=$diag" for uplo in ('L', 'U'), trans in ('N', 'T', 'C'),
            diag in ('N', 'U')
        args = ((uplo, Const), (trans, Const), (diag, Const))
        @testset "forward $Tret, A $TA, B $TB" for Tret in ACTIVITIES, TA in ACTIVITIES, TB in ACTIVITIES
            are_activities_compatible(Tret, TA, TB) || continue
            test_forward(trtrs_out, Tret, args..., (A, TA), (B, TB))
        end
        # Complex trtrs is not handled in reverse mode.
        T <: Real || continue
        @testset "reverse $Tret, A $TA, B $TB" for Tret in ACTIVITIES, TA in ACTIVITIES, TB in ACTIVITIES
            are_activities_compatible(Tret, TA, TB) || continue
            test_reverse(trtrs_out, Tret, args..., (A, TA), (B, TB))
        end
    end
end

# The triangular matrix is built inside, as W(P(M)) with P one of identity, transpose and
# adjoint.
ldiv_2arg(W, P, M, B) = ldiv!(W(P(M)), copy(B))
ldiv_3arg(W, P, M, B) = ldiv!(similar(B), W(P(M)), B)

@testset "ldiv! with $W ($T)" for T in (Float64, ComplexF64),
        W in (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
    M = trtrs_matrix(T, 3)
    B = rand(T, 3)
    @testset "$f, wrapping $P(M)" for f in (ldiv_2arg, ldiv_3arg), P in (identity, transpose, adjoint)
        args = ((W, Const), (P, Const))
        activities = (
            (Duplicated, Const, Duplicated), (Duplicated, Duplicated, Const),
            (Duplicated, Duplicated, Duplicated), (BatchDuplicated, BatchDuplicated, BatchDuplicated),
        )
        @testset "forward $Tret, M $TM, B $TB" for (Tret, TM, TB) in activities
            test_forward(f, Tret, args..., (M, TM), (B, TB))
        end
        T <: Real || continue
        @testset "reverse $Tret, M $TM, B $TB" for (Tret, TM, TB) in activities
            test_reverse(f, Tret, args..., (M, TM), (B, TB))
        end
    end
end

# -xᵀ (L Lᵀ)⁻¹ x / 2 and -xᵀ (Lᵀ L)⁻¹ x / 2, with Hessians -(L Lᵀ)⁻¹ and -(Lᵀ L)⁻¹.
tri_solve(x, L) = -sum(abs2, L \ x) / 2
tri_solve_adjoint(x, L) = -sum(abs2, L' \ x) / 2
tri_solve_transpose(x, L) = -sum(abs2, transpose(L) \ x) / 2
tri_solve_ldiv(x, L) = -sum(abs2, ldiv!(L, copy(x))) / 2
tri_solve_3arg(x, L) = -sum(abs2, ldiv!(similar(x), L, x)) / 2

function tri_grad!(dx, f, x, L)
    fill!(dx, 0)
    autodiff_deferred(Reverse, Const(f), Active, Duplicated(x, dx), Const(L))
    return nothing
end

function tri_grad_active!(dx, dL, f, x, L)
    fill!(dx, 0)
    fill!(parent(dL), 0)
    autodiff_deferred(Reverse, Const(f), Active, Duplicated(x, dx), Duplicated(L, dL))
    return nothing
end

# Hessian-vector product along v, forward over reverse.
function tri_hvp(mode, f, x, v, L)
    dx, hv = zeros(length(x)), zeros(length(x))
    autodiff(mode, Const(tri_grad!), Const, Duplicated(dx, hv), Const(f), Duplicated(x, copy(v)), Const(L))
    return dx, hv
end

# The same with L active too, which also gives the tangent along v of the gradient for L.
function tri_hvp_active(mode, f, x, v, L)
    dL, ddL, dx, hv = make_zero(L), make_zero(L), zeros(length(x)), zeros(length(x))
    autodiff(
        mode, Const(tri_grad_active!), Const,
        Duplicated(dx, hv), Duplicated(dL, ddL), Const(f), Duplicated(x, copy(v)), Const(L)
    )
    return hv, parent(ddL)
end

# On Julia 1.12, `\` with a lower triangular matrix solves through ldiv!(L, B), which branches
# at run time on whether L is also upper triangular. Forward over reverse through the solve in
# that branch needs runtime activity: the reverse pass keeps the constant matrix on its tape,
# as it does for any custom rule given a constant array.
lower_backslash(f, W) =
    (f === tri_solve && W in (LowerTriangular, UnitLowerTriangular)) ||
    (f in (tri_solve_adjoint, tri_solve_transpose) && W in (UpperTriangular, UnitUpperTriangular))

@testset "Triangular solve, forward and forward over reverse ($W)" for W in
    (LowerTriangular, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular)
    L = W(trtrs_matrix(Float64, 4))
    Lfull = Matrix(L)
    x = [0.3, -0.7, 1.1, 0.4]
    v = [1.0, 0.5, -0.25, 2.0]
    @testset "$f" for (f, H) in (
            (tri_solve, -inv(Lfull * Lfull')),
            (tri_solve_adjoint, -inv(Lfull' * Lfull)),
            (tri_solve_transpose, -inv(Lfull' * Lfull)),
            (tri_solve_ldiv, -inv(Lfull * Lfull')),
            (tri_solve_3arg, -inv(Lfull * Lfull')),
        )
        @test autodiff(Forward, f, Duplicated(x, v), Const(L))[1] ≈ dot(H * x, v)
        @test autodiff(set_runtime_activity(Forward), f, Duplicated(x, v), Const(L))[1] ≈ dot(H * x, v)
        @test gradient(Forward, f, x, Const(L))[1] ≈ H * x
        @test gradient(Reverse, f, x, Const(L))[1] ≈ H * x

        gradL(y) = parent(gradient(Reverse, f, Const(y), L)[2])
        fd = central_fdm(5, 1)(t -> gradL(x + t * v), 0.0)

        dx, hv = tri_hvp(set_runtime_activity(Forward), f, x, v, L)
        @test dx ≈ H * x
        @test hv ≈ H * v
        hv, ddL = tri_hvp_active(set_runtime_activity(Forward), f, x, v, L)
        @test hv ≈ H * v
        @test ddL ≈ fd rtol = 1.0e-6

        if VERSION < v"1.11" || !lower_backslash(f, W)
            dx, hv = tri_hvp(Forward, f, x, v, L)
            @test dx ≈ H * x
            @test hv ≈ H * v
            hv, ddL = tri_hvp_active(Forward, f, x, v, L)
            @test hv ≈ H * v
            @test ddL ≈ fd rtol = 1.0e-6
        elseif VERSION >= v"1.12"
            @test_broken tri_hvp(Forward, f, x, v, L)[2] ≈ H * v
            @test_broken tri_hvp_active(Forward, f, x, v, L)[1] ≈ H * v
        end
    end
end
