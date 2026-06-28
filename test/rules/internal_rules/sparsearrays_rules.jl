using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using SparseArrays
using Test

const T_test = Float64
struct RHS_test{M}
    A1::M
    A2::M
end
(r::RHS_test)(du, u, p, t) = (mul!(du, r.A1, u, -p[1], zero(T_test)); mul!(du, r.A2, u, p[2], one(T_test)); nothing)

mutable struct Integrator_test{RU, uType, pType}
    f::RU
    u::uType
    p::pType
    t::T_test
    dt::T_test
end

function step_test!(integ::Integrator_test)
    du = zeros(T_test, 2)
    integ.f(du, integ.u, integ.p, integ.t)
    integ.u .+= integ.dt .* du
    integ.t += integ.dt
end

function solve_custom_test(integ)
    step_test!(integ)
    step_test!(integ)
    return integ.u[1] + integ.u[2]
end

const U_sparse_test = RHS_test(sparse(T_test[0.0 1.0; 0.0 0.0]), sparse(T_test[0.0 0.0; 1.0 0.0]))
function f_sparse_test(p)
    integ = Integrator_test(U_sparse_test, T_test[3.0, 4.0], p, 0.0, 0.1)
    return solve_custom_test(integ)
end

function f_with_sparse_constructor(p)
    A = sparse([1], [1], [1.0], 2, 2)
    return p[1] * sum(A)
end

@testset "Sparse construction" begin
    p = [2.0]
    dp = Enzyme.make_zero(p)
    Enzyme.autodiff(
           Enzyme.set_runtime_activity(Enzyme.Reverse),
           f_with_sparse_constructor,
           Active,
           Duplicated(p, dp),
       )[1]
    @test dp ≈ [1.0]
end

function test_sparse(M, v, α, β)
    tout = promote_type(eltype(M), eltype(v), typeof(α), typeof(β))
    if v isa AbstractVector
        C = zeros(tout, size(M, 1))
    else
        C = zeros(tout, size(M, 1), size(v, 2))
    end


    for Tret in (Duplicated, BatchDuplicated), TM in (Const, Duplicated, BatchDuplicated), Tv in (Const, Duplicated, BatchDuplicated),
            Tα in (Const, Active), Tβ in (Const, Active)

        are_activities_compatible(Tret, Tret, TM, Tv, Tα, Tβ) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, TM), (v, Tv), (α, Tα), (β, Tβ))
    end

    for Tret in (Duplicated, BatchDuplicated), TM in (Const, Duplicated, BatchDuplicated),
            Tv in (Const, Duplicated, BatchDuplicated), bα in (true, false), bβ in (true, false)
        are_activities_compatible(Tret, Tret, TM, Tv) || continue
        test_reverse(LinearAlgebra.mul!, Tret, (C, Tret), (M, Const), (v, Tv), (bα, Const), (bβ, Const))
    end

    return test_reverse(LinearAlgebra.mul!, Const, (C, Const), (M, Const), (v, Const), (α, Active), (β, Active))
end

@testset "SparseArrays spmatvec reverse rule" begin
    Ts = ComplexF64

    M0 = [
        0.0   1.50614;
        0.0  -0.988357;
        0.0   0.0
    ]


    M = SparseMatrixCSC((M0 .+ 2im * M0))
    v = rand(Ts, 2)
    α = rand(Ts)
    β = rand(Ts)

    # Purely complex
    test_sparse(M, v, α, β)

    # Purely real
    test_sparse(real(M), real(v), real(α), real(β))

    # Now test mixed. We only need to test what the variables are active
    C = zeros(ComplexF64, size(M0, 1))
    TB = (Duplicated, BatchDuplicated)
    for T in TB
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (v, T), (α, Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (real(v), T), (α, Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (real(v), T), (α, Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (v, T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (v, T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (real(v), T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (real(M), T), (real(v), T), (real(α), Active), (β, Active))
        test_reverse(LinearAlgebra.mul!, T, (C, T), (M, T), (v, T), (α, Active), (real(β), Active))
    end
end

@testset "SparseArrays spmatmat reverse rule" begin
    Ts = ComplexF64

    M0 = [
        0.0   1.50614;
        0.0  -0.988357;
        0.0   0.0
    ]


    M = SparseMatrixCSC((M0 .+ 2im * M0))
    v = rand(Ts, 2, 2)
    α = rand(Ts)
    β = rand(Ts)

    # Now all the code paths are already tested in the vector case so we just make sure that
    # general matrix multiplication works
    test_sparse(M, v, α, β)

end

@testset "Sparse mul! with aliased shadow (primal corruption)" begin
    p = [1.0, 2.0]
    dp_sparse = Enzyme.make_zero(p)
    
    # Before the fix, this would corrupt U_sparse_test.A1.nzval and produce wrong gradients.
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), f_sparse_test, Active, Duplicated(p, dp_sparse))
    
    @test dp_sparse ≈ [-0.94, 0.53]
    @test U_sparse_test.A1.nzval == [1.0]
end

