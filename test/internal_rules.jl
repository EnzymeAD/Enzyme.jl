module InternalRules

using Enzyme
using Enzyme.EnzymeRules
using EnzymeTestUtils
using FiniteDifferences
using LinearAlgebra
using SparseArrays
using Test
import Random

struct TPair
    a::Float64
    b::Float64
end

function sorterrfn(t, x)
    function lt(a, b)
        return a.a < b.a
    end
    return first(sortperm(t, lt=lt)) * x
end

@testset "Sort rules" begin
    function f1(x)
        a = [1.0, 3.0, x]
        sort!(a)
        return a[2]
    end

    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1] == 1
    @test autodiff(Forward, f1, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1"=1.0, var"2"=2.0)
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 1
    @test autodiff(Forward, f1, Duplicated(4.0, 1.0))[1] == 0
    @test autodiff(Forward, f1, BatchDuplicated(4.0, (1.0, 2.0)))[1] == (var"1"=0.0, var"2"=0.0)
    @test autodiff(Reverse, f1, Active, Active(4.0))[1][1] == 0

    function f2(x)
        a = [1.0, -3.0, -x, -2x, x]
        sort!(a; rev=true, lt=(x, y) -> abs(x) < abs(y) || (abs(x) == abs(y) && x < y))
        return sum(a .* [1, 2, 3, 4, 5])
    end

    @test autodiff(Forward, f2, Duplicated(2.0, 1.0))[1] == -3
    @test autodiff(Forward, f2, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1"=-3.0, var"2"=-6.0)
    @test autodiff(Reverse, f2, Active, Active(2.0))[1][1] == -3

    function f3(x)
        a = [2.0, 2.5, x, 1.0]
        return partialsort(a, 2)
    end

    @test autodiff(Forward, f3, Duplicated(1.5, 1.0))[1] == 1.0
    @test autodiff(Forward, f3, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1"=1.0, var"2"=2.0)
    @test autodiff(Reverse, f3, Active(1.5))[1][1] == 1.0
    @test autodiff(Reverse, f3, Active(2.5))[1][1] == 0.0

    function f4(x)
        a = [2.0, 2.5, x, x / 2]
        y = partialsort(a, 1:2)
        return sum(y)
    end

    @test autodiff(Forward, f4, Duplicated(1.5, 1.0))[1] == 1.5
    @static if VERSION < v"1.7-" || VERSION >= v"1.8-"
        @test autodiff(Forward, f4, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1"=1.5, var"2"=3.0)
    end
    @test autodiff(Reverse, f4, Active(1.5))[1][1] == 1.5
    @test autodiff(Reverse, f4, Active(4.0))[1][1] == 0.5
    @test autodiff(Reverse, f4, Active(6.0))[1][1] == 0.0

    dd = Duplicated([TPair(1, 2), TPair(2, 3), TPair(0, 1)], [TPair(0, 0), TPair(0, 0), TPair(0, 0)])
    res = Enzyme.autodiff(Reverse, sorterrfn, dd, Active(1.0))

    @test res[1][2] ≈ 3
    @test dd.dval[1].a ≈ 0
    @test dd.dval[1].b ≈ 0
    @test dd.dval[2].a ≈ 0
    @test dd.dval[2].b ≈ 0
    @test dd.dval[3].a ≈ 0
    @test dd.dval[3].b ≈ 0
end

@testset "Linear Solve" begin
    A = Float64[2 3; 5 7]
    dA = zero(A)
    b = Float64[11, 13]
    db = zero(b)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(\)}, Duplicated, Duplicated{typeof(A)}, Duplicated{typeof(b)})

    tape, primal, shadow = forward(Const(\), Duplicated(A, dA), Duplicated(b, db))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Duplicated(A, dA), Duplicated(b, db), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test dA ≈ (-z * transpose(y))
    @test db ≈ z

    db = zero(b)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(\)}, Duplicated, Const{typeof(A)}, Duplicated{typeof(b)})

    tape, primal, shadow = forward(Const(\), Const(A), Duplicated(b, db))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Const(A), Duplicated(b, db), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test db ≈ z

    dA = zero(A)

    forward, pullback = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(\)}, Duplicated, Duplicated{typeof(A)}, Const{typeof(b)})

    tape, primal, shadow = forward(Const(\), Duplicated(A, dA), Const(b))

    dy = Float64[17, 19]
    copyto!(shadow, dy)

    pullback(Const(\), Duplicated(A, dA), Const(b), tape)

    z = transpose(A) \ dy

    y = A \ b
    @test dA ≈ (-z * transpose(y))
end

function tr_solv(A, B, uplo, trans, diag, idx)
  B = copy(B)
  LAPACK.trtrs!(uplo, trans, diag, A, B)
  return @inbounds B[idx]
end


@testset "Reverse triangular solve" begin
	A = [0.7550523937508613 0.7979976952197996 0.29318222271218364; 0.4416768066117529 0.4335305304334933 0.8895389673238051; 0.07752980210005678 0.05978245503334367 0.4504482683752542]
	B = [0.10527381151977078 0.5450388247476627 0.3179106723232359 0.43919576779182357 0.20974326586875847; 0.7551160501548224 0.049772782182839426 0.09284926395551141 0.07862188927391855 0.17346407477062986; 0.6258040138863172 0.5928022963567454 0.24251650865340169 0.6626410383247967 0.32752198021506784]
    for idx in 1:15
    for uplo in ('L', 'U')
    for diag in ('N', 'U')
    for trans in ('N', 'T')
        dA = zero(A)
        dB = zero(B)	
        Enzyme.autodiff(Reverse, tr_solv, Duplicated(A, dA), Duplicated(B, dB), Const(uplo),Const(trans), Const(diag), Const(idx))
        fA = FiniteDifferences.grad(central_fdm(5, 1), A->tr_solv(A, B, uplo, trans, diag, idx), A)[1]
        fB = FiniteDifferences.grad(central_fdm(5, 1), B->tr_solv(A, B, uplo, trans, diag, idx), B)[1]

		if max(abs.(dA)...) >= 1e-10 || max(abs.(fA)...) >= 1e-10
			@test dA ≈ fA
		end
		if max(abs.(dB)...) >= 1e-10 || max(abs.(fB)...) >= 1e-10
			@test dB ≈ fB
		end
    end
    end
    end
    end
end

function chol_lower0(x)
  c = copy(x)
  C, info = LinearAlgebra.LAPACK.potrf!('L', c)
  return c[2,1]
end

function chol_upper0(x)
  c = copy(x)
  C, info = LinearAlgebra.LAPACK.potrf!('U', c)
  return c[1,2]
end

@testset "Cholesky PotRF" begin
    x = reshape([1.0, -0.10541615131279458, 0.6219810761363638, 0.293343219811946, -0.10541615131279458, 1.0, -0.05258941747718969, 0.34629296878264443, 0.6219810761363638, -0.05258941747718969, 1.0, 0.4692436399208845, 0.293343219811946, 0.34629296878264443, 0.4692436399208845, 1.0], 4, 4)
     dL = zero(x)
     dL[2, 1] = 1.0
 
     @test Enzyme.gradient(Reverse, chol_lower0, x) ≈  [0.05270807565639164 0.0 0.0 0.0; 1.0000000000000024 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0] 
     
     @test reshape(collect(Enzyme.gradient(Forward, chol_lower0, x)), 4, 4) ≈  [0.05270807565639164 0.0 0.0 0.0; 1.0000000000000024 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0] 

     @test FiniteDifferences.grad(central_fdm(5, 1), chol_lower0, x)[1] ≈ [0.05270807565639164 0.0 0.0 0.0; 1.0000000000000024 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
     
     @test reshape(collect(Enzyme.gradient(Forward, chol_upper0, x)), 4, 4) ≈ [0.05270807565639728 0.9999999999999999 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
     @test Enzyme.gradient(Reverse, chol_upper0, x) ≈ [0.05270807565639728 0.9999999999999999 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
     @test FiniteDifferences.grad(central_fdm(5, 1), chol_upper0, x)[1] ≈ [0.05270807565639728 0.9999999999999999 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
end


function tchol_lower(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('L', c)
    return c[row, col]
end
function tchol_upper(x, row, col)
    c = copy(x)
    C, info = LinearAlgebra.LAPACK.potrf!('U', c)
    return c[row, col]
end

@testset "Cholesky PotRF 3x3" begin

    x = [1.0 0.13147601759884564 0.5282944836504488; 0.13147601759884564 1.0 0.18506733179093515; 0.5282944836504488 0.18506733179093515 1.0]
    for i in 1:size(x, 1)
        for j in 1:size(x, 2)
             reverse_grad  = Enzyme.gradient(Reverse, x -> tchol_lower(x, i, j), x)
             forward_grad  = reshape(collect(Enzyme.gradient(Forward, x -> tchol_lower(x, i, j), x)), size(x))
             finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tchol_lower(x, i, j), x)[1]
             @test reverse_grad  ≈ finite_diff 
             @test forward_grad  ≈ finite_diff 
             
             reverse_grad  = Enzyme.gradient(Reverse, x -> tchol_upper(x, i, j), x)
             forward_grad  = reshape(collect(Enzyme.gradient(Forward, x -> tchol_upper(x, i, j), x)), size(x))
             finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tchol_upper(x, i, j), x)[1]
             @test reverse_grad  ≈ finite_diff 
             @test forward_grad  ≈ finite_diff
        end
    end
end

function tcholsolv_lower(A, B, i)
    c = copy(B)
    C, info = LinearAlgebra.LAPACK.potrs!('L', A, c)
    return c[i]
end
function tcholsolv_upper(A, B, i)
    c = copy(B)
    C, info = LinearAlgebra.LAPACK.potrs!('U', A, c)
    return c[i]
end

@testset "Cholesky PotRS 3x5" begin

    x = [1.0 0.13147601759884564 0.5282944836504488; 0.13147601759884564 1.0 0.18506733179093515; 0.5282944836504488 0.18506733179093515 1.0]
    for i in 1:15
         B = [3.1 2.7 5.9 2.4 1.6; 7.9 8.2 1.3 9.4 5.5; 4.7 2.9 9.8 7.1 4.3]
         reverse_grad  = Enzyme.gradient(Reverse, B -> tcholsolv_lower(x, B, i), B)
         # forward_grad  = reshape(collect(Enzyme.gradient(Forward, B -> tcholsolv_lower(x, B, i), B)), size(B))
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), B -> tcholsolv_lower(x, B, i), B)[1]
         @test reverse_grad  ≈ finite_diff 
         # @test forward_grad  ≈ finite_diff 
         
         reverse_grad  = Enzyme.gradient(Reverse, B -> tcholsolv_upper(x, B, i), B)
         # forward_grad  = reshape(collect(Enzyme.gradient(Forward, B -> tcholsolv_upper(x, B, i), B)), size(B))
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), B -> tcholsolv_upper(x, B, i), B)[1]
         @test reverse_grad  ≈ finite_diff 
         # @test forward_grad  ≈ finite_diff

         reverse_grad  = Enzyme.gradient(Reverse, x -> tcholsolv_lower(x, B, i), x)
         #forward_grad  = reshape(collect(Enzyme.gradient(Forward, x -> tcholsolv_lower(x, B, i), x)), size(x))
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tcholsolv_lower(x, B, i), x)[1]
         @test reverse_grad  ≈ finite_diff 
         #@test forward_grad  ≈ finite_diff 
         # 
         reverse_grad  = Enzyme.gradient(Reverse, x -> tcholsolv_upper(x, B, i), x)
         #forward_grad  = reshape(collect(Enzyme.gradient(Forward, x -> tcholsolv_upper(x, B, i), x)), size(x))
         finite_diff = FiniteDifferences.grad(central_fdm(5, 1), x -> tcholsolv_upper(x, B, i), x)[1]
         @test reverse_grad  ≈ finite_diff 
         #@test forward_grad  ≈ finite_diff
    end
end

@static if VERSION > v"1.8"
    @testset "cholesky" begin
        activities = (Const, Duplicated, BatchDuplicated)
        function _square(A)
            S = A * adjoint(A)
            S[diagind(S)] .= real.(S[diagind(S)]) # workaround for issue #1456:
            return S
        end
        @testset for (Te, TSs) in (
            Float64 => (Symmetric, Hermitian),
            ComplexF64 => (Hermitian,),
        ), TA in activities, Tret in activities
            @testset "without wrapper arguments" begin
                A = rand(Te, 5, 5)
                are_activities_compatible(Tret, TA) || continue
                test_forward(cholesky ∘ _square, Tret, (A, TA))
                test_reverse(cholesky ∘ _square, Tret, (A, TA))
            end
            @testset "with wrapper arguments" for TS in TSs, uplo in (:U, :L)
                _A = collect(exp(TS(I + rand(Te, 5, 5))))
                A = TS(_A, uplo)
                are_activities_compatible(Tret, TA) || continue
                test_forward(cholesky, Tret, (A, TA); fdm=FiniteDifferences.forward_fdm(5, 1))
                test_reverse(cholesky, Tret, (A, TA))
            end
        end
    end

    @testset "Linear solve for `Cholesky`" begin
        activities = (Const, Duplicated, DuplicatedNoNeed, BatchDuplicated,
                      BatchDuplicatedNoNeed)
        @testset for Te in (Float64, ComplexF64), uplo in ('L', 'U')
            C = Cholesky(I + rand(Te, 5, 5), uplo, 0) # add `I` for numerical stability
            B = rand(Te, 5, 5)
            b = rand(Te, 5)
            @testset for TC in activities,
                         TB in activities,
                         Tret in (Const, Duplicated, BatchDuplicated)

                @testset "$(size(_B))" for _B in (B, b)
                    are_activities_compatible(Tret, TC, TB) || continue
                    # Non-uniform activities are disabled due to unresolved questions
                    # see https://github.com/EnzymeAD/Enzyme.jl/issues/1411
                    Tret == TC == TB || continue
                    test_forward(\, Tret, (C, TC), (_B, TB))
                    test_reverse(\, Tret, (C, TC), (_B, TB))
                end
            end
            @testset for TC in activities,
                         TB in activities,
                         Tret in (Const, Duplicated, BatchDuplicated)

                @testset "$(size(_B))" for _B in (B, b)
                    are_activities_compatible(Tret, TC, TB) || continue
                    # Non-uniform activities are disabled due to unresolved questions
                    # see https://github.com/EnzymeAD/Enzyme.jl/issues/1411
                    Tret == TC == TB || continue
                    test_forward(ldiv!, Tret, (C, TC), (_B, TB))
                    test_reverse(ldiv!, Tret, (C, TC), (_B, TB))
                end
            end
        end
    end

@testset "Linear solve for triangular matrices" begin
    @testset for T in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular),
        TE in (Float64, ComplexF64), sizeB in ((3,), (3, 3))
        n = sizeB[1]
        M = rand(TE, n, n)
        B = rand(TE, sizeB...)
        Y = zeros(TE, sizeB...)
        A = T(M)
        @testset "test through constructor" begin
            _A = T(A)
            f!(Y, A, B, ::T) where {T} = ldiv!(Y, T(A), B)
            for TY in (Const, Duplicated, BatchDuplicated),
                TM in (Const, Duplicated, BatchDuplicated),
                TB in (Const, Duplicated, BatchDuplicated)
                are_activities_compatible(Const, TY, TM, TB) || continue
                test_reverse(f!, TY, (Y, TY), (M, TM), (B, TB), (_A, Const))
            end
        end
        @testset "test through `Adjoint` wrapper (regression test for #1306)" begin
            # Test that we get the same derivative for `M` as for the adjoint of its
            # (materialized) transpose. It's the same matrix, but represented differently
            function f!(Y, A, B)
                ldiv!(Y, A, B)
                return nothing
            end
            A1 = T(M)
            A2 = T(conj(permutedims(M))')
            dA1 = make_zero(A1)
            dA2 = make_zero(A2)
            dB1 = make_zero(B)
            dB2 = make_zero(B)
            dY1 = rand(TE, sizeB...)
            dY2 = copy(dY1)
            autodiff(Reverse, f!, Duplicated(Y, dY1), Duplicated(A1, dA1), Duplicated(B, dB1))
            autodiff(Reverse, f!, Duplicated(Y, dY2), Duplicated(A2, dA2), Duplicated(B, dB2))
            @test dA1.data ≈ dA2.data
            @test dB1 ≈ dB2
        end
    end
end
end

@testset "rand and randn rules" begin
    # Distributed as x + unit normal + uniform
    struct MyDistribution
        x::Float64
    end

    Random.rand(rng::Random.AbstractRNG, d::MyDistribution) = d.x + randn() + rand()
    Random.rand(d::MyDistribution) = rand(Random.default_rng(), d)

    # Outer rand should be differentiated through, and inner rand and randn should be ignored.
    @test autodiff(Enzyme.Reverse, x -> rand(MyDistribution(x)), Active, Active(1.0)) == ((1.0,),)
end

end # InternalRules
