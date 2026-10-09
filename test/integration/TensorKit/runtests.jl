using Test, TensorKit, Random, Enzyme, EnzymeTestUtils, TupleTools
using TensorKit.VectorInterface: One, Zero

Enzyme.Compiler.RunAttributor[] = false

default_tol(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
default_tol(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-5

function randindextuple(N::Int, k::Int = rand(0:N))
    @assert 0 ≤ k ≤ N
    _p = randperm(N)
    return (tuple(_p[1:k]...), tuple(_p[(k + 1):end]...))
end

function randcircshift(N₁::Int, N₂::Int, k::Int = rand(0:(N₁ + N₂)))
    N = N₁ + N₂
    @assert 0 ≤ k ≤ N
    p = TupleTools.vcat(ntuple(identity, N₁), reverse(ntuple(identity, N₂) .+ N₁))
    n = rand(0:N)
    _p = TupleTools.circshift(p, n)
    return (tuple(_p[1:k]...), reverse(tuple(_p[(k + 1):end]...)))
end

Vtr = (ℂ^2, (ℂ^3)', ℂ^4, ℂ^3, (ℂ^2)')
VRepU₁ = (
    Vect[U1Irrep](0 => 2, 1 => 2, -1 => 2),
    Vect[U1Irrep](0 => 1, 1 => 1, -1 => 1)',
    Vect[U1Irrep](0 => 3, 1 => 1, -1 => 1),
    Vect[U1Irrep](0 => 1, 1 => 2, -1 => 1)',
    Vect[U1Irrep](0 => 1, 1 => 1, -1 => 2),
)
I_Hubbard = FermionParity ⊠ SU2Irrep ⊠ U1Irrep
VfHubbard = (
    Vect[I_Hubbard]((0, 0, 0) => 3, (1, 1 // 2, -1) => 1),
    Vect[I_Hubbard]((0, 0, 0) => 1, (0, 0, -2) => 1, (0, 1, 0) => 1)',
    Vect[I_Hubbard]((1, 1 // 2, -1) => 1, (1, 1 // 2, +1) => 1, (0, 1, -2) => 1, (0, 1, +2) => 1),
    Vect[I_Hubbard]((0, 0, 0) => 2, (1, 1 // 2, +1) => 1, (1, 1 // 2, -1) => 1)',
    Vect[I_Hubbard]((0, 0, 0) => 1, (1, 1 // 2, +1) => 1, (1, 3 // 2, -1) => 1),
)
I_A4Z4 = A4Irrep ⊠ Z4Element{2}
VRepA4Twistedℤ₄ = (
    Vect[I_A4Z4]((0, 0) => 2, (1, 1) => 1, (2, 3) => 0, (3, 2) => 1),
    Vect[I_A4Z4]((0, 0) => 1, (1, 1) => 1, (2, 3) => 1, (3, 2) => 0)',
    Vect[I_A4Z4]((0, 0) => 2, (1, 1) => 1, (2, 3) => 1, (3, 2) => 1),
    Vect[I_A4Z4]((0, 0) => 0, (1, 1) => 1, (2, 3) => 1, (3, 2) => 1)',
    Vect[I_A4Z4]((0, 0) => 0, (1, 1) => 2, (2, 3) => 0, (3, 2) => 1),
)
ad_spacelist = (Vtr, VRepU₁, VfHubbard, VRepA4Twistedℤ₄)

eltypes = (Float64, ComplexF64)

Tαs = (Active, Const)
Tβs = (Active, Const)

@testset "Index Manipulations" begin
    @testset "$(TensorKit.type_repr(sectortype(eltype(V)))) ($T) TA ($TA)" for V in ad_spacelist, T in eltypes, TA in (Duplicated,)
        atol = default_tol(T)
        rtol = default_tol(T)
        A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
        has_braiding = BraidingStyle(sectortype(eltype(V))) isa HasBraiding
        symmetricbraiding = BraidingStyle(sectortype(eltype(V))) isa SymmetricBraiding
        α = randn(T)
        β = randn(T)
        #=@testset "flip and twist" begin # turn off for now until Enzyme can get compilation times down
            if has_braiding
                if !(T <: Real && !(sectorscalartype(sectortype(A)) <: Real))
                    EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
                    EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
                    EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), (1, Const); atol, rtol)
                    EnzymeTestUtils.test_reverse(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol)
                    EnzymeTestUtils.test_forward(twist!, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
                    EnzymeTestUtils.test_forward(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
                    EnzymeTestUtils.test_forward(twist!, TA, (A, TA), (1, Const); atol, rtol)
                    EnzymeTestUtils.test_forward(twist!, TA, (A, TA), ([1, 3], Const); atol, rtol)
                end
                EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
                EnzymeTestUtils.test_reverse(flip, TA, (A, TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
                EnzymeTestUtils.test_reverse(flip, TA, (A, TA), (1, Const); atol, rtol)
                EnzymeTestUtils.test_reverse(flip, TA, (A, TA), ([1, 3], Const); atol, rtol)

                EnzymeTestUtils.test_forward(flip, TA, (A, TA), (1, Const); atol, rtol, fkwargs = (inv = false,))
                EnzymeTestUtils.test_forward(flip, TA, (A, TA), ([1, 3], Const); atol, rtol, fkwargs = (inv = true,))
                EnzymeTestUtils.test_forward(flip, TA, (A, TA), (1, Const); atol, rtol)
                EnzymeTestUtils.test_forward(flip, TA, (A, TA), ([1, 3], Const); atol, rtol)
            end
        end
        @testset "transpose" begin
            # repeat a couple times to get some distribution of arrows
            p = randcircshift(numout(A), numin(A))
            C = randn!(transpose(A, p))
            EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (One(), Const), (Zero(), Const); atol, rtol)
            @testset for Tα in Tαs, Tβ in Tβs
                EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                if !(T <: Real)
                    EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (A, Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                    EnzymeTestUtils.test_reverse(TensorKit.transpose!, Duplicated, (copy(C), Duplicated), (real(A), Duplicated), (p, Const), (real(α), Tα), (β, Tβ); atol, rtol)
                end
            end
        end=#
        symmetricbraiding && @testset "permute" begin
            A = randn(T, V[1] ⊗ V[2] ← (V[3] ⊗ V[4] ⊗ V[5])')
            p = randindextuple(numind(A))
            C = randn!(permute(A, p))
            @testset for Tα in Tαs, Tβ in Tβs
                EnzymeTestUtils.test_reverse(TensorKit.permute!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (α, Tα), (β, Tβ); atol, rtol)
            end
        end
    end
end
