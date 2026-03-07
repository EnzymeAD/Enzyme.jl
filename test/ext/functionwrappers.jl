using Enzyme, Test
using FunctionWrappers: FunctionWrapper

@testset "FunctionWrappers Extension" begin

    # In-place (IIP) test function: du[1] = p[1] * u[1]^2
    f!(du, u, p) = (du[1] = p[1] * u[1]^2; nothing)

    # Out-of-place (OOP) test function: returns p[1] * x^2
    f_oop(x, p) = p[1] * x^2

    @testset "IIP Forward Mode" begin
        fw = FunctionWrapper{Nothing,Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}(f!)

        u = [2.0]; du = zeros(1); p = [3.0]
        ddu = zeros(1); du_u = [1.0]

        # Differentiate through FunctionWrapper
        Enzyme.autodiff(Forward, fw, Const{Nothing},
            Duplicated(du, ddu), Duplicated(u, du_u), Const(p))

        # Compare with raw function
        u2 = [2.0]; du2 = zeros(1); ddu2 = zeros(1); du_u2 = [1.0]
        Enzyme.autodiff(Forward, f!, Const{Nothing},
            Duplicated(du2, ddu2), Duplicated(u2, du_u2), Const(p))

        @test ddu ≈ ddu2
        # ddu[1] should be d/du(p*u^2) * du_u = 3.0 * 2 * 2.0 * 1.0 = 12.0
        @test ddu[1] ≈ 12.0
    end

    @testset "IIP Reverse Mode" begin
        fw = FunctionWrapper{Nothing,Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}(f!)

        u = [2.0]; du = zeros(1); p = [3.0]
        ddu = [1.0]; du_u = zeros(1)

        Enzyme.autodiff(Reverse, fw, Const{Nothing},
            Duplicated(du, ddu), Duplicated(u, du_u), Const(p))

        # Compare with raw function
        u2 = [2.0]; du2 = zeros(1); ddu2 = [1.0]; du_u2 = zeros(1)
        Enzyme.autodiff(Reverse, f!, Const{Nothing},
            Duplicated(du2, ddu2), Duplicated(u2, du_u2), Const(p))

        @test du_u ≈ du_u2
        # du/du[1] of (du[1] = p[1]*u[1]^2) with seed ddu[1]=1.0:
        # = p[1] * 2 * u[1] = 3.0 * 2 * 2.0 = 12.0
        @test du_u[1] ≈ 12.0
    end

    @testset "OOP Forward Mode" begin
        fw_oop = FunctionWrapper{Float64,Tuple{Float64,Vector{Float64}}}(f_oop)

        x = 3.0; p = [2.0]
        dx = 1.0

        res = Enzyme.autodiff(Forward, fw_oop, Duplicated,
            Duplicated(x, dx), Const(p))

        # Compare with raw function
        res2 = Enzyme.autodiff(Forward, f_oop, Duplicated,
            Duplicated(x, dx), Const(p))

        @test res[1] ≈ res2[1]
        # d/dx(p*x^2) = 2*p*x = 2*2.0*3.0 = 12.0
        @test res[1] ≈ 12.0
    end

    @testset "OOP Reverse Mode" begin
        fw_oop = FunctionWrapper{Float64,Tuple{Float64,Vector{Float64}}}(f_oop)

        x = 3.0; p = [2.0]

        res = Enzyme.autodiff(Reverse, fw_oop, Active,
            Active(x), Const(p))

        # Compare with raw function
        res2 = Enzyme.autodiff(Reverse, f_oop, Active,
            Active(x), Const(p))

        @test res[1][1] ≈ res2[1][1]
        # d/dx(p*x^2) = 2*p*x = 2*2.0*3.0 = 12.0
        @test res[1][1] ≈ 12.0
    end

end
