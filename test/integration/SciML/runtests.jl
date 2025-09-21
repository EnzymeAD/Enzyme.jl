using Enzyme, OrdinaryDiffEqTsit5, StaticArrays, DiffEqBase, ForwardDiff, Test
using OrdinaryDiffEq, SciMLSensitivity, Zygote
using LinearSolve, LinearAlgebra

@testset "Direct Differentiation of Explicit ODE Solve" begin
    function lorenz!(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
    end

    _saveat =  SA[0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]

    function f_dt(y::Array{Float64}, u0::Array{Float64})
        tspan = (0.0, 3.0)
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(lorenz!, u0, tspan)
        sol = DiffEqBase.solve(prob, Tsit5(), saveat = _saveat, sensealg = DiffEqBase.SensitivityADPassThrough(), abstol=1e-12, reltol=1e-12)
        y .= sol[1,:]
        return nothing
    end;

    function f_dt(u0)
        tspan = (0.0, 3.0)
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(lorenz!, u0, tspan)
        sol = DiffEqBase.solve(prob, Tsit5(), saveat = _saveat, sensealg = DiffEqBase.SensitivityADPassThrough(), abstol=1e-12, reltol=1e-12)
        sol[1,:]
    end;

    u0 = [1.0; 0.0; 0.0]
    fdj = ForwardDiff.jacobian(f_dt, u0)

    ezj = stack(map(1:3) do i
        d_u0 = zeros(3)
        dy = zeros(13)
        y  = zeros(13)
        d_u0[i] = 1.0
        Enzyme.autodiff(Forward, f_dt,  Duplicated(y, dy), Duplicated(u0, d_u0));
        dy
    end)

    @test ezj ≈ fdj

    function f_dt2(u0)
        tspan = (0.0, 3.0)
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(lorenz!, u0, tspan)
        sol = DiffEqBase.solve(prob, Tsit5(), dt=0.1, saveat = _saveat, sensealg = DiffEqBase.SensitivityADPassThrough(), abstol=1e-12, reltol=1e-12)
        sum(sol[1,:])
    end

    fdg = ForwardDiff.gradient(f_dt2, u0)
    d_u0 = zeros(3)
    Enzyme.autodiff(Reverse, f_dt2,  Active, Duplicated(u0, d_u0));

    @test d_u0 ≈ fdg
end

@testset "SciMLSensitivity Adjoint Interface" begin
    Enzyme.API.typeWarning!(false)

    odef(du, u, p, t) = du .= u .* p
    prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

    struct senseloss0{T}
        sense::T
    end
    function (f::senseloss0)(u0p)
        prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
        sum(solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.1))
    end
    u0p = [2.0, 3.0]
    du0p = zeros(2)
    dup = Zygote.gradient(senseloss0(InterpolatingAdjoint()), u0p)[1]
    Enzyme.autodiff(Reverse, senseloss0(InterpolatingAdjoint()), Active, Duplicated(u0p, du0p))
    @test du0p ≈ dup
end

@testset "LinearSolve Adjoints" begin
    n = 4
    A = rand(n, n);
    dA = zeros(n, n);
    b1 = rand(n);
    db1 = zeros(n);

    function f(A, b1; alg = LUFactorization())
        prob = LinearProblem(A, b1)

        sol1 = solve(prob, alg)

        s1 = sol1.u
        norm(s1)
    end

    f(A, b1) # Uses BLAS

    Enzyme.autodiff(Reverse, f, Duplicated(copy(A), dA), Duplicated(copy(b1), db1))
    dA2 = ForwardDiff.gradient(x -> f(x, eltype(x).(b1)), copy(A))
    db12 = ForwardDiff.gradient(x -> f(eltype(x).(A), x), copy(b1))

    @test dA ≈ dA2
    @test db1 ≈ db12
end
