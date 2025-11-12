using Enzyme
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

function kinetic_energy(model, ν, K, dt=1)
    vitd = VerticallyImplicitTimeDiscretization()
    new_closure = ScalarDiffusivity(vitd; ν)
    tracer_names = keys(model.tracers)
    new_closure = Oceananigans.TurbulenceClosures.with_tracers(tracer_names, new_closure)
    model.closure = new_closure

    for n = 1:10
        time_step!(model, dt)
    end

    compute!(K)

    return @allowscalar first(K)
end

@testset "Column model with ScalarDiffusivity" begin
    ν = 1e-3
    grid = RectilinearGrid(size=128, z=(-64, 64), topology=(Flat, Flat, Bounded))
    vitd = VerticallyImplicitTimeDiscretization()
    closure = ScalarDiffusivity(vitd; ν)
    m = HydrostaticFreeSurfaceModel(; grid, closure, coriolis=FPlane(f=1e-4),
        tracers=:b, buoyancy=BuoyancyTracer())

    N² = 1e-6
    bᵢ(z) = N² * z
    uᵢ(z) = exp(-z^2 / 10)
    set!(m, b=bᵢ, u=uᵢ)

    u, v, w = m.velocities
    ke = (u^2 + v^2) / 2
    K = Field(Integral(ke))

    dm = Enzyme.make_zero(m)
    dK = Enzyme.make_zero(K)

    dKdν = autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    kinetic_energy, Active,
                    Duplicated(m, dm),
                    Active(ν),
                    Duplicated(K, dK))

    @test dKdν[1][1] != 0
end