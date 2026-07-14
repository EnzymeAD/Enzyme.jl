# SpeedyWeather.jl integration example
# Sensitivity Analysis of a single time step of the PrimitiveWetModel
#
# For the test itself, we test that Enzyme doesn't error and gradients are nonzero and make some physical sense

using SpeedyWeather, Enzyme, Test

spectral_grid = SpectralGrid(trunc = 32, nlayers = 8)             # define resolution
model = PrimitiveWetModel(; spectral_grid, physics=false)         # construct model
# physics = false to accelate the test
simulation = initialize!(model)
initialize!(simulation)
run!(simulation, period = Day(20))

(; prognostic_variables, diagnostic_variables, model) = simulation
(; Δt, Δt_millisec) = model.time_stepping
dt = 2Δt

progn = prognostic_variables
diagn = diagnostic_variables

# do the scaling again because we need it for the timestepping when calling it manually
SpeedyWeather.scale!(progn, diagn, model.planet.radius)

dprogn = zero(progn)
ddiag = make_zero(diagn)
dmodel = make_zero(model)

# Temperature One-Hot
seed_point = 443    # seed point
ddiag.grid.temp_grid[seed_point, 8] = 1

# Sensitivity Analysis of Temperature at a single grid point (one-hot seed) for a single timestep
autodiff(Enzyme.Reverse, SpeedyWeather.timestep!, Const, Duplicated(progn, dprogn), Duplicated(diagn, ddiag), Const(dt), Duplicated(model, dmodel))

vor_grid = transform(dprogn.vor[:, :, 2], model.spectral_transform)

# nonzero
@test sum(abs, vor_grid) > 0

# localized around the seed point
# sensitivty has to be high around the seed point and low far away
# both, in vertical and horiztonal direction
@test abs(vor_grid[seed_point, 8]) > abs(vor_grid[seed_point - 2, 8])
@test abs(vor_grid[seed_point, 8]) > abs(vor_grid[seed_point + 2, 8])
@test abs(vor_grid[seed_point, 8]) > abs(vor_grid[seed_point - 200, 8])
@test abs(vor_grid[seed_point, 8]) > abs(vor_grid[seed_point + 200, 8])
@test abs(vor_grid[seed_point, 8]) > abs(vor_grid[seed_point, 4])
@test abs(vor_grid[seed_point, 8]) > abs(vor_grid[seed_point, 1])
