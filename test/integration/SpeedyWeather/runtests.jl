# SpeedyWeather.jl integration example
# Sensitivity Analysis of Temperature at a single grid point (one-hot seed)
# over the full integration of the PrimitiveWetModel over N timesteps
# Note: reducing N, or reducing trunc will not reduce compile time of the gradient
# we could reduce model complexity a bit by excluding some parameterizations
#
# For the test itself, we test that Enzyme doesn't error and gradients are nonzero

using SpeedyWeather, Enzyme, Checkpointing

# Parse command line argument for N (number of timesteps)
const N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5

spectral_grid = SpectralGrid(trunc = 32, nlayers = 8)          # define resolution
model = PrimitiveWetModel(; spectral_grid)                 # construct model
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

function checkpointed_timesteps!(progn::PrognosticVariables, diagn, model, N_steps, checkpoint_scheme::Scheme, lf1 = 2, lf2 = 2)

    @ad_checkpoint checkpoint_scheme for _ in 1:N_steps
        SpeedyWeather.timestep!(progn, diagn, 2 * model.time_stepping.Δt, model, lf1, lf2)
    end

    return nothing
end

checkpoint_scheme = Revolve(N)

# Temperature One-Hot
d_progn = zero(progn)
d_model = make_zero(model)
d_diag = make_zero(diagn)
seed_point = 443    # seed point
d_diag.grid.temp_grid[seed_point, 8] = 1

# Sensitivity Analysis of Temperature at a single grid point (one-hot seed)
autodiff(Enzyme.Reverse, checkpointed_timesteps!, Const, Duplicated(progn, d_progn), Duplicated(diagn, d_diag), Duplicated(model, d_model), Const(N), Const(checkpoint_scheme))

vor_grid = transform(d_progn.vor[:, :, 2], model.spectral_transform)

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
