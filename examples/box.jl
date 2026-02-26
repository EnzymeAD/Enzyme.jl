# # Enzyme for adjoint tutorial: Stommel three-box ocean model

# The goal of this tutorial is to teach about a specific usage of Enzyme's automatic
# differentiation capabilities, and will be centered around the Stommel ocean model. This
# is a nice example to see how powerful Enzyme is, and the ability of it to take a
# derivative of a complicated function (namely one that has many parts and parameters).
# This tutorial will focus first on the computations and getting Enzyme running,
# for those interested a mathematical explanation of the model and what an adjoint
# variable is will be provided at the end.

# # Brief model overview

# The Stommel box model can be viewed as a watered down full ocean model. In our example, we have three
# boxes (Box One, Box Two, and Box Three) and we model the transport of fluid between
# them. The full equations of our system are given by:
#
# ```math
# \begin{aligned}
#    U &= u_0 \left\{ \rho_2 - \left[ \rho_1 + (1 - \delta) \rho_3 \right] \right\} \\
#    \rho_i &= -\alpha T_i + \beta S_i, \; \; \; \; i = 1, 2, 3
# \end{aligned}
# ```
#
# for the **transport** U and **densities** ``\rho``, and then the time derivatives
#
# ```math
# \begin{aligned}
#    \dot{T_1} &= U(T_3 - T_1)/V_1 + \gamma (T_1^* - T_1 ) & \dot{S_1} &= U(S_3 - S_1)/V_1 + FW_1/V_1 \\
#    \dot{T_2} &= U(T_1 - T_2)/V_2 + \gamma (T_2^* - T_2 ) & \dot{S_2} &= U(S_1 - S_2)/V_2 + FW_2/V_2 \\
#    \dot{T_3} &= U(T_2 - T_3)/V_3 & \dot{S_3} &= U(S_2 - S_3)/V_3
# \end{aligned}
# ```
#
# for positive transport, ``U > 0``, and
#
# ```math
# \begin{aligned}
#    \dot{T_1} &= U(T_2 - T_1)/V_1 + \gamma (T_1^* - T_1) & \dot{S_1} &= U(S_2 - S_1)/V_1 + FW_1/V_1 \\
#    \dot{T_2} &= U(T_3 - T_2)/V_2 + \gamma (T_2^* - T_2 ) & \dot{S_2} &= U(S_3 - S_2)/V_2 + FW_2/V_2 \\
#    \dot{T_3} &= U(T_1 - T_3)/V_3 & \dot{S_3} &= U(S_1 - S_3)/V_3
# \end{aligned}
# ```
#
# for ``U \leq 0``.
# The only force driving our system is a density gradient generated via temperature
# and salinity differences between the boxes. This makes it a really easy
# model to play around with! With this in mind, the model is run
# forward with the steps:
#
# 1) Compute densities
# 2) Compute transport
# 3) Compute time derivatives of the box temperatures and salinities
# 4) Update the state vector
#
# We'll start by going through the model setup step by step, then providing a few test
# cases with Enzyme.

# # Model setup

# ## Model dependencies

# Let's first add the necessary packages to run everything

using Enzyme

# ## Initialize constants

# The system equations have quite a few constants that appear, here we initialize them
# for later use. We'll do this in a Julia way: we have an empty structure that will hold
# all the parameters, and a function (we'll call this `setup`) that initializes them.
# This means that, so long as we don't need to change parameters, we only need to run `setup`
# once.

struct ModelParameters

    ## handy to have constants
    day::Float64
    year::Float64

    ## Information related to the boxes
    boxlength::Vector{Float64}      ## Vector with north-south size of each box  [cm]
    boxdepth::Vector{Float64}       ## "          " the depth of each box  [cm]
    boxwidth::Float64               ## "          " the width of each box  [cm]
    boxarea::Vector{Float64}        ## "          " the area of each box   [cm^2]
    boxvol::Vector{Float64}         ## "          " the volume of each box   [cm^3]

    delta::Float64                  ## Constant ratio depth(box1) / (depth(box1) + depth(box3))

    ## Parameters that appear in the box model equations
    u0::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64

    ## Coefficient for the Robert filter smoother
    rf_coeff::Float64

    ## Freshwater forcing
    FW::Vector{Float64}

    ## Restoring atmospheric temperatures and salinities
    Tstar::Vector{Float64}
    Sstar::Vector{Float64}

end

#-

function setup()

    blength = [5000.0e5; 1000.0e5; 5000.0e5]
    bdepth = [1.0e5; 5.0e5; 4.0e5]

    delta = bdepth[1] / (bdepth[1] + bdepth[3])

    bwidth = 4000.0 * 1.0e5  ## box width, centimeters

    ## box areas
    barea = blength .* bwidth

    ## box volumes
    bvolume = barea .* bdepth

    ## parameters that are used to ensure units are in CGS (cent-gram-sec)

    day = 3600.0 * 24.0
    year = day * 365.0
    Sv = 1.0e12                       ## one Sverdrup (a unit of ocean transport), 1e6 meters^3/second

    ## parameters that appear in box model equations
    u0 = 16.0 * Sv / 0.0004
    alpha = 1668.0e-7
    beta = 0.7811e-3

    gamma = 1 / (300 * day)

    ## robert filter coefficient for the smoother part of the timestep
    robert_filter_coeff = 0.25

    ## freshwater forcing
    FW = [(100 / year) * 35.0 * barea[1]; -(100 / year) * 35.0 * barea[1]]

    ## restoring atmospheric temperatures
    Tstar = [22.0; 0.0]
    Sstar = [36.0; 34.0]

    structure_with_parameters = ModelParameters(
        day,
        year,
        blength,
        bdepth,
        bwidth,
        barea,
        bvolume,
        delta,
        u0,
        alpha,
        beta,
        gamma,
        robert_filter_coeff,
        FW,
        Tstar,
        Sstar
    )

    return structure_with_parameters

end

# ## Define model functions

# Here we define functions that will calculate quantities used in the forward steps.

"""
Function to compute transport.

### Arguments
- `rho`: the density vector

### Returns
- `U`: transport value
"""
function compute_transport(rho, params)
    (; delta,  u0) = params
    U = u0 * (rho[2] - (delta * rho[1] + (1 - delta) * rho[3]))
    return U
end

#-

"""
Function to compute density.

### Arguments
- `state`: vector of `[T1; T2; T3; S1; S2; S3]`

### Returns
- `rho`
"""
function compute_density(state, params)
    (; alpha, beta) = params
    rho = -alpha * state[1:3] + beta * state[4:6]
    return rho
end

#-

# Lastly, we define a function that takes one step forward.

"""
Compute the state update.

### Arguments
- `state_now` = [T1(t), T2(t), ..., S3(t)]
- `state_old` = [T1(t-dt), ..., S3(t-dt)]
- `u` = transport(t)
- `dt` = time step

### Returns
- `state_new`: [T1(t+dt), ..., S3(t+dt)]
"""
function compute_update(state_now, state_old, u, params, dt)

    dstate_now_dt = zeros(6)
    state_new = zeros(6)
    (; boxvol, gamma, Tstar, FW) = params

    ## first computing the time derivatives of the various temperatures and salinities
    if u > 0

        dstate_now_dt[1] = u * (state_now[3] - state_now[1]) / boxvol[1] + gamma * (Tstar[1] - state_now[1])
        dstate_now_dt[2] = u * (state_now[1] - state_now[2]) / boxvol[2] + gamma * (Tstar[2] - state_now[2])
        dstate_now_dt[3] = u * (state_now[2] - state_now[3]) / boxvol[3]

        dstate_now_dt[4] = u * (state_now[6] - state_now[4]) / boxvol[1] + FW[1] / boxvol[1]
        dstate_now_dt[5] = u * (state_now[4] - state_now[5]) / boxvol[2] + FW[2] / boxvol[2]
        dstate_now_dt[6] = u * (state_now[5] - state_now[6]) / boxvol[3]


    elseif u <= 0

        dstate_now_dt[1] = u * (state_now[2] - state_now[1]) / boxvol[1] + gamma * (Tstar[1] - state_now[1])
        dstate_now_dt[2] = u * (state_now[3] - state_now[2]) / boxvol[2] + gamma * (Tstar[2] - state_now[2])
        dstate_now_dt[3] = u * (state_now[1] - state_now[3]) / boxvol[3]

        dstate_now_dt[4] = u * (state_now[5] - state_now[4]) / boxvol[1] + FW[1] / boxvol[1]
        dstate_now_dt[5] = u * (state_now[6] - state_now[5]) / boxvol[2] + FW[2] / boxvol[2]
        dstate_now_dt[6] = u * (state_now[4] - state_now[6]) / boxvol[3]

    end

    ## update fldnew using a version of Euler's method
    state_new .= state_old + 2.0 * dt * dstate_now_dt

    return state_new
end

# ## Define forward functions

# Finally, we create two functions, the first of which computes and stores all the
# states of the system, and the second will take just a single step forward.

# Let's start with the standard forward function. This is just going to be used
# to store the states at every timestep:

function integrate(state_now, state_old, dt, M, parameters)

    ## Because of the adjoint problem we're setting up, we need to store both the states before
    ## and after the Robert filter smoother has been applied
    states_before = [state_old]
    states_after = [state_old]

    for t in 1:M

        rho = compute_density(state_now, parameters)
        u = compute_transport(rho, parameters)
        state_new = compute_update(state_now, state_old, u, parameters, dt)

        ## Applying the Robert filter smoother (needed for stability)
        state_new_smoothed = state_now + parameters.rf_coeff * (state_new - 2.0 * state_now + state_old)

        push!(states_after, state_new_smoothed)
        push!(states_before, state_new)

        ## cycle the "now, new, old" states
        state_old = state_new_smoothed
        state_now = state_new

    end

    return states_after, states_before
end

# Now, for the purposes of Enzyme, it would be convenient for us to have a function
# that runs a single step of the model forward rather than the whole integration.
# This would allow us to save as many of the adjoint variables as we wish when running the adjoint method,
# although for the example we'll discuss later we technically only need one of them
function one_step_forward(state_now, state_old, out_now, out_old, parameters, dt)

    state_new_smoothed = zeros(6)
    rho = compute_density(state_now, parameters)                             ## compute density
    u = compute_transport(rho, parameters)                                   ## compute transport
    state_new = compute_update(state_now, state_old, u, parameters, dt)      ## compute new state values

    ## Robert filter smoother
    state_new_smoothed[:] = state_now + parameters.rf_coeff * (state_new - 2.0 * state_now + state_old)

    out_old[:] = state_new_smoothed
    out_now[:] = state_new

    return nothing

end

# One difference to note is that `one_step_forward` now returns nothing, but is rather a function of both its input
# and output. Since the output of the function is a vector, we need to have this return nothing for Enzyme to work.
# Now we can move on to some examples using Enzyme.

# # Example 1: Simply using Enzyme

# For the first example let's just compute the gradient of our forward function and
# examine the output. We'll just run the model for one step, and take a `dt` of ten
# days. The initial conditions of the system are given as `Tbar` and `Sbar`. We run setup once here,
# and never have to run it again! (Unless we decide to change a parameter)

parameters = setup()

Tbar = [20.0; 1.0; 1.0]         ## initial temperatures
Sbar = [35.5; 34.5; 34.5]       ## initial salinities

## Running the model one step forward
states_after_smoother, states_before_smoother = integrate(
    copy([Tbar; Sbar]),
    copy([Tbar; Sbar]),
    10 * parameters.day,
    1,
    parameters
)

## Run Enzyme one time on `one_step_forward`
dstate_now = zeros(6)
dstate_old = zeros(6)
out_now = zeros(6); dout_now = ones(6)
out_old = zeros(6); dout_old = ones(6)

autodiff(
    Reverse,
    one_step_forward,
    Duplicated([Tbar; Sbar], dstate_now),
    Duplicated([Tbar; Sbar], dstate_old),
    Duplicated(out_now, dout_now),
    Duplicated(out_old, dout_old),
    Const(parameters),
    Const(10 * parameters.day)
)

# In order to run Enzyme on `one_step_forward`, we've needed to provide quite a few
# placeholders, and wrap everything in [`Duplicated`](@ref) as all components of our function
# are vectors, not scalars. Let's go through and see what Enzyme did with all
# of those placeholders.

# First we can look at what happened to the zero vectors `out_now` and `out_old`:

out_now
#-
out_old

# Comparing to the results of forward func:

states_before_smoother[2]
#-
states_after_smoother[2]

# we see that Enzyme has computed and stored exactly the output of the
# forward step. Next, let's look at `dstate_now`:

dstate_now

# Just a few numbers, but this is what makes AD so nice: Enzyme has exactly computed
# the derivative of all outputs with respect to the input `state_now`, evaluated at
# `state_now`, and acted with this gradient on what we gave as `dout_now` (in our case,
# all ones). Using AD notation for reverse mode, this is

# ```math
# \overline{\text{state_now}} = \left.\frac{\partial \text{out_now}}{\partial \text{state_now}}\right|_\text{state_now} \overline{\text{out_now}} + \left.\frac{\partial \text{out_old}}{\partial \text{state_now}}\right|_\text{state_now} \overline{\text{out_old}}
# ```

# We note here that had we initialized `dstate_now` and `dstate_old` as something else, our results
# will change. Let's multiply them by two and see what happens.

dstate_now_new = zeros(6)
dstate_old_new = zeros(6)
out_now = zeros(6); dout_now = 2 * ones(6)
out_old = zeros(6); dout_old = 2 * ones(6)
autodiff(
    Reverse,
    one_step_forward,
    Duplicated([Tbar; Sbar], dstate_now_new),
    Duplicated([Tbar; Sbar], dstate_old_new),
    Duplicated(out_now, dout_now),
    Duplicated(out_old, dout_old),
    Const(parameters),
    Const(10 * parameters.day)
)

# Now checking `dstate_now` and `dstate_old` we see

dstate_now_new

# What happened? Enzyme is actually taking the computed gradient and acting on what we
# give as input to `dout_now` and `dout_old`. Checking this, we see

2 * dstate_now

# and they match the new results. This exactly matches what we'd expect to happen since
# we scaled `dout_now` by two.

# # Example 2: Full sensitivity calculations

# Now we want to use Enzyme for a bit more than just a single derivative. Let's
# say we'd like to understand how sensitive the final temperature of Box One is to the initial
# salinity of Box Two. That is, given the function

# ```math
# J = (1,0,0,0,0,0)^T \cdot \mathbf{x}(t_f)
# ```
# we want Enzyme to calculate the derivative

# ```math
# \frac{\partial J}{\partial \mathbf{x}(0)}
# ```

# where ``x(t)`` is the state of the model at time t. If we think about ``x(t_f)`` as solely depending on the
# initial condition, then this derivative is really

# ```math
# \frac{\partial J}{\partial \mathbf{x}(0)} = \frac{\partial}{\partial \mathbf{x}(0)} \left( (1,0,0,0,0,0)^T \cdot L(\ldots(L(\mathbf{x}(0)))) \right)
# ```

# with ``L(x(t)) = x(t + dt)``, i.e. one forward step. One could expand this derivative with the chain rule (and it would be very complicated), but really this
# is where Enzyme comes in. Each run of autodiff on our forward function is one piece of this big chain rule done for us! We also note that the chain rule
# goes from the outside in, so we start with the derivative of the forward function at the final state, and work backwards until the initial state.
# To get Enzyme to do this, we complete the following steps:
# 1) Run the forward model and store outputs (in a real ocean model this wouldn't be
#       feasible and we'd need to use checkpointing)
# 2) Compute the initial derivative from the final state
# 3) Use Enzyme to work backwards until we reach the desired derivative.

# For simplicity we define a function that takes completes our AD steps

function compute_adjoint_values(states_before_smoother, states_after_smoother, M, parameters)

    dout_now = [0.0;0.0;0.0;0.0;0.0;0.0]
    dout_old = [1.0;0.0;0.0;0.0;0.0;0.0]

    for j in M:-1:1

        dstate_now = zeros(6)
        dstate_old = zeros(6)

        autodiff(
            Reverse,
            one_step_forward,
            Duplicated(states_before_smoother[j], dstate_now),
            Duplicated(states_after_smoother[j], dstate_old),
            Duplicated(zeros(6), dout_now),
            Duplicated(zeros(6), dout_old),
            Const(parameters),
            Const(10 * parameters.day)
        )

        if j == 1
            return dstate_now, dstate_old
        end

        dout_now = copy(dstate_now)
        dout_old = copy(dstate_old)

    end

    return
end

# First we integrate the model forward:

M = 10000                       ## Total number of forward steps to take
Tbar = [20.0; 1.0; 1.0]         ## initial temperatures
Sbar = [35.5; 34.5; 34.5]       ## initial salinities

states_after_smoother, states_before_smoother = integrate(
    copy([Tbar; Sbar]),
    copy([Tbar; Sbar]),
    10 * parameters.day,
    M,
    parameters
)

# Next, we pass all of our states to the AD function to get back to the desired derivative:

dstate_now, dstate_old = compute_adjoint_values(
    states_before_smoother,
    states_after_smoother,
    M,
    parameters
)

# And we're done! We were interested in sensitivity to the initial salinity of box
# two, which will live in what we've called `dstate_old`. Checking this value we see

dstate_old[5]

# As it stands this is just a number, but a good check that Enzyme has computed what we want
# is to approximate the derivative with a Taylor series. Specifically,
#
# ```math
# J(\mathbf{x}(0) + \varepsilon) \approx J(\mathbf{x}(0)) +
# \varepsilon \frac{\partial J}{\partial \mathbf{x}(0)}
# ```
#
# and a simple rearrangement yields
#
# ```math
# \frac{\partial J}{\partial \mathbf{x}(0)} \approx
# \frac{J(\mathbf{x}(0) + \varepsilon)  - J(\mathbf{x}(0))}{\varepsilon}
# ```
#
# Hopefully we see that the analytical values converge close to the one we
# found with Enzyme:

## unperturbed final state
use_to_check = states_after_smoother[M + 1]

initial_temperature = [20.0; 1.0; 1.0]
initial_salinity = [35.5; 34.5; 34.5]

## a loop to compute the perturbed final states
diffs = []
step_sizes = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
for eps in step_sizes

    state_new_smoothed = zeros(6)

    perturbed_initial_salinity = initial_salinity + [0.0; eps; 0.0]

    state_old = [initial_temperature; perturbed_initial_salinity]
    state_now = [20.0; 1.0; 1.0; 35.5; 34.5; 34.5]

    for t in 1:M

        rho = compute_density(state_now, parameters)
        u = compute_transport(rho, parameters)
        state_new = compute_update(state_now, state_old, u, parameters, 10 * parameters.day)

        state_new_smoothed[:] = state_now + parameters.rf_coeff * (state_new - 2.0 * state_now + state_old)

        state_old = state_new_smoothed
        state_now = state_new

    end

    push!(diffs, (state_old[1] - use_to_check[1]) / eps)

end

# Then checking what we found the derivative to be analytically:

diffs

# which comes very close to our calculated value. We can go further and check the
# percent difference to see

abs.(diffs .- dstate_old[5]) ./ dstate_old[5]

# and we get down to a percent difference on the order of ``{10}^{-5}``, showing Enzyme calculated
# the correct derivative. Success!
