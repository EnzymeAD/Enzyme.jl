# # [Physics-informed neural network from scratch](@id pinn)

# This tutorial trains a small physics-informed neural network (PINN) with nothing
# but plain Julia arrays and Enzyme. It is a Julia port of the PyTorch tutorial
# [*A PINN from scratch*](https://rimbach.gitlabpages.inria.fr/fabrique-ton-pinns/a-pinn-from-scratch/)
# from the course *Fabrique ton PINNs* (Inria); the problem, the network, the loss,
# the sampling strategy and all the numerical constants follow that tutorial, so you
# can read the two side by side. What the original does with `torch.autograd.grad`
# and `loss.backward()` we do with nested [`autodiff`](@ref) calls.
#
# The interesting part for Enzyme users is the *nesting*: the PDE residual needs
# second spatial derivatives of the network output, and the training step needs the
# gradient of that residual with respect to the network parameters. That is
# reverse mode over forward mode over forward mode, all handled by Enzyme.

# ## The problem
#
# We solve the parametric Poisson equation on a disk ``\Omega`` with centre ``c``
# and radius ``r``, for a parameter ``\mu`` in an interval ``M``:
#
# ```math
# \begin{aligned}
#   -\mu \Delta u &= f &&\text{in } \Omega \times M, \\
#   u &= g &&\text{on } \partial\Omega \times M,
# \end{aligned}
# ```
#
# with ``f(x, \mu) = \mu^2`` and ``g = 0``. The exact solution is
# ``u^*(x, \mu) = \tfrac{\mu}{4}\,\bigl(r^2 - \lVert x - c \rVert^2\bigr)``, which we
# use to check the result at the end.
#
# The PINN ansatz is a network ``u_\theta(x, \mu)`` that takes the two spatial
# coordinates and the parameter as inputs. It is trained by minimising
#
# ```math
# \tilde\ell(\theta) = \frac{1}{m}\sum_{i=1}^{m} \bigl(-\mu_i \Delta u_\theta(x_i, \mu_i) - f(x_i, \mu_i)\bigr)^2
#                    + \frac{w}{n}\sum_{j=1}^{n} \bigl(u_\theta(x_j, \mu_j) - g(x_j, \mu_j)\bigr)^2,
# ```
#
# where the first sum runs over collocation points in the interior and the second
# over points on the boundary, both resampled at every epoch.

using Enzyme
using LinearAlgebra
using Random
using Statistics

const c = (-0.5, 0.5)   # centre of the disk
const r = 1.0           # radius of the disk
const M = (0.5, 1.0)    # parameter domain

rng = Random.Xoshiro(0)
nothing #hide

# Throughout, a batch of points is stored column-wise: `xy` is a `2×N` matrix and
# `μ` a `1×N` matrix, so that `vcat(xy, μ)` is the `3×N` network input.

# ## The network
#
# A multilayer perceptron `3 → 16 → 16 → 1` with `tanh` activations. The parameters
# `θ` are a tuple of `(W, b)` named tuples, initialised like `torch.nn.Linear`.

function init_mlp(rng, dim_in::Int, hidden::Vector{Int}, dim_out::Int)
    sizes = [dim_in; hidden; dim_out]
    layers = map(1:length(sizes) - 1) do i
        din, dout = sizes[i], sizes[i + 1]
        bound = 1 / sqrt(din)
        (W = (2 .* rand(rng, dout, din) .- 1) .* bound,
         b = (2 .* rand(rng, dout) .- 1) .* bound)
    end
    return Tuple(layers)
end

function mlp(θ, inputs::AbstractMatrix)
    h = inputs
    for i in 1:length(θ) - 1
        h = tanh.(θ[i].W * h .+ θ[i].b)
    end
    return θ[end].W * h .+ θ[end].b
end

"Network evaluated at `N` points; returns a vector of length `N`."
u_theta(θ, xy, μ) = vec(mlp(θ, vcat(xy, μ)))

θ = init_mlp(rng, 3, [16, 16], 1)
n_params = sum(length(l.W) + length(l.b) for l in θ)

# ## Spatial derivatives with forward mode
#
# The PDE operator needs the Laplacian ``\Delta u = u_{xx} + u_{yy}`` of the network
# output at every collocation point. Since the ``i``-th output only depends on the
# ``i``-th column of `xy`, seeding the tangent of `xy` with a matrix `d` whose columns
# are all the same unit direction gives the directional derivative of every output at
# once, in a single forward-mode call. This is the same batching trick the PyTorch
# version uses when it calls `torch.autograd.grad` with a `ones_like` seed.
#
# A second forward-mode call over the first one gives the second directional
# derivative. `u` is passed in as an argument so the same helpers can be applied to
# the exact solution below.

function directional_derivative(u, θ, xy, μ, d)
    return first(autodiff(Forward, u, Duplicated(θ, make_zero(θ)),
                          Duplicated(xy, d), Duplicated(μ, zero(μ))))
end

function second_directional_derivative(u, θ, xy, μ, d)
    return first(autodiff(Forward, directional_derivative,
                          Const(u), Duplicated(θ, make_zero(θ)), Duplicated(xy, d),
                          Duplicated(μ, zero(μ)), Duplicated(d, zero(d))))
end

function laplacian(u, θ, xy, μ)
    N = size(xy, 2)
    ex = vcat(ones(1, N), zeros(1, N))
    ey = vcat(zeros(1, N), ones(1, N))
    return second_directional_derivative(u, θ, xy, μ, ex) .+
           second_directional_derivative(u, θ, xy, μ, ey)
end

# !!! note "Why `Duplicated` with a zero tangent instead of `Const`?"
#     We only want derivatives with respect to `xy` here, so `θ` and `μ` could be
#     marked `Const`. That is semantically fine even though the outer reverse pass
#     later differentiates through these calls with respect to `θ`. However, Enzyme.jl
#     currently mishandles one specific pattern in nested differentiation: a `Const`
#     argument that is passed by value (such as our tuple of layers) and indexed with
#     a runtime index (the loop in `mlp`) is treated as inactive by the *outer* pass as
#     well, which silently zeroes the gradient of the Laplacian term. See
#     [EnzymeAD/Enzyme.jl#3617](https://github.com/EnzymeAD/Enzyme.jl/issues/3617).
#     Passing `Duplicated(θ, make_zero(θ))` sidesteps this. The gradient below was
#     checked against finite differences.

# The PDE operators, right-hand sides and the exact solution. The parameter slot of
# `ustar` carries `(c, r)` so it can be fed through the same `laplacian` helper.

apply_left_hand_side(u, θ, xy, μ) = -vec(μ) .* laplacian(u, θ, xy, μ)
apply_left_hand_side_bc(u, θ, xy, μ) = u(θ, xy, μ)

f(xy, μ) = vec(μ) .^ 2
g(xy, μ) = zeros(size(xy, 2))

function ustar(cr, xy, μ)
    (c, r) = cr
    return vec(μ) ./ 4 .* (r^2 .- (xy[1, :] .- c[1]) .^ 2 .- (xy[2, :] .- c[2]) .^ 2)
end

# ## Samplers
#
# Interior points are drawn uniformly in the disk by rejection from the bounding box,
# boundary points through the polar parameterisation, and parameters uniformly in
# `M`.

function sample_Omega(rng, n, c, r)
    xy = Matrix{Float64}(undef, 2, n)
    k = 0
    while k < n
        x = c[1] + r * (2rand(rng) - 1)
        y = c[2] + r * (2rand(rng) - 1)
        if (x - c[1])^2 + (y - c[2])^2 <= r^2
            k += 1
            xy[1, k] = x
            xy[2, k] = y
        end
    end
    return xy
end

function sample_Omega_bc(rng, n, c, r)
    t = 2π .* rand(rng, 1, n)
    return vcat(c[1] .+ r .* cos.(t), c[2] .+ r .* sin.(t))
end

sample_parameter_domain(rng, n, M) = M[1] .+ (M[2] - M[1]) .* rand(rng, 1, n)

# Before training anything, check the derivative machinery on the exact solution:
# ``-\mu \Delta u^*`` must equal ``f = \mu^2`` exactly.

let
    xy = sample_Omega(rng, 5, c, r)
    μ = sample_parameter_domain(rng, 5, M)
    apply_left_hand_side(ustar, (c, r), xy, μ) ≈ f(xy, μ)
end

# ## The loss and its gradient with reverse mode
#
# The discretised loss is a plain Julia function of `θ`. Its gradient comes from one
# reverse-mode `autodiff` call that differentiates straight through the two nested
# forward-mode calls inside `laplacian`.
#
# The collocation arrays get zero shadows too. They are the analogue of
# `xy.requires_grad_()` in the tutorial: we discard their gradients, but every array
# that flows into the nested forward-mode calls has to be differentiable, otherwise
# Enzyme reports a runtime-activity error from `vcat`, which may return one of its
# inputs unchanged.

function ltilde(θ, xy, μ, xy_bc, μ_bc, w)
    residual = apply_left_hand_side(u_theta, θ, xy, μ) .- f(xy, μ)
    boundary = apply_left_hand_side_bc(u_theta, θ, xy_bc, μ_bc) .- g(xy_bc, μ_bc)
    return mean(abs2, residual) + w * mean(abs2, boundary)
end

function loss_and_gradient!(dθ, θ, xy, μ, xy_bc, μ_bc, w)
    make_zero!(dθ)
    _, loss = autodiff(ReverseWithPrimal, ltilde, Active,
                       Duplicated(θ, dθ),
                       Duplicated(xy, zero(xy)), Duplicated(μ, zero(μ)),
                       Duplicated(xy_bc, zero(xy_bc)), Duplicated(μ_bc, zero(μ_bc)),
                       Const(w))
    return loss
end

# ## Optimisation
#
# The tutorial uses Adam with PyTorch's default learning rate. To keep this example
# dependency-free we write the update ourselves; it modifies the parameter arrays in
# place.

mutable struct Adam{T}
    lr::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    m::T
    v::T
    t::Int
end
Adam(θ; lr = 1e-3, β1 = 0.9, β2 = 0.999, ϵ = 1e-8) =
    Adam(lr, β1, β2, ϵ, make_zero(θ), make_zero(θ), 0)

function step!(opt::Adam, θ, dθ)
    opt.t += 1
    for (layer, dlayer, mlayer, vlayer) in zip(θ, dθ, opt.m, opt.v), name in (:W, :b)
        p, g = getfield(layer, name), getfield(dlayer, name)
        m, v = getfield(mlayer, name), getfield(vlayer, name)
        @. m = opt.β1 * m + (1 - opt.β1) * g
        @. v = opt.β2 * v + (1 - opt.β2) * g^2
        @. p -= opt.lr * (m / (1 - opt.β1^opt.t)) / (sqrt(v / (1 - opt.β2^opt.t)) + opt.ϵ)
    end
    return θ
end

# The training loop resamples all collocation points every epoch, exactly like the
# original.

m = 1000      # interior collocation points
n = 2000      # boundary collocation points
w = 30.0      # weight of the boundary term
epochs = 2000

dθ = make_zero(θ)
opt = Adam(θ)
loss_history = Float64[]

for epoch in 1:epochs
    xy = sample_Omega(rng, m, c, r)
    μ = sample_parameter_domain(rng, m, M)
    xy_bc = sample_Omega_bc(rng, n, c, r)
    μ_bc = sample_parameter_domain(rng, n, M)

    loss = loss_and_gradient!(dθ, θ, xy, μ, xy_bc, μ_bc, w)
    step!(opt, θ, dθ)
    push!(loss_history, loss)

    if epoch % 200 == 0 || epoch == 1
        println("epoch ", lpad(epoch, 4), "  loss = ", round(loss; sigdigits = 4))
    end
end

# ## Checking the result
#
# Compare the trained network with the exact solution on fresh samples of
# ``\Omega \times M``.

let
    xy = sample_Omega(rng, 10_000, c, r)
    μ = sample_parameter_domain(rng, 10_000, M)
    err = u_theta(θ, xy, μ) .- ustar((c, r), xy, μ)
    println("relative L² error over Ω × M: ",
            round(norm(err) / norm(ustar((c, r), xy, μ)); sigdigits = 3))
end

# and along a line through the centre of the disk for ``\mu = 0.75``:

let
    xs = range(c[1] - r, c[1] + r; length = 9)
    xy = vcat(xs', fill(c[2], 1, length(xs)))
    μ = fill(mean(M), 1, length(xs))
    for (x, uθ, u) in zip(xs, u_theta(θ, xy, μ), ustar((c, r), xy, μ))
        println("x = ", lpad(round(x; digits = 2), 5), "   u_θ = ", lpad(round(uθ; digits = 4), 7),
                "   u* = ", lpad(round(u; digits = 4), 7))
    end
end

# The loss keeps decreasing at the end of the 2000 epochs, so more epochs or a
# learning-rate schedule improve the fit further, exactly as in the original
# tutorial.
