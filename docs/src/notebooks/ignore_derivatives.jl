### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ b72e9218-81ba-11f0-1eba-5bd949c7ade4
begin
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the local environment
    Pkg.activate(".")
    Pkg.instantiate()
    using PlutoUI, PlutoLinks
end

# ╔═╡ 9f5c0822-a19a-4c63-95e7-d2f066a7440f
@revise using Enzyme

# ╔═╡ a4453d23-6e31-451f-b2cd-97346accac82
@revise using EnzymeCore

# ╔═╡ 3f8e0394-9b27-40a4-bc4c-3f4f773d35dc
using LinearAlgebra: norm

# ╔═╡ bd0352c3-1b3c-42f5-ab93-7ca4cb67b9ad
begin
    using CairoMakie
    set_theme!(
        theme_latexfonts();
        fontsize = 16,
        Lines = (linewidth = 2,),
        markersize = 16
    )
end

# ╔═╡ df72e42f-7eec-476f-8ce5-72b09f620005
md"""
# Reproducing "Stabilizing backpropagation through time to learn complex physics"

Fig 1 from <https://openreview.net/pdf?id=bozbTTWcaw>
"""

# ╔═╡ 23a8503f-3c68-4523-aebe-a4ce4575a02b
import Enzyme: ignore_derivatives

# ╔═╡ fabba18a-b8d8-479d-babd-c18279273fb5
N(xᵢ, θ) = θ[1] * xᵢ^2 + θ[2] * xᵢ;

# ╔═╡ 9c773164-d4c4-404e-a1bd-abbb2bd9baa7
S(xᵢ, cᵢ) = xᵢ + cᵢ;

# ╔═╡ 5baa757c-c611-4d8b-ac37-4f97e585613e
function simulate(N, S, x₀, y, θ, n)
    xᵢ = x₀
    for i in 1:n
        cᵢ = N(xᵢ, θ)
        xᵢ = S(xᵢ, cᵢ)
    end
    return L = 1 / 2 * (xᵢ - y)^2
end

# ╔═╡ adf9ae2c-92b6-4826-bb01-12e46f365610
begin
    x₀ = -0.3
    y = 2.0
    n = 4
end;

# ╔═╡ bdc31f7f-fa2d-45d5-bc7a-843340ad5426
begin
    θ₁ = -4:0.01:4
    θ₂ = -4:0.01:4
end;

# ╔═╡ 074d5e04-5e6d-4c47-bfc6-e719756b0ef7
θ_space = collect(Iterators.product(θ₁, θ₂));

# ╔═╡ 3101e04b-7cbb-4a30-851d-c6183a65c8ae
L_space = simulate.(N, S, x₀, y, θ_space, n);

# ╔═╡ 0c0ebb20-e794-4545-b94c-e026cb7fa3e2
let
    fig, ax, hm = heatmap(
        θ₁, θ₂, L_space,
        colorscale = log10,
        colormap = Makie.Reverse(:Blues),
        colorrange = (10^-5, 10^5)
    )
    Colorbar(fig[:, end + 1], hm)

    fig
end

# ╔═╡ 45ee18f4-d6d3-40f4-bbc0-04cbd3b7b840
function ∇simulate(N, S, x₀, y, θ, n)
    dθ = MixedDuplicated(θ, Ref(Enzyme.make_zero(θ)))
    Enzyme.autodiff(Enzyme.Reverse, simulate, Const(N), Const(S), Const(x₀), Const(y), dθ, Const(n))
    return dθ.dval[]
end

# ╔═╡ ae6a671d-1559-4bff-af6e-78d2b54db020
function plot_gradientfield(N, S, x₀, y, θ₁, θ₂, n)
    θ_space = collect(Iterators.product(θ₁, θ₂))
    gradient_field = ∇simulate.(N, S, x₀, y, θ_space, n)

    fig, ax, hm = heatmap(
        θ₁, θ₂, map(norm, gradient_field),
        colorscale = log10,
        colormap = Makie.Reverse(:Blues),
        colorrange = (10^-3, 10^3)
    )
    Colorbar(fig[:, end + 1], hm)

    streamplot!(
        ax, (θ) -> -∇simulate(N, S, x₀, y, θ, n), θ₁, θ₂,
        alpha = 0.5,
        colorscale = log10, color = p -> :red,
        arrow_size = 10
    )
    return fig
end

# ╔═╡ be852753-126d-42fa-a55c-c907f5dce99d
plot_gradientfield(N, S, x₀, y, θ₁, θ₂, n)

# ╔═╡ 873e7792-99a1-4472-92c2-6fc32e2889fa
N_stop(xᵢ, θ) = θ[1] * ignore_derivatives(xᵢ^2) + θ[2] * ignore_derivatives(xᵢ);

# ╔═╡ d71a22cc-c1f3-4425-8a6f-442a0bc4f215
plot_gradientfield(N_stop, S, x₀, y, θ₁, θ₂, n)

# ╔═╡ Cell order:
# ╟─df72e42f-7eec-476f-8ce5-72b09f620005
# ╠═b72e9218-81ba-11f0-1eba-5bd949c7ade4
# ╠═9f5c0822-a19a-4c63-95e7-d2f066a7440f
# ╠═23a8503f-3c68-4523-aebe-a4ce4575a02b
# ╠═a4453d23-6e31-451f-b2cd-97346accac82
# ╠═3f8e0394-9b27-40a4-bc4c-3f4f773d35dc
# ╠═bd0352c3-1b3c-42f5-ab93-7ca4cb67b9ad
# ╠═fabba18a-b8d8-479d-babd-c18279273fb5
# ╠═9c773164-d4c4-404e-a1bd-abbb2bd9baa7
# ╠═5baa757c-c611-4d8b-ac37-4f97e585613e
# ╠═adf9ae2c-92b6-4826-bb01-12e46f365610
# ╠═bdc31f7f-fa2d-45d5-bc7a-843340ad5426
# ╠═074d5e04-5e6d-4c47-bfc6-e719756b0ef7
# ╠═3101e04b-7cbb-4a30-851d-c6183a65c8ae
# ╠═0c0ebb20-e794-4545-b94c-e026cb7fa3e2
# ╠═45ee18f4-d6d3-40f4-bbc0-04cbd3b7b840
# ╠═ae6a671d-1559-4bff-af6e-78d2b54db020
# ╠═be852753-126d-42fa-a55c-c907f5dce99d
# ╠═873e7792-99a1-4472-92c2-6fc32e2889fa
# ╠═d71a22cc-c1f3-4425-8a6f-442a0bc4f215
