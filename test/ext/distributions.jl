using Enzyme, Test, Distributions

# `Product` sums `logpdf` over its distributions through a closure that captures the
# active distributions and the constant data; the extension overlays a method without it.
# Runtime activity is deliberately off: without the overlay these calls throw.
product_logpdf(θ, y) = logpdf(product_distribution(Exponential.(θ)), y)

@testset "Distributions" begin
    for n in (5, 3000) # a short sum, and one long enough for pairwise summation
        θ = rand(n) .+ 0.5
        y = rand(n)
        d = product_logpdf(θ, y)
        dθ = @. -1 / θ + y / θ^2

        g = Enzyme.gradient(Forward, product_logpdf, θ, Const(y); chunk = Val(16))
        @test g[1] ≈ dθ

        g = Enzyme.gradient(Reverse, product_logpdf, θ, Const(y))
        @test g[1] ≈ dθ

        dθ2 = zero(θ)
        _, v = Enzyme.autodiff(ReverseWithPrimal, product_logpdf, Active, Duplicated(θ, dθ2), Const(y))
        @test dθ2 ≈ dθ
        @test v ≈ d
    end
end
