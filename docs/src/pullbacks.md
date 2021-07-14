# Implementing pullbacks

Enzyme's [`autodiff`](@ref) function can only handle functions with scalar output. To implement pullbacks (back-propagation of gradients/tangents) for array-valued functions, use a mutating function that returns `nothing` and stores it's result in one of the arguments, which must be passed wrapped in a [`Duplicated`](@ref).

## Example

Given a function `mymul!` that performs the equivalent of `R = A * B` for matrices `A` and `B`, and given a gradient (tangent) `∂z_∂R`, we can compute `∂z_∂A` and `∂z_∂B` like this:

```@example pullback
using Enzyme

function mymul!(R, A, B)
    @assert axes(A,2) == axes(B,1)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end
    @inbounds for j in axes(B, 2), i in axes(A, 1)
        @inbounds @simd for k in axes(A,2)
            R[i,j] += A[i,k] * B[k,j]
        end
    end
    nothing
end


A = rand(5, 3)
B = rand(3, 7)

R = zeros(size(A,1), size(B,2))
∂z_∂R = rand(size(R)...)  # Some gradient/tangent passed to us

∂z_∂A = zero(A)
∂z_∂B = zero(B)

Enzyme.autodiff(mymul!, Const, Duplicated(R, ∂z_∂R), Duplicated(A, ∂z_∂A), Duplicated(B, ∂z_∂B))
```

Now we have:

```@example pullback
R ≈ A * B            &&
∂z_∂A ≈ ∂z_∂R * B'   &&  # equivalent to Zygote.pullback(*, A, B)[2](∂z_∂R)[1]
∂z_∂B ≈ A' * ∂z_∂R       # equivalent to Zygote.pullback(*, A, B)[2](∂z_∂R)[2]
```

Note that the result of the backpropagation is *added to* `∂z_∂A` and `∂z_∂B`, they act as accumulators for gradient information.
