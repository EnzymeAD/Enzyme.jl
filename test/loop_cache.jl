using Enzyme, Test

# A scan that seeds its result from the first element runs that step before its
# loop, so the loop is `for i in 2:m` and runs `m - 1` times. Nested in another
# loop with the same `us` in every outer step, Enzyme allocates the inner loop's
# caches once, before the outer loop, sized by `(m - 1)` times the outer count.
# With `us` empty the inner loop never runs and that count is negative.
@inline function peeled_inner_scan(init, us, x)
    m = length(us)
    pending = Vector{Vector{Float64}}(undef, m + 1)
    pending[1] = init
    c = init
    if m >= 1
        c = c .+ x .* us[1]
        pending[2] = c
        for i in 2:m
            c = c .+ x .* us[i]
            pending[i + 1] = c
        end
    end
    return pending[end]
end

function nested_peeled_scans(xs, us, w)
    carry = zeros(2) .* w[1]
    total = 0.0
    for k in eachindex(xs)
        carry = peeled_inner_scan(carry .* w[1], us, xs[k])
        total += sum(carry)
    end
    return total + sum(w)
end

# The summed carry follows σ ← w₁σ + x·Σus, so the value and its derivative in
# w₁ follow in closed form.
function nested_peeled_scans_reference(xs, us, w)
    mass = sum(sum, us; init = 0.0)
    σ = dσ = total = dtotal = 0.0
    for x in xs
        σ, dσ = w[1] * σ + mass * x, σ + w[1] * dσ
        total += σ
        dtotal += dσ
    end
    return total + sum(w), [dtotal + 1, 1.0, 1.0]
end

@testset "Loop caches allocated before a skipped inner loop" begin
    w = [1.3, 2.0, 0.5]
    # Outer lengths of at least two keep the count at or below -2, where an
    # unclamped count fails with an `OutOfMemoryError`; -1 corrupts the heap.
    for us in (Vector{Float64}[], [[1.0, 2.0]], [[1.0, 2.0], [0.5, -1.0]]),
            xs in ([0.3, -0.2], [0.3, -0.2, 0.5])
        value, gradient = nested_peeled_scans_reference(xs, us, w)
        dw = zero(w)
        res = autodiff(
            ReverseWithPrimal, nested_peeled_scans, Active,
            Const(xs), Const(us), Duplicated(w, dw)
        )
        @test res[2] ≈ value
        @test dw ≈ gradient
    end
end
