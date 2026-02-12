using Enzyme
using Flux
using Zygote
using Test
using NNlib
using StableRNGs
using Random

# generic loss function for any Flux model
generic_loss_function(model, x, ps, st) = sum(abs2, first(model(x, ps, st)))

# compute gradients using Enzyme
function compute_enzyme_gradient(model, x, ps, st)
    return Enzyme.gradient(
        Enzyme.set_runtime_activity(Reverse),
        generic_loss_function,
        Const(model),
        x,
        ps,
        Const(st),
    )[2:3]
end

# compute gradients using Zygote
function compute_zygote_gradient(model, x, ps, st)
    _, dx, dps, _ = Zygote.gradient(generic_loss_function, model, x, ps, st)
    return dx, dps
end

# compare Enzyme gradients with Zygote gradients
function test_enzyme_gradients(model, x, ps, st)
    dx, dps = compute_enzyme_gradient(model, x, ps, st)
    dx_zygote, dps_zygote = compute_zygote_gradient(model, x, ps, st)

    @test check_approx(dx, dx_zygote; atol = 1.0f-3, rtol = 1.0f-3)
    @test check_approx(dps, dps_zygote; atol = 1.0f-3, rtol = 1.0f-3)
end

# small list of models to test
const MODELS_LIST = [
    # simple Dense layer
    (Dense(2, 3), randn(Float32, 2, 4)),

    # small Chain
    (Chain(Dense(2, 4, relu), Dense(4, 2)), randn(Float32, 2, 3)),

    # simple Conv layer
    (Conv((3, 3), 2 => 2), randn(Float32, 5, 5, 2, 1)),
]


@testset "Enzyme Flux Integration" begin
    for (i, (model, x)) in enumerate(MODELS_LIST)
        @testset "[$i] $(nameof(typeof(model)))" begin
            # set up parameters and state
            ps = Flux.trainable(model)
            st = nothing

            # run the gradient test
            test_enzyme_gradients(model, x, ps, st)
        end
    end
end
