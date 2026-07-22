using Enzyme
using Flux
using Zygote
using Test
using Functors: fmapstructure_with_path, fmap, isleaf
using Random

# Generic loss for any Flux model: sum of squares of the model output.
generic_loss(model, x) = sum(abs2, model(x))

# A subtree that holds no float arrays is treated as a leaf and skipped, since
# Zygote (and Enzyme's Zygote-like gradient) may return `nothing` for whole
# subtrees; this keeps us from recursing into such a pruned branch.
function _contains_no_numerical(kp, x)
    count = 0
    fmap(x) do y
        y isa AbstractArray{<:AbstractFloat} && (count += 1)
        return y
    end
    return count == 0
end

# Compare two Zygote-like gradient structures leaf by leaf. `exclude` stops the
# tandem walk at genuine leaves and at numeric-free subtrees, so `@test` fires on
# every float-array leaf present in both structures.
function check_equal_leaves(a, b; rtol = 1.0f-3, atol = 1.0f-3)
    exclude(kp, x) = isleaf(x) || _contains_no_numerical(kp, x)
    fmapstructure_with_path(a, b; exclude) do kp, x, y
        x isa AbstractArray{<:AbstractFloat} && @test x ≈ y rtol = rtol atol = atol
        return nothing
    end
    return nothing
end

# Compare Enzyme gradients (w.r.t. both the model and the input) against Zygote.
function test_enzyme_gradients(model, x; rtol = 1.0f-3, atol = 1.0f-3)
    grad_zygote = Flux.gradient(generic_loss, AutoZygote(), model, x)
    grad_enzyme = Flux.gradient(generic_loss, AutoEnzyme(), model, x)
    return check_equal_leaves(grad_zygote, grad_enzyme; rtol, atol)
end

# Models mirror the list exercised by Flux's own test suite, see
# https://github.com/FluxML/Flux.jl/blob/master/test/test_module.jl
Random.seed!(0)
const TEST_MODELS = [
    (Dense(2 => 4), randn(Float32, 2, 1), "Dense"),
    (Chain(Dense(2 => 4, tanh), Dense(4 => 3)), randn(Float32, 2, 1), "Chain(Dense, Dense)"),
    (f64(Chain(Dense(2 => 4), Dense(4 => 2))), randn(Float64, 2, 1), "f64(Chain(Dense, Dense))"),
    (Flux.Scale([1.0f0 2.0f0 3.0f0 4.0f0], true, abs2), randn(Float32, 2, 1), "Flux.Scale"),
    (Conv((3, 3), 2 => 3), randn(Float32, 3, 3, 2, 1), "Conv"),
    (Chain(Conv((3, 3), 2 => 3), Conv((3, 3), 3 => 1, tanh)), rand(Float32, 5, 5, 2, 1), "Chain(Conv, Conv)"),
    (Chain(Conv((4, 4), 2 => 2, pad = SamePad()), MeanPool((5, 5), pad = SamePad())), rand(Float32, 5, 5, 2, 2), "Chain(Conv, MeanPool)"),
    (Maxout(() -> Dense(5 => 4, tanh), 3), randn(Float32, 5, 1), "Maxout"),
    (SkipConnection(Dense(2 => 2), vcat), randn(Float32, 2, 3), "SkipConnection"),
    (Flux.Bilinear((2, 2) => 3), randn(Float32, 2, 1), "Bilinear"),
    (ConvTranspose((3, 3), 3 => 2, stride = 2), rand(Float32, 5, 5, 3, 1), "ConvTranspose"),
    (LayerNorm(2), randn(Float32, 2, 10), "LayerNorm"),
    (BatchNorm(2), randn(Float32, 2, 10), "BatchNorm"),
    (first ∘ MultiHeadAttention(16), randn32(16, 20, 2), "MultiHeadAttention"),
    (RNN(3 => 2), randn(Float32, 3, 2), "RNN"),
    (LSTM(3 => 5), randn(Float32, 3, 2), "LSTM"),
    (GRU(3 => 5), randn(Float32, 3, 10), "GRU"),
    (Chain(RNN(3 => 4), RNN(4 => 3)), randn(Float32, 3, 2), "Chain(RNN, RNN)"),
    (Chain(LSTM(3 => 5), LSTM(5 => 3)), randn(Float32, 3, 2), "Chain(LSTM, LSTM)"),
]

@testset "Enzyme Flux Integration" begin
    @testset "[$(i)] $(name)" for (i, (model, x, name)) in enumerate(TEST_MODELS)
        Flux.trainmode!(model)
        test_enzyme_gradients(model, x)
    end
end
