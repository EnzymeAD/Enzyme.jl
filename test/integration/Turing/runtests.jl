module TuringIntegrationTests

using ADTypes: AutoEnzyme
using DynamicPPL
using Enzyme: Enzyme
import ForwardDiff
using StableRNGs: StableRNG
using Test
using Turing

adtypes = (
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward)),
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)),
)

# Some supplements to DynamicPPL.TestUtils.ALL_MODELS.
@model function assume_normal()
    a ~ Normal()
end
dppl_lda = begin
    v = 100      # words
    k = 5        # topics
    m = 10       # number of docs
    alpha = ones(k)
    beta = ones(v)
    phi = rand(Dirichlet(beta), k)
    theta = rand(Dirichlet(alpha), m)
    doc_lengths = rand(Poisson(1_000), m)
    n = sum(doc_lengths)
    w = Vector{Int}(undef, n)
    doc = Vector{Int}(undef, n)
    for i in 1:m
        local idx = sum(doc_lengths[1:(i - 1)]) # starting index for inner loop
        for j in 1:doc_lengths[i]
            z = rand(Categorical(theta[:, i]))
            w[idx + j] = rand(Categorical(phi[:, z]))
            doc[idx + j] = i
        end
    end
    @model function dppl_lda(k, m, w, doc, alpha, beta)
        theta ~ product_distribution(fill(Dirichlet(alpha), m))
        phi ~ product_distribution(fill(Dirichlet(beta), k))
        log_phi_dot_theta = log.(phi * theta)
        @addlogprob! sum(log_phi_dot_theta[CartesianIndex.(w, doc)])
    end
    dppl_lda
end
MODELS = [
    DynamicPPL.TestUtils.ALL_MODELS...,
    assume_normal(),
    dppl_lda(k, m, w, doc, alpha, beta),
]

@testset "AD on logdensity" begin
    # This code is essentially what Turing's HMC/NUTS samplers use internally
    @testset "$(model.f)" for model in MODELS
        @testset "AD type: $(adtype)" for adtype in adtypes
            @test DynamicPPL.TestUtils.AD.run_ad(model, adtype; rng = StableRNG(468), test = true, benchmark = false) isa Any
        end
    end
end

@testset "AD / Gibbs sampling" begin
    # The code to differentiate for the Gibbs sampler is slightly different from the
    # HMC/NUTS samplers (even though each individual variable is sampled with HMC) so we
    # have to test it separately.
    @testset "AD type: $(adtype)" for adtype in adtypes
        spl = Gibbs(
            @varname(s) => HMC(0.1, 10; adtype = adtype),
            @varname(m) => HMC(0.1, 10; adtype = adtype),
        )
        @testset "model=$(model.f)" for model in DynamicPPL.TestUtils.DEMO_MODELS
            @info "Sampling model=$(model.f) with AD type=$(adtype)"
            @test sample(StableRNG(468), model, spl, 2; progress = false) isa Any
        end
    end
end

end # module
