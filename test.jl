using CUDA, Enzyme, Zygote, Flux

struct Linear{W,B}
    weight::W
    bias::B
end
Linear(in::Integer, out::Integer) = Linear(randn(Float32, out, in), randn(Float32, out))
(m::Linear)(x) = m.weight * x .+ m.bias

model = Linear(2, 2) |> gpu
dmodel = Duplicated(model, Enzyme.make_zero(model))
x = rand(Float32, 2) |> gpu
f(m, x) = sum(m(x))
ad = Enzyme.set_runtime_activity(Reverse)
Enzyme.autodiff(ad, Const(f), Active, dmodel, Const(x))
g = dmodel.dval
g.weight

# g2 = Zygote.gradient(f, model, x)[1]
# @assert collect(g2.weight) ≈ collect(g1.weight) atol=1e-4, rtol=1e-4