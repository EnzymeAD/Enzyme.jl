using Test
using Random: MersenneTwister
using SymbolicRegression
using SymbolicRegression.ConstantOptimizationModule: Evaluator, GradEvaluator, specialized_options
using DynamicExpressions
using DifferentiationInterface: AutoEnzyme
using ForwardDiff
using Enzyme

rng = MersenneTwister(0)
X = rand(rng, 2, 16)
true_params = reshape([0.5], 1, 1)
true_constants = [2.6, -0.2]
init_params = reshape([0.1], 1, 1)
init_constants = [2.5, -0.5]

n = size(X, 2)

function model(x1, x2, c1, c2, p1)
    return x1 * x1 - cos(c1 * x2 + c2) + p1
end

y = [
    model(X[1, i], X[2, i], true_constants[1], true_constants[2], true_params[1]) for
    i in 1:n
]

dataset = Dataset(X, y)

function loss(theta)
    c1, c2, p1 = theta
    pred = [model(X[1, i], X[2, i], c1, c2, p1) for i in 1:n]
    return sum(abs2, pred .- y) / n
end

theta0 = vcat(init_constants, init_params)
ref_val = loss(theta0)
ref_grad = ForwardDiff.gradient(loss, theta0)

options = Options(; unary_operators=[cos], binary_operators=[+, *, -], autodiff_backend=:Enzyme)

ex = @parse_expression(
    x * x - cos(2.5 * y + -0.5) + p1,
    operators = options.operators,
    expression_type = ParametricExpression,
    variable_names = ["x", "y"],
    extra_metadata = (parameter_names=["p1"], parameters=init_params),
)

x0, refs = get_scalar_constants(ex)
G = zero(x0)

f = Evaluator(ex, refs, dataset, specialized_options(options))
fg! = GradEvaluator(f, AutoEnzyme())

val = fg!(nothing, G, x0)

@test val ≈ ref_val
@test G ≈ ref_grad
