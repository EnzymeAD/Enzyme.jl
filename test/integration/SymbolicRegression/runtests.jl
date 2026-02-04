using Test
using Random: MersenneTwister
using SymbolicRegression
using Enzyme

rng = MersenneTwister(0)
X = rand(rng, 2, 16)
# Choose a constant target so constant-optimization is guaranteed to run,
# exercising `autodiff_backend=:Enzyme` deterministically.
y = fill(1.0, size(X, 2))

dataset = Dataset(
    X,
    y;
    variable_names=["x1", "x2"],
    extra=(class=fill(1, size(X, 2)),),
)

options = Options(
    binary_operators=[+, *, -],
    unary_operators=[],
    populations=1,
    population_size=20,
    ncycles_per_iteration=5,
    maxsize=8,
    autodiff_backend=:Enzyme,
    optimizer_probability=1.0,
    seed=0,
    deterministic=true,
    verbosity=0,
    save_to_file=false,
)

hall_of_fame = equation_search(
    dataset;
    niterations=2,
    options=options,
    parallelism=:serial,
    runtests=false,
    progress=false,
)

best_loss = minimum(
    member.loss for (member, exists) in zip(hall_of_fame.members, hall_of_fame.exists) if
    exists
)

@test best_loss < 1e-8
