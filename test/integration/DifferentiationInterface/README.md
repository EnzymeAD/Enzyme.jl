# DI integration tests for Enzyme

This folder contains tests to ensure that changes to Enzyme do not break integration with [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) (DI).

## Relevant source files

The test utilities used here come from the sibling package [DifferentiationInterfaceTest](https://github.com/JuliaDiff/DifferentiationInterface.jl/tree/main/DifferentiationInterfaceTest) (DIT).
Correctness checking itself is implemented in [`src/tests/correctness_eval.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/ed5655a90bf9f3a6092904070d353a9d705ebdc4/DifferentiationInterfaceTest/src/tests/correctness_eval.jl), which is where you will see test errors originating.
Test scenarios are located in [`src/scenarios`](https://github.com/JuliaDiff/DifferentiationInterface.jl/tree/ed5655a90bf9f3a6092904070d353a9d705ebdc4/DifferentiationInterfaceTest/src/scenarios) (especially [`src/scenarios/default.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/ed5655a90bf9f3a6092904070d353a9d705ebdc4/DifferentiationInterfaceTest/src/scenarios/default.jl) and [`src/scenarios/modify.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/ed5655a90bf9f3a6092904070d353a9d705ebdc4/DifferentiationInterfaceTest/src/scenarios/modify.jl)) and in package extensions.
The structure of a `Scenario` is defined in [`src/scenarios/scenario.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl/blob/ed5655a90bf9f3a6092904070d353a9d705ebdc4/DifferentiationInterfaceTest/src/scenarios/scenario.jl).

Scenario generation relies on internals of DifferentiationInterfaceTest, which is why its version is pinned in the `Project.toml`.

## Interpreting test errors

The most common test error you will see looks like

```julia
Correctness: Test Failed at .../src/tests/correctness_eval.jl:...
  Expression: res1_out1_noval ≈ scen.res1
```

Each test scenario `scen` contains a first-order result `res1` and a second-order result `res2`, which are the reference values we compare our autodiff results (i.e. the output of `DI.gradient` or `DI.jacobian`) against.
The suffixes of the left-hand term are defined as follows:

- `_out` for the output of the operator, `_in` for the input if it is in-place
- `1` for the first call to the operator, `2` for the second call (which is used to check that the preparation object has not been altered and can safely be reused)
- `_val` if the operator also returns the value of the function (like `DI.value_and_gradient`), `_noval` otherwise

As you can see, several variants of each operator are tested, so a single bug will give rise to many different errors. In addition, different preparation mechanisms are also cycled through.
The testset summary at the end of the CI log is probably the right place to start hunting down issues.

## What to do if a test fails

Open an issue on the DI repo with a link to the relevant PR or CI log.