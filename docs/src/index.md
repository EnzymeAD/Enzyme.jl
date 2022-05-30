```@meta
CurrentModule = Enzyme
```

# Enzyme

Documentation for [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), the Julia bindings for [Enzyme](https://github.com/EnzymeAD/enzyme).

Enzyme performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme.jl can be installed in the usual way Julia packages are installed:

```
] add Enzyme
```

The Enzyme binary dependencies will be installed automatically via Julia's binary actifact system.

The Enzyme.jl API revolves around the function [`autodiff`](@ref), see it's documentation for details and a usage example. Also see [Implementing pullbacks](@ref) on how to use Enzyme.jl to implement back-propagation for functions with non-scalar results.


## Caveats / Known-issues

### CUDA.jl support

[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is only support on Julia v1.7.0 and onwards. On 1.6 attempting to differentiate CUDA kernel functions, will not use device overloads
correctly and thus return fundamentally wrong results.
