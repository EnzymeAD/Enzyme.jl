```@meta
CurrentModule = Enzyme
```

# Enzyme

Documentation for [Enzyme.jl](https://github.com/wsmoses/Enzyme.jl), the Julia bindings for [Enzyme](https://github.com/wsmoses/enzyme).

Enzyme performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme.jl can be installed in the usual way Julia packages are installed:

```
] add Enzyme
```

The Enzyme binary dependencies will be installed automatically via Julia's binary actifact system.

The Enzyme.jl API revolves around the function [`autodiff`](@ref), see it's documentation for details and a usage example. Also see [Implementing pullbacks](@ref)](@ref) on how to use Enzyme.jl to implement back-propagation for functions with non-scalar results.
