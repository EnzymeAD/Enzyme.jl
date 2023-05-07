```@meta
CurrentModule = Enzyme
DocTestSetup = quote
    using Enzyme
end
```

# Enzyme

Documentation for [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), the Julia bindings for [Enzyme](https://github.com/EnzymeAD/enzyme).

Enzyme performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability to perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

Enzyme.jl can be installed in the usual way Julia packages are installed:

```
] add Enzyme
```

The Enzyme binary dependencies will be installed automatically via Julia's binary actifact system.

The Enzyme.jl API revolves around the function [`autodiff`](@ref), see its documentation for details and a usage example. Also see [Implementing pullbacks](@ref) on how to use Enzyme.jl to implement back-propagation for functions with non-scalar results.

## Getting started

```jldoctest rosenbrock
julia> rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2
rosenbrock (generic function with 1 method)

julia> rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rosenbrock_inp (generic function with 1 method)
```

### Reverse mode

The return value of reverse mode is a tuple that contains as a first value
the derivative value of the active inputs and optionally the primal return value.

```jldoctest rosenbrock
julia> autodiff(Reverse, rosenbrock, Active, Active(1.0), Active(2.0))
((-400.0, 200.0),)

julia> autodiff(ReverseWithPrimal, rosenbrock, Active, Active(1.0), Active(2.0))
((-400.0, 200.0), 100.0)
```

```jldoctest rosenbrock
julia> x = [1.0, 2.0]
2-element Vector{Float64}:
 1.0
 2.0

julia> dx = [0.0, 0.0]
2-element Vector{Float64}:
 0.0
 0.0

julia> autodiff(Reverse, rosenbrock_inp, Active, Duplicated(x, dx))
((nothing,),)

julia> dx
2-element Vector{Float64}:
 -400.0
  200.0
```

Both the inplace and "normal" variant return the gradient. The difference is that with
[`Active`](@ref) the gradient is returned and with [`Duplicated`](@ref) the gradient is accumulated in place.

### Forward mode
The return value of forward mode with a `Duplicated` return is a tuple containing as the first value
the primal return value and as the second value the derivative.

In forward mode `Duplicated(x, 0.0)` is equivalent to `Const(x)`,
except that we can perform more optimizations for `Const`.

```jldoctest rosenbrock
julia> autodiff(Forward, rosenbrock, Duplicated, Const(1.0), Duplicated(3.0, 1.0))
(400.0, 400.0)

julia> autodiff(Forward, rosenbrock, Duplicated, Duplicated(1.0, 1.0), Const(3.0))
(400.0, -800.0)
```

Of note, when we seed both arguments at once the tangent return is the sum of both.

```jldoctest rosenbrock
julia> autodiff(Forward, rosenbrock, Duplicated, Duplicated(1.0, 1.0), Duplicated(3.0, 1.0))
(400.0, -400.0)
```

We can also use forward mode with our inplace method.

```jldoctest rosenbrock
julia> x = [1.0, 3.0]
2-element Vector{Float64}:
 1.0
 3.0

julia> dx = [1.0, 1.0]
2-element Vector{Float64}:
 1.0
 1.0

julia> autodiff(Forward, rosenbrock_inp, Duplicated, Duplicated(x, dx))
(400.0, -400.0)
```

Note the seeding through `dx`.

#### Vector forward mode

We can also use vector mode to calculate both derivatives at once.

```jldoctest rosenbrock
julia> autodiff(Forward, rosenbrock, BatchDuplicated, BatchDuplicated(1.0, (1.0, 0.0)), BatchDuplicated(3.0, (0.0, 1.0)))
(400.0, (var"1" = -800.0, var"2" = 400.0))

julia> x = [1.0, 3.0]
2-element Vector{Float64}:
 1.0
 3.0

julia> dx_1 = [1.0, 0.0]; dx_2 = [0.0, 1.0];

julia> autodiff(Forward, rosenbrock_inp, BatchDuplicated, BatchDuplicated(x, (dx_1, dx_2)))
(400.0, (var"1" = -800.0, var"2" = 400.0))
```

## Caveats / Known-issues

### Activity of temporary storage

If you pass in any temporary storage which may be involved in an active computation to a function you want to differentiate, you must also pass in a duplicated temporary storage for use in computing the derivatives. 

```julia
function f(x, tmp, n)
    tmp[1] = 1
    for i in 1:n
        tmp[1] *= x
    end
    tmp[1]
end

# Incorrect [ returns (0.0,) ]
Enzyme.autodiff(f, Active(1.2), Const(Vector{Float64}(undef, 1)), Const(5))

# Correct [ returns (10.367999999999999,) == 1.2^4 * 5 ]
Enzyme.autodiff(f, Active(1.2), Duplicated(Vector{Float64}(undef, 1), Vector{Float64}(undef, 1)), Const(5))
```

### CUDA.jl support

[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is only supported on Julia v1.7.0 and onwards. On v1.6, attempting to differentiate CUDA kernel functions will not use device overloads
correctly and thus returns fundamentally wrong results.

### Sparse Arrays

At the moment there is limited support for sparse linear algebra operations. Sparse arrays may be used, but care must be taken because backing arrays drop zeros in Julia (unless told not to).

```julia
using SparseArrays

a=sparse([2.0])
f(a)=sum(a)

# Incorrect: SparseMatrixCSC drops explicit zeros
# returns 1-element SparseVector{Float64, Int64} with 0 stored entries
da=sparse([0.0])

# Correct: Prevent SparseMatrixCSC from dropping zeros
# returns 1-element SparseVector{Float64, Int64} with 1 stored entry:
#  [1]  =  0.0
da=sparsevec([1], [0.0])

Enzyme.autodiff(Reverse, f, Active, Duplicated(a, da))
@show da
```
