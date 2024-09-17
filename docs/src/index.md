```@meta
CurrentModule = Enzyme
DocTestSetup = quote
    using Enzyme
end
```

# Enzyme

Documentation for [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl), the Julia bindings for [Enzyme](https://github.com/EnzymeAD/enzyme).

Enzyme performs automatic differentiation (AD) of statically analyzable LLVM. It is highly-efficient and its ability to perform AD on optimized code allows Enzyme to meet or exceed the performance of state-of-the-art AD tools.

## Getting started

Enzyme.jl can be installed in the usual way Julia packages are installed:

```
] add Enzyme
```

The Enzyme binary dependencies will be installed automatically via Julia's binary artifact system.

The Enzyme.jl API revolves around the function [`autodiff`](@ref).
For some common operations, Enzyme additionally wraps [`autodiff`](@ref) in several convenience functions; e.g., [`gradient`](@ref) and [`jacobian`](@ref).

The tutorial below covers the basic usage of these functions.
For a complete overview of Enzyme's functionality, see the [API reference](@ref) documentation.
Also see [Implementing pullbacks](@ref) on how to implement back-propagation for functions with non-scalar results.

We will try a few things with the following functions:

```jldoctest rosenbrock
julia> rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2
rosenbrock (generic function with 1 method)

julia> rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rosenbrock_inp (generic function with 1 method)
```

## Reverse mode

The return value of reverse mode [`autodiff`](@ref) is a tuple that contains as a first value
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

## Forward mode

The return value when using `ForwardWithPrimal` is a tuple containing as the first value
the derivative return value and as the second value the original value.

The return value when using `Forward` is a single-element tuple containing the derivative.

In forward mode `Duplicated(x, 0.0)` is equivalent to `Const(x)`,
except that we can perform more optimizations for `Const`.

```jldoctest rosenbrock
julia> autodiff(ForwardWithPrimal, rosenbrock, Const(1.0), Duplicated(3.0, 1.0))
(400.0, 400.0)

julia> autodiff(Forward, rosenbrock, Const(1.0), Duplicated(3.0, 1.0))
(400.0,)

julia> autodiff(ForwardWithPrimal, rosenbrock, Duplicated(1.0, 1.0), Const(3.0))
(-800.0, 400.0)

julia> autodiff(Forward, rosenbrock, Duplicated(1.0, 1.0), Const(3.0))
(-800.0,)
```

Of note, when we seed both arguments at once the tangent return is the sum of both.

```jldoctest rosenbrock
julia> autodiff(ForwardWithPrimal, rosenbrock, Duplicated(1.0, 1.0), Duplicated(3.0, 1.0))
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

### Vector forward mode

We can also use vector mode to calculate both derivatives at once.

```jldoctest rosenbrock
julia> autodiff(ForwardWithPrimal, rosenbrock, BatchDuplicated(1.0, (1.0, 0.0)), BatchDuplicated(3.0, (0.0, 1.0)))
(400.0, (var"1" = -800.0, var"2" = 400.0))

julia> x = [1.0, 3.0]
2-element Vector{Float64}:
 1.0
 3.0

julia> dx_1 = [1.0, 0.0]; dx_2 = [0.0, 1.0];

julia> autodiff(ForwardWithPrimal, rosenbrock_inp, BatchDuplicated(x, (dx_1, dx_2)))
(400.0, (var"1" = -800.0, var"2" = 400.0))
```

## Gradient Convenience functions

!!! note
    While the convenience functions discussed below use [`autodiff`](@ref) internally, they are generally more limited in their functionality. Beyond that, these convenience functions may also come with performance penalties; especially if one makes a closure of a multi-argument function instead of calling the appropriate multi-argument [`autodiff`](@ref) function directly.

Key convenience functions for common derivative computations are [`gradient`](@ref) (and its inplace variant [`gradient!`](@ref)).
Like [`autodiff`](@ref), the mode (forward or reverse) is determined by the first argument.

The functions [`gradient`](@ref) and [`gradient!`](@ref) compute the gradient of function with vector input and scalar return.

Gradient functions take a mode as the first argument. If the mode is `Reverse` or `Forward`, the return type is a tuple of gradients of each argument. 
If the mode is `ReverseWithPrimal` or `ForwardWithPrimal`, the return type is a named tuple containing both the derivatives and the original return result.

```jldoctest rosenbrock
julia> gradient(Reverse, rosenbrock_inp, [1.0, 2.0])
([-400.0, 200.0],)

julia> gradient(ReverseWithPrimal, rosenbrock_inp, [1.0, 2.0])
(derivs=[-400.0, 200.0], val=100.0)

julia> # inplace variant
       dx = [0.0, 0.0];
       gradient!(Reverse, dx, rosenbrock_inp, [1.0, 2.0])
([-400.0, 200.0],)

julia> dx
2-element Vector{Float64}:
 -400.0
  200.0

julia> gradient(Forward, rosenbrock_inp, [1.0, 2.0])
([-400.0, 200.0],)

julia> gradient(ForwardWithPrimal, rosenbrock_inp, [1.0, 2.0])
(derivs=[-400.0, 200.0], val=100.0)

julia> # in forward mode, we can also optionally pass a chunk size
       # to specify the number of derivatives computed simulateneously
       # using vector forward mode
       gradient(Forward, rosenbrock_inp, [1.0, 2.0]; chunk=Val(1))
([-400.0, 200.0],)
```

## Jacobian Convenience functions

The function [`jacobian`](@ref) computes the Jacobian of a function vector input and vector return.
Like [`autodiff`](@ref) and [`gradient`](@ref), the mode (forward or reverse) is determined by the first argument.

Again like [`gradient`](@ref), if the mode is `Reverse` or `Forward`, the return type is a tuple of jacobians of each argument. 
If the mode is `ReverseWithPrimal` or `ForwardWithPrimal`, the return type is a named tuple containing both the derivatives and the original return result.

Both forward and reverse modes take an optional chunk size to compute several derivatives simultaneously using vector mode, and reverse mode optionally takes `n_outs` which describes the shape of the output value.

```jldoctest rosenbrock
julia> foo(x) = [rosenbrock_inp(x), prod(x)];

julia> jacobian(Reverse, foo, [1.0, 2.0]) 
([-400.0  200.0; 2.0    1.0],)

julia> jacobian(ReverseWithPrimal, foo, [1.0, 2.0]) 
(derivs=([-400.0  200.0; 2.0    1.0],), val=[100.0, 2.0])

julia> jacobian(Reverse, foo, [1.0, 2.0]; chunk=Val(2)) 
([-400.0  200.0; 2.0    1.0],)

julia> jacobian(Reverse, foo, [1.0, 2.0]; chunk=Val(2), n_outs=Val((2,)))
([-400.0  200.0; 2.0    1.0],)

julia> jacobian(Forward, foo, [1.0, 2.0])
([-400.0  200.0; 2.0    1.0],)

julia> jacobian(Forward, foo, [1.0, 2.0], chunk=Val(2))
([-400.0  200.0; 2.0    1.0],)
```

## Hessian Vector Product Convenience functions

Enzyme provides convenience functions for second-order derivative computations, like [`hvp`](@ref) to compute Hessian vector products. Mathematically, this computes $H(x) v$, where $H$ is the hessian operator.

Unlike [`autodiff`](@ref) and [`gradient`](@ref), a mode is not specified. Here, Enzyme will choose to perform forward over reverse mode (generally the fastest for this type of operation).

```jldoctest hvp; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
julia> f(x) = sin(x[1] * x[2]);

julia> hvp(f, [2.0, 3.0], [5.0, 2.7])
2-element Vector{Float64}:
 19.69268826373025
 16.201003759768003
```

Enzyme also provides an in-place variant which will store the hessian vector product in a pre-allocated array (this will, however, still allocate another array for storing an intermediate gradient).

```jldoctest hvp2; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
julia> f(x) = sin(x[1] * x[2])
f (generic function with 1 method)

julia> res = Vector{Float64}(undef, 2);

julia> hvp!(res, f, [2.0, 3.0], [5.0, 2.7]);

julia> res
2-element Vector{Float64}:
 19.69268826373025
 16.201003759768003
```

Finally. Enzyme provides a second in-place variant which simultaneously computes both the hessian vector product, and the gradient. This function uses no additional allocation, and is much more efficient than separately computing the hvp and the gradient.

```jldoctest hvp3; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
julia> f(x) = sin(x[1] * x[2]);

julia> res = Vector{Float64}(undef, 2);

julia> grad = Vector{Float64}(undef, 2);

julia> hvp_and_gradient!(res, grad, f, [2.0, 3.0], [5.0, 2.7])

julia> res
2-element Vector{Float64}:
 19.69268826373025
 16.201003759768003

julia> grad
2-element Vector{Float64}:
 2.880510859951098
 1.920340573300732
```
