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
(-400.0, 400.0)

julia> autodiff(Forward, rosenbrock, Duplicated(1.0, 1.0), Duplicated(3.0, 1.0))
(-400.0,)
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

julia> autodiff(ForwardWithPrimal, rosenbrock_inp, Duplicated, Duplicated(x, dx))
(-400.0, 400.0)
```

Note the seeding through `dx`.

### Vector forward mode

We can also use vector mode to calculate both derivatives at once.

```jldoctest rosenbrock
julia> autodiff(ForwardWithPrimal, rosenbrock, BatchDuplicated(1.0, (1.0, 0.0)), BatchDuplicated(3.0, (0.0, 1.0)))
((var"1" = -800.0, var"2" = 400.0), 400.0)

julia> x = [1.0, 3.0]
2-element Vector{Float64}:
 1.0
 3.0

julia> dx_1 = [1.0, 0.0]; dx_2 = [0.0, 1.0];

julia> autodiff(ForwardWithPrimal, rosenbrock_inp, BatchDuplicated(x, (dx_1, dx_2)))
((var"1" = -800.0, var"2" = 400.0), 400.0)
```
## Convenience functions (gradient, jacobian, hessian)

### Gradient Convenience functions

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
(derivs = ([-400.0, 200.0],), val = 100.0)

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
(derivs = ([-400.0, 200.0],), val = 100.0)
```

In forward mode, we can also optionally pass a chunk size to specify the number of derivatives computed simulateneously using vector forward mode.
```jldoctest rosenbrock
julia> gradient(Forward, rosenbrock_inp, [1.0, 2.0]; chunk=Val(1))
([-400.0, 200.0],)
```

### Jacobian Convenience functions

The function [`jacobian`](@ref) computes the Jacobian of a function vector input and vector return.
Like [`autodiff`](@ref) and [`gradient`](@ref), the mode (forward or reverse) is determined by the first argument.

Again like [`gradient`](@ref), if the mode is `Reverse` or `Forward`, the return type is a tuple of jacobians of each argument. 
If the mode is `ReverseWithPrimal` or `ForwardWithPrimal`, the return type is a named tuple containing both the derivatives and the original return result.

Both forward and reverse modes take an optional chunk size to compute several derivatives simultaneously using vector mode, and reverse mode optionally takes `n_outs` which describes the shape of the output value.

```jldoctest rosenbrock
julia> foo(x) = [rosenbrock_inp(x), prod(x)];

julia> jacobian(Reverse, foo, [1.0, 2.0]) 
([-400.0 200.0; 2.0 1.0],)

julia> jacobian(ReverseWithPrimal, foo, [1.0, 2.0]) 
(derivs = ([-400.0 200.0; 2.0 1.0],), val = [100.0, 2.0])

julia> jacobian(Reverse, foo, [1.0, 2.0]; chunk=Val(2)) 
([-400.0 200.0; 2.0 1.0],)

julia> jacobian(Reverse, foo, [1.0, 2.0]; chunk=Val(2), n_outs=Val((2,)))
([-400.0 200.0; 2.0 1.0],)

julia> jacobian(Forward, foo, [1.0, 2.0])
([-400.0 200.0; 2.0 1.0],)

julia> jacobian(Forward, foo, [1.0, 2.0], chunk=Val(2))
([-400.0 200.0; 2.0 1.0],)
```

### Hessian Vector Product Convenience functions

Enzyme provides convenience functions for second-order derivative computations, like [`hvp`](@ref) to compute Hessian vector products. Mathematically, this computes $H(x) v$, where $H$ is the hessian operator.

Unlike [`autodiff`](@ref) and [`gradient`](@ref), a mode is not specified. Here, Enzyme will choose to perform forward over reverse mode (generally the fastest for this type of operation).

```jldoctest hvp; filter = r"([0-9]+\.[0-9]{8})[0-9]+" => s"\1***"
julia> f(x) = sin(x[1] * x[2]);

julia> hvp(f, [2.0, 3.0], [5.0, 2.7])
2-element Vector{Float64}:
 19.69268826373025
 16.201003759768003
```

Enzyme also provides an in-place variant which will store the hessian vector product in a pre-allocated array (this will, however, still allocate another array for storing an intermediate gradient).

```jldoctest hvp2; filter = r"([0-9]+\.[0-9]{8})[0-9]+" => s"\1***"
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

```jldoctest hvp3; filter = r"([0-9]+\.[0-9]{8})[0-9]+" => s"\1***"
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

## Defining rules

While Enzyme will automatically generate derivative functions for you, there may be instances in which it is necessary or helpful to define custom derivative rules. Enzyme has three primary ways for defining derivative rules: inactive annotations, [`EnzymeRules.@easy_rule`](@ref) macro definitions, general purpose derivative rules, and importing from `ChainRules`.

### Inactive Annotations

The simplest custom derivative is simply telling Enzyme that a given function does not need to be differentiated. For example, consider computing `det(Unitary Matrix)`. The determinant is always 1 so the derivative is always zero. Without this high level mathematical insight, the default rule Enzyme generates will add up a bunch of numbers that eventually come to zero. Instead of unnecessarily doing this work, we can just tell Enzyme that the derivative is always zero.

In autodiff-parlance we are telling Enzyme that the given result is `inactive` (aka makes no impact on the derivative). This can be done as follows:

```julia

# Our existing function and types
struct UnitaryMatrix
    ...
end

det(::UnitaryMatrix) = ...

using Enzyme.EnzymeRules

EnzymeRules.inactive(::typeof(det), ::UnitaryMatrix) = true
```

Specifically, we define a new overload of the method [`EnzymeRules.inactive`](@ref) where the first argument is the type of the function being marked inactive, and the corresponding arguments match the arguments we want to overload the method for. This enables us, for example, to only mark the determinant of the `UnitaryMatrix` class here as inactive, and not the determinant of a general Matrix.

Enzyme also supports a second way to mark things inactive, where the marker is "less strong" and not guaranteed to apply if other optimizations might otherwise simplify the code first.

```julia
EnzymeRules.inactive_noinl(::typeof(det), ::UnitaryMatrix) = true
```

### [Easy Rules](@id man-easy-rule)

The recommended way for writing rules for most use cases is through the [`EnzymeRules.@easy_rule`](@ref) macro. This macro enables users to write derivatives for any functions which only read from their arguments (e.g. do not overwrite memory), and has numbers, matricies of numbers, or tuples thereof as arguments/result types. 

When writing an [`EnzymeRules.@easy_rule`](@ref) one first describes the function signature one wants the derivative rule to apply to. In each subsequent line, one should write a tuple, where each element of the tuple represents the derivative of the corresponding input argument. In that sense writing an [`EnzymeRules.@easy_rule`](@ref) is equivalent to specifying the Jacobian. Inside of this tuple, one can call arbitrary Julia code.

One can also define certain arguments as not having a derivative via `@Constant`. 

For more information see the [`EnzymeRules.@easy_rule`](@ref) documentation.

```jldoctest easyrules
julia> using Enzyme

julia> f(x, y) = (x*x, cos(y) * x);

julia> Enzyme.EnzymeRules.@easy_rule(f(x,y),
           (2*x, @Constant),        # df1/dx, #df1/dy
           (cos(y), x * sin(y)),    # df2/dx, #df2/dy
       )

julia> g(x, y) = f(x, y)[2];

julia> Enzyme.gradient(Reverse, g, 2.0, 3.0)
(-0.9899924966004454, 0.2822400161197344)
```

Enzyme will automatically generate efficient derivatives for forward mode, reverse mode, batched forward and reverse mode, overwritten data, inactive inputs, and more from the given specification macro.

### General Purpose EnzymeRules

Finally Enzyme supports general-purpose EnzymeRules. For a given function, one can specify arbitrary behavior to occur when differentiting a given function. This is useful if you want to write efficient derivatives for mutating code, are handling funky behavior like GPU/distributed runtime calls, and more.

Like before, Enzyme takes a specification of the function the rule applies to, and passes various configuration data for full user-level customization.

```jldoctest genrules
julia> using Enzyme

julia> mysin(x) = sin(x);

julia> function Enzyme.EnzymeRules.forward(config, ::Const{typeof(mysin)}, ::Type, x)
           # If we don't need the original result, let's avoid computing it (and print)
           if !needs_primal(config)
               println("Avoiding computing sin!")
               return cos(x.val) * x.dval
           else
               println("Still computing sin")
               return Duplicated(sin(x.val), cos(x.val) * x.dval)
           end
       end

julia> function mysquare(x)
           y = mysin(x)
           return y*y
       end;

julia> Enzyme.gradient(Forward, mysin, 2.0)
Avoiding computing sin!
(-0.4161468365471424,)

julia> # Since d/dx sin(x)^2 = 2 * sin(x) * sin'(x)
       # so the original result is still needed
       Enzyme.gradient(Forward, mysquare, 2.0)
Still computing sin
(-0.7568024953079283,)
```

For more information, see [the custom rule docs](@ref custom_rules), [`EnzymeRules.forward`](@ref),  [`EnzymeRules.augmented_primal`](@ref), and [`EnzymeRules.reverse`](@ref).

### Importing ChainRules

Enzyme can also import rules from the `ChainRules` ecosystem. This is often helpful when first getting started, though it will generally be much more efficient to write either an [`EnzymeRules.@easy_rule`](@ref) or general custom rule.

Enzyme can import the forward rule, reverse rule, or both.

```jldoctest chainrule
using Enzyme, ChainRulesCore

f(x) = sin(x)
ChainRulesCore.@scalar_rule f(x)  (cos(x),)

# Import the reverse rule for float32
Enzyme.@import_rrule typeof(f) Float32

# Import the forward rule for float32
Enzyme.@import_frule typeof(f) Float32

# output
```

See the docs on [`Enzyme.@import_frule`](@ref) and [`Enzyme.@import_rrule`](@ref) for more information.
