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

The Enzyme binary dependencies will be installed automatically via Julia's binary artifact system.

The Enzyme.jl API revolves around the function [`autodiff`](@ref).
For some common operations, Enzyme additionally wraps [`autodiff`](@ref) in several convenience functions; e.g., [`gradient`](@ref) and [`jacobian`](@ref).

The tutorial below covers the basic usage of these functions.
For a complete overview of Enzyme's functionality, see the [API](@ref) documentation.
Also see [Implementing pullbacks](@ref) on how to implement back-propagation for functions with non-scalar results.

## Getting started

```jldoctest rosenbrock
julia> rosenbrock(x, y) = (1.0 - x)^2 + 100.0 * (y - x^2)^2
rosenbrock (generic function with 1 method)

julia> rosenbrock_inp(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
rosenbrock_inp (generic function with 1 method)
```

### Reverse mode

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

### Convenience functions

!!! note
    While the convenience functions discussed below use [`autodiff`](@ref) internally, they are generally more limited in their functionality. Beyond that, these convenience functions may also come with performance penalties; especially if one makes a closure of a multi-argument function instead of calling the appropriate multi-argument [`autodiff`](@ref) function directly.

Key convenience functions for common derivative computations are [`gradient`](@ref) (and its inplace variant [`gradient!`](@ref)) and [`jacobian`](@ref).
Like [`autodiff`](@ref), the mode (forward or reverse) is determined by the first argument.

The functions [`gradient`](@ref) and [`gradient!`](@ref) compute the gradient of function with vector input and scalar return.

```jldoctest rosenbrock
julia> gradient(Reverse, rosenbrock_inp, [1.0, 2.0])
2-element Vector{Float64}:
 -400.0
  200.0

julia> # inplace variant
       dx = [0.0, 0.0];
       gradient!(Reverse, dx, rosenbrock_inp, [1.0, 2.0])
2-element Vector{Float64}:
 -400.0
  200.0

julia> dx
2-element Vector{Float64}:
 -400.0
  200.0

julia> gradient(Forward, rosenbrock_inp, [1.0, 2.0])
(-400.0, 200.0)

julia> # in forward mode, we can also optionally pass a chunk size
       # to specify the number of derivatives computed simulateneously
       # using vector forward mode
       chunk_size = Val(2)
       gradient(Forward, rosenbrock_inp, [1.0, 2.0], chunk_size)
(-400.0, 200.0)
```

The function [`jacobian`](@ref) computes the Jacobian of a function vector input and vector return.

```jldoctest rosenbrock
julia> foo(x) = [rosenbrock_inp(x), prod(x)];

julia> output_size = Val(2) # here we have to provide the output size of `foo` since it cannot be statically inferred
       jacobian(Reverse, foo, [1.0, 2.0], output_size) 
2×2 Matrix{Float64}:
 -400.0  200.0
    2.0    1.0

julia> chunk_size = Val(2) # By specifying the optional chunk size argument, we can use vector inverse mode to propogate derivatives of multiple outputs at once.
       jacobian(Reverse, foo, [1.0, 2.0], output_size, chunk_size)
2×2 Matrix{Float64}:
 -400.0  200.0
    2.0    1.0

julia> jacobian(Forward, foo, [1.0, 2.0])
2×2 Matrix{Float64}:
 -400.0  200.0
    2.0    1.0

julia> # Again, the optinal chunk size argument allows us to use vector forward mode
       jacobian(Forward, foo, [1.0, 2.0], chunk_size)
2×2 Matrix{Float64}:
 -400.0  200.0
    2.0    1.0
```

## Caveats / Known-issues

### Activity of temporary storage / Activity Unstable Code

If you pass in any temporary storage which may be involved in an active computation to a function you want to differentiate, you must also pass in a duplicated temporary storage for use in computing the derivatives. For example, consider the following function which uses a temporary buffer to compute the result.

```jldoctest storage
function f(x, tmp, k, n)
    tmp[1] = 1.0
    for i in 1:n
        tmp[k] *= x
    end
    tmp[1]
end

# output

f (generic function with 1 method)
```

Marking the argument for `tmp` as Const (aka non-differentiable) means that Enzyme believes that all variables loaded from or stored into `tmp` must also be non-differentiable, since all values inside a non-differentiable variable must also by definition be non-differentiable.
```jldoctest storage
Enzyme.autodiff(Reverse, f, Active(1.2), Const(Vector{Float64}(undef, 1)), Const(1), Const(5))  # Incorrect

# output

((0.0, nothing, nothing, nothing),)
```

Passing in a dupliacted (e.g. differentiable) variable for `tmp` now leads to the correct answer.

```jldoctest storage
Enzyme.autodiff(Reverse, f, Active(1.2), Duplicated(Vector{Float64}(undef, 1), Vector{Float64}(undef, 1)), Const(1), Const(5))  # Correct (returns 10.367999999999999 == 1.2^4 * 5)

# output

((10.367999999999999, nothing, nothing, nothing),)
```

However, even if we ignore the semantic guarantee provided by marking `tmp` as constant, another issue arises. When computing the original function, intermediate computations (like in `f` above) can use `tmp` for temporary storage. When computing the derivative, Enzyme also needs additional temporary storage space for the corresponding derivative variables as well. If `tmp` is marked as Const, Enzyme does not have any temporary storage space for the derivatives!

Recent versions of Enzyme will attempt to error when they detect these latter types of situations, which we will refer to as `activity unstable`. This term is chosen to mirror the Julia notion of type-unstable code (e.g. where a type is not known at compile time). If an expression is activity unstable, it could either be constant, or active, depending on data not known at compile time. For example, consider the following:

```julia
function g(cond, active_var, constant_var)
  if cond
    return active_var
  else
    return constant_var
end

Enzyme.autodiff(Forward, g, Const(condition), Duplicated(x, dx), Const(y))
```

The returned value here could either by constant or duplicated, depending on the runtime-defined value of `cond`. If `cond` is true, Enzyme simply returns the shadow of `active_var` as the derivative. However, if `cond` is false, there is no derivative shadow for `constant_var` and Enzyme will throw a "Mismatched activity" error. For some simple types, e.g. a float Enzyme can circumvent this issue, for example by returning the float 0. Similarly, for some types like the Symbol type, which are never differentiable, such a shadow value will never be used, and Enzyme can return the original "primal" value as its derivative.  However, for arbitrary data structures, Enzyme presently has no generic mechanism to resolve this.

For example consider a third function:
```julia
function h(cond, active_var, constant_var)
  return [g(cond, active_var, constant_var), g(cond, active_var, constant_var)]
end

Enzyme.autodiff(Forward, h, Const(condition), Duplicated(x, dx), Const(y))
```

Enzyme provides a nice utility `Enzyme.make_zero` which takes a data structure and constructs a deepcopy of the data structure with all of the floats set to zero and non-differentiable types like Symbols set to their primal value. If Enzyme gets into such a "Mismatched activity" situation where it needs to return a differentiable data structure from a constant variable, it could try to resolve this situation by constructing a new shadow data structure, such as with `Enzyme.make_zero`. However, this still can lead to incorrect results. In the case of `h` above, suppose that `active_var` and `consant_var` are both arrays, which are mutable (aka in-place) data types. This means that the return of `h` is going to either be `result = [active_var, active_var]` or `result = [constant_var, constant_var]`.  Thus an update to `result[1][1]` would also change `result[2][1]` since `result[1]` and `result[2]` are the same array. 

If one created a new zero'd copy of each return from `g`, this would mean that the derivative `dresult` would have one copy made for the first element, and a second copy made for the second element. This could lead to incorrect results, and is unfortunately not a general resolution. However, for non-mutable variables (e.g. like floats) or non-differrentiable types (e.g. like Symbols) this problem can never arise.

Instead, Enzyme has a special mode known as "Runtime Activity" which can handle these types of situations. It can come with a minor performance reduction, and is therefore off by default. It can be enabled with `Enzyme.API.runtimeActivity!(true)` right after importing Enzyme for the first time. 

The way Enzyme's runtime activity resolves this issue is to return the original primal variable as the derivative whenever it needs to denote the fact that a variable is a constant. As this issue can only arise with mutable variables, they must be represented in memory via a pointer. All addtional loads and stores will now be modified to first check if the primal pointer is the same as the shadow pointer, and if so, treat it as a constant. Note that this check is not saying that the same arrays contain the same values, but rather the same backing memory represents both the primal and the shadow (e.g. `a === b` or equivalently `pointer(a) == pointer(b)`). 

Enabling runtime activity does therefore, come with a sharp edge, which is that if the computed derivative of a function is mutable, one must also check to see if the primal and shadow represent the same pointer, and if so the true derivative of the function is actually zero.

Generally, the preferred solution to these type of activity unstable codes should be to make your variables all activity-stable (e.g. always containing differentiable memory or always containing non-differentiable memory). However, with care, Enzyme does support "Runtime Activity" as a way to differentiate these programs without having to modify your code.

### Mixed Activity

Sometimes in Reverse mode (but not forward mode), you may see an error `Type T has mixed internal activity types` for some type. This error arises when a variable in a computation cannot be fully represented as either a Duplicated or Active variable.

Active variables are used for immutable variables (like `Float64`), whereas Duplicated variables are used for mutable variables (like `Vector{Float64}`). Speciically, since Active variables are immutable, functions with Active inputs will return the adjoint of that variable. In contrast Duplicated variables will have their derivatives `+=`'d in place.

This error indicates that you have a type, like `Tuple{Float, Vector{Float64}}` that has immutable components and mutable components. Therefore neither Active nor Duplicated can be used for this type.

Internally, by virtue of working at the LLVM level, most Julia types are represented as pointers, and this issue does not tend to arise within code fully differentiated by Enzyme internally. However, when a program needs to interact with Julia API's (e.g. as arguments to a custom rule, a type unstable call, or the outermost function being differentiated), Enzyme must adhere to Julia's notion of immutability and will throw this error rather than risk an incorrect result.

For example, consider the following code, which has a type unstable call to `myfirst`, passing in a mixed type `Tuple{Float64, Vector{Float64}}`.

```julia
@noinline function myfirst(tup::T) where T
    return tup[1]
end

function f(x::Float64)
    vec = [x]
    tup = (x, vec)
    Base.inferencebarrier(myfirst)(tup)::Float64
end

Enzyme.autodiff(Reverse, f, Active, Active(3.1))
```

When this situation arises, it is often easiest to resolve it by adding a level of indirection to ensure the entire variable is mutable. For example, one could enclose this variable in a reference, such as `Ref{Tuple{Float, Vector{Float64}}}`, like as follows.


```julia
@noinline function myfirst_ref(tup_ref::T) where T
    tup = tup_ref[]
    return tup[1]
end

function f2(x::Float64)
    vec = [x]
    tup = (x, vec)
    tup_ref = Ref(tup)
    Base.inferencebarrier(myfirst_ref)(tup_ref)::Float64
end

Enzyme.autodiff(Reverse, f2, Active, Active(3.1))
```

### CUDA.jl support

[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is only supported on Julia v1.7.0 and onwards. On v1.6, attempting to differentiate CUDA kernel functions will not use device overloads
correctly and thus returns fundamentally wrong results.

### Sparse Arrays

At the moment there is limited support for sparse linear algebra operations. Sparse arrays may be used, but care must be taken because backing arrays drop zeros in Julia (unless told not to).

```jldoctest sparse
using SparseArrays
a = sparse([2.0])
da1 = sparse([0.0]) # Incorrect: SparseMatrixCSC drops explicit zeros
Enzyme.autodiff(Reverse, sum, Active, Duplicated(a, da1))
da1

# output

1-element SparseVector{Float64, Int64} with 0 stored entries
```

```jldoctest sparse
da2 = sparsevec([1], [0.0]) # Correct: Prevent SparseMatrixCSC from dropping zeros
Enzyme.autodiff(Reverse, sum, Active, Duplicated(a, da2))
da2

# output

1-element SparseVector{Float64, Int64} with 1 stored entry:
  [1]  =  1.0
```

Sometimes, determining how to perform this zeroing can be complicated.
That is why Enzyme provides a helper function `Enzyme.make_zero` that does this automatically.

### Complex Numbers

Differentiation of a function which returns a complex number is ambiguous, because there are several different gradients which may be desired. Rather than assume a specific of these conventions and potentially result in user error when the resulting derivative is not the desired one, Enzyme forces users to specify the desired convention by returning a real number instead.

Consider the function `f(z) = z*z`. If we were to differentiate this and have real inputs and outputs, the derivative `f'(z)` would be unambiguously `2*z`. However, consider breaking down a complex number down into real and imaginary parts. Suppose now we were to call `f` with the explicit real and imaginary components, `z = x + i y`. This means that `f` is a function that takes an input of two values and returns two values `f(x, y) = u(x, y) + i v(x, y)`. In the case of `z*z` this means that `u(x,y) = x*x-y*y` and `v(x,y) = 2*x*y`.


If we were to look at all first-order derivatives in total, we would end up with a 2x2 matrix (i.e. Jacobian), the derivative of each output wrt each input. Let's try to compute this, first by hand, then with Enzyme.

```
grad u(x, y) = [d/dx u, d/dy u] = [d/dx x*x-y*y, d/dy x*x-y*y] = [2*x, -2*y];
grad v(x, y) = [d/dx v, d/dy v] = [d/dx 2*x*y, d/dy 2*x*y] = [2*y, 2*x];
```

Reverse mode differentiation computes the derivative of all inputs with respect to a single output by propagating the derivative of the return to its inputs. Here, we can explicitly differentiate with respect to the real and imaginary results, respectively, to find this matrix.

```jldoctest complex
f(z) = z * z

# a fixed input to use for testing
z = 3.1 + 2.7im

grad_u = Enzyme.autodiff(Reverse, z->real(f(z)), Active, Active(z))[1][1]
grad_v = Enzyme.autodiff(Reverse, z->imag(f(z)), Active, Active(z))[1][1]

(grad_u, grad_v)
# output
(6.2 - 5.4im, 5.4 + 6.2im)
```

This is somewhat inefficient, since we need to call the forward pass twice, once for the real part, once for the imaginary. We can solve this using batched derivatives in Enzyme, which computes several derivatives for the same function all in one go. To make it work, we're going to need to use split mode, which allows us to provide a custom derivative return value.

```jldoctest complex
fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(f)}, Active, Active{ComplexF64})

# Compute the reverse pass seeded with a differntial return of 1.0 + 0.0im
grad_u = rev(Const(f), Active(z), 1.0 + 0.0im, fwd(Const(f), Active(z))[1])[1][1]
# Compute the reverse pass seeded with a differntial return of 0.0 + 1.0im
grad_v = rev(Const(f), Active(z), 0.0 + 1.0im, fwd(Const(f), Active(z))[1])[1][1]

(grad_u, grad_v)

# output
(6.2 - 5.4im, 5.4 + 6.2im)
```

Now let's make this batched

```jldoctest complex
fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWidth(ReverseSplitNoPrimal, Val(2)), Const{typeof(f)}, Active, Active{ComplexF64})

# Compute the reverse pass seeded with a differential return of 1.0 + 0.0im and 0.0 + 1.0im in one go!
rev(Const(f), Active(z), (1.0 + 0.0im, 0.0 + 1.0im), fwd(Const(f), Active(z))[1])[1][1]

# output
(6.2 - 5.4im, 5.4 + 6.2im)
```

In contrast, Forward mode differentiation computes the derivative of all outputs with respect to a single input by providing a differential input. Thus we need to seed the shadow input with either 1.0 or 1.0im, respectively. This will compute the transpose of the matrix we found earlier.

```
d/dx f(x, y) = d/dx [u(x,y), v(x,y)] = d/dx [x*x-y*y, 2*x*y] = [ 2*x, 2*y];
d/dy f(x, y) = d/dy [u(x,y), v(x,y)] = d/dy [x*x-y*y, 2*x*y] = [-2*y, 2*x];
```

```jldoctest complex
d_dx = Enzyme.autodiff(Forward, f, Duplicated(z, 1.0+0.0im))[1]
d_dy = Enzyme.autodiff(Forward, f, Duplicated(z, 0.0+1.0im))[1]

(d_dx, d_dy)

# output
(6.2 + 5.4im, -5.4 + 6.2im)
```

Again, we can go ahead and batch this.
```jldoctest complex
Enzyme.autodiff(Forward, f, BatchDuplicated(z, (1.0+0.0im, 0.0+1.0im)))[1]

# output
(var"1" = 6.2 + 5.4im, var"2" = -5.4 + 6.2im)
```

Taking Jacobians with respect to the real and imaginary results is fine, but for a complex scalar function it would be really nice to have a single complex derivative. More concretely, in this case when differentiating `z*z`, it would be nice to simply return `2*z`. However, there are four independent variables in the 2x2 jacobian, but only two in a complex number. 

Complex differentiation is often viewed in the lens of directional derivatives. For example, what is the derivative of the function as the real input increases, or as the imaginary input increases. Consider the derivative along the real axis, $\texttt{lim}_{\Delta x \rightarrow 0} \frac{f(x+\Delta x, y)-f(x, y)}{\Delta x}$. This simplifies to $\texttt{lim}_{\Delta x \rightarrow 0} \frac{u(x+\Delta x, y)-u(x, y) + i \left[ v(x+\Delta x, y)-v(x, y)\right]}{\Delta x} = \frac{\partial}{\partial x} u(x,y) + i\frac{\partial}{\partial x} v(x,y)$. This is exactly what we computed by seeding forward mode with a shadow of `1.0 + 0.0im`.

For completeness, we can also consider the derivative along the imaginary axis  $\texttt{lim}_{\Delta y \rightarrow 0} \frac{f(x, y+\Delta y)-f(x, y)}{i\Delta y}$. Here this simplifies to $\texttt{lim}_{u(x, y+\Delta y)-u(x, y) + i \left[ v(x, y+\Delta y)-v(x, y)\right]}{i\Delta y} = -i\frac{\partial}{\partial y} u(x,y) + \frac{\partial}{\partial y} v(x,y)$. Except for the $i$ in the denominator of the limit, this is the same as the result of Forward mode, when seeding x with a shadow of `0.0 + 1.0im`. We can thus compute the derivative along the real axis by multiplying our second Forward mode call by `-im`.

```jldoctest complex
d_real = Enzyme.autodiff(Forward, f, Duplicated(z, 1.0+0.0im))[1]
d_im   = -im * Enzyme.autodiff(Forward, f, Duplicated(z, 0.0+1.0im))[1]

(d_real, d_im)

# output
(6.2 + 5.4im, 6.2 + 5.4im)
```

Interestingly, the derivative of `z*z` is the same when computed in either axis. That is because this function is part of a special class of functions that are invariant to the input direction, called holomorphic. 

Thus, for holomorphic functions, we can simply seed Forward-mode AD with a shadow of one for whatever input we are differenitating. This is nice since seeding the shadow with an input of one is exactly what we'd do for real-valued funtions as well.

Reverse-mode AD, however, is more tricky. This is because holomorphic functions are invariant to the direction of differentiation (aka the derivative inputs), not the direction of the differential return.

However, if a function is holomorphic, the two derivative functions we computed above must be the same. As a result, $\frac{\partial}{\partial x} u = \frac{\partial}{\partial y} v$ and $\frac{\partial}{\partial y} u = -\frac{\partial}{\partial x} v$. 

We saw earlier, that performing reverse-mode AD with a return seed of `1.0 + 0.0im` yielded `[d/dx u, d/dy u]`. Thus, for a holomorphic function, a real-seeded Reverse-mode AD computes `[d/dx u, -d/dx v]`, which is the complex conjugate of the derivative.


```jldoctest complex
conj(grad_u)

# output

6.2 + 5.4im
```

In the case of a scalar-input scalar-output function, that's sufficient. However, most of the time one uses reverse mode, it involves either several inputs or outputs, perhaps via memory. This case requires additional handling to properly sum all the partial derivatives from the use of each input and apply the conjugate operator at only the ones relevant to the differential return.

For simplicity, Enzyme provides a helper utlity `ReverseHolomorphic` which performs Reverse mode properly here, assuming that the function is indeed holomorphic and thus has a well-defined single derivative.

```jldoctest complex
Enzyme.autodiff(ReverseHolomorphic, f, Active, Active(z))[1][1]

# output

6.2 + 5.4im
```

For even non-holomorphic functions, complex analysis allows us to define $\frac{\partial}{\partial z} = \frac{1}{2}\left(\frac{\partial}{\partial x} - i \frac{\partial}{\partial y} \right)$. For non-holomorphic functions, this allows us to compute `d/dz`.  Let's consider `myabs2(z) = z * conj(z)`. We can compute the derivative wrt z of this in Forward mode as follows, which as one would expect results in a result of `conj(z)`:

```jldoctest complex
myabs2(z) = z * conj(z)

dabs2_dx, dabs2_dy = Enzyme.autodiff(Forward, myabs2, BatchDuplicated(z, (1.0 + 0.0im, 0.0 + 1.0im)))[1]
(dabs2_dx - im * dabs2_dy) / 2

# output

3.1 - 2.7im
```

Similarly, we can compute `d/d conj(z) = d/dx + i d/dy`.

```jldoctest complex
(dabs2_dx + im * dabs2_dy) / 2

# output

3.1 + 2.7im
```

Computing this in Reverse mode is more tricky. Let's expand `f` in terms of `u` and `v`. $\frac{\partial}{\partial z} f = \frac12 \left( [u_x + i v_x] - i [u_y + i v_y] \right) = \frac12 \left( [u_x + v_y] + i [v_x - u_y] \right)$. Thus `d/dz = (conj(grad_u) + im * conj(grad_v))/2`.

```jldoctest complex
abs2_fwd, abs2_rev = Enzyme.autodiff_thunk(ReverseSplitWidth(ReverseSplitNoPrimal, Val(2)), Const{typeof(myabs2)}, Active, Active{ComplexF64})

# Compute the reverse pass seeded with a differential return of 1.0 + 0.0im and 0.0 + 1.0im in one go!
gradabs2_u, gradabs2_v = abs2_rev(Const(myabs2), Active(z), (1.0 + 0.0im, 0.0 + 1.0im), abs2_fwd(Const(myabs2), Active(z))[1])[1][1]

(conj(gradabs2_u) + im * conj(gradabs2_v)) / 2

# output

3.1 - 2.7im
```

For `d/d conj(z)`, $\frac12 \left( [u_x + i v_x] + i [u_y + i v_y] \right) = \frac12 \left( [u_x - v_y] + i [v_x + u_y] \right)$. Thus `d/d conj(z) = (grad_u + im * grad_v)/2`.

```jldoctest complex
(gradabs2_u + im * gradabs2_v) / 2

# output

3.1 + 2.7im
```

Note: when writing rules for complex scalar functions, in reverse mode one needs to conjugate the differential return, and similarly the true result will be the conjugate of that value (in essence you can think of reverse-mode AD as working in the conjugate space).