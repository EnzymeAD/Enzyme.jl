```@meta
CurrentModule = Enzyme
DocTestSetup = quote
    using Enzyme
end
```

# Frequently asked questions

## Implementing pullbacks

In combined reverse mode, Enzyme's [`autodiff`](@ref) function can only handle functions with scalar output (this is not true for split reverse mode, aka `autodiff_thunk`).
To implement pullbacks (back-propagation of gradients/tangents) for array-valued functions, use a mutating function that returns `nothing` and stores its result in one of the arguments, which must be passed wrapped in a [`Duplicated`](@ref).
Regardless of AD mode, this mutating function will be much more efficient anyway than one which allocates the output.

Given a function `mymul!` that performs the equivalent of `R = A * B` for matrices `A` and `B`, and given a gradient (tangent) `∂z_∂R`, we can compute `∂z_∂A` and `∂z_∂B` like this:

```@example pullback
using Enzyme, Random

function mymul!(R, A, B)
    @assert axes(A,2) == axes(B,1)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end
    @inbounds for j in axes(B, 2), i in axes(A, 1)
        @inbounds @simd for k in axes(A,2)
            R[i,j] += A[i,k] * B[k,j]
        end
    end
    nothing
end

Random.seed!(1234)
A = rand(5, 3)
B = rand(3, 7)

R = zeros(size(A,1), size(B,2))
∂z_∂R = rand(size(R)...)  # Some gradient/tangent passed to us
∂z_∂R0 = copyto!(similar(∂z_∂R), ∂z_∂R)  # exact copy for comparison

∂z_∂A = zero(A)
∂z_∂B = zero(B)

Enzyme.autodiff(Reverse, mymul!, Const, Duplicated(R, ∂z_∂R), Duplicated(A, ∂z_∂A), Duplicated(B, ∂z_∂B))
```

Now we have:

```@example pullback
R ≈ A * B            &&
∂z_∂A ≈ ∂z_∂R0 * B'  &&  # equivalent to Zygote.pullback(*, A, B)[2](∂z_∂R)[1]
∂z_∂B ≈ A' * ∂z_∂R0      # equivalent to Zygote.pullback(*, A, B)[2](∂z_∂R)[2]
```

Note that the result of the backpropagation is *added to* `∂z_∂A` and `∂z_∂B`, they act as accumulators for gradient information.

## Identical types in `Duplicated` / Memory Layout

Enzyme checks that `x` and `∂f_∂x` have the same types when constructing objects of type `Duplicated`, `DuplicatedNoNeed`, `BatchDuplicated`, etc.
This is not a mathematical or practical requirement within Enzyme, but rather a guardrail to prevent user error.
The memory locations of the shadow `∂f_∂x` can only be accessed in the derivative function `∂f` if the corresponding memory locations of the variable `x` are accessed by the function `f`.
Imposing that the variable `x` and shadow `∂f_∂x` have the same type is a heuristic way to ensure that they have the same data layout.
This helps prevent some user errors, for instance when the provided shadow cannot be accessed at the relevant memory locations.

In some ways, type equality is too strict: two different types can have the same data layout.
For instance, a vector and a view of a matrix column are arranged identically in memory.
But in other ways it is not strict enough.
Suppose you have a function `f(x) = x[7]`.
If you call `Enzyme.autodiff(Reverse, f, Duplicated(ones(10), ones(1))`, the type check alone will not be sufficient.
Since the original code accesses `x[7]`, the derivative code will try to set `∂f_∂x[7]`.
The length is not encoded in the type, so Julia cannot provide a high-level error before running `autodiff`, and the user may end up with a segfault (or other memory error) when running the generated derivative code.
Another typical example is sparse arrays, for which the sparsity pattern of `x` and `∂f_∂x` should be identical.

To make sure that `∂f_∂x` has the right data layout, create it with `∂f_∂x = Enzyme.make_zero(x)`.

### Circumventing Duplicated Restrictions / Advanced Memory Layout

Advanced users may leverage Enzyme's memory semantics (only touching locations in the shadow that were touched in the primal) for additional performance/memory savings, at the obvious cost of potential safety if used incorrectly.

Consider the following function that loads from offset 47 of a `Ptr`

```jldoctest dup
function f(ptr)
    x = unsafe_load(ptr, 47)
    x * x
end

ptr = Base.reinterpret(Ptr{Float64}, Libc.malloc(100*sizeof(Float64)))
unsafe_store!(ptr, 3.14, 47)

f(ptr)

# output
9.8596
```

The recommended (and guaranteed sound) way to differentiate this is to pass in a shadow pointer that is congruent with the primal. That is to say, its length (and recursively for any sub types) are equivalent to the primal.

```jldoctest dup
ptr = Base.reinterpret(Ptr{Float64}, Libc.malloc(100*sizeof(Float64)))
unsafe_store!(ptr, 3.14, 47)
dptr = Base.reinterpret(Ptr{Float64}, Libc.calloc(100*sizeof(Float64), 1))

autodiff(Reverse, f, Duplicated(ptr, dptr))

unsafe_load(dptr, 47)

# output
6.28
```

However, since we know the original function only reads from one float64, we could choose to only allocate a single float64 for the shadow, as long as we ensure that loading from offset 47 (the only location accessed) is in bounds.

```jldoctest dup
ptr = Base.reinterpret(Ptr{Float64}, Libc.malloc(100*sizeof(Float64)))
unsafe_store!(ptr, 3.14, 47)
dptr = Base.reinterpret(Ptr{Float64}, Libc.calloc(sizeof(Float64), 1))

# offset the pointer to have unsafe_load(dptr, 47) access the 0th byte of dptr
# since julia one indexes we subtract 46 * sizeof(Float64) here
autodiff(Reverse, f, Duplicated(ptr, dptr - 46 * sizeof(Float64)))

# represents the derivative of the 47'th elem of ptr, 
unsafe_load(dptr)

# output
6.28
```

However, this style of optimization is not specific to Enzyme, or AD, as one could have done the same thing on the primal code where it only passed in one float. The difference, here however, is that performing these memory-layout tricks safely in Enzyme requires understanding the access patterns of the generated derivative code -- like discussed here.


```jldoctest dup
ptr = Base.reinterpret(Ptr{Float64}, Libc.calloc(sizeof(Float64), 1))
unsafe_store!(ptr, 3.14)
# offset the pointer to have unsafe_load(ptr, 47) access the 0th byte of dptr
# again since julia one indexes we subtract 46 * sizeof(Float64) here
f(ptr - 46 * sizeof(Float64))

# output
9.8596
```

## CUDA support

[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) is only supported on Julia v1.7.0 and onwards. On v1.6, attempting to differentiate CUDA kernel functions will not use device overloads correctly and thus returns fundamentally wrong results.

Specifically, differentiating within device kernels is supported. See our [cuda tests](https://github.com/EnzymeAD/Enzyme.jl/blob/main/test/cuda.jl) for some examples. 

Differentiating through a heterogeneous (e.g. combined host and device) code presently requires defining a custom derivative that tells Enzyme that differentiating an `@cuda` call is done by performing `@cuda` of its generated derivative. For an example of this in Enzyme-C++ see [here](https://enzyme.mit.edu/getting_started/CUDAGuide/). Automating this for a better experience for CUDA.jl requires an update to [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl/pull/1869/files), and is now available for Kernel Abstractions.

Differentiating host-side code when accesses device memory (e.g. `sum(CuArray)`) is not yet supported, but in progress.

## Linear Algebra

Enzyme supports presently some, but not all of Julia's linear algebra library. This is because some of Julia's linear algebra library is not pure Julia code and calls external functions such as BLAS, LAPACK, CuBLAS, SuiteSparse, etc.

For all BLAS functions, Enzyme will generate a correct derivative function. If it is a `gemm` (matmul), `gemv` (matvec), `dot` (dot product), `axpy` (vector add and scale), and a few others, Enzyme will generate a fast derivative using another corresponding BLAS call.  For other BLAS functions, Enzyme will presently emit a warning `Fallback BLAS [functionname]` that indicates that Enzyme will differentiate this function by differentiating a serial implementation of BLAS. This will still work for all BLAS codes, but may be slower on a parallel platform.

Other libraries do not yet have derivatives (either fast or fallback) implemented within Enzyme. Supporting these is not a fundamental limitation, but requires implementing a rule in Enzyme describing how to differentiate them. Contributions welcome!

## Sparse arrays

Differentiating code using sparse arrays is supported, but care must be taken because backing arrays drop zeros in Julia (unless told not to).

```jldoctest sparse
julia> using SparseArrays

julia> a = sparse([2.0])
1-element SparseVector{Float64, Int64} with 1 stored entry:
  [1]  =  2.0

julia> da1 = sparse([0.0]); # Incorrect: SparseMatrixCSC drops explicit zeros

julia> Enzyme.autodiff(Reverse, sum, Active, Duplicated(a, da1));

julia> da1
1-element SparseVector{Float64, Int64} with 0 stored entries

julia> da2 = sparsevec([1], [0.0]); # Correct: Prevent SparseMatrixCSC from dropping zeros

julia> Enzyme.autodiff(Reverse, sum, Active, Duplicated(a, da2));

julia> da2
1-element SparseVector{Float64, Int64} with 1 stored entry:
  [1]  =  1.0
```

Sometimes, determining how to perform this zeroing can be complicated.
That is why Enzyme provides a helper function `Enzyme.make_zero` that does this automatically.

```jldoctest sparse
julia> Enzyme.make_zero(a)
1-element SparseVector{Float64, Int64} with 1 stored entry:
  [1]  =  0.0

julia> Enzyme.gradient(Reverse, sum, a)[1] # This calls make_zero(a)
1-element SparseVector{Float64, Int64} with 1 stored entry:
  [1]  =  1.0
```

Some Julia libraries sparse linear algebra libraries call out to external C code like SuiteSparse which we don't presently implement derivatives for (we have some but have yet to complete all). If that case happens, Enzyme will throw a "no derivative found" error at the callsite of that function. This isn't a fundamental limitation, and is easily resolvable by writing a custom rule or internal Enzyme support. Help is certainly welcome :).

### Advanced Sparse arrays

Essentially the way Enzyme represents all data structures, including sparse data structures, is to have the shadow (aka derivative) memory be the same memory layout as the primal. Suppose you have an input data structure `x`. The derivative of `x` at byte offset 12 will be stored in the shadow `dx` at byte offset 12, etc.

This has the nice property that the storage for the derivative, including all intermediate computations, is the same as that of the primal (ignoring caching requirements for reverse mode).

It also means that any arbitrary data structure can be differentiated with respect to, and we don’t have any special handling required to register every data structure one could create.

This representation does have some caveats (e.g. see Identical types in `Duplicated` above).

Sparse data structures are often represented with say a `Vector{Float64}` that holds the actual elements, and a `Vector{Int}` that specifies the index `n` the backing array that corresponds to the true location in the overall vector.

We have no explicit special cases for sparse Data structures, so the layout semantics mentioned above is indeed what Enzyme uses.

Thus the derivative of a sparse array is to have a second backing array of the same size, and another `Vector{Int}` (of the same offsets).

As a concrete example, suppose we have the following: `x = { 3 : 2.7, 10 : 3.14 }`. In other words, a sparse data structure with two elements, one at index 3, another at index 10. This could be represented with the backing array being `[2.7, 3.14]` and the index array being `[3, 10]`.

A correctly zero-initialized shadow data structure would be to have a backing array of size 2 with zero’s, and an index array again being `[3, 10]`.

In this form the second element of the derivative backing array is used to store/represent the derivative of the second element of the original backing array, in other words the derivative at index 10.

Like mentioned above, a caveat here is that this correctly zero’d initializer is not the default produced by `sparse([0.0])` as this drops the zero elements from the backing array. `Enzyme.make_zero` recursively goes through your data structure to generate the shadows of the correct structure (and in this case would make a new backing array of appropriate size). The `make_zero` function is not special cased to sparsity, but just comes out as a result.

Internally, when differentiating a function this is the type of data structure that Enzyme builds and uses to represent variables. However, at the Julia level that there’s a bit of a sharp edge.

Consider a function `f(A(x))` where `x` is a scalar or dense input, `A(x)` returns a sparse array, and `f(A(x))` returns a scalar loss. 

The derivative that Enzyme creates for `A(x)` would create both the backing/index arrays for the original result A, as well as the equal sized backing/index arrays for the derivative.

For any program which generates sparse data structures internally, like the total program `f(A(x))`, this will always give you the answer you expect. Moreover, the memory requirements of the derivative will be the same as the primal (other AD tools will blow up the memory usage and construct dense derivatives where the primal was sparse).

The added caveat, however, comes when you differentiate a top level function that has a sparse array input. For example, consider the sparse `sum` function which adds up all elements. While in one definition, this function represents summing up all elements of the virtual sparse array (including the zero's which are never materialized), in a more literal sense this `sum` function will only add elements 3 and 10 of the input sparse array -- the only two nonzero elements -- or equivalently the sum of the whole backing array. Correspondingly Enzyme will update the sparse shadow data structure to mark both elements 3 and 10 as having a derivative of 1 (or more literally set all the elements of the backing array to derivative 1). These are the only variables that Enzyme needs to update, since they are the only variables read (and thus the only ones which have a non-zero derivative). Thus any function which may call this method and compose via the chain rule will only ever read the derivative of these two elements. This is why this memory-safe representation composes within Enzyme, though may produce counter-intuitive reuslts at the top level.

If the name we gave to this data structure wasn’t "SparseArray" but instead "MyStruct" this is precisely the answer we would have desired. However, since the sparse array printer prints zeros for elements outside of the sparse backing array, this isn’t what one would expect. Making a nicer user conversion from Enzyme’s form of differential data structures, to the more natural "Julia" form where there is a semantic mismatch between what Julia intends a data structure to mean by name, and what is being discussed [here](https://github.com/EnzymeAD/Enzyme.jl/issues/1334).

The benefit of this representation is that : (1) all of our rules compose correctly (you get the correct answer for `f(A(x)`), (2) without the need to special case any sparse code, and (3) with the same memory/performance expectations as the original code.

## Activity of temporary storage

If you pass in any temporary storage which may be involved in an active computation to a function you want to differentiate, you must also pass in a duplicated temporary storage for use in computing the derivatives. For example, consider the following function which uses a temporary buffer to compute the result.

```jldoctest storage
julia> function f(x, tmp, k, n)
           tmp[1] = 1.0
           for i in 1:n
               tmp[k] *= x
           end
           tmp[1]
       end
f (generic function with 1 method)
```

Marking the argument for `tmp` as `Const` (aka non-differentiable) means that Enzyme believes that all variables loaded from or stored into `tmp` must also be non-differentiable, since all values inside a non-differentiable variable must also by definition be non-differentiable.
```jldoctest storage
julia> Enzyme.autodiff(Reverse, f, Active(1.2), Const(Vector{Float64}(undef, 1)), Const(1), Const(5))  # Incorrect
((0.0, nothing, nothing, nothing),)
```

Passing in a duplicated (e.g. differentiable) variable for `tmp` now leads to the correct answer.

```jldoctest storage
julia> Enzyme.autodiff(Reverse, f, Active(1.2), Duplicated(Vector{Float64}(undef, 1), zeros(1)), Const(1), Const(5))  # Correct (returns 10.367999999999999 == 1.2^4 * 5)
((10.367999999999999, nothing, nothing, nothing),)
```

## [Runtime Activity](@id faq-runtime-activity)

When computing the derivative of mutable variables, Enzyme also needs additional temporary storage space for the corresponding derivative variables. If an argument `tmp` is marked as `Const`, Enzyme does not have any temporary storage space for the derivatives!

Enzyme will error when they detect these latter types of situations, which we will refer to as `activity unstable`. This term is chosen to mirror the Julia notion of type-unstable code (e.g. where a type is not known at compile time). If an expression is activity unstable, it could either be constant, or active, depending on data not known at compile time. For example, consider the following:

```@example runtime
using Enzyme

function g(cond, active_var, constant_var)
    if cond
        return active_var
    else
        return constant_var
    end
end

x, dx = [1.0], [2.0];
y = [3.0];
condition = false;  # return the constant variable
try  #hide
Enzyme.autodiff(Forward, g, Const(condition), Duplicated(x, dx), Const(y))
catch err; showerror(stderr, err); end  #hide
```

The returned value here could either by constant or duplicated, depending on the runtime-defined value of `cond`. If `cond` is true, Enzyme simply returns the shadow of `active_var` as the derivative. However, if `cond` is false, there is no derivative shadow for `constant_var` and Enzyme will throw a `EnzymeRuntimeActivityError` error. For some simple types, e.g. a float Enzyme can circumvent this issue, for example by returning the float 0. Similarly, for some types like the Symbol type, which are never differentiable, such a shadow value will never be used, and Enzyme can return the original "primal" value as its derivative.  However, for arbitrary data structures, Enzyme presently has no generic mechanism to resolve this.

For example consider a third function:

```@example runtime
function h(cond, active_var, constant_var)
  return [g(cond, active_var, constant_var), g(cond, active_var, constant_var)]
end

try  #hide
Enzyme.autodiff(Forward, h, Const(condition), Duplicated(x, dx), Const(y))
catch err; showerror(stderr, err); end  #hide
```

Enzyme provides a nice utility `Enzyme.make_zero` which takes a data structure and constructs a deepcopy of the data structure with all of the floats set to zero and non-differentiable types like Symbols set to their primal value. If Enzyme gets into such a "Mismatched activity" situation where it needs to return a differentiable data structure from a constant variable, it could try to resolve this situation by constructing a new shadow data structure, such as with `Enzyme.make_zero`. However, this still can lead to incorrect results. In the case of `h` above, suppose that `active_var` and `constant_var` are both arrays, which are mutable (aka in-place) data types. This means that the return of `h` is going to either be `result = [active_var, active_var]` or `result = [constant_var, constant_var]`.  Thus an update to `result[1][1]` would also change `result[2][1]` since `result[1]` and `result[2]` are the same array. 

If one created a new zero'd copy of each return from `g`, this would mean that the derivative `dresult` would have one copy made for the first element, and a second copy made for the second element. This could lead to incorrect results, and is unfortunately not a general resolution. However, for non-mutable variables (e.g. like floats) or non-differrentiable types (e.g. like Symbols) this problem can never arise.

Instead, Enzyme has a special mode known as "Runtime Activity" which can handle these types of situations. It can come with a minor performance reduction, and is therefore off by default. It can be enabled with by setting runtime activity to true in a desired differentiation mode.

The way Enzyme's runtime activity resolves this issue is to return the original primal variable as the derivative whenever it needs to denote the fact that a variable is a constant. As this issue can only arise with mutable variables, they must be represented in memory via a pointer. All addtional loads and stores will now be modified to first check if the primal pointer is the same as the shadow pointer, and if so, treat it as a constant. Note that this check is not saying that the same arrays contain the same values, but rather the same backing memory represents both the primal and the shadow (e.g. `a === b` or equivalently `pointer(a) == pointer(b)`).

Enabling runtime activity does therefore, come with a sharp edge, which is that if the computed derivative of a function is mutable, one must also check to see if the primal and shadow represent the same pointer, and if so the true derivative of the function is actually zero.

Generally, the preferred solution to these type of activity unstable codes should be to make your variables all activity-stable (e.g. always containing differentiable memory or always containing non-differentiable memory). However, with care, Enzyme does support "Runtime Activity" as a way to differentiate these programs without having to modify your code. One can enable runtime activity for your code by changing the mode, such as

```@example runtime
dout, out = Enzyme.autodiff(Enzyme.set_runtime_activity(ForwardWithPrimal), g, Const(condition), Duplicated(x, dx), Const(y))
```

However, care must be taken to check derivative aliasing afterwards:

```@example runtime
dout === out  # if true and pointer-like, the actual derivative is zero 
```

## Mixed activity

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

## [Strong Zero](@id faq-strong-zero)

By default, Enzyme (and essentially all other) AD tools may assume that intermediate values are finite and not nan. This is beneficial for performance, but also can lead to some non-intuitive behaviors.

Consider the following code snippet:

```jldoctest strongzero
julia> function f(x)
           y = 1.0 / x
           return min(1.0, y)
       end;

julia> f(0.0)
1.0
```

When evaluated at `x=0.0`, `y=Inf`. However due to the `min` the result will be a finite value. 

Let's consider what happens for the derivative by applying Enzyme. 

```jldoctest strongzero
julia> Enzyme.gradient(Reverse, f, 0.0)
(NaN,)
```

We find the derivative is NaN, which is perhaps less than helpful. Let's look at why this happens by writing out the chain rule ourselves.


```jldoctest strongzero
function grad_f(x)
    y = 1.0 / x
    z = min(1.0, y)

    # Initialization of the shadow derivatives (which will be +='d into)
    dx = dy = dz = 0.0

    # derivative of return
    dz = 1.0

    # derivative of min(1.0, y)
    dy += (1.0 > y) ? dz : 0.0
    dz = 0.0

    # derivative of 1.0 / x
    dx += dy * - 1.0 / x^2 
    dy = 0.0

    return dx
end

grad_f(0.0)

# output
NaN
```

The problematic point for us that creates the NaN is the derivative rule for `1.0 / x`. In particular it computes `dy * - 1.0 / x^2 `. The right hand side `-1.0 / x^2` is of course infinite (in this case `-Inf`). However, `dy = 0.0` since we computed that the term was not used in the final returned expression. Multiplying these together indeed produces a `NaN`. The problem here, is that in this case the fact that we didn't use the value of `y` in a differentiable way should bind more **tightly** -- in other words, if the partial derivative with respect to y is zero (aka `dy == 0.0`), the partial derivative with respect to all operands of y should be zero.

This is exactly what the strong zero mode of Enzyme does. It tells the derivative rules to perform an additional runtime check that the derivative to propagate is non-zero. It comes with a nontrivial additional compute cost as a result, but ensures correctness in cases like this where intermediate values of a computation (even if unused) may be infinite or `NaN`.

One can use this from Enzyme.jl as follows and get the intended result:

```jldoctest strongzero
julia> Enzyme.gradient(set_strong_zero(Reverse), f, 0.0)
(-0.0,)
```

## Complex numbers

Differentiation of a function which returns a complex number is ambiguous, because there are several different gradients which may be desired. Rather than assume a specific of these conventions and potentially result in user error when the resulting derivative is not the desired one, Enzyme forces users to specify the desired convention by returning a real number instead.

Consider the function `f(z) = z*z`. If we were to differentiate this and have real inputs and outputs, the derivative `f'(z)` would be unambiguously `2*z`. However, consider breaking down a complex number down into real and imaginary parts. Suppose now we were to call `f` with the explicit real and imaginary components, `z = x + i y`. This means that `f` is a function that takes an input of two values and returns two values `f(x, y) = u(x, y) + i v(x, y)`. In the case of `z*z` this means that `u(x,y) = x*x-y*y` and `v(x,y) = 2*x*y`.


If we were to look at all first-order derivatives in total, we would end up with a 2x2 matrix (i.e. Jacobian), the derivative of each output wrt each input. Let's try to compute this, first by hand, then with Enzyme.

```
grad u(x, y) = [d/dx u, d/dy u] = [d/dx x*x-y*y, d/dy x*x-y*y] = [2*x, -2*y];
grad v(x, y) = [d/dx v, d/dy v] = [d/dx 2*x*y, d/dy 2*x*y] = [2*y, 2*x];
```

Reverse mode differentiation computes the derivative of all inputs with respect to a single output by propagating the derivative of the return to its inputs. Here, we can explicitly differentiate with respect to the real and imaginary results, respectively, to find this matrix.

```jldoctest complex
julia> f(z) = z * z;

julia> z = 3.1 + 2.7im; # a fixed input to use for testing

julia> grad_u = Enzyme.autodiff(Reverse, z->real(f(z)), Active, Active(z))[1][1]
6.2 - 5.4im

julia> grad_v = Enzyme.autodiff(Reverse, z->imag(f(z)), Active, Active(z))[1][1]
5.4 + 6.2im
```

This is somewhat inefficient, since we need to call the forward pass twice, once for the real part, once for the imaginary. We can solve this using batched derivatives in Enzyme, which computes several derivatives for the same function all in one go. To make it work, we're going to need to use split mode, which allows us to provide a custom derivative return value.

```jldoctest complex
julia> fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(f)}, Active, Active{ComplexF64});

julia> # Compute the reverse pass seeded with a differential return of 1.0 + 0.0im
       grad_u = rev(Const(f), Active(z), 1.0 + 0.0im, fwd(Const(f), Active(z))[1])[1][1]
6.2 - 5.4im

julia> # Compute the reverse pass seeded with a differential return of 0.0 + 1.0im
       grad_v = rev(Const(f), Active(z), 0.0 + 1.0im, fwd(Const(f), Active(z))[1])[1][1]
5.4 + 6.2im
```

Now let's make this batched

```jldoctest complex
fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWidth(ReverseSplitNoPrimal, Val(2)), Const{typeof(f)}, Active, Active{ComplexF64})

# Compute the reverse pass seeded with a differential return of 1.0 + 0.0im and 0.0 + 1.0im in one go!
rev(Const(f), Active(z), (1.0 + 0.0im, 0.0 + 1.0im), fwd(Const(f), Active(z))[1])[1][1]

# output
(6.2 - 5.4im, 5.4 + 6.2im)
```

In contrast, Forward mode differentiation computes the derivative of all outputs with respect to a single input by providing a differential input. Thus we need to seed the shadow input with either `1.0` or `1.0im`, respectively. This will compute the transpose of the matrix we found earlier.

```
d/dx f(x, y) = d/dx [u(x,y), v(x,y)] = d/dx [x*x-y*y, 2*x*y] = [ 2*x, 2*y];
d/dy f(x, y) = d/dy [u(x,y), v(x,y)] = d/dy [x*x-y*y, 2*x*y] = [-2*y, 2*x];
```

```jldoctest complex
julia> d_dx = Enzyme.autodiff(Forward, f, Duplicated(z, 1.0+0.0im))[1]
6.2 + 5.4im

julia> d_dy = Enzyme.autodiff(Forward, f, Duplicated(z, 0.0+1.0im))[1]
-5.4 + 6.2im
```

Again, we can go ahead and batch this.
```jldoctest complex
julia> Enzyme.autodiff(Forward, f, BatchDuplicated(z, (1.0+0.0im, 0.0+1.0im)))[1]
(var"1" = 6.2 + 5.4im, var"2" = -5.4 + 6.2im)
```

Taking Jacobians with respect to the real and imaginary results is fine, but for a complex scalar function it would be really nice to have a single complex derivative. More concretely, in this case when differentiating `z*z`, it would be nice to simply return `2*z`. However, there are four independent variables in the 2x2 jacobian, but only two in a complex number. 

Complex differentiation is often viewed in the lens of directional derivatives. For example, what is the derivative of the function as the real input increases, or as the imaginary input increases. Consider the derivative along the real axis, $\texttt{lim}_{\Delta x \rightarrow 0} \frac{f(x+\Delta x, y)-f(x, y)}{\Delta x}$. This simplifies to $\texttt{lim}_{\Delta x \rightarrow 0} \frac{u(x+\Delta x, y)-u(x, y) + i \left[ v(x+\Delta x, y)-v(x, y)\right]}{\Delta x} = \frac{\partial}{\partial x} u(x,y) + i\frac{\partial}{\partial x} v(x,y)$. This is exactly what we computed by seeding forward mode with a shadow of `1.0 + 0.0im`.

For completeness, we can also consider the derivative along the imaginary axis  $\texttt{lim}_{\Delta y \rightarrow 0} \frac{f(x, y+\Delta y)-f(x, y)}{i\Delta y}$. Here this simplifies to $\texttt{lim}_{u(x, y+\Delta y)-u(x, y) + i \left[ v(x, y+\Delta y)-v(x, y)\right]}{i\Delta y} = -i\frac{\partial}{\partial y} u(x,y) + \frac{\partial}{\partial y} v(x,y)$. Except for the $i$ in the denominator of the limit, this is the same as the result of Forward mode, when seeding x with a shadow of `0.0 + 1.0im`. We can thus compute the derivative along the real axis by multiplying our second Forward mode call by `-im`.

```jldoctest complex
julia> d_real = Enzyme.autodiff(Forward, f, Duplicated(z, 1.0+0.0im))[1]
6.2 + 5.4im

julia> d_im   = -im * Enzyme.autodiff(Forward, f, Duplicated(z, 0.0+1.0im))[1]
6.2 + 5.4im
```

Interestingly, the derivative of `z*z` is the same when computed in either axis. That is because this function is part of a special class of functions that are invariant to the input direction, called holomorphic. 

Thus, for holomorphic functions, we can simply seed Forward-mode AD with a shadow of one for whatever input we are differenitating. This is nice since seeding the shadow with an input of one is exactly what we'd do for real-valued funtions as well.

Reverse-mode AD, however, is more tricky. This is because holomorphic functions are invariant to the direction of differentiation (aka the derivative inputs), not the direction of the differential return.

However, if a function is holomorphic, the two derivative functions we computed above must be the same. As a result, $\frac{\partial}{\partial x} u = \frac{\partial}{\partial y} v$ and $\frac{\partial}{\partial y} u = -\frac{\partial}{\partial x} v$. 

We saw earlier, that performing reverse-mode AD with a return seed of `1.0 + 0.0im` yielded `[d/dx u, d/dy u]`. Thus, for a holomorphic function, a real-seeded Reverse-mode AD computes `[d/dx u, -d/dx v]`, which is the complex conjugate of the derivative.


```jldoctest complex
julia> conj(grad_u)
6.2 + 5.4im
```

In the case of a scalar-input scalar-output function, that's sufficient. However, most of the time one uses reverse mode, it involves either several inputs or outputs, perhaps via memory. This case requires additional handling to properly sum all the partial derivatives from the use of each input and apply the conjugate operator at only the ones relevant to the differential return.

For simplicity, Enzyme provides a helper utlity `ReverseHolomorphic` which performs Reverse mode properly here, assuming that the function is indeed holomorphic and thus has a well-defined single derivative.

```jldoctest complex
julia> Enzyme.autodiff(ReverseHolomorphic, f, Active, Active(z))[1][1]
6.2 + 5.4im
```

For even non-holomorphic functions, complex analysis allows us to define $\frac{\partial}{\partial z} = \frac{1}{2}\left(\frac{\partial}{\partial x} - i \frac{\partial}{\partial y} \right)$. For non-holomorphic functions, this allows us to compute `d/dz`.  Let's consider `myabs2(z) = z * conj(z)`. We can compute the derivative wrt z of this in Forward mode as follows, which as one would expect results in a result of `conj(z)`:

```jldoctest complex
julia> myabs2(z) = z * conj(z);

julia> dabs2_dx, dabs2_dy = Enzyme.autodiff(Forward, myabs2, BatchDuplicated(z, (1.0 + 0.0im, 0.0 + 1.0im)))[1]
(var"1" = 6.2 + 0.0im, var"2" = 5.4 + 0.0im)

julia> (dabs2_dx - im * dabs2_dy) / 2
3.1 - 2.7im
```

Similarly, we can compute `d/d conj(z) = d/dx + i d/dy`.

```jldoctest complex
julia> (dabs2_dx + im * dabs2_dy) / 2
3.1 + 2.7im
```

Computing this in Reverse mode is more tricky. Let's expand `f` in terms of `u` and `v`. $\frac{\partial}{\partial z} f = \frac12 \left( [u_x + i v_x] - i [u_y + i v_y] \right) = \frac12 \left( [u_x + v_y] + i [v_x - u_y] \right)$. Thus `d/dz = (conj(grad_u) + im * conj(grad_v))/2`.

```jldoctest complex
julia> abs2_fwd, abs2_rev = Enzyme.autodiff_thunk(
           ReverseSplitWidth(ReverseSplitNoPrimal, Val(2)),
           Const{typeof(myabs2)},
           Active,
           Active{ComplexF64}
       );

julia> # Compute the reverse pass seeded with a differential return of 1.0 + 0.0im and 0.0 + 1.0im in one go!
       gradabs2_u, gradabs2_v = abs2_rev(
           Const(myabs2),
           Active(z),
           (1.0 + 0.0im, 0.0 + 1.0im),
           abs2_fwd(Const(myabs2), Active(z))[1]
       )[1][1]
(6.2 + 5.4im, 0.0 + 0.0im)

julia> (conj(gradabs2_u) + im * conj(gradabs2_v)) / 2
3.1 - 2.7im
```

For `d/d conj(z)`, $\frac12 \left( [u_x + i v_x] + i [u_y + i v_y] \right) = \frac12 \left( [u_x - v_y] + i [v_x + u_y] \right)$. Thus `d/d conj(z) = (grad_u + im * grad_v)/2`.

```jldoctest complex
julia> (gradabs2_u + im * gradabs2_v) / 2
3.1 + 2.7im
```

Note: when writing rules for complex scalar functions, in reverse mode one needs to conjugate the differential return, and similarly the true result will be the conjugate of that value (in essence you can think of reverse-mode AD as working in the conjugate space).

## What types are differentiable?

Enzyme tracks differentiable dataflow through values. Specifically Enzyme tracks differentiable data in base types like `Float32`, `Float64`, `Float16`, `BFloat16`, etc.

As a simple example:

```jldoctest types
julia> f(x) = x * x;

julia> Enzyme.autodiff(Forward, f, Duplicated(3.0, 1.0))
(6.0,)
```

Enzyme also tracks differentiable data in any types containing these base types (e.g. floats). For example, consider a struct or array containing floats.

```jldoctest types
julia> struct Pair
           lhs::Float64
           rhs::Float64
       end

julia> f_pair(x) = x.lhs * x.rhs;

julia> Enzyme.autodiff(Forward, f_pair, Duplicated(Pair(3.0, 2.0), Pair(1.0, 0.0)))
(2.0,)
```

```jldoctest types
julia> Enzyme.autodiff(Forward, sum, Duplicated([1.0, 2.0, 3.0], [5.0, 0.0, 100.0]))
(105.0,)
```

A differentiable data structure can be arbitrarily complex, such as a linked list.


```jldoctest types

struct LList
    prev::Union{Nothing, LList}
    value::Float64
end

function make_list(x::Vector)
   result = nothing
   for value in reverse(x)
      result = LList(result, value)
   end
   return result
end

function list_sum(list::Union{Nothing, LList})
   result = 0.0
   while list != nothing
     result += list.value
     list = list.prev
   end
   return result
end

list = make_list([1.0, 2.0, 3.0])
dlist = make_list([5.0, 0.0, 100.0])

Enzyme.autodiff(Forward, list_sum, Duplicated(list, dlist))

# output

(105.0,)
```

Presently Enzyme only considers floats as base types. As a result, Enzyme does not support differentiating data contained in `Int`s, `String`s, or `Val`s. If it is desirable for Enzyme to add a base type, please open an issue.

```jldoctest types
julia> f_int(x) = x * x;

julia> Enzyme.autodiff(Forward, f_int, Duplicated, Duplicated(3, 1))
ERROR: Return type `Int64` not marked Const, but type is guaranteed to be constant
```

```jldoctest types
julia> f_str(x) = parse(Float64, x) * parse(Float64, x);

julia> autodiff(Forward, f_str, Duplicated("1.0", "1.0"))
(0.0,)
```

```jldoctest types
julia> f_val(::Val{x}) where x = x * x;

julia> autodiff(Forward, f_val, Duplicated(Val(1.0), Val(1.0)))
ERROR: Type of ghost or constant type Duplicated{Val{1.0}} is marked as differentiable.
```

## Finalizers

Julia supports attaching finalizers to objects (see the listing below for an example) 

```julia
mutable struct Obj
    x::Float64
    function Obj(x)
        o = new(x)
        finalizer(o) do o
            # do someting with o
        end
        return o
    end
end
```

When Enzyme encounters a code like:

```julia
function f(x)
    o = Obj(x)
    # computations over o
    return o.x
end

autodiff(Forward, f, Duplicated(1.0, 1.0))
```

Enzyme has to allocate a shadow object for `o` and in the process encounters the finalizer being attached to the primal object.
Now the question is what should Enzyme do with the finalizer for the shadow objects? One option would be to simply ignore it,
but finalizers are often used for resource management (like manually allocating memory) and thus we would leak resources that are attached
to the shadow object. Instead, we define finalizers to be inactive (contain no instructions that are relevant with respect to AD),
yet we must attach them to the shadow object to release resources attached to them. 

