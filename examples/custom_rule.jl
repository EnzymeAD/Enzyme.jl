# # [Enzyme custom rules tutorial](@id custom_rules)
#
# !!! note "More Examples"
#     The tutorial below focuses on a simple setting to illustrate the basic concepts of writing custom rules.
#     For more complex custom rules beyond the scope of this tutorial, you may take inspiration from the following in-the-wild examples:
#     - [Enzyme internal rules](https://github.com/EnzymeAD/Enzyme.jl/tree/main/src/internal_rules)
#     - [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl/blob/main/ext/EnzymeExt.jl)
#     - [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl/blob/main/ext/LinearSolveEnzymeExt.jl)
#     - [NNlib.jl](https://github.com/FluxML/NNlib.jl/blob/master/ext/NNlibEnzymeCoreExt/NNlibEnzymeCoreExt.jl)
#
# The goal of this tutorial is to give a simple example of defining a custom rule with Enzyme.
# Specifically, our goal will be to write custom rules for the following function `f`:

function f(y, x)
    y .= x .^ 2
    return sum(y)
end

# Our function `f` populates its first input `y` with the element-wise square of `x`.
# In addition, it returns `sum(y)` as output. What a sneaky function!

# In this case, Enzyme can differentiate through `f` automatically. For example, using forward mode:

using Enzyme
x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2 # function to differentiate

## derivative of g with respect to x[1]
autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx))
#-
## derivative of y with respect to x[1] when g is run
dy

# (See the [AutoDiff API tutorial](autodiff.md) for more information on using `autodiff`.)

# But there may be special cases where we need to write a custom rule to help Enzyme out.
# Let's see how to write a custom rule for `f`!

# !!! warning "Don't use custom rules unnecessarily!"
#     Enzyme can efficiently handle a wide range of constructs, and so a custom rule should only be required
#     in certain special cases. For example, a function may make a foreign call that Enzyme cannot differentiate,
#     or we may have higher-level mathematical knowledge that enables us to write a more efficient rule.
#     Even in these cases, try to make your custom rule encapsulate the minimum possible construct that Enzyme
#     cannot differentiate, rather than expanding the scope of the rule unnecessarily.
#     For pedagogical purposes, we will disregard this principle here and go ahead and write a custom rule for `f` :)

# ## Defining our first rule

# First, we import the functions [`EnzymeRules.forward`](@ref), [`EnzymeRules.augmented_primal`](@ref),
# and [`EnzymeRules.reverse`](@ref).
# We need to overload `forward` in order to define a custom forward rule, and we need to overload
# `augmented_primal` and `reverse` in order to define a custom reverse rule.

import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

# In this section, we write a simple forward rule to start out:

function forward(config::FwdConfig, func::Const{typeof(f)}, ::Type{<:Duplicated}, y::Duplicated, x::Duplicated)
    println("Using custom rule!")
    ret = func.val(y.val, x.val)
    y.dval .= 2 .* x.val .* x.dval
    return Duplicated(ret, sum(y.dval))
end

# In the signature of our rule, we have made use of `Enzyme`'s activity annotations. Let's break down each one:
# - the [`EnzymeRules.FwdConfig`](@ref) configuration passes certain compile-time information about differentiation procedure (the width, and if we're using runtime activity),
# - the [`Const`](@ref) annotation on `f` indicates that we accept a function `f` that does not have a derivative component,
#   which makes sense since `f` is not a closure with data that could be differentiated.
# - the [`Duplicated`](@ref) annotation given in the second argument annotates the return value of `f`. This means that
#   our `forward` function should return an output of type `Duplicated`, containing the original output `sum(y)` and its derivative.
# - the [`Duplicated`](@ref) annotations for `x` and `y` mean that our `forward` function handles inputs `x` and `y`
#   which have been marked as `Duplicated`. We should update their shadows with their derivative contributions.

# In the logic of our forward function, we run the original function, populate `y.dval` (the shadow of `y`),
# and finally return a `Duplicated` for the output as promised. Let's see our rule in action!
# With the same setup as before:

x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

## function to differentiate
g(y, x) = f(y, x)^2

#-

## derivative of g with respect to x[1]
autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx))

#-

## derivative of y with respect to x[1] when g is run
dy

# We see that our custom forward rule has been triggered and gives the same answer as before.

# ## Handling more activities

# Our custom rule applies for the specific set of activities that are annotated for `f` in the above `autodiff` call.
# However, Enzyme has a number of other annotations. Let us consider a particular example, where the output
# has a [`DuplicatedNoNeed`](@ref) annotation. This means we are only interested in its derivative, not its value.
# To squeeze out the last drop of performance, the below rule avoids computing the output of the original function and
# just computes its derivative.

function forward(config, func::Const{typeof(f)}, ::Type{<:DuplicatedNoNeed}, y::Duplicated, x::Duplicated)
    println("Using custom rule with DuplicatedNoNeed output.")
    y.val .= x.val .^ 2
    y.dval .= 2 .* x.val .* x.dval
    return sum(y.dval)
end

# Our rule is triggered, for example, when we call `autodiff` directly on `f`, as the return value's derivative isn't needed:

x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

## derivative of f with respect to x[1]
autodiff(Forward, f, Duplicated(y, dy), Duplicated(x, dx))
#-
## derivative of y with respect to x[1] when f is run
dy

# !!! note "Custom rule dispatch"
#     When multiple custom rules for a function are defined, the correct rule is chosen using
#     [Julia's multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/#Methods).
#     In particular, it is important to understand that the custom rule does not *determine* the
#     activities of the inputs and the return value: rather, `Enzyme` decides the activity annotations independently,
#     and then *dispatches* to the custom rule handling the activities, if one exists.
#     If a custom rule is specified for the correct function/argument types, but not the correct activity annotation,
#     a runtime error will be thrown alerting the user to the missing activity rule rather than silently ignoring the rule."

# Finally, it may be that either `x`, `y`, or the return value are marked as [`Const`](@ref), in which case we can simply return the original result. However, Enzyme also may determine the return is not differentiable and also not needed for other computations, in which case we should simply return nothing.
#
# We can in fact handle this case, along with the previous two cases, all together in a single rule by leveraging utility functions [`EnzymeRules.needs_primal`](@ref) and [`EnzymeRules.needs_shadow`](@ref), which return true if the original return or the derivative is needed to be returned, respectively:

Base.delete_method.(methods(forward, (Const{typeof(f)}, Vararg{Any}))) # delete our old rules

function forward(
        config, func::Const{typeof(f)}, RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}},
        y::Union{Const, Duplicated}, x::Union{Const, Duplicated}
    )
    println("Using our general custom rule!")
    y.val .= x.val .^ 2
    if !(x isa Const) && !(y isa Const)
        y.dval .= 2 .* x.val .* x.dval
    elseif !(y isa Const)
        make_zero!(y.dval)
    end
    dret = !(y isa Const) ? sum(y.dval) : zero(eltype(y.val))
    if needs_primal(config) && needs_shadow(config)
        return Duplicated(sum(y.val), dret)
    elseif needs_primal(config)
        return sum(y.val)
    elseif needs_shadow(config)
        return dret
    else
        return nothing
    end
end

# Let's try out our rule:

x = [3.0, 1.0]
dx = [1.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2 # function to differentiate

## derivative of g with respect to x[1]
autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx))
#-
## derivative of g with respect to x[1], with y annotated Const
autodiff(Forward, g, Const(y), Duplicated(x, dx))
#-
## derivative of g with respect to x[1], with x and y annotated Const
autodiff(Forward, g, Const(y), Const(x));

# Note that there are also exist batched duplicated annotations for forward mode, namely [`BatchDuplicated`](@ref)
# and [`BatchDuplicatedNoNeed`](@ref), which are not covered in this tutorial.

# ## Defining a reverse-mode rule

# Let's look at how to write a simple reverse-mode rule!
# First, we write a method for [`EnzymeRules.augmented_primal`](@ref):

function augmented_primal(
        config::RevConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active},
        y::Duplicated, x::Duplicated
    )
    println("In custom augmented primal rule.")
    ## Compute primal
    if needs_primal(config)
        primal = func.val(y.val, x.val)
    else
        y.val .= x.val .^ 2 # y still needs to be mutated even if primal not needed!
        primal = nothing
    end
    ## Save x in tape if x will be overwritten
    if overwritten(config)[3]
        tape = copy(x.val)
    else
        tape = nothing
    end
    ## Return an AugmentedReturn object with shadow = nothing
    return AugmentedReturn(primal, nothing, tape)
end

# Let's unpack our signature for `augmented_primal` :
# * We accepted a [`EnzymeRules.RevConfig`](@ref) object with a specified width of 1, which means that our rule does not support batched reverse mode.
# * We annotated `f` with [`Const`](@ref) as usual.
# * We dispatched on an [`Active`](@ref) annotation for the return value. This is a special annotation for scalar values, such as our return value,
#   that indicates that that we care about the value's derivative but we need not explicitly allocate a mutable shadow since it is a scalar value.
# * We annotated `x` and `y` with [`Duplicated`](@ref), similar to our first simple forward rule.

# Now, let's unpack the body of our `augmented_primal` rule:
# * We checked if the `config` requires the primal. If not, we need not compute the return value, but we make sure to mutate `y` in all cases.
# * We checked if `x` could possibly be overwritten using the `Overwritten` attribute of [`EnzymeRules.RevConfig`](@ref).
#   If so, we save the elements of `x` on the `tape` of the returned [`EnzymeRules.AugmentedReturn`](@ref) object.
# * We return a shadow of `nothing` since the return value is [`Active`](@ref) and hence does not need a shadow.

# Now, we write a method for [`EnzymeRules.reverse`](@ref):

function reverse(
        config::RevConfigWidth{1}, func::Const{typeof(f)}, dret::Active, tape,
        y::Duplicated, x::Duplicated
    )
    println("In custom reverse rule.")
    ## retrieve x value, either from original x or from tape if x may have been overwritten.
    xval = overwritten(config)[3] ? tape : x.val
    ## accumulate dret into x's shadow. don't assign!
    x.dval .+= 2 .* xval .* dret.val
    ## also accumulate any derivative in y's shadow into x's shadow.
    x.dval .+= 2 .* xval .* y.dval
    make_zero!(y.dval)
    return (nothing, nothing)
end

# Let's make a few observations about our reverse rule:
# * The activities used in the signature correspond to what we used for `augmented_primal`.
# * However, for [`Active`](@ref) return types such as in this case, we now receive an *instance* `dret` of [`Active`](@ref) for the return type, not just a type annotation,
#   which stores the derivative value for `ret` (not the original return value!). For the other annotations (e.g. [`Duplicated`](@ref)), we still receive only the type.
#   In that case, if necessary a reference to the shadow of the output should be placed on the tape in `augmented_primal`.
# * Using `dret.val` and `y.dval`, we accumulate the backpropagated derivatives for `x` into its shadow `x.dval`.
#   Note that we have to accumulate from both `y.dval` and `dret.val`. This is because in reverse-mode AD we have to sum up the derivatives from all uses:
#   if `y` was read after our function, we need to consider derivatives from that use as well.
# * We zero-out `y`'s shadow.  This is because `y` is overwritten within `f`, so there is no derivative w.r.t. to the `y` that was originally inputted.
# * Finally, since all derivatives are accumulated *in place* (in the shadows of the [`Duplicated`](@ref) arguments), these derivatives must not be communicated via the return value.
#   Hence, we return `(nothing, nothing)`. If, instead, one of our arguments was annotated as [`Active`](@ref), we would have to provide its derivative at the corresponding index in the tuple returned.

# Finally, let's see our reverse rule in action!

x = [3.0, 1.0]
dx = [0.0, 0.0]
y = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2

autodiff(Reverse, g, Duplicated(y, dy), Duplicated(x, dx))

## derivative of g with respect to x
dx
#-
## derivative of g with respect to y
dy

# Let's also try a function which mutates `x` after running `f`, and also uses `y` directly rather than only `ret` after running `f`
# (but ultimately gives the same result as above):

function h(y, x)
    ret = f(y, x)
    x .= x .^ 2
    return ret * sum(y)
end

x = [3.0, 1.0]
y = [0.0, 0.0]
make_zero!(dx)
make_zero!(dy)

autodiff(Reverse, h, Duplicated(y, dy), Duplicated(x, dx))

## derivative of h with respect to x
dx
#-
## derivative of h with respect to y
dy

# ## Marking functions inactive

# If we want to tell Enzyme that the function call does not affect the differentiation result in any form
# (i.e. not by side effects or through its return values), we can simply use [`EnzymeRules.inactive`](@ref).
# So long as there exists a matching dispatch to [`EnzymeRules.inactive`](@ref), the function will be considered inactive.
# For example:

printhi() = println("Hi!")
EnzymeRules.inactive(::typeof(printhi), args...) = nothing

function k(x)
    printhi()
    return x^2
end

autodiff(Forward, k, Duplicated(2.0, 1.0))

# Or for a case where we incorrectly mark a function inactive:

double(x) = 2 * x
EnzymeRules.inactive(::typeof(double), args...) = nothing

## mathematically should be 3.0, inactive rule causes it to be 1.0
autodiff(Forward, x -> x + double(x), Duplicated(2.0, 1.0))

# ## Testing our rules

# We can test our rules using finite differences using [`EnzymeTestUtils.test_forward`](@ref)
# and [`EnzymeTestUtils.test_reverse`](@ref).

using EnzymeTestUtils, Test

@testset "f rules" begin
    @testset "forward" begin
        @testset for RT in (Const, DuplicatedNoNeed, Duplicated),
                Tx in (Const, Duplicated),
                Ty in (Const, Duplicated)

            x = [3.0, 1.0]
            y = [0.0, 0.0]
            test_forward(g, RT, (x, Tx), (y, Ty))
        end
    end
    @testset "reverse" begin
        @testset for RT in (Active,),
                Tx in (Duplicated,),
                Ty in (Duplicated,),
                fun in (g, h)

            x = [3.0, 1.0]
            y = [0.0, 0.0]
            test_reverse(fun, RT, (x, Tx), (y, Ty))
        end
    end
end

# In any package that implements Enzyme rules using EnzymeRules, it is recommended to add
# EnzymeTestUtils as a test dependency to test the rules.
