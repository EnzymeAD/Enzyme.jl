# # Enzyme custom rules tutorial

# The goal of this tutorial is to give a simple example of defining a custom rule with Enzyme.
# Specifically, our goal will be to write custom rules for the following function `f`:

function f(y, x)
    y .= x.^2
    return sum(y) 
end

# Our function `f` populates its first input `y` with the element-wise square of `x`.
# In addition, it returns `sum(y)` as output. What a sneaky function!

# In this case, Enzyme can differentiate through `f` automatically. For example, using forward mode:

using Enzyme
x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2 # function to differentiate 

@show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1] when g is run

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

function forward(func::Const{typeof(f)}, ::Type{<:Duplicated}, y::Duplicated, x::Duplicated)
    println("Using custom rule!")
    ret = func.val(y.val, x.val)
    y.dval .= 2 .* x.val .* x.dval
    return Duplicated(ret, sum(y.dval)) 
end

# In the signature of our rule, we have made use of `Enzyme`'s activity annotations. Let's break down each one:
# - the [`Const`](@ref) annotation on `f` indicates that we accept a function `f` that does not have a derivative component,
#   which makes sense since `f` is not a closure with data that could be differentiated. 
# - the [`Duplicated`](@ref) annotation given in the second argument annotates the return value of `f`. This means that
#   our `forward` function should return an output of type `Duplicated`, containing the original output `sum(y)` and its derivative.
# - the [`Duplicated`](@ref) annotations for `x` and `y` mean that our `forward` function handles inputs `x` and `y`
#   which have been marked as `Duplicated`. We should update their shadows with their derivative contributions. 

# In the logic of our forward function, we run the original function, populate `y.dval` (the shadow of `y`), 
# and finally return a `Duplicated` for the output as promised. Let's see our rule in action! 
# With the same setup as before:

x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2 # function to differentiate

@show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1] when g is run

# We see that our custom forward rule has been triggered and gives the same answer as before.

# ## Handling more activities 

# Our custom rule applies for the specific set of activities that are annotated for `f` in the above `autodiff` call. 
# However, Enzyme has a number of other annotations. Let us consider a particular example, where the output
# has a [`DuplicatedNoNeed`](@ref) annotation. This means we are only interested in its derivative, not its value.
# To squeeze out the last drop of performance, the below rule avoids computing the output of the original function and 
# just computes its derivative.

function forward(func::Const{typeof(f)}, ::Type{<:DuplicatedNoNeed}, y::Duplicated, x::Duplicated)
    println("Using custom rule with DuplicatedNoNeed output.")
    y.val .= x.val.^2 
    y.dval .= 2 .* x.val .* x.dval
    return sum(y.dval)
end

# Our rule is triggered, for example, when we call `autodiff` directly on `f`, as the return value's derivative isn't needed:

x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

@show autodiff(Forward, f, Duplicated(y, dy), Duplicated(x, dx)) # derivative of f w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1] when f is run

# !!! note "Custom rule dispatch"
#     When multiple custom rules for a function are defined, the correct rule is chosen using 
#     [Julia's multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/#Methods).
#     In particular, it is important to understand that the custom rule does not *determine* the
#     activities of the inputs and the return value: rather, `Enzyme` decides the activity annotations independently,
#     and then *dispatches* to the custom rule handling the activities, if one exists.
#     If a custom rule is specified for the correct function/argument types, but not the correct activity annotation, 
#     a runtime error will be thrown alerting the user to the missing activity rule rather than silently ignoring the rule."

# Finally, it may be that either `x`, `y`, or the return value are marked as [`Const`](@ref). We can in fact handle this case, 
# along with the previous two cases, all together in a single rule:

Base.delete_method.(methods(forward, (Const{typeof(f)}, Vararg{Any}))) # delete our old rules

function forward(func::Const{typeof(f)}, RT::Type{<:Union{Const, DuplicatedNoNeed, Duplicated}}, 
                 y::Union{Const, Duplicated}, x::Union{Const, Duplicated})
    println("Using our general custom rule!")
    y.val .= x.val.^2 
    if !(x isa Const) && !(y isa Const)
        y.dval .= 2 .* x.val .* x.dval
    elseif !(y isa Const) 
        y.dval .= 0
    end
    dret = !(y isa Const) ? sum(y.dval) : zero(eltype(y.val))
    if RT <: Const
        return nothing
    elseif RT <: DuplicatedNoNeed
        return dret 
    else
        return Duplicated(sum(y.val), dret)
    end
end

# Let's try out our rule:

x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2 # function to differentiate 

@show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g w.r.t. x[1]
@show autodiff(Forward, g, Const(y), Duplicated(x, dx)) # derivative of g w.r.t. x[1], with y annotated Const
@show autodiff(Forward, g, Const(y), Const(x)); # derivative of g w.r.t. x[1], with x and y annotated Const

# Note that there are also exist batched duplicated annotations for forward mode, namely [`BatchDuplicated`](@ref)
# and [`BatchDuplicatedNoNeed`](@ref), which are not covered in this tutorial.

# ## Defining a reverse-mode rule

# Let's look at how to write a simple reverse-mode rule! 
# First, we write a method for [`EnzymeRules.augmented_primal`](@ref):

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active},
                          y::Duplicated, x::Duplicated)
    println("In custom augmented primal rule.")
    ## Compute primal
    if needs_primal(config)
        primal = func.val(y.val, x.val)
    else
        y.val .= x.val.^2 # y still needs to be mutated even if primal not needed!
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
# * We accepted a [`EnzymeRules.Config`](@ref) object with a specified width of 1, which means that our rule does not support batched reverse mode.
# * We annotated `f` with [`Const`](@ref) as usual.
# * We dispatched on an [`Active`](@ref) annotation for the return value. This is a special annotation for scalar values, such as our return value,
#   that indicates that that we care about the value's derivative but we need not explicitly allocate a mutable shadow since it is a scalar value.
# * We annotated `x` and `y` with [`Duplicated`](@ref), similar to our first simple forward rule.

# Now, let's unpack the body of our `augmented_primal` rule:
# * We checked if the `config` requires the primal. If not, we need not compute the return value, but we make sure to mutate `y` in all cases.
# * We checked if `x` could possibly be overwritten using the `Overwritten` attribute of [`EnzymeRules.Config`](@ref). 
#   If so, we save the elements of `x` on the `tape` of the returned [`EnzymeRules.AugmentedReturn`](@ref) object.
# * We return a shadow of `nothing` since the return value is [`Active`](@ref) and hence does not need a shadow.

# Now, we write a method for [`EnzymeRules.reverse`](@ref):

function reverse(config::ConfigWidth{1}, func::Const{typeof(f)}, dret::Active, tape,
                 y::Duplicated, x::Duplicated)
    println("In custom reverse rule.")
    ## retrieve x value, either from original x or from tape if x may have been overwritten.
    xval = overwritten(config)[3] ? tape : x.val 
    ## accumulate into x's shadow, don't assign!
    x.dval .+= 2 .* xval .* dret.val 
    return ()
end

# The activities used in the signature match what we used for `augmented_primal`. 
# One key difference is that we now receive an *instance* `dret` of [`Active`](@ref) for the return type, not just a type annotation.
# Here, `dret.val` stores the derivative value for `dret` (not the original return value!).
# Using this derivative value, we accumulate the backpropagated derivatives for `x` into its shadow. 
# Note that we do not accumulate anything into `y`'s shadow! This is because `y` is overwritten within `f`, so there is no derivative
# w.r.t. to the `y` that was originally inputted.

# Finally, let's see our reverse rule in action!

x  = [3.0, 1.0]
dx = [0.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2

autodiff(Reverse, g, Duplicated(y, dy), Duplicated(x, dx))
@show dx # derivative of g w.r.t. x
@show dy; # derivative of g w.r.t. y

# Let's also try a function which mutates `x` after running `f`:

function h(y, x)
    ret = f(y, x)
    x .= x.^2
    return ret^2
end

x  = [3.0, 1.0]
y  = [0.0, 0.0]
dx .= 0
dy .= 0

autodiff(Reverse, h, Duplicated(y, dy), Duplicated(x, dx))
@show dx # derivative of h w.r.t. x
@show dy; # derivative of h w.r.t. y

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

double(x) = 2*x
EnzymeRules.inactive(::typeof(double), args...) = nothing

autodiff(Forward, x -> x + double(x), Duplicated(2.0, 1.0)) # mathematically should be 3.0, inactive rule causes it to be 1.0