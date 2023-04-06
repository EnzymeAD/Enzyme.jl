# # Enzyme custom rules tutorial

# The goal of this tutorial is to give a simple example of defining a custom rule with Enzyme.
# Specifically, our goal will be to write a custom rule for the following function `f`:

function f(y, x)
    y .= x.^2
    return sum(y) 
end

# Our function `f` populates its first input `y` with the element-wise square of `x`.
# In addition to doing this mutation, it returns `sum(y)` as output. What a sneaky function!

# In this case, Enzyme can differentiate through `f` automatically. For example, using forward mode:

using Enzyme
x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

g(y, x) = f(y, x)^2 # function to differentiate 

@show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1] when g is run

# (See the [AutoDiff API tutorial](..autodiff.md) for more information on using `autodiff`.)

# However, while Enzyme can efficiently handle a wide range of constructs, we may encounter cases where
# we would like to use a custom rule for `f`. For example, `f` may make a foreign call that Enzyme cannot 
# differentiate, or we may have special higher-level knowledge about `f` that enables us to write a more 
# efficient rule. So let's see how to write a custom rule for `f`.

# ## Defining our first rule 

# First, we import the functions [`EnzymeRules.forward`](@ref), [`EnzymeRules.augmented_primal`](@ref),
# and [`EnzymeRules.reverse`](@ref).
# We need to overload `forward` in order to define a custom forward rule, and we need to overload
# `augmented_primal` and `reverse` in order to define a custom reverse rule.

import Enzyme.EnzymeRules: forward, reverse, augmented_primal

# In this section, we write a simple forward rule to start out:

function forward(func::Const{typeof(f)}, ::Type{<:Duplicated}, y::Duplicated, x::Duplicated)
    println("Using custom rule!")
    out = func.val(y.val, x.val)
    y.dval .= 2 .* x.val .* x.dval
    return Duplicated(out, sum(y.dval)) 
end

# In the signature of our rule, we have made use of `Enzyme`'s activity annotations. Let's break down each one:
# - the [`Const`](@ref) annotation on `f` indicates that we accept a function `f` that does not have a derivative component,
#   which makes sense since `f` itself does not depend on any parameters.
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

# !!! note
#     The `autodiff` call is not currently automatically recompiled when a custom rule is defined.
#     As a workaround, when interactively developing custom rules, make sure to redefine the primal function
#     when editing a custom rule in order to trigger recompilation of the `autodiff` call.
#     See [Issue #696](https://github.com/EnzymeAD/Enzyme.jl/issues/696) for more information.

# ## Handling more activities 

# Our custom rule applies for the specific set of activities that are annotated for `f` in the above `autodiff` call. 
# However, Enzyme has a number of other annotations. Let us consider a particular case as an example, where the output
# has a [`DuplicatedNoNeed`](@ref) annotation. This means we are only interested in its derivative, not its value.
# To squeeze the last drop of performance, the below rule avoids computing the output of the original function and 
# just computes its derivative.

function forward(func::Const{typeof(f)}, ::Type{<:DuplicatedNoNeed}, y::Duplicated, x::Duplicated)
    println("Using custom rule with DuplicatedNoNeed output.")
    y.val .= x.val.^2 
    y.dval .= 2 .* x.val .* x.dval
    return sum(y.dval)
end

# This rule is triggered, for example, when we call `autodiff` directly on `f`:

x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = [0.0, 0.0]
dy = [0.0, 0.0]

@show autodiff(Forward, f, Duplicated(y, dy), Duplicated(x, dx)) # derivative of f w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1] when f is run

# Finally, it may be that either `x` or `y`  are marked as [`Const`](@ref). We can in fact handle this case, along with
# the previous two cases, together in a single rule:

function forward(func::Const{typeof(f)}, RT::Type{<:Union{DuplicatedNoNeed, Duplicated}}, 
                 y::Union{Const, Duplicated}, x::Union{Const, Duplicated})
    println("Using custom rule!")
    y.val .= x.val.^2 
    if !(x <: Const) && !(y <: Const)
        y.dval .= 2 .* x.val .* x.dval
    elseif !(y <: Const) 
        y.dval .= 0
    end
    if RT <: DuplicatedNoNeed
        return sum(y.dval)
    else
        return Duplicated(sum(y.val), sum(y.dval))
    end
end

# Note that there are also exist batched duplicated annotations for forward mode, i.e. [`BatchDuplicated`](@ref)
# and [`BatchDuplicatedNoNeed`](@ref), which are not covered in this tutorial.

# ## Reverse-mode

# Finally, let's look at how to write a reverse-mode rule! First, we define [`EnzymeRules.augmented_primal`](@ref):

# TODO

# function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(f)}, ::Type{<:Active}, x::Active, y::Active)
#     if needs_primal(config)
#         return AugmentedReturn(func.val(x.val), nothing, nothing)
#     else
#         return AugmentedReturn(nothing, nothing, nothing)
#     end
# end

# TODO: code dump rest of DuplicatedNoNeed, batch rules, reverse-mode rules (get accumulation v.s. assignment of shadows correct for this one).
