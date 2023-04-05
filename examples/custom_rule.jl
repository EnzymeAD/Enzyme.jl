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

@show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of g's output w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1]

# (See the [AutoDiff API tutorial](..autodiff.md) for more information on using `autodiff`.)

# However, while Enzyme can efficiently handle a wide range of constructs, we may encounter cases where
# we would like to use a custom rule for `f`. For example, `f` may make a foreign call that Enzyme cannot 
# differentiate, or we may have special higher-level knowledge about `f` that enables us to write a more 
# efficient rule. So let's see how to write a custom rule for `f`.

# ## Defining our first rule 

# First, we import the functions [`EnzymeRules.forward`](@ref) and [`EnzymeRules.reverse`](@ref).
# We will need to overload these functions in order to define our custom rules.

import Enzyme.EnzymeRules: forward, reverse

# In this section, we write a simple forward rule:

function forward(func::Const{typeof(f)}, ::Type{<:Duplicated}, y::Duplicated, x::Duplicated)
    println("Using custom rule!")
    out = func.val(y.val, x.val)
    y.dval .+= 2 .* x.val .* x.dval
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

@show autodiff(Forward, g, Duplicated(y, dy), Duplicated(x, dx)) # derivative of output w.r.t. x[1]
@show dy; # derivative of y w.r.t. x[1]

# We see that our custom forward rule has been triggered and gives the same answer as before.

# ## A more comprehensive set of rules 

# Our custom rule applies for the specific set of activities that are triggered in the above example. However,
# Enzyme has a number of other annotations. And of course, we'd like rules for reverse-mode too!. 
# Below we define a more comprehensive set of custom rules for our function `f`:

# TODO: code dump DuplicatedNoNeed, batch rules, reverse-mode rules (get accumulation v.s. assignment of shadows correct for this one).
