# # AutoDiff API

# The goal of this tutorial is to give users already familiar with automatic
# differentiation (AD) an overview
# of the Enzyme differentiation API for the following differentiation modes
# * Reverse mode
# * Forward mode
# * Forward over reverse mode
# * Vector Forward over reverse mode
# # Defining a function
# Enzyme differentiates arbitrary multivariate vector functions as the most
# general case in automatic differentiation
# ```math
# f: \mathbb{R}^n \rightarrow \mathbb{R}^m, y = f(x)
# ```
# For simplicity we define a vector function with ``m=1``. However, this
# tutorial can easily be applied to arbitrary ``m \in \mathbb{N}``.
using Enzyme

function f(x::Array{Float64}, y::Array{Float64})
    y[1] = x[1] * x[1] + x[2] * x[1]
    return nothing
end;

# # Reverse mode
# The reverse model in AD is defined as
# ```math
# \begin{aligned}
# y &= f(x) \\
# \bar{x} &= \bar{y} \cdot \nabla f(x)
# \end{aligned}
# ```
# bar denotes an adjoint variable. Note that executing an AD in reverse mode
# computes both ``y`` and the adjoint ``\bar{x}``.
x  = [2.0, 2.0]
bx = [0.0, 0.0]
y  = [0.0]
by = [1.0];

# Enzyme stores the value and adjoint of a variable in an object of type
# `Duplicated` where the first element represent the value and the second the
# adjoint. Evaluating the reverse model using Enzyme is done via the following
# call.
Enzyme.autodiff(f, Duplicated(x, bx), Duplicated(y, by));
# This yields the gradient of `f` in `bx` at point `x = [2.0, 2.0]`. `by` is called the seed and has
# to be set to ``1.0`` in order to compute the gradient. Let's save the gradient for later.
g = copy(bx)

# # Forward mode
# The forward model in AD is defined as
# ```math
# \begin{aligned}
# y &= f(x) \\
# \dot{y} &= \nabla f(x) \cdot x
# \end{aligned}
# ```
# To obtain the first element of the gradient using the forward model we have to
# seed ``\dot{x}`` with ``\dot{x} = [1.0,0.0]``
x  = [2.0, 2.0]
dx = [1.0, 0.0]
y  = [0.0]
dy = [0.0];
# In the forward mode the second element of `Duplicated` stores the tangent.
Enzyme.autodiff(Forward, f, Duplicated(x, dx), Duplicated(y, dy));

# We can now verify that indeed the reverse mode and forward mode yield the same
# result for the first component of the gradient. Note that to acquire the full
# gradient one needs to execute the forward model a second time with the seed
# `dx` set to `[0.0,1.0]`.

# Let's verify whether the reverse and forward model agree.
g[1] == dy[1]

# # Forward over reverse
# The forward over reverse (FoR) model is obtained by applying the forward model
# to the reverse model using the chain rule for the product in the adjoint statement.
# ```math
# \begin{aligned}
# y &= f(x) \\
# \dot{y} &= f(x) \cdot \dot{x} \\
# \bar{x} &= \bar{y} \cdot \nabla f(x) \\
# \dot{\bar{x}} &= \bar{y} \cdot \nabla^2 f(x) \cdot \dot{x} + \dot{\bar{y}} \cdot \nabla f(x)
# \end{aligned}
# ```
# To obtain the first column/row of the Hessian ``\nabla^2 f(x)`` we have to
# seed ``\dot{\bar{y}}`` with ``[0.0]``, ``\bar{y}`` with ``[1.0]`` and ``\dot{x}`` with ``[1.0, 0.0]``.

y = [0.0]
x = [2.0, 2.0]

dy = [0.0]
dx = [1.0, 0.0]

bx = [0.0, 0.0]
by = [1.0]
dbx = [0.0, 0.0]
dby = [0.0]

Enzyme.autodiff(
    Forward,
    (x,y) -> Enzyme.autodiff_deferred(Reverse, f, x, y),
    Duplicated(Duplicated(x, bx), Duplicated(dx, dbx)),
    Duplicated(Duplicated(y, by), Duplicated(dy, dby)),
)

# The FoR model also computes the forward model from before, giving us again the first component of the gradient.
g[1] == dy[1]
# In addition we now have the first row/column of the Hessian.
dbx[1] == 2.0
dbx[2] == 1.0

# # Vector forward over reverse
# The vector FoR allows us to propagate several tangents at once through the
# second-order model. This allows us the acquire the Hessian in one autodiff
# call. The multiple tangents are organized in tuples. Following the same seeding strategy as before, we now seed both
# in the `vdx[1]=[1.0, 0.0]` and `vdx[2]=[0.0, 1.0]` direction. These tuples have to be put into a `BatchDuplicated` type.
y = [0.0]
x = [2.0, 2.0]

vdy = ([0.0],[0.0])
vdx = ([1.0, 0.0], [0.0, 1.0])

bx = [0.0, 0.0]
by = [1.0]
vdbx = ([0.0, 0.0], [0.0, 0.0])
vdby = ([0.0], [0.0]);

# The `BatchedDuplicated` objects are constructed using the broadcast operator
# on our tuples of `Duplicated` for the tangents.
Enzyme.autodiff(
    Forward,
    (x,y) -> Enzyme.autodiff_deferred(Reverse, f, x, y),
    BatchDuplicated(Duplicated(x, bx), Duplicated.(vdx, vdbx)),
    BatchDuplicated(Duplicated(y, by), Duplicated.(vdy, vdby)),
);

# Again we obtain the first-order gradient.
g[1] == vdy[1][1]
# We have now the first row/column of the Hessian
vdbx[1][1] == 2.0

vdbx[1][2] == 1.0
# as well as the second row/column
vdbx[2][1] == 1.0

vdbx[2][2] == 0.0
