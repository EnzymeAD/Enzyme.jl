
using Enzyme


function h(x, y)
    return sum(x .^ 2) + y^3
end

# FORWARD
x  = [3.0, 1.0]
dx = [1.0, 0.0]
y  = 1.0
dy = 1.0

@show autodiff(Forward, h, Duplicated, Duplicated(x, dx), Duplicated(y, dy)) # first return: h(x,y)
                                                                             # second return: ∂h/∂x1*dx1 + ∂h/∂x2*dx2 + ∂h/∂y*dy


# BATCHED FORWARD
dx1 = [1.0, 0.0]
dx2 = [0.0, 1.0]
dx0 = [0.0, 0.0]
ret = autodiff(Forward, h, BatchDuplicated, BatchDuplicated(x, (dx1, dx2, dx0)), BatchDuplicated(y, (0.0, 0.0, 1.0))) 
@show ret

# REVERSE
dx = [0.0, 0.0]
ret = autodiff(ReverseWithPrimal, h, Active, Duplicated(x, dx), Active(y))
@show ret
@show ret[2]  # primal
@show dx      # accumulated ∂h/∂x
@show ret[1][2] # ∂h/∂y

dx = [0.0, 0.0]
ret = autodiff(ReverseWithPrimal, h, Active, Duplicated(x, dx), Active(y))
@show ret
@show ret[2]  # primal
@show dx      # accumulated ∂h/∂x
@show ret[1][2] # ∂h/∂y

