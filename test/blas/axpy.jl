using Enzyme
using LinearAlgebra
using Test

@testset "BLAS axpy" begin
    x1 = [0.0, 0.0, 0.0]
    x2 = [0.0, 0.0, 0.0]
    x3 = [0.0, 0.0, 0.0]
    x4 = [0.0, 0.0, 0.0]
    x5 = [0.0, 0.0, 0.0]
    dx1 = [0.0, 0.0, 0.0]
    dx2 = [0.0, 0.0, 0.0]
    dx3 = [0.0, 0.0, 0.0]
    dx4 = [0.0, 0.0, 0.0]
    dx5 = [0.0, 0.0, 0.0]

    y1 = [0.0, 0.0, 0.0]
    y2 = [1.0, 0.0, 0.0]
    y3 = [1.0, 1.0, 0.0]
    y4 = [1.0, 1.0, 1.0]
    y5 = [1.0, 1.0, 1.0]

    dy1 = [0.0, 0.0, 0.0]
    dy2 = [1.0, 0.0, 0.0]
    dy3 = [1.0, 1.0, 1.0]
    dy4 = [1.0, 1.0, 2.0]
    dy5 = [1.0, 1.0, 0.0]

    #after: dy1 = [0.0, 0.0, 0.0]
    #after: dy2 = [1.0, 0.0, 0.0]
    #after: dy3 = [1.0, 1.0, 1.0]
    #after: dy4 = [1.0, 1.0, 2.0]
    #after: dy5 = [1.0, -1.0, 2.0]


    ret1 = autodiff(Reverse, BLAS.axpy!, Const, Const(0.0), Const(x1), Duplicated(y1, dy1))[1][1]
    ret2 = autodiff(Reverse, BLAS.axpy!, Const, Const(0.0), Const(x2), Duplicated(y2, dy2))[1][1]
    ret3 = autodiff(Reverse, BLAS.axpy!, Const, Const(0.0), Const(x3), Duplicated(y3, dy3))[1][1]
    ret4 = autodiff(Reverse, BLAS.axpy!, Const, Const(0.0), Const(x4), Duplicated(y4, dy4))[1][1]
    ret5 = autodiff(Reverse, BLAS.axpy!, Const, Const(0.0), Const(x5), Duplicated(y5, dy5))[1][1]
    @show dy1 
    @show dy2 
    @show dy3 
    @show dy4 
    @show dy5 

    x1 = [1.0, 2.0, 3.0]
    x2 = [2.0, 3.0, 4.0]
    x3 = [2.0, 4.0, 6.0]
    x4 = [2.0, 4.0, 6.0]
    x5 = [2.0, 4.0, 6.0]

    y1 = [0.0, 0.0, 0.0]
    y2 = [1.0, 0.0, 0.0]
    y3 = [1.0, 1.0, 0.0]
    y4 = [1.0, 1.0, 1.0]
    y5 = [1.0, -1.0, 1.0]
    
    dx1 = [0.0, 0.0, 0.0]
    dx2 = [1.0, 0.0, 0.0]
    dx3 = [1.0, 1.0, 1.0]
    dx4 = [1.0, 1.0, 2.0]
    dx5 = [1.0, -1.0, 2.0]

    dy1 = [0.0, 0.0, 0.0]
    dy2 = [1.0, 0.0, 0.0]
    dy3 = [1.0, 1.0, 1.0]
    dy4 = [1.0, 1.0, 2.0]
    dy5 = [1.0, -1.0, 2.0]

    ret1 = autodiff(Reverse, BLAS.axpy!, Const, Active(1.0), Duplicated(x1, dx1), Duplicated(y1, dy1))[1][1]
    ret2 = autodiff(Reverse, BLAS.axpy!, Const, Active(1.0), Duplicated(x2, dx2), Duplicated(y2, dy2))[1][1]
    ret3 = autodiff(Reverse, BLAS.axpy!, Const, Active(1.0), Duplicated(x3, dx3), Duplicated(y3, dy3))[1][1]
    ret4 = autodiff(Reverse, BLAS.axpy!, Const, Active(1.0), Duplicated(x4, dx4), Duplicated(y4, dy4))[1][1]
    ret5 = autodiff(Reverse, BLAS.axpy!, Const, Active(1.0), Duplicated(x5, dx5), Duplicated(y5, dy5))[1][1]
    @show ret1
    @show ret2
    @show ret3
    @show ret4
    @show ret5
    @show dx1 
    @show dx2 
    @show dx3 
    @show dx4 
    @show dx5 
    @show dy1 
    @show dy2 
    @show dy3 
    @show dy4 
    @show dy5 
    # dx1 = [0.0, 0.0, 0.0]
    # dx2 = [2.0, 0.0, 0.0]
    # dx3 = [2.0, 2.0, 2.0]
    # dx4 = [2.0, 2.0, 4.0]
    # dx5 = [2.0, -2.0, 4.0]
    # dy1 = [0.0, 0.0, 0.0]
    # dy2 = [1.0, 0.0, 0.0]
    # dy3 = [1.0, 1.0, 1.0]
    # dy4 = [1.0, 1.0, 2.0]
    # dy5 = [1.0, -1.0, 2.0]
    # ret1 = 0.0
    # ret2 = 2.0
    # ret3 = 12.0
    # ret4 = 18.0
    # ret5 = 10.0
    


    
    x1 = [1.0, 2.0, 3.0]
    x2 = [2.0, 3.0, 4.0]
    x3 = [2.0, 4.0, 6.0]
    x4 = [2.0, 4.0, 6.0]
    x5 = [2.0, 4.0, 6.0]
    
    y1 = [0.0, 0.0, 0.0]
    y2 = [1.0, 0.0, 0.0]
    y3 = [1.0, 1.0, 0.0]
    y4 = [1.0, 1.0, 1.0]
    y5 = [1.0, -1.0, 1.0]
    
    dx1 = [0.0, 0.0, 0.0]
    dx2 = [1.0, 0.0, 0.0]
    dx3 = [1.0, 1.0, 1.0]
    dx4 = [1.0, 1.0, 2.0]
    dx5 = [1.0, -1.0, 2.0]
    
    dy1 = [0.0, 0.0, 0.0]
    dy2 = [1.0, 0.0, 0.0]
    dy3 = [1.0, 1.0, 1.0]
    dy4 = [1.0, 1.0, 2.0]
    dy5 = [1.0, -1.0, 2.0]
    
    ret1 = autodiff(Reverse, BLAS.axpy!, Const, Active(2.0), Duplicated(x1, dx1), Duplicated(y1, dy1))[1][1]
    ret2 = autodiff(Reverse, BLAS.axpy!, Const, Active(2.0), Duplicated(x2, dx2), Duplicated(y2, dy2))[1][1]
    ret3 = autodiff(Reverse, BLAS.axpy!, Const, Active(2.0), Duplicated(x3, dx3), Duplicated(y3, dy3))[1][1]
    ret4 = autodiff(Reverse, BLAS.axpy!, Const, Active(2.0), Duplicated(x4, dx4), Duplicated(y4, dy4))[1][1]
    ret5 = autodiff(Reverse, BLAS.axpy!, Const, Active(2.0), Duplicated(x5, dx5), Duplicated(y5, dy5))[1][1]
    @show ret1
    @show ret2
    @show ret3
    @show ret4
    @show ret5
    @show dx1 
    @show dx2 
    @show dx3 
    @show dx4 
    @show dx5 
    @show dy1 
    @show dy2 
    @show dy3 
    @show dy4 
    @show dy5 
    # ret1 = 0.0
    # ret2 = 2.0
    # ret3 = 12.0
    # ret4 = 18.0
    # ret5 = 10.0
    # dx1 = [0.0, 0.0, 0.0]
    # dx2 = [3.0, 0.0, 0.0]
    # dx3 = [3.0, 3.0, 3.0]
    # dx4 = [3.0, 3.0, 6.0]
    # dx5 = [3.0, -3.0, 6.0]
    # dy1 = [0.0, 0.0, 0.0]
    # dy2 = [1.0, 0.0, 0.0]
    # dy3 = [1.0, 1.0, 1.0]
    # dy4 = [1.0, 1.0, 2.0]
    # dy5 = [1.0, -1.0, 2.0]


end
