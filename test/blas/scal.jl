using Enzyme
using LinearAlgebra
using Test

@testset "BLAS scal" begin

    x1 = [1.0, 2.0, 3.0]
    x2 = [2.0, 3.0, 4.0]
    x3 = [2.0, 4.0, 6.0]
    x4 = [2.0, 4.0, 6.0]
    x5 = [2.0, 4.0, 6.0]
    
    dx1 = [0.0, 0.0, 0.0]
    dx2 = [1.0, 0.0, 0.0]
    dx3 = [1.0, 1.0, 1.0]
    dx4 = [1.0, 1.0, 2.0]
    dx5 = [1.0, -1.0, 2.0]

    ret1 = autodiff(Reverse, BLAS.scal!, Const, Active(1.0), Duplicated(x1, dx1))[1][1]
    ret2 = autodiff(Reverse, BLAS.scal!, Const, Active(1.0), Duplicated(x2, dx2))[1][1]
    ret3 = autodiff(Reverse, BLAS.scal!, Const, Active(1.0), Duplicated(x3, dx3))[1][1]
    ret4 = autodiff(Reverse, BLAS.scal!, Const, Active(1.0), Duplicated(x4, dx4))[1][1]
    ret5 = autodiff(Reverse, BLAS.scal!, Const, Active(1.0), Duplicated(x5, dx5))[1][1]
    @test ret1 == 0.0
    @test ret2 == 2.0
    @test ret3 == 12.0
    @test ret4 == 18.0
    @test ret5 == 10.0
    @test dx1 == [0.0, 0.0, 0.0]
    @test dx2 == [1.0, 0.0, 0.0]
    @test dx3 == [1.0, 1.0, 1.0]
    @test dx4 == [1.0, 1.0, 2.0]
    @test dx5 == [1.0, -1.0, 2.0]


    
    x1 = [1.0, 2.0, 3.0]
    x2 = [2.0, 3.0, 4.0]
    x3 = [2.0, 4.0, 6.0]
    x4 = [2.0, 4.0, 6.0]
    x5 = [2.0, 4.0, 6.0]
    
    dx1 = [0.0, 0.0, 0.0]
    dx2 = [1.0, 0.0, 0.0]
    dx3 = [1.0, 1.0, 1.0]
    dx4 = [1.0, 1.0, 2.0]
    dx5 = [1.0, -1.0, 2.0]
    
    ret1 = autodiff(Reverse, BLAS.scal!, Const, Active(2.0), Duplicated(x1, dx1))[1][1]
    ret2 = autodiff(Reverse, BLAS.scal!, Const, Active(2.0), Duplicated(x2, dx2))[1][1]
    ret3 = autodiff(Reverse, BLAS.scal!, Const, Active(2.0), Duplicated(x3, dx3))[1][1]
    ret4 = autodiff(Reverse, BLAS.scal!, Const, Active(2.0), Duplicated(x4, dx4))[1][1]
    ret5 = autodiff(Reverse, BLAS.scal!, Const, Active(2.0), Duplicated(x5, dx5))[1][1]
    @test ret1 == 0.0
    @test ret2 == 2.0
    @test ret3 == 12.0
    @test ret4 == 18.0
    @test ret5 == 10.0
    @test dx1 == [0.0, 0.0, 0.0]
    @test dx2 == [2.0, 0.0, 0.0]
    @test dx3 == [2.0, 2.0, 2.0]
    @test dx4 == [2.0, 2.0, 4.0]
    @test dx5 == [2.0, -2.0, 4.0]

end
