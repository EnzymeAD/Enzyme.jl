using Enzyme: Enzyme
using Reactant: Reactant, @jit
using Test

square(x) = x .^ 2
enzyme_jacobian(x) = only(Enzyme.jacobian(Enzyme.Forward, square, x))

@testset "Forward Jacobian" begin
    x = Reactant.to_rarray(Float32[1, 2])
    @test Array(@jit enzyme_jacobian(x)) == Float32[2 0; 0 4]
end
