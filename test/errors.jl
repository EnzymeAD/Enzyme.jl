using Enzyme, Test

array_square(x) = 2 .* x

@testset "Array of Pointer Copy" begin
	@test_throws Enzyme.Compiler.EnzymeNonScalarReturnException Enzyme.gradient(Reverse, array_square, [2.0])
end
