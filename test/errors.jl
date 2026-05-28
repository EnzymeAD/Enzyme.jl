using Enzyme, Test

array_square(x) = 2 .* x

@testset "Array of Pointer Copy" begin
	@test_throws Enzyme.Compiler.EnzymeNonScalarReturnException Enzyme.gradient(Reverse, array_square, [2.0])
end


function sumsin(x)
	return sin(sum(x))
end

@testset "Incorrect thunk arguments" begin
	fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(sumsin)}, Active, Duplicated{Vector{Float64}})

	@test_throws Enzyme.Compiler.ThunkCallError fwd(Duplicated([1.0], [2.0]))

	@test_throws Enzyme.Compiler.ThunkCallError fwd(Const(sumsin), Duplicated([1.0], [2.0]), Active(3.14))
end

only_int(x::Int) = x * 2

@testset "LookupError" begin
	@test_throws Enzyme.LookupError Enzyme.my_methodinstance(Reverse, typeof(only_int), Tuple{Float64})

	err = Enzyme.LookupError(typeof(only_int), Tuple{Float64}, UInt(1))
	msg = sprint(showerror, err)
	@test occursin("no method matching", msg)
end


