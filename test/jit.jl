using Enzyme, Test

jit_cube(x) = x * x * x

# Thunks of one function for different jobs. Since JuliaLang/julia#60031, Julia's codegen gives
# all of them the same entry name.
jit_sum_of_thunks(x) = autodiff(Forward, jit_cube, Duplicated(x, 1.0))[1] +
    autodiff(Forward, jit_cube, Duplicated(Float32(x), 1.0f0))[1] +
    autodiff(Reverse, jit_cube, Active(x))[1][1]

@testset "Thunks with the same entry name" begin
    @test autodiff(Reverse, jit_cube, Active(2.0))[1][1] ≈ 12.0
    @test autodiff(Reverse, jit_cube, Active(2.0f0))[1][1] ≈ 12.0f0
    @test autodiff(Reverse, jit_cube, Const(2.0))[1][1] === nothing
    @test jit_sum_of_thunks(2.0) ≈ 36.0

    # One compilation imports the three thunks.
    @test autodiff(Reverse, jit_sum_of_thunks, Active(2.0))[1][1] ≈ 36.0
    @test autodiff(Forward, jit_sum_of_thunks, Duplicated(2.0, 1.0))[1] ≈ 36.0
end

jit_unstable(x) = Base.inferencebarrier(x)[1]^2

@testset "InlineABI resolves the globals of Enzyme" begin
    # Julia's JIT compiles this thunk. The thunk refers to Julia values that Enzyme defines
    # in the JIT, like `ejl_enz_runtime_generic_rev`.
    x = [2.0]
    dx = [0.0]
    autodiff(Enzyme.set_abi(Reverse, Enzyme.InlineABI), jit_unstable, Active, Duplicated(x, dx))
    @test dx ≈ [4.0]
end
