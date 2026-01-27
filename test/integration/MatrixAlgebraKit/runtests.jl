module MatrixAlgebraKitIntegrationTests

rng = StableRNG(23)

"""
Enum type for choosing Enzyme autodiff modes.
"""
@enum ModeSelector Neither Forward Reverse Both

"""
Type for specifying a test case for `Enzyme.gradient`.
The test will check the accuracy of the gradient of `func` at `value` against `finitediff`,
with both forward and reverse mode autodiff. `name` is for diagnostic printing.
`runtime_activity` and `broken` are for specifying whether to use
`Enzyme.set_runtime_activity` or not and whether the test is broken. Both of them taken
values `Neither`, `Forward`, `Reverse` or `Both`, to specify which mode to apply the setting
to. `splat` is for specifying whether to call the function as `func(value)` or as
`func(value...)`.
A constructor is also provided for giving a `Distribution` instead of a function, in which
case the function is `x -> logpdf(distribution, x)`.
Default values are `name=nothing` or `name=string(nameof(typeof(distribution)))`,
`runtime_activity=Neither`, `broken=Neither` and `splat=false`.
"""
struct TestCase
    func::Function
    value
    name::Union{String, Nothing}
    runtime_activity::ModeSelector
    broken::ModeSelector
    splat::Bool
end


end
