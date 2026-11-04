using Enzyme, Test

module ModuleLookupValues
    square(x) = x * x
    cube(x) = x * x * x
    coefficient = 2.0
    integer = 2
    values = [2.0, 3.0]
    mixture = (2.0, [3.0])
end

module OtherModuleLookupValues
    square(x) = x * x * x
end

module_binding = ModuleLookupValues
const constant_module_binding = ModuleLookupValues
dynamic_module_square(x) = module_binding.square(x)
constant_module_square(x) = constant_module_binding.square(x)
module_property(x, mod, name) = getproperty(mod, name)(x)
module_field(x, mod, name) = getfield(mod, name)(x)
module_global(x, mod, name) = getglobal(mod, name)(x)
module_ordered_global(x, mod, name, order) = getglobal(mod, name, order)(x)
module_coefficient(x, mod, name) = x * getglobal(mod, name)
module_array(x, mod, name) = x * sum(getglobal(mod, name))
module_return(mod, name) = getglobal(mod, name)::Vector{Float64}

function module_mixed(x, mod, name)
    value = getglobal(mod, name)
    return x * (value[1] + sum(value[2]))
end

function module_global!(out, x, mod, name)
    out[] = getglobal(mod, name)(x[])
    return nothing
end

@testset "Dynamic module lookup" begin
    @testset for mode in (Reverse, Forward, set_runtime_activity(Reverse), set_runtime_activity(Forward))
        @test Enzyme.gradient(mode, constant_module_square, 3.0) == (6.0,)
        @test Enzyme.gradient(mode, dynamic_module_square, 3.0) == (6.0,)

        @testset for f in (module_property, module_field, module_global)
            @test Enzyme.gradient(mode, f, 3.0, Const(ModuleLookupValues), Const(:square))[1] == 6.0
            @test Enzyme.gradient(mode, f, 3.0, Const(ModuleLookupValues), Const(:cube))[1] == 27.0
            @test Enzyme.gradient(mode, f, 3.0, Const(OtherModuleLookupValues), Const(:square))[1] == 27.0
            @test_throws UndefVarError Enzyme.gradient(mode, f, 3.0, Const(ModuleLookupValues), Const(:undefined_binding))
        end

        @testset for order in (:monotonic, :acquire)
            @test Enzyme.gradient(
                mode, module_ordered_global, 3.0,
                Const(ModuleLookupValues), Const(:square), Const(order)
            )[1] == 6.0
        end
        @test_throws ConcurrencyViolationError Enzyme.gradient(
            mode, module_ordered_global, 3.0,
            Const(ModuleLookupValues), Const(:square), Const(:release)
        )

        # Module identity is not fixed by its type or by a non-const binding.
        global module_binding = OtherModuleLookupValues
        try
            @test Enzyme.gradient(mode, dynamic_module_square, 3.0) == (27.0,)
        finally
            global module_binding = ModuleLookupValues
        end
        @test Enzyme.gradient(mode, dynamic_module_square, 3.0) == (6.0,)
    end
end

@testset "Global values are read without differentiating the module" begin
    @testset for mode in (Reverse, Forward, set_runtime_activity(Reverse), set_runtime_activity(Forward))
        @test Enzyme.gradient(mode, module_coefficient, 3.0, Const(ModuleLookupValues), Const(:coefficient))[1] == 2.0
        @test Enzyme.gradient(mode, module_coefficient, 3.0, Const(ModuleLookupValues), Const(:integer))[1] == 2.0
        @test Enzyme.gradient(mode, module_array, 3.0, Const(ModuleLookupValues), Const(:values))[1] == 5.0
        @test ModuleLookupValues.values == [2.0, 3.0]
        @test Enzyme.gradient(mode, module_mixed, 3.0, Const(ModuleLookupValues), Const(:mixture))[1] == 5.0
        @test ModuleLookupValues.mixture == (2.0, [3.0])

        Core.setglobal!(ModuleLookupValues, :coefficient, 4.0)
        try
            @test Enzyme.gradient(mode, module_coefficient, 3.0, Const(ModuleLookupValues), Const(:coefficient))[1] == 4.0
        finally
            Core.setglobal!(ModuleLookupValues, :coefficient, 2.0)
        end
    end
end

@testset "Returned global has independent batch shadows" begin
    mode = Enzyme.ReverseSplitWidth(ReverseSplitWithPrimal, Val(2))
    fwd, rev = autodiff_thunk(
        mode, Const{typeof(module_return)}, BatchDuplicated{Vector{Float64}, 2},
        Const{Module}, Const{Symbol}
    )
    tape, value, shadows = fwd(Const(module_return), Const(ModuleLookupValues), Const(:values))
    @test value === ModuleLookupValues.values
    @test shadows[1] == shadows[2] == zeros(2)
    @test shadows[1] !== shadows[2]
    @test shadows[1] !== value
    shadows[1][1] = 1.0
    shadows[2][2] = 3.0
    @test rev(Const(module_return), Const(ModuleLookupValues), Const(:values), tape) == ((nothing, nothing),)
    @test ModuleLookupValues.values == [2.0, 3.0]
end

@testset "Batched module lookup" begin
    derivatives = autodiff(
        Forward, module_global, BatchDuplicated(3.0, (1.0, 2.0)),
        Const(ModuleLookupValues), Const(:square)
    )[1]
    @test Tuple(derivatives) == (6.0, 12.0)

    derivatives = autodiff(
        Forward, module_coefficient, BatchDuplicated(3.0, (1.0, 2.0)),
        Const(ModuleLookupValues), Const(:coefficient)
    )[1]
    @test Tuple(derivatives) == (2.0, 4.0)

    x = Ref(3.0)
    dx = (Ref(0.0), Ref(0.0))
    out = Ref(0.0)
    dout = (Ref(1.0), Ref(3.0))
    autodiff(
        Reverse, module_global!, BatchDuplicated(out, dout), BatchDuplicated(x, dx),
        Const(ModuleLookupValues), Const(:square)
    )
    @test out[] == 9.0
    @test (dx[1][], dx[2][]) == (6.0, 18.0)
end
