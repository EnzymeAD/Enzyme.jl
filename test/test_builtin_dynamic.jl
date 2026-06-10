module TestBuiltinDynamic
    using Enzyme
    using Test

    mutable struct MyStructBuiltinDynamic
        x::Float64
    end

    const f_ref_builtin_dynamic = Ref{Any}(Core.getfield)

    function loss_builtin_dynamic(obj)
        f = f_ref_builtin_dynamic[]
        val = f(obj, :x)::Float64
        return val * val
    end

    function run_test()
        @testset "Builtin dynamic dispatch runtime AD" begin
            obj = MyStructBuiltinDynamic(2.0)
            d_obj = MyStructBuiltinDynamic(0.0)
            autodiff(Reverse, loss_builtin_dynamic, Duplicated(obj, d_obj))
            @test d_obj.x ≈ 4.0
        end
    end
end

TestBuiltinDynamic.run_test()
