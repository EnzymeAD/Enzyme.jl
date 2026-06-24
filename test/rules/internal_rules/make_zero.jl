using Enzyme
using Test

@testset "make_zero forward rule" begin
    # 1. Primitive Float64
    f_float(x) = make_zero(x) + x
    @test Enzyme.autodiff(Forward, f_float, Duplicated(2.0, 3.0)) == (3.0,)
    
    res_float = Enzyme.autodiff(Forward, f_float, BatchDuplicated, BatchDuplicated(2.0, (3.0, 5.0)))[1]
    @test res_float[1] ≈ 3.0
    @test res_float[2] ≈ 5.0

    # 2. Vector{Float64}
    f_vector(x) = sum(make_zero(x)) + sum(x)
    @test Enzyme.autodiff(Forward, f_vector, Duplicated([1.0, 2.0], [10.0, 20.0])) == (30.0,)
    
    res_vector = Enzyme.autodiff(Forward, f_vector, BatchDuplicated, BatchDuplicated([1.0, 2.0], ([10.0, 20.0], [100.0, 200.0])))[1]
    @test res_vector[1] ≈ 30.0
    @test res_vector[2] ≈ 300.0

    # 3. Tuple
    f_tuple(x) = sum(make_zero((x, 2x))) + x
    @test Enzyme.autodiff(Forward, f_tuple, Duplicated(2.0, 3.0)) == (3.0,)

    # 4. Custom Mutable Struct
    mutable struct MyStruct
        a::Float64
        b::Vector{Float64}
    end
    function f_struct(s::MyStruct)
        sz = make_zero(s)
        return sz.a + sum(sz.b) + s.a + sum(s.b)
    end
    s = MyStruct(2.0, [3.0, 4.0])
    ds = MyStruct(10.0, [100.0, 1000.0])
    @test Enzyme.autodiff(Forward, f_struct, Duplicated(s, ds)) == (1110.0,)

    @testset "Nested AD with undef-able field struct" begin
        RA_R = Enzyme.set_runtime_activity(Enzyme.Reverse)
        RA_F = Enzyme.set_runtime_activity(Enzyme.Forward)
        struct Par
            w::Vector{Float64}
            extra::Any
        end
        loss1(p) = sum(abs2, p.w)
        p = Par([1.0, 2.0, 3.0], "ignored")
        dp = Par([1.0, 0.0, 0.0], "ignored")
        res = Enzyme.autodiff(RA_F, Enzyme.Const(x -> Enzyme.gradient(RA_R, loss1, x)[1]), Enzyme.Duplicated(p, dp))
        dpar = res[1]
        @test dpar.w ≈ [2.0, 0.0, 0.0]
    end

    @testset "Reverse mode make_zero rule" begin
        struct MyRevStruct
            data::Vector{Float64}
        end
        function f_rev(x::Vector{Float64})
            s = MyRevStruct(x)
            z = Enzyme.make_zero(s)
            return sum(z.data) + sum(x)
        end
        @test Enzyme.gradient(Enzyme.Reverse, Const(f_rev), [1.0, 2.0]) ≈ [1.0, 1.0]
    end
    
end
