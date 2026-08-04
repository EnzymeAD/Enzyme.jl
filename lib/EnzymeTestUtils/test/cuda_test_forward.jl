# TODO needs https://github.com/JuliaGPU/CUDA.jl/pull/3214 to work

#=using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using MetaTesting
using Test

f_tuple(x) = (-3 * x[1], 2 * x[2])
f_namedtuple(x) = (s = abs2(x.a), c = x.b^3)
f_struct(x::TestStruct) = TestStruct((x.a .* x.x) .^ 3, x.a^4)
function f_mut_fwd!(y, x, a)
    map!(xi -> xi * a, y, x)
    return y
end

f_kwargs_fwd(x; a = 3.0, kwargs...) = a .* x .^ 2

function f_kwargs_fwd!(x; kwargs...)
    copyto!(x, f_kwargs_fwd(x; kwargs...))
    return nothing
end

function EnzymeRules.forward(
        config,
        func::Const{typeof(f_kwargs_fwd)},
        RT::Type{
            <:Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed},
        },
        x::Union{Const, Duplicated, BatchDuplicated};
        a = 4.0, # mismatched keyword
        incorrect_primal = false,
        incorrect_tangent = false,
        incorrect_batched_tangent = false,
        kwargs...,
    )
    if RT <: Const
        return func.val(x.val; a = (incorrect_primal ? a - 1 : a), kwargs...)
    end
    dval = if x isa Duplicated
        2 * (incorrect_tangent ? (a + 2) : a) .* x.val .* x.dval
    elseif x isa BatchDuplicated
        map(x.dval) do dx
            2 * (incorrect_batched_tangent ? (a - 2) : a) .* x.val .* dx
        end
    else
        (incorrect_tangent | incorrect_batched_tangent) ? 2 * x.val : zero(a) * x.val
    end

    if RT <: Union{DuplicatedNoNeed, BatchDuplicatedNoNeed}
        return dval
    else
        val = func.val(x.val; a = (incorrect_primal ? a - 1 : a), kwargs...)
        RT <: Duplicated && return Duplicated(val, dval)
        RT <: BatchDuplicated && return BatchDuplicated(val, dval)
    end
end

@testset "test_forward" begin
    @testset "tests pass for functions with no rules" begin
        @testset "unary function tests" begin
            combinations = [
                "vector arguments" => (CuVector, f_array),
                "matrix arguments" => (CuMatrix, f_array),
                "multidimensional array arguments" => (CuArray{<:Any, 3}, f_array),
            ]
            sz = (2, 3, 4)
            @testset "$name" for (name, (TT, fun)) in combinations
                @testset for Tret in (
                            Const,
                            Duplicated,
                            DuplicatedNoNeed,
                            BatchDuplicated,
                            BatchDuplicatedNoNeed,
                        ),
                        Tx in (Const, Duplicated, BatchDuplicated),
                        T in (Float32, Float64, ComplexF64)

                    # skip invalid combinations
                    are_activities_compatible(Tret, Tx) || continue

                    x = CuArray(randn(T, sz[1:ndims(TT)]))
                    atol = rtol = sqrt(eps(real(T)))
                    runtime_activity = TT <: TestStruct && (Tret <: Const)
                    test_forward(fun, Tret, (x, Tx); atol, rtol, runtime_activity)
                end
            end
        end

        @testset "multi-argument function" begin
            @testset for Tret in (
                        Const,
                        Duplicated,
                        DuplicatedNoNeed,
                        BatchDuplicated,
                        BatchDuplicatedNoNeed,
                    ),
                    Tx in (Const, Duplicated, BatchDuplicated),
                    Ta in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                # skip invalid combinations
                are_activities_compatible(Tret, Tx, Ta) || continue

                x = CuArray(randn(T, 3))
                a = randn(T)
                atol = rtol = sqrt(eps(real(T)))

                @test !fails() do
                    test_forward(f_multiarg, Tret, (x, Tx), (a, Ta); atol, rtol)
                end
            end
        end

        @testset "structured array inputs/outputs" begin
            @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                # if some are batch, none must be duplicated
                are_activities_compatible(Tret, Tx) || continue

                x = Hermitian(CuArray(randn(T, 5, 5)))

                atol = rtol = sqrt(eps(real(T)))
                test_forward(f_structured_array, Tret, (x, Tx); atol, rtol)
            end
        end

        @testset "equivalent arrays in output" begin
            function f(x)
                z = x * 2
                return (z, z)
            end
            x = CuArray(randn(2, 3))
            @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Tx) || continue
                test_forward(f, Tret, (x, Tx))
            end
        end

        @testset "arrays sharing memory in output" begin
            function f(x)
                z = x * 2
                return (z, z)
            end
            x = CuArray(randn(2, 3))
            @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Tx) || continue
                test_forward(f, Tret, (x, Tx))
            end
        end

        @testset "mutating function" begin
            sz = (2, 3)
            @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated),
                    Ta in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                # if some are batch, all non-Const must be batch
                are_activities_compatible(Tret, Tx, Ta) || continue
                # since y is returned, it needs the same activity as the return type
                Ty = Tret

                x = CuArray(randn(T, sz))
                y = CuArray(zeros(T, sz))
                a = randn(T)

                atol = rtol = sqrt(eps(real(T)))
                @test !fails() do
                    test_forward(f_mut_fwd!, Tret, (y, Ty), (x, Tx), (a, Ta); atol, rtol, runtime_activity = true)
                end
            end
        end

        @testset "incorrect mutated argument detected" begin
            @testset for Tx in (Const, Duplicated)
                x = CuArray(randn(3))
                a = randn()

                test_forward(f_kwargs_fwd!, Const, (x, Tx); fkwargs = (; a))
                fkwargs = (; a, incorrect_primal = true)
                @test fails() do
                    return test_forward(f_kwargs_fwd!, Const, (x, Tx); fkwargs)
                end
            end
        end

        @testset "mutated callable" begin
            n = 3
            @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tc in (Const, Duplicated, BatchDuplicated),
                    Ty in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF64)

                # if some are batch, all non-Const must be batch
                are_activities_compatible(Tret, Tc, Ty) || continue

                c = MutatedCallable(CuArray(randn(T, n)))
                y = CuArray(randn(T, n))

                atol = rtol = sqrt(eps(real(T)))
                @test !fails() do
                    test_forward((c, Tc), Tret, (y, Ty); atol, rtol)
                end
            end
        end
    end
end=#
