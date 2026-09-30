using Enzyme
using EnzymeTestUtils
using LinearAlgebra
using MetaTesting
using Test
using CUDA

# Stands in for `f_array`, which is `sum(abs2, x)`. With no rule for that `f`, Enzyme
# descends into the CUDA reduction kernel, which fails to compile on Julia 1.10 and 1.11.
# A plain `sum` goes to the `GPUArrays._mapreduce` rule instead.
f_gpu_array(x) = sum(x)

f_output_tangent(x) = 2 .* x

# Stands in for `f_structured_array`. CUDA cannot compile the broadcast kernel for
# `LowerTriangular(y.data) .*= 2` on Julia 1.10 and 1.11, so scale the same triangle with a
# dense mask instead.
function f_gpu_structured_array(x::Hermitian)
    y = x * 3
    n = size(y, 1)
    tri = y.uplo == 'U' ? tril(ones(real(eltype(y)), n, n)) : triu(ones(real(eltype(y)), n, n))
    # mutate the unused triangle, which ensures that our Jacobian differs from FiniteDifferences
    y.data .*= 1 .+ CuArray(tri)
    return y
end

# Reverse mode cannot yet go through a GPU broadcast that also captures a scalar, as in
# `abs2.(a .* x)`, `c.x ./= s` or `map!(xi -> xi * a, y, x)`. Enzyme passes the broadcast to
# the KernelAbstractions kernel as a `MixedDuplicated`, whose host `Ref` shadow cannot be
# converted for the device. See EnzymeAD/Enzyme.jl#3704 and
# JuliaGPU/KernelAbstractions.jl#806. Turn the scalar bcast tests on once that works.

function f_mut_rev!(y, x, a)
    map!(xi -> xi * a, y, x)
    return y
end

f_kwargs_rev(x; a = 3.0, kwargs...) = a .* x .^ 2

function f_kwargs_rev!(x; kwargs...)
    copyto!(x, f_kwargs_rev(x; kwargs...))
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(f_kwargs_rev)},
        RT::Type{<:Union{Const, Duplicated, DuplicatedNoNeed}},
        x::Union{Const, Duplicated};
        a = 4.0, # mismatched keyword
        incorrect_primal = false,
        incorrect_tape = false,
        kwargs...,
    )
    xtape = incorrect_tape ? x.val * 3 : copy(x.val)
    if EnzymeRules.needs_primal(config) || EnzymeRules.needs_shadow(config)
        val = func.val(x.val; a = (incorrect_primal ? a - 1 : a), kwargs...)
    else
        val = nothing
    end
    primal = EnzymeRules.needs_primal(config) ? val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(val) : nothing
    tape = (xtape, shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(f_kwargs_rev)},
        dret::Type{<:Union{Const, Duplicated, DuplicatedNoNeed}},
        tape,
        x::Union{Const, Duplicated};
        a = 4.0, # mismatched keyword
        incorrect_tangent = false,
        kwargs...,
    )
    xval, dval = tape
    if !(x isa Const) && (dval !== nothing)
        x.dval .+= 2 .* (incorrect_tangent ? (a + 2) : a) .* dval .* xval
    end
    return (nothing,)
end

@testset "test_reverse" begin
    @testset "tests pass for functions with no rules" begin
        @testset "unary function tests" begin
            combinations = [
                "vector arguments" => (CuVector, f_gpu_array),
                "matrix arguments" => (CuMatrix, f_gpu_array),
                "multidimensional array arguments" => (CuArray{<:Any, 3}, f_gpu_array),
            ]
            sz = (2, 3, 4)
            @testset "$name" for (name, (TT, fun)) in combinations
                @testset for Tret in (Active,),
                        Tx in (Const, Duplicated, BatchDuplicated),
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                    are_activities_compatible(Tret, Tx) || continue
                    x = CuArray(randn(T, sz[1:ndims(TT)]))
                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse(fun, Tret, (x, Tx); atol, rtol)
                end
            end
        end

        # also, with only `Const` inputs Enzyme wrongly asks for runtime activity
        @static if VERSION >= v"1.12.0-rc"
            @testset "multi-argument function" begin
                @testset for Tret in (Const, Duplicated),
                        Tx in (Const, Duplicated),
                        Ta in (Const, Active),
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                    x = CuArray(randn(T, 3))
                    a = randn(T)
                    atol = rtol = sqrt(eps(real(T)))
                    @test !fails() do
                        test_reverse(f_multiarg, Tret, (x, Tx), (a, Ta); atol, rtol)
                    end
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
                # as for the arrays in output below, Enzyme reports a need for runtime
                # activity here when `x` is `Const`
                test_reverse(
                    f_gpu_structured_array, Tret, (x, Tx); atol, rtol, runtime_activity = true
                )
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
                # unlike the CPU equivalent, Enzyme reports a need for runtime activity
                # here when `x` is `Const`
                test_reverse(f, Tret, (x, Tx); runtime_activity = true)
            end
        end

        @testset "arrays sharing memory in output" begin
            function f(x)
                z = x * 2
                return (z, vec(z))
            end
            x = CuArray(randn(2, 3))
            @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated)

                are_activities_compatible(Tret, Tx) || continue
                # unlike the CPU equivalent, Enzyme reports a need for runtime activity
                # here when `x` is `Const`
                test_reverse(f, Tret, (x, Tx); runtime_activity = true)
            end
        end

        @testset "device output_tangent" begin
            # `output_tangent` is the only way the cotangent reaches `j′vp` already on the
            # device, where it is read one element at a time.
            # only a `Duplicated` return takes a cotangent; with a `Const` return finite
            # differences would still apply `output_tangent`, but Enzyme rightly ignores it
            @testset for Tret in (Duplicated,),
                    T in (Float64, ComplexF64)

                x = CuArray(randn(T, 3))
                ȳ = CuArray(randn(T, 3))
                atol = rtol = sqrt(eps(real(T)))
                test_reverse(
                    f_output_tangent, Tret, (x, Duplicated); output_tangent = ȳ, atol, rtol
                )
            end
        end

        # also, with a `Duplicated` `y` Enzyme emits invalid IR, which aborts the process
        @static if VERSION >= v"1.12.0-rc"
            @testset "mutating function" begin
                sz = (2, 3)
                @testset for Ty in (Const, Duplicated, BatchDuplicated),
                        Tx in (Const, Duplicated, BatchDuplicated),
                        Ta in (Const, Active),
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                    # if some are batch, none must be duplicated
                    are_activities_compatible(Ty, Tx, Ta) || continue

                    x = CuArray(randn(T, sz))
                    y = CuArray(zeros(T, sz))
                    a = randn(T)

                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse(f_mut_rev!, Ty, (y, Ty), (x, Tx), (a, Ta); atol, rtol, runtime_activity = true)
                end
            end
           @testset "mutated callable" begin
                n = 3
                @testset for Tret in (Const, Active),
                        Tc in (Const, Duplicated),
                        Ty in (Const, Duplicated),
                        T in (Float32, Float64, ComplexF64)

                    # if some are batch, none must be duplicated
                    are_activities_compatible(Tret, Tc, Ty) || continue

                    c = MutatedCallable(CuArray(randn(T, n)))
                    y = CuArray(randn(T, n))

                    atol = rtol = sqrt(eps(real(T)))

                    if Tc <: BatchDuplicated && Ty <: BatchDuplicated
                        @test !fails() do
                            test_reverse((c, Tc), Tret, (y, Ty); atol, rtol)
                        end
                    else
                        @test !fails() do
                            test_reverse((c, Tc), Tret, (y, Ty); atol, rtol)
                        end
                    end
                end
            end
        end
    end

    @testset "kwargs correctly forwarded" begin
        @testset for Tx in (Const, Duplicated)
            x = CuArray(randn(3))
            a = randn()

            @test fails() do
                test_reverse(f_kwargs_rev, Duplicated, (x, Tx))
            end
            test_reverse(f_kwargs_rev, Duplicated, (x, Tx); fkwargs = (; a))
        end
    end

    @testset "incorrect mutated argument detected" begin
        @testset for Tx in (Const, Duplicated)
            x = CuArray(randn(3))
            a = randn()

            test_reverse(f_kwargs_rev!, Const, (x, Tx); fkwargs = (; a))
            fkwargs = (; a, incorrect_primal = true)
            @test fails() do
                test_reverse(f_kwargs_rev!, Const, (x, Tx); fkwargs)
            end
        end
    end

    @testset "incorrect tangent detected" begin
        @testset for Tx in (Duplicated,)
            x = CuArray(randn(3))
            a = randn()

            test_reverse(f_kwargs_rev, Duplicated, (x, Tx); fkwargs = (; a))
            fkwargs = (; a, incorrect_tangent = true)
            @test fails() do
                test_reverse(f_kwargs_rev, Duplicated, (x, Tx); fkwargs)
            end
        end
    end
end
