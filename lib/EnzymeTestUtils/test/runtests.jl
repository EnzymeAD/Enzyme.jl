using Enzyme
using EnzymeTestUtils
using MetaTesting
using Test

f_array(x) = sum(exp, x)
f_tuple(x) = (-3 * x[1], 2 * x[2])
f_namedtuple(x) = (s=sin(x.a), c=cos(x.b))
struct TestStruct{X,A}
    x::X
    a::A
end
f_struct(x::TestStruct) = TestStruct(sinh.(x.a .* x.x), exp(x.a))
f_multiarg(x::AbstractArray, a) = sin.(a .* x)
function f_mut!(y, x, a)
    y .= x .* a
    return y
end

f_kwargs(x; a=3.0, kwargs...) = a .* x .^ 2

struct MutatedCallable{T}
    x::T
end
function (c::MutatedCallable)(y)
    s = c.x'y
    c.x ./= s
    return s
end

function EnzymeRules.forward(
    func::Const{typeof(f_kwargs)},
    RT::Type{
        <:Union{Const,Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}
    },
    x::Union{Const,Duplicated,BatchDuplicated};
    a=4.0, # mismatched keyword
    incorrect_primal=false,
    incorrect_tangent=false,
    incorrect_batched_tangent=false,
    kwargs...,
)
    if RT <: Const
        return func.val(x.val; a=(incorrect_primal ? a - 1 : a), kwargs...)
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

    if RT <: Union{DuplicatedNoNeed,BatchDuplicatedNoNeed}
        return dval
    else
        val = func.val(x.val; a=(incorrect_primal ? a - 1 : a), kwargs...)
        return RT(val, dval)
    end
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.ConfigWidth{1},
    func::Const{typeof(f_kwargs)},
    RT::Type{<:Union{Const,Duplicated,DuplicatedNoNeed}},
    x::Union{Const,Duplicated};
    a=4.0, # mismatched keyword
    incorrect_primal=false,
    incorrect_tape=false,
    kwargs...,
)
    xtape = incorrect_tape ? x.val * 3 : copy(x.val)
    if EnzymeRules.needs_primal(config) || EnzymeRules.needs_shadow(config)
        val = func.val(x.val; a=(incorrect_primal ? a - 1 : a), kwargs...)
    else
        val = nothing
    end
    primal = EnzymeRules.needs_primal(config) ? val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? zero(val) : nothing
    tape = (xtape, shadow)
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    config::EnzymeRules.ConfigWidth{1},
    func::Const{typeof(f_kwargs)},
    dret::Type{<:Union{Const,Duplicated,DuplicatedNoNeed}},
    tape,
    x::Union{Const,Duplicated};
    a=4.0, # mismatched keyword
    incorrect_tangent=false,
    kwargs...,
)
    xval, dval = tape
    if !(x isa Const) && (dval !== nothing)
        x.dval .+= 2 .* (incorrect_tangent ? (a + 2) : a) .* dval .* xval
    end
    return (nothing,)
end

@testset "EnzymeRules testing functions" begin
    @testset "test_forward" begin
        @testset "tests pass for functions with no rules" begin
            @testset "unary function tests" begin
                combinations = [
                    "vector arguments" => (Vector, f_array),
                    "matrix arguments" => (Matrix, f_array),
                    "multidimensional array arguments" => (Array{<:Any,3}, f_array),
                    "tuple argument and return" => (Tuple, f_tuple),
                    "namedtuple argument and return" => (NamedTuple, f_namedtuple),
                    # "struct argument and return" => (TestStruct, f_struct),
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
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                        # skip invalid combinations
                        all_or_no_batch(Tret, Tx) || continue

                        if TT <: Array
                            x = randn(T, sz[1:ndims(TT)])
                        elseif TT <: Tuple
                            x = (randn(T), randn(T))
                        elseif TT <: NamedTuple
                            x = (a=randn(T), b=randn(T))
                        else  # TT <: TestStruct
                            x = TestStruct(randn(T, 5), randn(T))
                        end
                        atol = rtol = sqrt(eps(real(T)))
                        test_forward(fun, Tret, (x, Tx); atol, rtol)
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
                    all_or_no_batch(Tret, Tx, Ta) || continue

                    x = randn(T, 3)
                    a = randn(T)
                    atol = rtol = sqrt(eps(real(T)))
                    test_forward(f_multiarg, Tret, (x, Tx), (a, Ta); atol, rtol)
                end
            end

            @testset "mutating function" begin
                Enzyme.API.runtimeActivity!(true)
                sz = (2, 3)
                @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tx in (Const, Duplicated, BatchDuplicated),
                    Ta in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # if some are batch, all non-Const must be batch
                    all_or_no_batch(Tret, Tx, Ta) || continue
                    # since y is returned, it needs the same activity as the return type
                    Ty = Tret

                    x = randn(T, sz)
                    y = zeros(T, sz)
                    a = randn(T)

                    atol = rtol = sqrt(eps(real(T)))
                    test_forward(f_mut!, Tret, (y, Ty), (x, Tx), (a, Ta); atol, rtol)
                end
                Enzyme.API.runtimeActivity!(false)
            end

            @testset "mutated callable" begin
                n = 3
                @testset for Tret in (Const, Duplicated, BatchDuplicated),
                    Tc in (Const, Duplicated, BatchDuplicated),
                    Ty in (Const, Duplicated, BatchDuplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # if some are batch, all non-Const must be batch
                    all_or_no_batch(Tret, Tc, Ty) || continue

                    c = MutatedCallable(randn(T, n))
                    y = randn(T, n)

                    atol = rtol = sqrt(eps(real(T)))
                    test_forward((c, Tc), Tret, (y, Ty); atol, rtol)
                end
            end
        end

        @testset "kwargs correctly forwarded" begin
            @testset for Tret in (Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated)

                all_or_no_batch(Tret, Tx) || continue

                x = randn(3)
                a = randn()

                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx))
                end
                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
            end
        end

        @testset "incorrect primal detected" begin
            @testset for Tret in (Duplicated, BatchDuplicated),
                Tx in (Const, Duplicated, BatchDuplicated)

                all_or_no_batch(Tret, Tx) || continue

                x = randn(3)
                a = randn()

                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_primal=true)
                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect tangent detected" begin
            @testset for Tret in (Duplicated, DuplicatedNoNeed), Tx in (Const, Duplicated)
                x = randn(3)
                a = randn()

                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_tangent=true)
                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect batch tangent detected" begin
            @testset for Tret in (BatchDuplicated, BatchDuplicatedNoNeed),
                Tx in (Const, BatchDuplicated)

                x = randn(3)
                a = randn()

                test_forward(f_kwargs, Tret, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_batched_tangent=true)
                @test fails() do
                    test_forward(f_kwargs, Tret, (x, Tx); fkwargs)
                end
            end
        end
    end

    @testset "test_reverse" begin
        @testset "tests pass for functions with no rules" begin
            @testset "unary function tests" begin
                combinations = [
                    "vector arguments" => (Vector, f_array),
                    "matrix arguments" => (Matrix, f_array),
                    "multidimensional array arguments" => (Array{<:Any,3}, f_array),
                ]
                sz = (2, 3, 4)
                @testset "$name" for (name, (TT, fun)) in combinations
                    @testset for Tret in (Active, Const),
                        Tx in (Const, Duplicated),
                        T in (Float32, Float64, ComplexF32, ComplexF64)

                        x = randn(T, sz[1:ndims(TT)])
                        atol = rtol = sqrt(eps(real(T)))
                        test_reverse(fun, Tret, (x, Tx); atol, rtol)
                    end
                end
            end

            @testset "multi-argument function" begin
                @testset for Tret in (Const, Duplicated),
                    Tx in (Const, Duplicated),
                    Ta in (Const, Active),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    x = randn(T, 3)
                    a = randn(T)
                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse(f_multiarg, Tret, (x, Tx), (a, Ta); atol, rtol)
                end
            end

            @testset "mutating function" begin
                sz = (2, 3)
                Enzyme.API.runtimeActivity!(true)
                @testset for Ty in (Const, Duplicated),
                    Tx in (Const, Duplicated),
                    Ta in (Const, Active),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    # return value is nothing
                    Tret = Const

                    x = randn(T, sz)
                    y = zeros(T, sz)
                    a = randn(T)

                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse(f_mut!, Tret, (y, Ty), (x, Tx), (a, Ta); atol, rtol)
                end
                Enzyme.API.runtimeActivity!(false)
            end

            @testset "mutated callable" begin
                n = 3
                @testset for Tret in (Const, Active),
                    Tc in (Const, Duplicated),
                    Ty in (Const, Duplicated),
                    T in (Float32, Float64, ComplexF32, ComplexF64)

                    c = MutatedCallable(randn(T, n))
                    y = randn(T, n)

                    atol = rtol = sqrt(eps(real(T)))
                    test_reverse((c, Tc), Tret, (y, Ty); atol, rtol)
                end
            end
        end

        @testset "kwargs correctly forwarded" begin
            @testset for Tx in (Const, Duplicated)
                x = randn(3)
                a = randn()

                @test fails() do
                    test_reverse(f_kwargs, Duplicated, (x, Tx))
                end
                test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs=(; a))
            end
        end

        @testset "incorrect primal detected" begin
            @testset for Tx in (Const, Duplicated)
                x = randn(3)
                a = randn()

                test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_primal=true)
                @test fails() do
                    test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect tangent detected" begin
            @testset for Tx in (Duplicated,)
                x = randn(3)
                a = randn()

                test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_tangent=true)
                @test fails() do
                    test_reverse(f_kwargs, Duplicated, (x, Tx); fkwargs)
                end
            end
        end

        @testset "incorrect tape detected" begin
            @testset for Tx in (Duplicated,)
                x = randn(3)
                a = randn()

                function f_kwargs_overwrite(x; kwargs...)
                    y = f_kwargs(x; kwargs...)
                    x[1] = 0.0
                    return y
                end

                test_reverse(f_kwargs_overwrite, Duplicated, (x, Tx); fkwargs=(; a))
                fkwargs = (; a, incorrect_tape=true)
                @test fails() do
                    test_reverse(f_kwargs_overwrite, Duplicated, (x, Tx); fkwargs)
                end
            end
        end
    end
end
