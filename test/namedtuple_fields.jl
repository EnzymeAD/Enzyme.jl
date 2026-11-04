using Enzyme, Test

@inline function namedtuple_store!(destination, index, value)
    copyto!(selectdim(destination, ndims(destination), index), value)
    return destination
end

struct RecordAccumulator{T}
    out::T
end

function (acc::RecordAccumulator)(x)
    acc.out[1] = 3acc.out[1] + sum(x)
    return nothing
end

function (acc::RecordAccumulator)(x, y)
    acc.out[1] = 3acc.out[1] + sum(x) + sum(y)
    return nothing
end

# These methods do not apply to the original NamedTuple calls. A compiler
# replacement must not introduce them by converting the operands to Tuples.
Base.foreach(::RecordAccumulator, ::Tuple) = error("Tuple-only foreach")
Base.foreach(::RecordAccumulator, ::Tuple, ::Tuple) = error("Tuple-only foreach")

function namedtuple_foreach_loss(x, records)
    out = zeros(eltype(x), 1)
    result = foreach(RecordAccumulator(out), records(x)...)
    result === nothing || error("foreach must return nothing")
    return only(out)
end

single_record(x) = ((; a = x, b = sum(abs2, x), c = 2 .* x),)
unequal_records(x) = ((; a = x, b = sum(abs2, x), c = 2 .* x), (2 .* x, 3 .* x))
empty_records(x) = ((;), (; a = x, b = sum(abs2, x)))

@testset "NamedTuple foreach preserves order and shortest length" begin
    x = [1.0, 2.0, 3.0]
    for (records, expected) in (
            (single_record, 11 .+ 6 .* x),
            (unequal_records, 12 .+ 2 .* x),
            (empty_records, zero(x)),
        )
        dx = zero(x)
        primal = autodiff(ReverseWithPrimal, namedtuple_foreach_loss, Active, Duplicated(x, dx), Const(records))[2]
        @test primal == namedtuple_foreach_loss(x, records)
        @test dx == expected
        @test autodiff(Forward, namedtuple_foreach_loss, Duplicated(x, ones(3)), Const(records))[1] == sum(expected)
        @test x == [1.0, 2.0, 3.0]
    end
end

struct CustomRecordAccumulator{T}
    out::T
end

(acc::CustomRecordAccumulator)(x) = (acc.out[1] += sum(x); nothing)

function custom_record_loss(x)
    out = zeros(eltype(x), 1)
    foreach(CustomRecordAccumulator(out), (; a = x, b = sum(abs2, x)))
    return only(out)
end

@testset "NamedTuple foreach before a new method" begin
    x = [1.0, 2.0, 3.0]
    dx = zero(x)
    autodiff(Reverse, custom_record_loss, Active, Duplicated(x, dx))
    @test dx == 1 .+ 2 .* x
end

function Base.foreach(acc::CustomRecordAccumulator, record::NamedTuple)
    acc(record.a)
    acc(record.a)
    return nothing
end

@testset "NamedTuple foreach keeps a method added after compilation" begin
    x = [1.0, 2.0, 3.0]
    dx = zero(x)
    primal = autodiff(ReverseWithPrimal, custom_record_loss, Active, Duplicated(x, dx))[2]
    @test primal == 2sum(x)
    @test dx == fill(2.0, 3)
end

struct RuleRecordAccumulator{T}
    out::T
end

(acc::RuleRecordAccumulator)(x) = (acc.out[1] += sum(x); nothing)

function rule_record_loss(x)
    out = zeros(eltype(x), 1)
    foreach(RuleRecordAccumulator(out), (; a = x, b = sum(abs2, x)))
    return only(out)
end

@testset "NamedTuple foreach before a new rule" begin
    x = [1.0, 2.0, 3.0]
    @test autodiff(Forward, rule_record_loss, Duplicated(x, ones(3)))[1] == 15.0
end

function Enzyme.EnzymeRules.forward(
        config::Enzyme.EnzymeRules.FwdConfig,
        ::Const{typeof(foreach)},
        ::Type{<:Const},
        acc::Duplicated{<:RuleRecordAccumulator},
        record::Duplicated{<:NamedTuple},
    )
    foreach(acc.val, record.val)
    acc.dval.out[1] += 10sum(record.dval.a)
    return nothing
end

@testset "NamedTuple foreach keeps a rule added after compilation" begin
    x = [1.0, 2.0, 3.0]
    @test autodiff(Forward, rule_record_loss, Duplicated(x, ones(3)))[1] == 30.0
end

@inline function namedtuple_foreach_store!(destination, index, value)
    foreach((out, item) -> namedtuple_store!(out, index, item), destination, value)
    return destination
end

@inline function namedtuple_direct_store!(destination, index, value)
    namedtuple_store!(destination.tail, index, value.tail)
    namedtuple_store!(destination.total, index, value.total)
    return destination
end

function namedtuple_batch(x, store)
    out = (; tail = similar(x, size(x, 1) - 1, size(x, 2)), total = similar(x, size(x, 2)))
    for j in axes(x, 2)
        column = view(x, :, j)
        value = (; tail = column[2:end], total = sum(abs2, column))
        store(out, j, value)
    end
    return out
end

namedtuple_total_loss(x, store) = sum(namedtuple_batch(x, store).total)

function namedtuple_both_loss(x, store)
    out = namedtuple_batch(x, store)
    return sum(abs2, out.tail) + sum(out.total)
end

function namedtuple_index_loss(x, index)
    record = (; array = x, scalar = sum(abs2, x))
    return sum(record[index])
end

function namedtuple_lazy_index_loss(x, index)
    index > 2 && return zero(eltype(x))
    return namedtuple_index_loss(x, index)
end

@testset "Heterogeneous NamedTuple fields" begin
    @testset "Record storage / $T / width=$width / $store / $loss" for
        T in (Float32, Float64), width in (1, 4),
            store in (namedtuple_foreach_store!, namedtuple_direct_store!),
            loss in (namedtuple_total_loss, namedtuple_both_loss)
        x = reshape(T.(1:(4 * width)), width, 4)
        saved = copy(x)
        @test namedtuple_batch(x, store) ==
            (; tail = x[2:end, :], total = vec(sum(abs2, x; dims = 1)))
        expected = 2 .* x
        if loss === namedtuple_both_loss
            expected[2:end, :] .*= 2
        end
        dx = zero(x)
        primal = autodiff(ReverseWithPrimal, loss, Active, Duplicated(x, dx), Const(store))[2]
        @test primal == loss(x, store)
        @test dx == expected
        @test x == saved
    end

    @testset "Runtime index / $T" for T in (Float32, Float64)
        x = T[1, 2, 3]
        saved = copy(x)
        for index in (1, 2)
            expected = index == 1 ? ones(T, length(x)) : 2 .* x
            dx = zero(x)
            primal = autodiff(ReverseWithPrimal, namedtuple_index_loss, Active, Duplicated(x, dx), Const(index))[2]
            @test primal == namedtuple_index_loss(x, index)
            @test dx == expected
            @test x == saved
        end
        for index in (0, 3)
            @test_throws BoundsError namedtuple_index_loss(x, index)
            @test_throws BoundsError autodiff(Reverse, namedtuple_index_loss, Active, Duplicated(x, zero(x)), Const(index))
        end
        dx = zero(x)
        @test autodiff(ReverseWithPrimal, namedtuple_lazy_index_loss, Active, Duplicated(x, dx), Const(3))[2] == 0
        @test iszero(dx)
        @test x == saved
    end
end
