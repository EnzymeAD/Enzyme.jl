using Enzyme
using Enzyme: batchify_activity
using Test

@testset "Batchify activity" begin
    @test batchify_activity(Active{Float64}, Val(2)) == Active{Float64}
    @test batchify_activity(Duplicated{Vector{Float64}}, Val(2)) == BatchDuplicated{Vector{Float64},2}
    @test batchify_activity(DuplicatedNoNeed{Vector{Float64}}, Val(2)) == BatchDuplicatedNoNeed{Vector{Float64},2}
    @test batchify_activity(MixedDuplicated{Tuple{Float64,Vector{Float64}}}, Val(2)) == BatchMixedDuplicated{Tuple{Float64,Vector{Float64}},2}
end

# the base case is a function returning (a(x, y), b(x, y))

a(x::Vector{Float64}, y::Float64) = sum(abs2, x) * y
b(x::Vector{Float64}, y::Float64) = sum(x) * abs2(y)

struct MyStruct
    bar::Float64
    foo::Float64
end

mutable struct MyMutableStruct
    bar::Float64
    foo::Float64
end

Base.:(==)(s1::MyMutableStruct, s2::MyMutableStruct) = s1.bar == s2.bar && s1.foo == s2.foo

struct MyMixedStruct
    bar::Float64
    foo::Vector{Float64}
end

f1(x, y) = a(x, y) + b(x, y)
f2(x, y) = [a(x, y), b(x, y)]
f3(x, y) = (a(x, y), b(x, y))
f4(x, y) = MyStruct(a(x, y), b(x, y))
f5(x, y) = MyMutableStruct(a(x, y), b(x, y))
f6(x, y) = MyMixedStruct(a(x, y), [b(x, y)])

x = [1.0, 2.0, 3.0]
y = 4.0

# output seeds, (a,b) case

da = 5.0
db = 7.0
das = (5.0, 11.0)
dbs = (7.0, 13.0)

# input derivatives, (a,b) case

dx_ref = da * 2x * y .+ db * abs2(y)
dy_ref = da * sum(abs2, x) + db * sum(x) * 2y
dxs_ref = (
    das[1] * 2x * y .+ dbs[1] * abs2(y),
    das[2] * 2x * y .+ dbs[2] * abs2(y)
)
dys_ref = (
    das[1] * sum(abs2, x) + dbs[1] * sum(x) * 2y,
    das[2] * sum(abs2, x) + dbs[2] * sum(x) * 2y
)

# input derivatives, (a+b) case

dx1_ref = (da + db) * (2x * y .+ abs2(y))
dy1_ref = (da + db) * (sum(abs2, x) + sum(x) * 2y)
dxs1_ref = (
    (das[1] + dbs[1]) * (2x * y .+ abs2(y)),
    (das[2] + dbs[2]) * (2x * y .+ abs2(y))
)
dys1_ref = (
    (das[1] + dbs[1]) * (sum(abs2, x) + sum(x) * 2y),
    (das[2] + dbs[2]) * (sum(abs2, x) + sum(x) * 2y)
)

# output seeds, weird cases

dz1 = da + db
dzs1 = das .+ dbs

dz2 = [da, db]
dzs2 = ([das[1], dbs[1]], [das[2], dbs[2]])

dz3 = (da, db)
dzs3 = ((das[1], dbs[1]), (das[2], dbs[2]))

dz4 = MyStruct(da, db)
dzs4 = (MyStruct(das[1], dbs[1]), MyStruct(das[2], dbs[2]))

dz5 = MyMutableStruct(da, db)
dzs5 = (MyMutableStruct(das[1], dbs[1]), MyMutableStruct(das[2], dbs[2]))

dz6 = MyMixedStruct(da, [db])
dzs6 = (MyMixedStruct(das[1], [dbs[1]]), MyMixedStruct(das[2], [dbs[2]]))

# validation

function validate_seeded_autodiff(f, dz, dzs)
    @testset for mode in (Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal)
        @testset "Simple" begin
            dx = make_zero(x)
            dinputs_and_maybe_result = autodiff(mode, Const(f), Seed(dz), Duplicated(x, dx), Active(y))
            dinputs = first(dinputs_and_maybe_result)
            @test isnothing(dinputs[1])
            if f === f1
                @test dinputs[2] == dy1_ref
                @test dx == dx1_ref
            else
                @test dinputs[2] == dy_ref
                @test dx == dx_ref
            end
            if Enzyme.Split(mode) == ReverseSplitWithPrimal
                @test last(dinputs_and_maybe_result) == f(x, y)
            end
        end

        @testset "Batch" begin
            dxs = (make_zero(x), make_zero(x))
            dinputs_and_maybe_result = autodiff(mode, Const(f), BatchSeed(dzs), BatchDuplicated(x, dxs), Active(y))
            dinputs = first(dinputs_and_maybe_result)
            @test isnothing(dinputs[1])
            if f === f1
                @test dinputs[2][1] == dys1_ref[1]
                @test dinputs[2][2] == dys1_ref[2]
                @test dxs[1] == dxs1_ref[1]
                @test dxs[2] == dxs1_ref[2]
            else
                @test dinputs[2][1] == dys_ref[1]
                @test dinputs[2][2] == dys_ref[2]
                @test dxs[1] == dxs_ref[1]
                @test dxs[2] == dxs_ref[2]
            end
            if Enzyme.Split(mode) == ReverseSplitWithPrimal
                @test last(dinputs_and_maybe_result) == f(x, y)
            end
        end
    end
end

@testset "Scalar output" begin
    validate_seeded_autodiff(f1, dz1, dzs1)
end;

@testset "Vector output" begin
    validate_seeded_autodiff(f2, dz2, dzs2)
end;

@testset "Tuple output" begin
    validate_seeded_autodiff(f3, dz3, dzs3)
end;

@testset "Struct output" begin
    validate_seeded_autodiff(f4, dz4, dzs4)
end;

@testset "Mutable struct output" begin
    validate_seeded_autodiff(f5, dz5, dzs5)
end;

@testset "Mixed struct output" begin
    validate_seeded_autodiff(f6, dz6, dzs6)  # TODO: debug this
end;
