using Enzyme
using EnzymeTestUtils
import Random
using Test

struct TPair
    a::Float64
    b::Float64
end

function sorterrfn(t, x)
    function lt(a, b)
        return a.a < b.a
    end
    return first(sortperm(t, lt = lt)) * x
end

@testset "Sort rules" begin
    function f1(x)
        a = [1.0, 3.0, x]
        sort!(a)
        return a[2]
    end

    @test autodiff(Forward, f1, Duplicated(2.0, 1.0))[1] == 1
    @test autodiff(Forward, f1, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1" = 1.0, var"2" = 2.0)
    @test autodiff(Reverse, f1, Active, Active(2.0))[1][1] == 1
    @test autodiff(Forward, f1, Duplicated(4.0, 1.0))[1] == 0
    @test autodiff(Forward, f1, BatchDuplicated(4.0, (1.0, 2.0)))[1] == (var"1" = 0.0, var"2" = 0.0)
    @test autodiff(Reverse, f1, Active, Active(4.0))[1][1] == 0

    function f2(x)
        a = [1.0, -3.0, -x, -2x, x]
        sort!(a; rev = true, lt = (x, y) -> abs(x) < abs(y) || (abs(x) == abs(y) && x < y))
        return sum(a .* [1, 2, 3, 4, 5])
    end

    @test autodiff(Forward, f2, Duplicated(2.0, 1.0))[1] == -3
    @test autodiff(Forward, f2, BatchDuplicated(2.0, (1.0, 2.0)))[1] == (var"1" = -3.0, var"2" = -6.0)
    @test autodiff(Reverse, f2, Active, Active(2.0))[1][1] == -3

    function f3(x)
        a = [2.0, 2.5, x, 1.0]
        return partialsort(a, 2)
    end

    @test autodiff(Forward, f3, Duplicated(1.5, 1.0))[1] == 1.0
    @test autodiff(Forward, f3, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1" = 1.0, var"2" = 2.0)
    @test autodiff(Reverse, f3, Active(1.5))[1][1] == 1.0
    @test autodiff(Reverse, f3, Active(2.5))[1][1] == 0.0

    function f4(x)
        a = [2.0, 2.5, x, x / 2]
        y = partialsort(a, 1:2)
        return sum(y)
    end

    @test autodiff(Forward, f4, Duplicated(1.5, 1.0))[1] == 1.5
    @test autodiff(Forward, f4, BatchDuplicated(1.5, (1.0, 2.0)))[1] == (var"1" = 1.5, var"2" = 3.0)
    @test autodiff(Reverse, f4, Active(1.5))[1][1] == 1.5
    @test autodiff(Reverse, f4, Active(4.0))[1][1] == 0.5
    @test autodiff(Reverse, f4, Active(6.0))[1][1] == 0.0

    dd = Duplicated([TPair(1, 2), TPair(2, 3), TPair(0, 1)], [TPair(0, 0), TPair(0, 0), TPair(0, 0)])
    res = Enzyme.autodiff(Reverse, sorterrfn, dd, Active(1.0))

    @test res[1][2] ≈ 3
    @test dd.dval[1].a ≈ 0
    @test dd.dval[1].b ≈ 0
    @test dd.dval[2].a ≈ 0
    @test dd.dval[2].b ≈ 0
    @test dd.dval[3].a ≈ 0
    @test dd.dval[3].b ≈ 0
end

function sort_resize(x)
    l = copy(x)
    sort!(l)
    resize!(l, 2)
    return l[1] + 2 * l[2]
end

function sort_unique(x)
    l = copy(x)
    push!(l, x[1])
    unique!(sort!(l))
    return l[1] + 2 * l[2]
end

function sort_push_pop(x)
    l = copy(x)
    sort!(l)
    push!(l, 0.0)
    pop!(l)
    return l[1] + 2 * l[2]
end

function sort_push(x)
    l = copy(x)
    sort!(l)
    for _ in 1:20
        push!(l, 1.0)
    end
    return l[1] + 2 * l[2]
end

function sort_popfirst(x)
    l = copy(x)
    sort!(l)
    popfirst!(l)
    return l[1] + 2 * l[2]
end

function sort_deleteat(x)
    l = copy(x)
    sort!(l)
    deleteat!(l, 1)
    return l[1] + 2 * l[2]
end

function partialsort_resize(x)
    l = copy(x)
    partialsort!(l, 1:2)
    resize!(l, 2)
    return l[1] + 2 * l[2]
end

function partialsort_index_resize(x)
    l = copy(x)
    m = partialsort!(l, 2)
    resize!(l, 1)
    return 3 * m + l[1]
end

function sort_data(k, d)
    l = Float64[d[j] for j in eachindex(d)]
    push!(l, d[1] + 1.0)
    unique!(sort!(l))
    s = 0.0
    for v in l
        s += k[1] * v
    end
    return s
end

@testset "Sort rules followed by an in-place resize" begin
    x = [3.0, 1.0, 4.0, 2.0, 5.0]
    two_smallest = [0.0, 1.0, 0.0, 2.0, 0.0]
    second_and_third = [2.0, 0.0, 0.0, 1.0, 0.0]
    @test Enzyme.gradient(Reverse, sort_resize, x)[1] == two_smallest
    @test Enzyme.gradient(Reverse, sort_unique, x)[1] == two_smallest
    @test Enzyme.gradient(Reverse, sort_push_pop, x)[1] == two_smallest
    @test Enzyme.gradient(Reverse, sort_push, x)[1] == two_smallest
    if VERSION >= v"1.11"
        @test Enzyme.gradient(Reverse, sort_popfirst, x)[1] == second_and_third
        @test Enzyme.gradient(Reverse, sort_deleteat, x)[1] == second_and_third
    else
        # no reverse rule for jl_array_del_beg / jl_array_del_at on Julia 1.10
        @test_broken Enzyme.gradient(Reverse, sort_popfirst, x)[1] == second_and_third
        @test_broken Enzyme.gradient(Reverse, sort_deleteat, x)[1] == second_and_third
    end
    @test Enzyme.gradient(Reverse, partialsort_resize, x)[1] == two_smallest
    @test Enzyme.gradient(Reverse, partialsort_index_resize, x)[1] == [0.0, 1.0, 0.0, 3.0, 0.0]
    @test Enzyme.gradient(Reverse, sort_data, [2.0], Const([3.0, 1.0, 3.0, 2.0]))[1] == [10.0]
    @test x == [3.0, 1.0, 4.0, 2.0, 5.0]
end
