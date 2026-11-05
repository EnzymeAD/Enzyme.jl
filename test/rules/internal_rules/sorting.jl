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

    z = Float32[3.0 1.0 -2.0; 2.0 4.0 0.5; 0.5 2.5 1.5]
    w = Float32[0.7 -1.2 0.3; 1.1 0.4 -0.8; -0.5 0.9 0.2]
    dx1 = Float32[1 0 0; 0 1 0; 0 0 1]
    dx2 = Float32[0 1 0; 0 0 1; 1 0 0]

    for d in (1, 2), rev in (false, true)
        inds = sortperm(z; dims = d, rev = rev)
        expected_grad = zeros(Float32, size(z))
        copyto!(view(expected_grad, inds), w)

        f_sort(x, wt) = sum(wt .* sort(x; dims = d, rev = rev))
        f_sort_accum(x, wt) = sum(wt .* sort(x; dims = d, rev = rev)) + 2 * sum(x)
        f_sort!(x, wt) = (y = copy(x); sort!(y; dims = d, rev = rev); sum(wt .* y))

        gz = zero(z)
        autodiff(Reverse, f_sort, Active, Duplicated(z, gz), Const(w))
        @test gz ≈ expected_grad

        gz_accum = zero(z)
        autodiff(Reverse, f_sort_accum, Active, Duplicated(z, gz_accum), Const(w))
        @test gz_accum ≈ expected_grad .+ 2

        gz1, gz2 = zero(z), zero(z)
        autodiff(Reverse, f_sort, Active, BatchDuplicated(z, (gz1, gz2)), Const(w))
        @test gz1 ≈ expected_grad
        @test gz2 ≈ expected_grad

        @test autodiff(Forward, f_sort, Duplicated(z, dx1), Const(w))[1] ≈ sum(expected_grad .* dx1)
        bdup = autodiff(Forward, f_sort, BatchDuplicated(z, (dx1, dx2)), Const(w))[1]
        @test bdup[1] ≈ sum(expected_grad .* dx1)
        @test bdup[2] ≈ sum(expected_grad .* dx2)

        gz_const = zero(w)
        autodiff(Reverse, f_sort, Active, Const(z), Duplicated(w, gz_const))
        @test gz_const ≈ sort(z; dims = d, rev = rev)
        @test autodiff(Forward, f_sort, Const(z), Duplicated(w, w))[1] ≈ sum(w .* sort(z; dims = d, rev = rev))

        gz_inplace = zero(z)
        autodiff(Reverse, f_sort!, Active, Duplicated(z, gz_inplace), Const(w))
        @test gz_inplace ≈ expected_grad

        gzi1, gzi2 = zero(z), zero(z)
        autodiff(Reverse, f_sort!, Active, BatchDuplicated(z, (gzi1, gzi2)), Const(w))
        @test gzi1 ≈ expected_grad
        @test gzi2 ≈ expected_grad
    end
end
