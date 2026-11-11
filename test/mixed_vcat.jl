using Enzyme, Test

mixed_vcat_loss(x) = sum(abs2, vcat(view(x, 1:1), exp.(view(x, 2:length(x)))))
mixed_vcat_reversed_loss(x) = sum(abs2, vcat(exp.(view(x, 2:length(x))), view(x, 1:1)))
mixed_vcat_strided_loss(x) = sum(abs2, vcat(view(x, 1:2:length(x)), exp.(view(x, 2:2:length(x)))))
mixed_vcat_empty_loss(x) = sum(abs2, vcat(view(x, 1:0), x))
mixed_vcat_three_loss(x) = sum(abs2, vcat(view(x, 1:1), exp.(view(x, 2:length(x))), ones(eltype(x), 2)))
mixed_vcat_integer_loss(x) = sum(abs2, vcat(view(x, 1:length(x)), [1, 2]))
mixed_vcat_ordered_loss(x) = sum(abs2, cumsum(vcat(view(x, 1:1), exp.(view(x, 2:length(x))))))
mixed_vcat_alias_loss(x) = sum(abs2, vcat(view(x, 1:1), x))

function mixed_tuple_loss(x)
    parts = (view(x, 1:1), copy(x))
    return sum(abs2, parts[length(x)])
end

function mixed_tuple_lazy_loss(x)
    parts = (view(x, 1:1), copy(x))
    length(x) > 2 && return zero(eltype(x))
    return sum(abs2, parts[length(x)])
end

function mixed_vcat_expected(x)
    expected = similar(x)
    expected[1] = 2x[1]
    for i in 2:length(x)
        expected[i] = 2exp(2x[i])
    end
    return expected
end

function mixed_vcat_strided_expected(x)
    expected = similar(x)
    for i in eachindex(x)
        expected[i] = isodd(i) ? 2x[i] : 2exp(2x[i])
    end
    return expected
end

function mixed_vcat_ordered_expected(x)
    # d(sum(y.^2))/dx_j = 2 * d(increment_j)/dx_j * sum(y[j:end]).
    y = cumsum(vcat(x[1:1], exp.(x[2:length(x)])))
    expected = similar(x)
    expected[1] = 2sum(y)
    for i in 2:length(x)
        expected[i] = 2exp(x[i]) * sum(view(y, i:length(y)))
    end
    return expected
end

@testset "Mixed vector and dense-array-view vcat" begin
    for T in (Float32, Float64), n in (1, 2, 7, 33)
        x = T[0.31sin(0.7i) - 0.17cos(0.3i) for i in 1:n]
        before = copy(x)
        alias_expected = 2 .* x
        alias_expected[1] += 2x[1]
        for (loss, expected) in (
                (mixed_vcat_loss, mixed_vcat_expected(x)),
                (mixed_vcat_reversed_loss, mixed_vcat_expected(x)),
                (mixed_vcat_strided_loss, mixed_vcat_strided_expected(x)),
                (mixed_vcat_empty_loss, 2 .* x),
                (mixed_vcat_three_loss, mixed_vcat_expected(x)),
                (mixed_vcat_integer_loss, 2 .* x),
                (mixed_vcat_ordered_loss, mixed_vcat_ordered_expected(x)),
                (mixed_vcat_alias_loss, alias_expected),
                (mixed_tuple_lazy_loss, n <= 2 ? 2 .* x : zero(x)),
            )
            @testset "$(nameof(loss)) / $T / n=$n" begin
                dx = zero(x)
                primal = autodiff(ReverseWithPrimal, loss, Active, Duplicated(x, dx))[2]
                @test primal ≈ loss(x)
                @test dx ≈ expected
                @test autodiff(Forward, loss, Duplicated(x, ones(T, length(x))))[1] ≈ sum(expected)
                @test x == before
            end
        end
    end
end

@testset "Runtime tuple bounds" begin
    for T in (Float32, Float64)
        for n in (1, 2)
            x = T[0.1i for i in 1:n]
            dx = zero(x)
            @test autodiff(ReverseWithPrimal, mixed_tuple_loss, Active, Duplicated(x, dx))[2] ≈ mixed_tuple_loss(x)
            @test dx ≈ 2 .* x
            @test autodiff(Forward, mixed_tuple_loss, Duplicated(x, ones(T, n)))[1] ≈ 2sum(x)
        end
        x = T[0.1, 0.2, 0.3]
        before = copy(x)
        @test_throws BoundsError mixed_tuple_loss(x)
        @test_throws BoundsError autodiff(ReverseWithPrimal, mixed_tuple_loss, Active, Duplicated(x, zero(x)))
        @test_throws BoundsError autodiff(Forward, mixed_tuple_loss, Duplicated(x, ones(T, length(x))))
        @test x == before
    end
end
