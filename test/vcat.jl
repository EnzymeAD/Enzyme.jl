using Enzyme, Test

vcat_exp_loss(u, c) = sum(vcat(c, exp.(u)))
vcat_exp_control(u, c) = sum(c) + sum(exp, u; init = 0.0)
vcat_squared_loss(a, b, c) = sum(abs2, vcat(a, b, c))
vcat_alias_loss(x, c) = sum(abs2, vcat(c, x, x, c))
vcat_exp_vector(u, c) = vcat(c, exp.(u))
vcat_many_loss(x, c) = sum(vcat(c, x, c, x, c, x, c, x, c, x, c, x, c, x, c, x))
vcat_single_loss(x) = sum(abs2, vcat(x))

@testset "Dense vcat preserves each argument's activity" begin
    for T in (Float32, Float64), n in (0, 1, 4)
        u = T[0.2 * i - 0.5 for i in 1:n]
        c = T[3.2, 5.1]
        for loss in (vcat_exp_loss, vcat_exp_control)
            du = fill(T(2), n)
            result = autodiff(ReverseWithPrimal, loss, Active, Duplicated(u, du), Const(c))
            @test result[2] ≈ loss(u, c)
            @test du ≈ fill(T(2), n) + exp.(u)
            @test u == T[0.2 * i - 0.5 for i in 1:n]
            @test c == T[3.2, 5.1]
        end

        tangent = fill(T(2), n)
        result = autodiff(ForwardWithPrimal, vcat_exp_vector, Duplicated, Duplicated(u, tangent), Const(c))
        @test result[2] ≈ vcat_exp_vector(u, c)
        @test result[1] ≈ vcat(zero(c), T(2) .* exp.(u))
        @test tangent == fill(T(2), n)
        @test c == T[3.2, 5.1]

        du1, du2 = zero(u), fill(T(2), n)
        autodiff(Reverse, vcat_exp_loss, Active, BatchDuplicated(u, (du1, du2)), Const(c))
        @test du1 ≈ exp.(u)
        @test du2 ≈ fill(T(2), n) + exp.(u)
        @test c == T[3.2, 5.1]

        result = autodiff(
            Forward, vcat_exp_loss, BatchDuplicated,
            BatchDuplicated(u, (ones(T, n), fill(T(2), n))), Const(c),
        )
        expected = sum(exp, u; init = zero(T))
        @test result[1][1] ≈ expected
        @test result[1][2] ≈ 2 * expected
    end

    for sizes in ((0, 0, 0), (0, 2, 1), (2, 0, 1), (2, 1, 0), (2, 1, 3))
        a, b, c = (collect(1.0:n) for n in sizes)
        da, db, dc = zero(a), zero(b), zero(c)
        autodiff(Reverse, vcat_squared_loss, Active, Duplicated(a, da), Const(b), Duplicated(c, dc))
        @test da ≈ 2 .* a
        @test dc ≈ 2 .* c
        @test b == collect(1.0:sizes[2])

        autodiff(Reverse, vcat_squared_loss, Active, Const(a), Duplicated(b, db), Const(c))
        @test db ≈ 2 .* b
        @test a == collect(1.0:sizes[1])
        @test c == collect(1.0:sizes[3])
    end

    x, c = [1.0, 2.0], [3.0]
    dx = zero(x)
    autodiff(Reverse, vcat_alias_loss, Active, Duplicated(x, dx), Const(c))
    @test dx ≈ 4 .* x
    @test c == [3.0]

    dx = zero(x)
    autodiff(set_runtime_activity(Reverse), vcat_alias_loss, Active, Duplicated(x, dx), Duplicated(c, c))
    @test dx ≈ 4 .* x
    @test c == [3.0]

    dx = zero(x)
    autodiff(Reverse, vcat_many_loss, Active, Duplicated(x, dx), Const(c))
    @test dx == fill(8.0, length(x))
    @test c == [3.0]

    dx = zero(x)
    autodiff(Reverse, vcat_single_loss, Active, Duplicated(x, dx))
    @test dx ≈ 2 .* x
end

struct VcatDispatchElement
    value::Float64
end

# A more specific method must retain its own semantics during differentiation.
Base.vcat(a::Vector{VcatDispatchElement}, b::Vector{VcatDispatchElement}) =
    [VcatDispatchElement(a[1].value + 2 * b[1].value)]
vcat_dispatch_loss(a, b) = only(vcat(a, b)).value

@testset "Dense vcat keeps custom dispatch" begin
    a, b = [VcatDispatchElement(3.0)], [VcatDispatchElement(5.0)]
    da, db = [VcatDispatchElement(0.0)], [VcatDispatchElement(0.0)]
    result = autodiff(ReverseWithPrimal, vcat_dispatch_loss, Active, Duplicated(a, da), Duplicated(b, db))
    @test result[2] == 13.0
    @test da[1].value == 1.0
    @test db[1].value == 2.0
end

struct VcatRuleElement
    value::Float64
end

vcat_rule_loss(a, b) = vcat(a, b)[1].value

@testset "Dense vcat before adding a rule" begin
    a, b = [VcatRuleElement(3.0)], [VcatRuleElement(5.0)]
    da = [VcatRuleElement(1.0)]
    result = autodiff(ForwardWithPrimal, vcat_rule_loss, Duplicated, Duplicated(a, da), Const(b))
    @test result[1] == 1.0
    @test result[2] == 3.0
end

function Enzyme.EnzymeRules.forward(
        config::Enzyme.EnzymeRules.FwdConfig,
        func::Const{typeof(vcat)},
        ::Type{<:Duplicated},
        a::Duplicated{Vector{VcatRuleElement}},
        b::Const{Vector{VcatRuleElement}},
    )
    primal = vcat(a.val, b.val)
    shadow = [VcatRuleElement(10 * a.dval[1].value), VcatRuleElement(0.0)]
    return Duplicated(primal, shadow)
end

@testset "Dense vcat keeps custom rules" begin
    a, b = [VcatRuleElement(3.0)], [VcatRuleElement(5.0)]
    da = [VcatRuleElement(1.0)]
    result = autodiff(ForwardWithPrimal, vcat_rule_loss, Duplicated, Duplicated(a, da), Const(b))
    @test result[1] == 10.0
    @test result[2] == 3.0
end
