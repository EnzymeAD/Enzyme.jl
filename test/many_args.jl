using Enzyme, Test

# Julia's optimizer turns a splatted call into a direct call only while the splatted
# tuple has at most 32 elements (`max_tuple_splat`). Entry points that splat their
# annotated arguments (plus the seed and tape) used to dispatch dynamically and allocate
# on every call beyond that. Such calls must cost the same as with a single argument.

const MANY_ARGS_N = 40

many_args_term(x, c) = x[1] * c[1] + x[2] * c[2]

# `f(x, c₁, …, cₙ)` and `h(a₁, …, aₙ)`, summed with nested binary `+` so that the
# objectives themselves do not allocate (an n-ary `+` goes through Base's `afoldl` loop).
function many_args_objective(n)
    cs = [Symbol(:c, i) for i in 1:n]
    body = foldl((a, b) -> :($a + $b), [:(many_args_term(x, $c)) for c in cs])
    return @eval (x, $(cs...)) -> $body
end
function many_active_objective(n)
    as = [Symbol(:a, i) for i in 1:n]
    body = foldl((a, b) -> :($a + $b), [:($(as[i]) * $i) for i in 1:n])
    return Core.eval(@__MODULE__, Expr(:->, Expr(:tuple, as...), body))
end

const MANY_ARGS_CALLS = (
    :reverse_primal, :reverse_unwrapped, :reverse_guessed, :reverse_batch,
    :reverse_deferred, :forward, :forward_deferred, :split,
)

for n in (1, MANY_ARGS_N)
    consts = [:(Const(cs[$i])) for i in 1:n]
    ctys = [:(Const{typeof(cs[$i])}) for i in 1:n]
    actives = [:(Active(as[$i])) for i in 1:n]
    @eval begin
        many_args_split_thunks(f::F, x, cs::NTuple{$n, Any}) where {F} = autodiff_thunk(
            ReverseSplitWithPrimal, Const{F}, Active, Duplicated{typeof(x)}, $(ctys...)
        )
        many_args_call(::Val{:reverse_primal}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff(ReverseWithPrimal, Const(f), Active, Duplicated(x, g), $(consts...))
        many_args_call(::Val{:reverse_unwrapped}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff(Reverse, f, Active, Duplicated(x, g), $(consts...))
        many_args_call(::Val{:reverse_guessed}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff(Reverse, Const(f), Duplicated(x, g), $(consts...))
        many_args_call(::Val{:reverse_batch}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff(Reverse, Const(f), Active, BatchDuplicated(x, (g, g)), $(consts...))
        many_args_call(::Val{:reverse_deferred}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff_deferred(Reverse, Const(f), Active, Duplicated(x, g), $(consts...))
        many_args_call(::Val{:forward}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff(Forward, Const(f), Duplicated(x, dx), $(consts...))
        many_args_call(::Val{:forward_deferred}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F} =
            autodiff_deferred(Forward, Const(f), Duplicated, Duplicated(x, dx), $(consts...))
        function many_args_call(::Val{:split}, f::F, g, x, dx, cs::NTuple{$n, Any}, thunks) where {F}
            fwd, rev = thunks
            tape, y, _ = fwd(Const(f), Duplicated(x, g), $(consts...))
            rev(Const(f), Duplicated(x, g), $(consts...), 1.0, tape)
            return y
        end
        many_active_call(h::H, as::NTuple{$n, Float64}) where {H} = autodiff(Reverse, h, Active, $(actives...))
    end
end

# Calls whose operands are stored concretely, so that measuring one allocates nothing
# besides the call itself.
struct ManyArgsCall{Name, F, C, T}
    f::F
    g::Vector{Float64}
    x::Vector{Float64}
    dx::Vector{Float64}
    cs::C
    thunks::T
end
ManyArgsCall{Name}(f::F, g, x, dx, cs::C, thunks::T) where {Name, F, C, T} =
    ManyArgsCall{Name, F, C, T}(f, g, x, dx, cs, thunks)
(c::ManyArgsCall{Name})() where {Name} =
    many_args_call(Val(Name), c.f, c.g, c.x, c.dx, c.cs, c.thunks)

struct ManyActiveCall{H, A}
    h::H
    as::A
end
(c::ManyActiveCall)() = many_active_call(c.h, c.as)

function warm_allocations(call::C) where {C}
    for _ in 1:3
        call()
    end
    return @allocated call()
end

# Defined at top level, so that the methods exist in the world the tests run in.
const MANY_ARGS_F = many_args_objective(MANY_ARGS_N)
const MANY_ARGS_F1 = many_args_objective(1)
const MANY_ACTIVE_H = many_active_objective(MANY_ARGS_N)
const MANY_ACTIVE_H1 = many_active_objective(1)

@testset "Entry points with $(MANY_ARGS_N + 1) annotated arguments" begin
    x = [0.3, -0.7]
    dx = [1.0, 0.0]
    g = zeros(2)
    cs = Tuple([0.1i, 0.2i] for i in 1:MANY_ARGS_N)
    expected = [sum(c[1] for c in cs), sum(c[2] for c in cs)]

    f = MANY_ARGS_F
    f1 = MANY_ARGS_F1
    thunks = many_args_split_thunks(f, x, cs)
    thunks1 = many_args_split_thunks(f1, x, cs[1:1])

    @testset "$name" for name in MANY_ARGS_CALLS
        call = ManyArgsCall{name}(f, g, x, dx, cs, thunks)
        fill!(g, 0)
        res = call()
        if name in (:forward, :forward_deferred)
            @test only(res) ≈ expected[1]
        elseif name === :reverse_batch
            @test g ≈ 2 .* expected
        else
            @test g ≈ expected
        end
        @test warm_allocations(call) ==
            warm_allocations(ManyArgsCall{name}(f1, g, x, dx, cs[1:1], thunks1))
    end

    @testset "Active arguments" begin
        as = Tuple(Float64.(1:MANY_ARGS_N))
        call = ManyActiveCall(MANY_ACTIVE_H, as)
        @test only(call()) == as
        @test warm_allocations(call) ==
            warm_allocations(ManyActiveCall(MANY_ACTIVE_H1, (1.0,)))
    end
end

many_args_width(args...) = Val(Enzyme.same_or_one(1, args...))
many_args_dynamic_width(T::DataType) = Enzyme.same_or_one(1, T)

@testset "Batch width of $(MANY_ARGS_N + 1) annotated arguments" begin
    x = [1.0]
    b2 = BatchDuplicated(x, (x, x))
    b3 = BatchDuplicated(x, (x, x, x))
    cs = ntuple(Returns(Const(x)), MANY_ARGS_N)
    @test Enzyme.same_or_one(1, cs...) == 1
    @test Enzyme.same_or_one(3, cs...) == 3
    @test Enzyme.same_or_one(1, cs..., b2) == 2
    @test Enzyme.same_or_one(1, typeof(b2), map(typeof, cs)...) == 2
    @test_throws ErrorException Enzyme.same_or_one(1, b2, cs..., b3)
    @test_throws ErrorException Enzyme.same_or_one(1, typeof(b2), typeof(b3))
    # The width is a compile-time constant, which every thunk type depends on.
    @test only(Base.return_types(many_args_width, typeof((b2, cs...)))) === Val{2}
    @test only(Base.return_types(many_args_width, typeof(cs))) === Val{1}
    # A type of unknown identity is resolved at run time.
    @test many_args_dynamic_width(typeof(b3)) == 3
    @test many_args_dynamic_width(Float64) == 1
end
