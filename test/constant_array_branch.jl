using Enzyme, LinearAlgebra, Test

# A branch selects a freshly allocated array with constant contents against an
# active array. Activity analysis proves the fresh array inactive, so its shadow
# is a zero array of the same shape rather than a runtime activity error.

@noinline consume_branch(v) = sum(abs2, v)

branch_literal(x, e) = sum(e ? Float64[] : x .* 2.0)
branch_literal_call(x, e) = consume_branch(e ? Float64[] : x .* 2.0)
branch_zeros0(x, e) = sum(e ? zeros(0) : x .* 2.0)
# With forced bounds checks, this can carry a separate phi of the arrays' data
# pointers and pass the fresh array to a readonly bounds-error path.
branch_zeros(x, e) = sum(e ? zeros(length(x)) : x .* 2.0)
branch_ones(x, e) = sum(e ? ones(length(x)) : x .* 2.0)
branch_fill(x, e) = sum(e ? fill(3.0, length(x)) : x .* 2.0)
branch_undef(x, e) = sum(e ? fill!(Vector{Float64}(undef, length(x)), 0.5) : x .* 2.0)
branch_similar0(x, e) = sum(e ? similar(x, 0) : x .* 2.0)
branch_matrix(x, e) = sum(e ? zeros(length(x), length(x)) : x .* x')
branch_complex(x, e) = sum(abs2, e ? zeros(ComplexF64, length(x)) : complex.(x, x))
branch_then_add(x, e) = sum(abs2, (e ? zeros(length(x)) : x .* 2.0) .+ x)
branch_return(x, e) = e ? zeros(length(x)) : x .* 2.0

@testset "Branch selecting a fresh constant array" begin
    x0 = [1.0, 2.0]
    cases = (
        (branch_literal, [0.0, 0.0], [2.0, 2.0]),
        (branch_literal_call, [0.0, 0.0], 8.0 .* x0),
        (branch_zeros0, [0.0, 0.0], [2.0, 2.0]),
        (branch_zeros, [0.0, 0.0], [2.0, 2.0]),
        (branch_ones, [0.0, 0.0], [2.0, 2.0]),
        (branch_fill, [0.0, 0.0], [2.0, 2.0]),
        (branch_undef, [0.0, 0.0], [2.0, 2.0]),
        (branch_similar0, [0.0, 0.0], [2.0, 2.0]),
        (branch_matrix, [0.0, 0.0], fill(sum(x0), 2) .* 2.0),
        (branch_complex, [0.0, 0.0], 4.0 .* x0),
        (branch_then_add, 2.0 .* x0, 18.0 .* x0),
    )
    for (f, dtrue, dfalse) in cases, e in (true, false)
        x = copy(x0)
        dx = zero(x)
        res = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), Const(e))
        @test res[2] ≈ f(x0, e)
        @test dx ≈ (e ? dtrue : dfalse)
        @test x == x0
    end

    for e in (true, false)
        expected = e ? 0.0 : 2.0
        @test autodiff(Forward, branch_zeros, Duplicated(copy(x0), [1.0, 0.0]), Const(e))[1] ≈ expected
        res = autodiff(ForwardWithPrimal, branch_zeros, Duplicated(copy(x0), [1.0, 0.0]), Const(e))
        @test res[1] ≈ expected
        @test res[2] ≈ branch_zeros(x0, e)

        bres = autodiff(
            Forward, branch_zeros, BatchDuplicated(copy(x0), ([1.0, 0.0], [0.0, 1.0])), Const(e)
        )[1]
        @test bres[1] ≈ expected
        @test bres[2] ≈ expected

        d1 = zero(x0)
        d2 = zero(x0)
        autodiff(Reverse, branch_zeros, Active, BatchDuplicated(copy(x0), (d1, d2)), Const(e))
        @test d1 ≈ fill(expected, 2)
        @test d2 ≈ fill(expected, 2)

        dx = zero(x0)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal,
            Const{typeof(branch_zeros)},
            Active,
            Duplicated{Vector{Float64}},
            Const{Bool},
        )
        tape, primal, _ = fwd(Const(branch_zeros), Duplicated(copy(x0), dx), Const(e))
        rev(Const(branch_zeros), Duplicated(copy(x0), dx), Const(e), 1.0, tape)
        @test primal ≈ branch_zeros(x0, e)
        @test dx ≈ fill(expected, 2)

        dret = autodiff(Forward, branch_return, Duplicated, Duplicated(copy(x0), [1.0, 0.0]), Const(e))[1]
        @test dret ≈ (e ? [0.0, 0.0] : [2.0, 0.0])
    end
end

# A loop carry starts as a fresh constant array and is replaced by an active value
# on each step, so the loop-exit phi meets both.
function carry_loop(x, k)
    v = zeros(k == 0 ? 0 : length(x))
    for i in 1:k
        v = x .* Float64(i) .+ v
    end
    return sum(v)
end
@noinline carry_step(x, v, i) = x .* Float64(i) .+ v
function carry_loop_call(x, k)
    v = zeros(0)
    for i in 1:k
        v = carry_step(x, i == 1 ? zeros(length(x)) : v, i)
    end
    return sum(abs2, v)
end

@testset "Loop carry starting as a fresh constant array" begin
    x0 = [1.0, 2.0]
    for k in (0, 1, 3)
        dx = zero(x0)
        autodiff(Reverse, carry_loop, Active, Duplicated(copy(x0), dx), Const(k))
        @test dx ≈ fill(Float64(sum(1:k)), 2)

        dx = zero(x0)
        autodiff(Reverse, carry_loop_call, Active, Duplicated(copy(x0), dx), Const(k))
        @test dx ≈ 2.0 .* x0 .* sum(1:k)^2
    end
end

# Active data reaching the fresh array through an alias makes it active, so these
# never take the zero-shadow path; their gradients must stay exact.
alias_store_through_branch(x, e) =
    (c = zeros(length(x)); v = e ? c : similar(x); v .= x .* 2.0; sum(c))
alias_store_after_branch(x, e) = (c = zeros(length(x)); v = e ? c : x .* 1.0; c .= x; sum(v))
alias_two_branches(x, e) =
    (c = zeros(length(x)); v = e ? c : x .* 2.0; w = e ? c : x .* 3.0; v[1] = x[1]; sum(w))
alias_push(x, e) = (v = e ? Float64[] : x .* 2.0; push!(v, x[1]); sum(v))
alias_vec(x, e) = (m = zeros(length(x), 1); v = e ? vec(m) : x .* 2.0; m[2, 1] = x[1]^2; sum(v))

@testset "Fresh constant array aliased with active data" begin
    x0 = [1.0, 2.0]
    cases = (
        (alias_store_through_branch, [2.0, 2.0], [0.0, 0.0]),
        (alias_store_after_branch, [1.0, 1.0], [1.0, 1.0]),
        (alias_two_branches, [1.0, 0.0], [3.0, 3.0]),
        (alias_push, [1.0, 0.0], [3.0, 2.0]),
        (alias_vec, [2.0, 0.0], [2.0, 2.0]),
    )
    for (f, dtrue, dfalse) in cases, e in (true, false)
        x = copy(x0)
        dx = zero(x)
        autodiff(Reverse, f, Active, Duplicated(x, dx), Const(e))
        @test dx ≈ (e ? dtrue : dfalse)
    end
end

# Whether `f()` returns rather than raising the runtime activity error; any other
# error propagates.
function differentiates(f)
    try
        f()
        return true
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        return false
    end
end

const BRANCH_GLOBAL = [1.0, 2.0]
branch_const_arg(x, c, e) = sum(e ? c : x .* 2.0)
branch_global(x, e) = sum(e ? BRANCH_GLOBAL : x .* 2.0)

# A `Const` argument merged with an active array has a zero-filled copy as its
# shadow, as when it is stored into active memory (test/const_element_store.jl): the
# gradient equals runtime activity's, and the argument is left unchanged.
@testset "Constant memory that is not a fresh allocation" begin
    x0 = [1.0, 2.0]
    for e in (true, false)
        c = [5.0, 6.0]
        dx = zero(x0)
        autodiff(Reverse, branch_const_arg, Active, Duplicated(copy(x0), dx), Const(c), Const(e))
        @test dx ≈ (e ? [0.0, 0.0] : [2.0, 2.0])
        @test c == [5.0, 6.0]
        drt = zero(x0)
        autodiff(
            set_runtime_activity(Reverse), branch_const_arg, Active,
            Duplicated(copy(x0), drt), Const(c), Const(e),
        )
        @test dx ≈ drt
        @test autodiff(Forward, branch_const_arg, Duplicated(copy(x0), [1.0, 0.0]), Const(c), Const(e))[1] ≈
            (e ? 0.0 : 2.0)
    end
    # Not built: a global merged with an active array keeps the runtime activity
    # error, because `constant_argument_data` recognizes only memory reached from an
    # argument.
    @test_broken differentiates(() -> autodiff(Reverse, branch_global, Active, Duplicated(copy(x0), zero(x0)), Const(true)))
end

# The fresh constant array holds references (arrays, or immutable structs with
# array fields). Its elements may alias other objects, so only an empty one has a
# fresh shadow; a non-empty one keeps the runtime activity error.
const NestedPair = NamedTuple{(:values, :state), Tuple{Vector{Float64}, Vector{Float64}}}

# Inlined, so the loop reads the branch's result directly as in a user's loop.
@inline function sum_nested(v)
    s = 0.0
    for a in v
        s += sum(a)
    end
    return s
end
@inline function sum_pairs(v)
    s = 0.0
    for a in v
        s += sum(a.values) + 2.0 * sum(a.state)
    end
    return s
end

nested_literal(x, e) = sum_nested(e ? Vector{Float64}[] : [x .* 2.0, x .* 3.0])
nested_pairs(x, e) =
    sum_pairs(e ? NestedPair[] : [(; values = x .* 2.0, state = x .* 3.0)])
nested_undef(x, e, n) = sum_nested(e ? Vector{Vector{Float64}}(undef, n) : [x .* 2.0])
nested_matrix(x, e) =
    sum_nested(e ? Matrix{Vector{Float64}}(undef, 0, 3) : reshape([x .* 2.0, x], 2, 1))
nested_select(x, e) = sum_nested(ifelse(e, Vector{Float64}[], [x .* 2.0]))
function nested_nonempty(x, e)
    v = e ? [zeros(length(x))] : [x .* 2.0]
    s = 0.0
    for a in v
        s += sum(a)
    end
    return s
end
# A matrix's size tuple is copied into its Array object.
function matrix_loop(x, e)
    v = e ? Matrix{Float64}(undef, 0, 3) : reshape(x .* 2.0, 1, 2)
    s = 0.0
    for a in v
        s += a
    end
    return s
end
nested_return(x, e) = e ? Vector{Float64}[] : [x .* 2.0]
function nested_push(x, e)
    v = e ? Vector{Float64}[] : [x .* 2.0]
    push!(v, x .* 3.0)
    return sum_nested(v)
end
function nested_carry(x, k)
    v = Vector{Float64}[]
    for i in 1:k
        v = [x .* Float64(i), x]
    end
    return sum_nested(v)
end

@testset "Branch selecting a fresh constant array of arrays" begin
    x0 = [1.0, 2.0]
    cases = (
        (nested_literal, [0.0, 0.0], [5.0, 5.0]),
        (nested_pairs, [0.0, 0.0], [8.0, 8.0]),
        (nested_matrix, [0.0, 0.0], [3.0, 3.0]),
        (matrix_loop, [0.0, 0.0], [2.0, 2.0]),
        (nested_select, [0.0, 0.0], [2.0, 2.0]),
        (nested_push, [3.0, 3.0], [5.0, 5.0]),
    )
    for (f, dtrue, dfalse) in cases, e in (true, false)
        x = copy(x0)
        dx = zero(x)
        res = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, dx), Const(e))
        @test res[2] ≈ f(x0, e)
        @test dx ≈ (e ? dtrue : dfalse)
        @test x == x0
    end

    for e in (true, false)
        expected = e ? 0.0 : 5.0
        @test autodiff(Forward, nested_literal, Duplicated(copy(x0), [1.0, 0.0]), Const(e))[1] ≈
            expected

        d1 = zero(x0)
        d2 = zero(x0)
        autodiff(Reverse, nested_literal, Active, BatchDuplicated(copy(x0), (d1, d2)), Const(e))
        @test d1 ≈ fill(expected, 2)
        @test d2 ≈ fill(expected, 2)

        dx = zero(x0)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal,
            Const{typeof(nested_literal)},
            Active,
            Duplicated{Vector{Float64}},
            Const{Bool},
        )
        tape, primal, _ = fwd(Const(nested_literal), Duplicated(copy(x0), dx), Const(e))
        rev(Const(nested_literal), Duplicated(copy(x0), dx), Const(e), 1.0, tape)
        @test primal ≈ nested_literal(x0, e)
        @test dx ≈ fill(expected, 2)

        dret = autodiff(Forward, nested_return, Duplicated, Duplicated(copy(x0), [1.0, 0.0]), Const(e))[1]
        @test dret == (e ? Vector{Float64}[] : [[2.0, 0.0]])
    end

    # The length is only known at run time. On Julia 1.11+ the branch then meets the
    # element pointer of either the shared empty memory or a fresh allocation; only
    # the edge from a fresh allocation raises.
    for (e, n, expected) in ((true, 0, 0.0), (false, 0, 2.0), (false, 3, 2.0))
        dx = zero(x0)
        autodiff(Reverse, nested_undef, Active, Duplicated(copy(x0), dx), Const(e), Const(n))
        @test dx ≈ fill(expected, 2)
    end

    for k in (0, 1, 3)
        dx = zero(x0)
        autodiff(Reverse, nested_carry, Active, Duplicated(copy(x0), dx), Const(k))
        @test dx ≈ fill(k == 0 ? 0.0 : k + 1.0, 2)
    end

    # refused: elements of a non-empty fresh array may alias objects whose shadow
    # differs from a zero copy, which static activity analysis cannot track;
    # correctness requires runtime activity (Enzyme FAQ, "Runtime Activity").
    dx = zero(x0)
    autodiff(Reverse, nested_nonempty, Active, Duplicated(copy(x0), dx), Const(false))
    @test dx ≈ [2.0, 2.0]
    @static if VERSION < v"1.11-"
        err = try
            autodiff(Reverse, nested_nonempty, Active, Duplicated(copy(x0), zero(x0)), Const(true))
            nothing
        catch e
            e
        end
        @test err isa Enzyme.Compiler.EnzymeRuntimeActivityError
        @test err.mi isa Core.MethodInstance
        @test occursin("Julia value causing error", sprint(showerror, err))
    end
    dx = zero(x0)
    autodiff(set_runtime_activity(Reverse), nested_nonempty, Active, Duplicated(copy(x0), dx), Const(true))
    @test dx ≈ [0.0, 0.0]
end

# The constant branch is a global, which keeps the runtime activity error, or a
# `Const` argument holding arrays, whose shadow is a zero-filled copy. Taking another
# branch must still return its derivative. The error, or the copy, used to be made
# in the block that branches to the merge, so it also fired for the other branches
# (Julia 1.10), or the error let post-optimization prove that the loop runs once and
# then drop the derivative (Julia 1.12).
const NESTED_GLOBAL = [[1.0, 1.0]]
const NESTED_GLOBAL_MATRIX = reshape([[1.0, 1.0]], 1, 1)
nested_global(x, e) = sum_nested(e ? NESTED_GLOBAL : [x .* 2.0])
nested_global_matrix(x, e) = sum_nested(e ? NESTED_GLOBAL_MATRIX : reshape([x .* 2.0], 1, 1))
nested_const_arg(x, c, e) = sum_nested(e ? c : [x .* 2.0])
nested_const_arg_flipped(x, c, e) = sum_nested(e ? [x .* 2.0] : c)
# The comparisons become a switch.
function nested_const_arg_switch(x, c, k)
    v = if k == 1
        c
    elseif k == 2
        [x .* 2.0]
    elseif k == 3
        [x .* 3.0]
    else
        [x .* 4.0]
    end
    return sum_nested(v)
end

@testset "Other branches beside a constant array of arrays" begin
    x0 = [1.0, 2.0]
    c = Const([[1.0, 1.0]])
    # (function, leading arguments, selector of the constant branch, selector of an
    # active branch, whether the constant branch is a global)
    cases = (
        (nested_global, (), true, false, true),
        (nested_global_matrix, (), true, false, true),
        (nested_const_arg, (c,), true, false, false),
        (nested_const_arg_flipped, (c,), false, true, false),
        (nested_const_arg_switch, (c,), 1, 2, false),
    )
    for (f, pre, constsel, activesel, isglobal) in cases
        if isglobal || VERSION >= v"1.11-"
            # Not built: a global keeps the runtime activity error, because
            # `constant_argument_data` recognizes only memory reached from an argument.
            # From Julia 1.11 the loop merges the argument's untracked data pointer
            # (`memoryref_data`), which has no zero-filled copy either.
            @test_broken differentiates(() -> autodiff(Reverse, f, Active, Duplicated(copy(x0), zero(x0)), pre..., Const(constsel)))
        else
            dx = zero(x0)
            autodiff(Reverse, f, Active, Duplicated(copy(x0), dx), pre..., Const(constsel))
            @test dx ≈ [0.0, 0.0]
            @test pre[1].val == [[1.0, 1.0]]
        end
        dx = zero(x0)
        autodiff(set_runtime_activity(Reverse), f, Active, Duplicated(copy(x0), dx), pre..., Const(constsel))
        @test dx ≈ [0.0, 0.0]

        dx = zero(x0)
        res = autodiff(ReverseWithPrimal, f, Active, Duplicated(copy(x0), dx), pre..., Const(activesel))
        @test res[2] ≈ 6.0
        @test dx ≈ [2.0, 2.0]

        d1 = zero(x0)
        d2 = zero(x0)
        autodiff(Reverse, f, Active, BatchDuplicated(copy(x0), (d1, d2)), pre..., Const(activesel))
        @test d1 ≈ [2.0, 2.0]
        @test d2 ≈ [2.0, 2.0]

        dx = zero(x0)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal,
            Const{typeof(f)},
            Active,
            Duplicated{Vector{Float64}},
            map(typeof, pre)...,
            Const{typeof(activesel)},
        )
        tape, primal, _ = fwd(Const(f), Duplicated(copy(x0), dx), pre..., Const(activesel))
        rev(Const(f), Duplicated(copy(x0), dx), pre..., Const(activesel), 1.0, tape)
        @test primal ≈ 6.0
        @test dx ≈ [2.0, 2.0]

        @test autodiff(Forward, f, Duplicated(copy(x0), [1.0, 0.0]), pre..., Const(activesel))[1] ≈ 2.0
    end
    for k in (3, 4)
        dx = zero(x0)
        autodiff(Reverse, nested_const_arg_switch, Active, Duplicated(copy(x0), dx), c, Const(k))
        @test dx ≈ [k, k]
    end
end

# A `Const` argument is merged with a fresh copy of itself, and the merged value goes
# with active data to a call that is not inlined, which activity analysis assumes
# may write active data into it. `getproperty(::Cholesky, :L)` returns
# `LowerTriangular(uplo == 'L' ? factors : copy(factors'))`, and on Julia 1.12 that
# copy goes through `Base.unalias` (`mightalias(dest, src) ? unaliascopy(src) : src`).
@noinline function scale_by_diagonal!(b, A)
    for i in eachindex(b)
        @inbounds b[i] *= A[i, i]
    end
    return b
end
copy_or_alias(A, e) = e ? A : copy(A')
copy_or_alias_scaled(x, A, e) = sum(abs2, scale_by_diagonal!(copy(x), copy_or_alias(A, e)))

cholesky_lower_solve(x, F, y) = sum(abs2, F.L \ (y .- x))
cholesky_upper_adjoint_solve(x, F, y) = sum(abs2, F.U' \ (y .- x))

@testset "Const argument merged with a fresh copy of itself" begin
    A = [2.0 1.0; 1.0 3.0]
    x0 = [1.0, 2.0]
    for e in (true, false)
        dx = zero(x0)
        autodiff(Reverse, copy_or_alias_scaled, Active, Duplicated(copy(x0), dx), Const(A), Const(e))
        @test dx ≈ 2 .* x0 .* [4.0, 9.0]
        @test A == [2.0 1.0; 1.0 3.0]
    end

    n = 4
    M = [sin(i + 2j) for i in 1:n, j in 1:n]
    S = M * M' + n * I
    y = collect(range(-1.0, 1.0; length = n))
    x1 = fill(0.1, n)
    expected = -2 .* (S \ (y .- x1))
    # `cholesky(Hermitian(S, :L))` copies only the lower triangle into `similar` storage,
    # so the unused upper triangle of `factors` is uninitialized on Julia 1.12 and 1.13
    # and may hold a NaN, which fails `==` against its own copy. Factoring a full copy in
    # place keeps every entry defined, so the `F.factors == factors` check below sees any
    # write into the `Const` factor.
    for F in (cholesky(S), cholesky!(Hermitian(copy(S), :L))),
            f in (cholesky_lower_solve, cholesky_upper_adjoint_solve)
        factors = copy(F.factors)
        dx = zero(x1)
        autodiff(Reverse, f, Active, Duplicated(copy(x1), dx), Const(F), Const(y))
        @test dx ≈ expected
        @test F.factors == factors
        @test autodiff(Forward, f, Duplicated(copy(x1), [1.0, 0.0, 0.0, 0.0]), Const(F), Const(y))[1] ≈ expected[1]

        d1 = zero(x1)
        d2 = zero(x1)
        autodiff(Reverse, f, Active, BatchDuplicated(copy(x1), (d1, d2)), Const(F), Const(y))
        @test d1 ≈ expected
        @test d2 ≈ expected

        dx = zero(x1)
        fwd, rev = autodiff_thunk(
            ReverseSplitWithPrimal, Const{typeof(f)}, Active,
            Duplicated{typeof(x1)}, Const{typeof(F)}, Const{typeof(y)},
        )
        tape, primal, _ = fwd(Const(f), Duplicated(copy(x1), dx), Const(F), Const(y))
        rev(Const(f), Duplicated(copy(x1), dx), Const(F), Const(y), 1.0, tape)
        @test primal ≈ f(x1, F, y)
        @test dx ≈ expected
    end
end
