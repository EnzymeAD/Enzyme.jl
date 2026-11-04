using Enzyme, Test

# The Attributor runs in `removeDeadArgs!`. Seeding it walks the code that must execute
# around every pointer and every value passed to a call, so its time grows with the square
# of a straight-line region. A batch of `W` tangents makes the differentiated region `W`
# times longer, so running it on differentiated code made the first call of a batched
# derivative quadratic in `W`: at a width of 64 a generated model density spent four fifths
# of its compile there. The compile time itself is too noisy to pin in a test, so these
# tests pin the stages the Attributor runs in, and that the derivative does not depend on it.

const ATTRIBUTOR_RUNS = Dict{Symbol, Int}()

function attributor_stage()
    names = Set{Symbol}(fr.func for fr in stacktrace(backtrace()))
    :fixup_callconv! in names && return :callconv
    :post_optimize! in names && return :post
    return :before_ad
end

# Derivatives compile at their first call, in the calling world, so this method is the one
# they run. It counts each run and then does what the pass does.
@eval Enzyme.Compiler function enzyme_attributor_pass!(mod::LLVM.Module)
    stage = Main.attributor_stage()
    Main.ATTRIBUTOR_RUNS[stage] = get(Main.ATTRIBUTOR_RUNS, stage, 0) + 1
    ccall((:RunAttributorOnModule, API.libEnzyme), Cvoid, (LLVM.API.LLVMModuleRef,), mod)
    return true
end

@noinline attributor_helper(a, b) = a * exp(b) + log1p(a * a)

# One function per compile: a derivative that is already compiled does not run the pass.
for name in (:attributor_density_off, :attributor_density_on, :attributor_density_wide)
    @eval function $name(x, c)
        s = 0.0
        for k in 1:6
            a = x[mod1(k, 3)] * c[1] + x[mod1(k + 1, 3)]
            b = attributor_helper(a, x[mod1(k + 2, 3)])
            s += b * b + exp(-a * a) - log1p(x[mod1(k + 1, 3)]^2) * c[2]
        end
        return s
    end
end

function attributor_gradient!(g, f, x, c)
    fill!(g, 0.0)
    autodiff_deferred(Reverse, Const(f), Active, Duplicated(x, g), Const(c))
    return nothing
end

# Forward over Reverse with `W` tangents, the Hessian's first `W` columns of it.
function attributor_hvp(f, ::Val{W}) where {W}
    x = [0.3, -0.7, 1.1]
    c = [0.4, 1.3]
    g = zero(x)
    dg = ntuple(_ -> zero(x), Val(W))
    dx = ntuple(k -> (v = zero(x); v[mod1(k, 3)] = 1.0; v), Val(W))
    autodiff(
        Forward, Const(attributor_gradient!), Const,
        BatchDuplicated(g, dg), Const(f), BatchDuplicated(x, dx), Const(c),
    )
    return g, dg
end

const ATTRIBUTOR_RUNS_BEFORE_AD = Enzyme.Compiler.RunAttributor[]

@testset "The Attributor does not run on differentiated code by default" begin
    @test !Enzyme.Compiler.RunAttributorAfterAD[]
    empty!(ATTRIBUTOR_RUNS)
    g_off, h_off = attributor_hvp(attributor_density_off, Val(3))
    # Julia 1.12 and later never run it. Where it does run it must still be seen here, or
    # the zeros below would prove nothing.
    @test (get(ATTRIBUTOR_RUNS, :before_ad, 0) > 0) == ATTRIBUTOR_RUNS_BEFORE_AD
    @test get(ATTRIBUTOR_RUNS, :callconv, 0) == 0
    @test get(ATTRIBUTOR_RUNS, :post, 0) == 0

    # The derivatives are the same with it. Machine code can differ, the floating-point
    # operations it computes do not.
    Enzyme.Compiler.RunAttributorAfterAD[] = true
    try
        empty!(ATTRIBUTOR_RUNS)
        g_on, h_on = attributor_hvp(attributor_density_on, Val(3))
        @test (get(ATTRIBUTOR_RUNS, :callconv, 0) > 0) == ATTRIBUTOR_RUNS_BEFORE_AD
        @test (get(ATTRIBUTOR_RUNS, :post, 0) > 0) == ATTRIBUTOR_RUNS_BEFORE_AD
        @test g_on == g_off
        @test all(h_on .== h_off)
    finally
        Enzyme.Compiler.RunAttributorAfterAD[] = false
    end
end

@testset "A wide batch agrees with the columns of a narrow one" begin
    empty!(ATTRIBUTOR_RUNS)
    _, narrow = attributor_hvp(attributor_density_off, Val(3))
    g, wide = attributor_hvp(attributor_density_wide, Val(12))
    @test get(ATTRIBUTOR_RUNS, :callconv, 0) == 0
    @test get(ATTRIBUTOR_RUNS, :post, 0) == 0
    for k in 1:12
        @test wide[k] == narrow[mod1(k, 3)]
    end

    # Central differences of the gradient, against the first column.
    x = [0.3, -0.7, 1.1]
    c = [0.4, 1.3]
    grad_call(y) = (d = zero(y); autodiff(Reverse, Const(attributor_density_off), Active, Duplicated(copy(y), d), Const(c)); d)
    h = 1.0e-6
    e = [1.0, 0.0, 0.0]
    @test isapprox(wide[1], (grad_call(x .+ h .* e) .- grad_call(x .- h .* e)) ./ (2h); rtol = 1.0e-7)
end
