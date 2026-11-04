using Enzyme, Test, InteractiveUtils

# Combined reverse and forward mode derivatives are compiled at their first call, not
# while the code calling `autodiff` is inferred: inference that the runtime triggers holds
# a process-wide lock, so compiling there stops every other thread from compiling (see
# `Enzyme.Compiler.LazyThunk`).

lazy_cube(x) = x^3 + sin(x)
lazy_grad(x) = autodiff(Reverse, lazy_cube, Active, Active(x))[1][1]
lazy_tangent(x) = autodiff(Forward, lazy_cube, Duplicated(x, 1.0))[1]
lazy_dcube(x) = 3 * x^2 + cos(x)

@testset "Inference does not compile the derivative" begin
    n = length(Enzyme.Compiler.cache)
    code_typed(lazy_grad, (Float64,))
    code_typed(lazy_tangent, (Float64,))
    @test length(Enzyme.Compiler.cache) == n
    @test lazy_grad(0.7) ≈ lazy_dcube(0.7)
    @test lazy_tangent(0.7) ≈ lazy_dcube(0.7)
    @test length(Enzyme.Compiler.cache) > n
    @test Base.return_types(lazy_grad, (Float64,)) == [Float64]
    @test Base.return_types(lazy_tangent, (Float64,)) == [Float64]
end

lazy_quartic(x) = x^4
lazy_inner(x) = autodiff(Reverse, lazy_quartic, Active, Active(x))[1][1]
lazy_inner_fwd(x) = autodiff(Forward, lazy_quartic, Duplicated(x, 1.0))[1]

@testset "Nested differentiation of a lazily compiled derivative" begin
    @test lazy_inner(2.0) ≈ 32.0
    @test lazy_inner_fwd(2.0) ≈ 32.0
    @test autodiff(Forward, lazy_inner, Duplicated(2.0, 1.0))[1] ≈ 48.0
    @test autodiff(Reverse, lazy_inner, Active, Active(2.0))[1][1] ≈ 48.0
    @test autodiff(Reverse, lazy_inner_fwd, Active, Active(2.0))[1][1] ≈ 48.0
end

lazy_scale(x) = 2x
lazy_scaled(x) = lazy_scale(x)^2
lazy_scaled_grad(x) = autodiff(Reverse, lazy_scaled, Active, Active(x))[1][1]
lazy_scaled_tangent(x) = autodiff(Forward, lazy_scaled, Duplicated(x, 1.0))[1]

lazy_factor(x) = 2.0
EnzymeRules.inactive(::typeof(lazy_factor), args...) = nothing
lazy_product(x) = x * lazy_factor(x)
lazy_product_grad(x) = autodiff(Reverse, lazy_product, Active, Active(x))[1][1]

@testset "Redefining a callee recompiles" begin
    @test lazy_scaled_grad(1.0) ≈ 8.0
    @test lazy_scaled_tangent(1.0) ≈ 8.0
    @test lazy_product_grad(1.5) ≈ 2.0
    @eval lazy_scale(x) = 3x
    @eval lazy_factor(x) = 5.0
    @test Base.invokelatest(lazy_scaled_grad, 1.0) ≈ 18.0
    @test Base.invokelatest(lazy_scaled_tangent, 1.0) ≈ 18.0
    @test Base.invokelatest(lazy_product_grad, 1.5) ≈ 5.0
end

lazy_shared(x) = x^5 + cos(x)
lazy_shared_grad(x) = autodiff(Reverse, lazy_shared, Active, Active(x))[1][1]

@testset "Concurrent first calls" begin
    results = fetch.([Threads.@spawn lazy_shared_grad(0.5) for _ in 1:8])
    @test all(r -> r ≈ 5 * 0.5^4 - sin(0.5), results)
end

# A callee reached by dynamic dispatch is differentiated through Enzyme's run-time rules,
# which build a split-mode derivative for it when the gradient first runs.
const lazy_dynamic_cube = Ref{Any}(lazy_cube)
const lazy_dynamic_quartic = Ref{Any}(lazy_quartic)
lazy_dynamic3(x) = lazy_dynamic_cube[](x)::Float64 * 2
lazy_dynamic4(x) = lazy_dynamic_quartic[](x)::Float64 * 2
lazy_dynamic_grad(f, x) = autodiff(Reverse, f, Active, Active(x))[1][1]

@testset "Dynamic dispatch to a lazily compiled callee" begin
    @test lazy_dynamic_grad(lazy_dynamic3, 0.7) ≈ 2 * lazy_dcube(0.7)
    @test lazy_dynamic_grad(lazy_dynamic3, 0.8) ≈ 2 * lazy_dcube(0.8)
    @test lazy_dynamic_grad(lazy_dynamic4, 2.0) ≈ 2 * 32.0
end

@testset "runtime_thunk compiles split mode at its first call" begin
    args = (
        Val(0), Const{typeof(lazy_shared)}, Active{Float64}, Tuple{Active{Float64}},
        Val(Enzyme.API.DEM_ReverseModePrimal), Val(1), Val((true, true)), Val(true), Val(false),
        Enzyme.FFIABI, Val(false), Val(false), Val(false),
    )
    n = length(Enzyme.Compiler.cache)
    code_typed(Enzyme.Compiler.runtime_thunk, map(Core.Typeof, args))
    @test length(Enzyme.Compiler.cache) == n
    fwd, rev = Enzyme.Compiler.runtime_thunk(args...)
    @test length(Enzyme.Compiler.cache) > n
    tape, primal, _ = fwd(Const(lazy_shared), Active(0.3))
    @test primal ≈ lazy_shared(0.3)
    @test rev(Const(lazy_shared), Active(0.3), 1.0, tape)[1][1] ≈ 5 * 0.3^4 - sin(0.3)
end

# What a derivative depends on beyond the primal (here the rule it calls, and a helper only
# that rule calls) is known only once it is compiled. A lazily compiled derivative still
# compiles again when that changes.
lazy_ruled(x) = x^2
lazy_ruled_helper(x) = 10.0
EnzymeRules.forward(config, ::Const{typeof(lazy_ruled)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(lazy_ruled(x.val), lazy_ruled_helper(x.val) + 2 * x.val * x.dval)
EnzymeRules.forward(config, ::Const{typeof(lazy_ruled)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    lazy_ruled_helper(x.val) + 2 * x.val * x.dval
lazy_ruled_tangent(x) = autodiff(Forward, lazy_ruled, Duplicated(x, 1.0))[1]

@testset "Changes the compiled derivative depends on recompile it" begin
    @test lazy_ruled_tangent(1.0) ≈ 12.0
    @eval lazy_ruled_helper(x) = 20.0
    @test Base.invokelatest(lazy_ruled_tangent, 1.0) ≈ 22.0
    while !isempty(methods(EnzymeRules.forward, Tuple{Any, Const{typeof(lazy_ruled)}, Vararg{Any}}))
        for m in methods(EnzymeRules.forward, Tuple{Any, Const{typeof(lazy_ruled)}, Vararg{Any}})
            Base.delete_method(m)
        end
    end
    @test Base.invokelatest(lazy_ruled_tangent, 1.0) ≈ 2.0
end
