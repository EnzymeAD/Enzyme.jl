using Test
using EnzymeCore
import EnzymeCore.EnzymeRules: forward, has_frule_from_sig

g(x) = x ^ 2
function forward(config, ::Const{typeof(g)}, ::Type{<:Const}, x::Const)
    return Const(g(x.val))
end

@test has_frule_from_sig(Base.signature_type(g, Tuple{Float64}))

f(;kwargs) = 1.0

function forward(config, ::Const{typeof(f)}, ::Type{<:Const}; kwargs...)
    return Const(f(; kwargs...))
end

@test has_frule_from_sig(Base.signature_type(f, Tuple{}))

data = [1.0, 2.0, 3.0, 4.0]

d = @view data[2:end]
y = @view data[3:end]
@test_skip @test_throws AssertionError Duplicated(d, y)

@test_throws ErrorException Active(data)
@test_skip @test_throws ErrorException Active(d)

@testset "EnzymeRules.runtime_active (#3713)" begin
    using EnzymeCore.EnzymeRules: FwdConfig, RevConfig, runtime_active, runtime_activity, strong_zero

    fwd_no_rta = FwdConfig{true, true, 1, false, false}()
    fwd_rta = FwdConfig{true, true, 1, true, true}()
    rev_no_rta = RevConfig{true, true, 1, (false, false), false, false}()
    rev_rta = RevConfig{true, true, 1, (false, false), true, true}()

    @test !runtime_activity(typeof(fwd_no_rta))
    @test runtime_activity(typeof(fwd_rta))
    @test !strong_zero(typeof(fwd_no_rta))
    @test strong_zero(typeof(fwd_rta))
    @test !runtime_activity(typeof(rev_no_rta))
    @test runtime_activity(typeof(rev_rta))
    @test !strong_zero(typeof(rev_no_rta))
    @test strong_zero(typeof(rev_rta))

    x = [1.0, 2.0]
    dx = [0.0, 0.0]
    dx2 = [0.0, 0.0]

    for cfg in (fwd_no_rta, typeof(fwd_no_rta), rev_no_rta, typeof(rev_no_rta))
        @test !runtime_active(cfg, Const(x))
        @test runtime_active(cfg, Active(1.0))
        @test runtime_active(cfg, Duplicated(x, dx))
        @test runtime_active(cfg, Duplicated(x, x))
        @test runtime_active(cfg, DuplicatedNoNeed(x, x))
        @test runtime_active(cfg, BatchDuplicated(x, (x, x)))
        @test runtime_active(cfg, BatchDuplicatedNoNeed(x, (x, x)))
        @test runtime_active(cfg, MixedDuplicated(1.0, Ref(0.0)))
        @test runtime_active(cfg, BatchMixedDuplicated(1.0, (Ref(0.0), Ref(0.0))))
    end

    for cfg in (fwd_rta, typeof(fwd_rta), rev_rta, typeof(rev_rta))
        @test !runtime_active(cfg, Const(x))
        @test runtime_active(cfg, Active(1.0))
        @test runtime_active(cfg, Duplicated(x, dx))
        @test !runtime_active(cfg, Duplicated(x, x))
        @test runtime_active(cfg, DuplicatedNoNeed(x, dx))
        @test !runtime_active(cfg, DuplicatedNoNeed(x, x))
        @test runtime_active(cfg, BatchDuplicated(x, (dx, dx2)))
        @test runtime_active(cfg, BatchDuplicated(x, (x, dx)))
        @test !runtime_active(cfg, BatchDuplicated(x, (x, x)))
        @test runtime_active(cfg, BatchDuplicatedNoNeed(x, (x, dx)))
        @test !runtime_active(cfg, BatchDuplicatedNoNeed(x, (x, x)))
        @test runtime_active(cfg, MixedDuplicated(1.0, Ref(0.0)))
        @test runtime_active(cfg, BatchMixedDuplicated(1.0, (Ref(0.0), Ref(0.0))))
    end
end
