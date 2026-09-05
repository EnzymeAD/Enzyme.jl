using Enzyme
using Enzyme.EnzymeRules
using Test

const C = Enzyme.Compiler
const I = Enzyme.Compiler.Interpreter

rc_plain(x) = 2x
rc_ruled(x) = 3x

fwd_epoch(world) = I.rule_epoch(EnzymeRules.forward, I.FWD_RULE_TT, world)

@testset "memo entries are shared across worlds" begin
    world = Base.get_world_counter()
    st = Tuple{typeof(rc_plain), Float64}
    @test !C.cached_has_frule(st, world, nothing)
    e0 = fwd_epoch(world)
    n0 = length(C.FRULE_MEMO.entries)
    @test n0 >= 1
    @test C.FRULE_MEMO.epoch == e0
    for i in 1:5
        # A new method bumps the world without touching any rule family.
        @eval $(Symbol(:rc_unrelated_, i))(x) = x + $i
        world = Base.get_world_counter()
        @test !C.cached_has_frule(st, world, nothing)
        @test fwd_epoch(world) == e0
        @test length(C.FRULE_MEMO.entries) == n0
    end
    # Signatures are also equal across worlds, so the token inputs do not churn.
    @test I.get_rule_signatures(EnzymeRules.forward, I.FWD_RULE_TT, world) ===
        I.get_rule_signatures(EnzymeRules.forward, I.FWD_RULE_TT, world - 1)
end

world_before_rule = Base.get_world_counter()
@test !C.cached_has_frule(Tuple{typeof(rc_ruled), Float64}, world_before_rule, nothing)
epoch_before_rule = fwd_epoch(world_before_rule)
rev_epoch_before_rule = I.rule_epoch(EnzymeRules.augmented_primal, I.REV_RULE_TT, world_before_rule)

function EnzymeRules.forward(config, ::Const{typeof(rc_ruled)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(rc_ruled(x.val), 3 * x.dval)
end
function EnzymeRules.forward(config, ::Const{typeof(rc_ruled)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated)
    return 3 * x.dval
end

@testset "a new rule retires the memo" begin
    world = Base.get_world_counter()
    st = Tuple{typeof(rc_ruled), Float64}
    @test fwd_epoch(world) == epoch_before_rule + 1
    @test C.cached_has_frule(st, world, nothing)
    # The memo was dropped as a whole: only the query just made is in it.
    @test C.FRULE_MEMO.epoch == epoch_before_rule + 1
    @test length(C.FRULE_MEMO.entries) == 1
    # Other families are untouched.
    @test I.rule_epoch(EnzymeRules.augmented_primal, I.REV_RULE_TT, world) == rev_epoch_before_rule
    # The old world still answers as it did, at its own epoch.
    @test !C.cached_has_frule(st, world_before_rule, nothing)
    @test autodiff(Forward, rc_ruled, Duplicated(1.0, 1.0))[1] == 3.0
end

@testset "check mode re-derives hits" begin
    C.CHECK_RULE_MEMO[] = true
    try
        world = Base.get_world_counter()
        st = Tuple{typeof(rc_ruled), Float64}
        @test C.cached_has_frule(st, world, nothing)
        @test C.cached_has_frule(st, world, nothing)
        @test !C.cached_is_inactive(st, world, nothing)
        @test !C.cached_is_inactive(st, world, nothing)
    finally
        C.CHECK_RULE_MEMO[] = false
    end
end

@testset "method table key strips the world" begin
    world = Base.get_world_counter()
    mt = Core.Compiler.InternalMethodTable(world)
    @test C.memo_method_table(mt) === nothing
    @test C.memo_method_table(nothing) === nothing
    omt = Core.Compiler.OverlayMethodTable(world, C.GPUCompiler.GLOBAL_METHOD_TABLE)
    @test C.memo_method_table(omt) === C.GPUCompiler.GLOBAL_METHOD_TABLE
    @test C.memo_method_table(Core.Compiler.OverlayMethodTable(world - 1, C.GPUCompiler.GLOBAL_METHOD_TABLE)) === C.memo_method_table(omt)
end
