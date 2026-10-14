module RuleInvalidation

using Test
using Enzyme
import .EnzymeRules: forward, inactive

issue696(x) = x^2
call_issue696(args...) = issue696(args...)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

# should invalidate cache for the previous result
forward(config, ::Const{typeof(issue696)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    10 + 2 * x.val * x.dval
forward(config, func::Const{typeof(issue696)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(func.val(x.val), 10 + 2 * x.val * x.dval)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 12.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 12.0

# should invalidate cache for the previous result again
forward(config, ::Const{typeof(issue696)}, ::Type{<:DuplicatedNoNeed}, x::Duplicated) =
    20 + 2 * x.val * x.dval
forward(config, func::Const{typeof(issue696)}, ::Type{<:Duplicated}, x::Duplicated) =
    Duplicated(func.val(x.val), 20 + 2 * x.val * x.dval)

@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 22.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 22.0

# check that `Base.delete_method` works as expected
# Loop required because on 1.12 delete_method uncovers the previous method
while !isempty(methods(forward, Tuple{Any, Const{typeof(issue696)}, Vararg{Any}}))
    for m in methods(forward, Tuple{Any, Const{typeof(issue696)}, Vararg{Any}})
        Base.delete_method(m)
    end
end
@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0

# now test invalidation for `inactive`
inactive(::typeof(issue696), args...) = nothing
@test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 0.0
@test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 0.0

# check that `Base.delete_method` works as expected
for m in methods(inactive, Tuple{typeof(issue696), Vararg{Any}})
    Base.delete_method(m)
end

# Deleting the `inactive` method invalidates its callers through the signature edges the
# thunk takes only since Julia 1.13.1 (JuliaLang/julia#63227). With `HAS_RULE_EDGES`, the
# thunk and the inferred code also take precise edges to the rule query for the signature,
# whose instance the deleted method's specializations invalidate.
if VERSION >= v"1.13.1-DEV" || Enzyme.Compiler.HAS_RULE_EDGES
    @test autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
    @test autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
else
    @test_broken autodiff(Forward, issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
    @test_broken autodiff(Forward, call_issue696, Duplicated(1.0, 1.0))[1] ≈ 2.0
end

# The rule queries are generated functions whose `CodeInstance`s are the cache of the
# answer, each with the world range it holds in and edges to the rule methods for the
# signature: a definition that is not a rule leaves an instance open, a rule for the
# signature closes it, and a rule for another signature leaves it alone.
function rule_query_instances(gen, sig)
    mi = nothing
    for m in Base.specializations(only(methods(gen)))
        m.specTypes == Tuple{typeof(gen), Type{sig}} && (mi = m)
    end
    cis = Core.CodeInstance[]
    (mi === nothing || !isdefined(mi, :cache)) && return cis
    ci = mi.cache
    while ci !== nothing
        (!hasfield(Core.CodeInstance, :owner) || ci.owner === nothing) && push!(cis, ci)
        ci = isdefined(ci, :next) ? ci.next : nothing
    end
    return cis
end

queried_a(x) = x + 1
queried_b(x) = x + 2
unrelated_definition() = nothing

@testset "rule queries are cached by world range" begin
    C = Enzyme.Compiler
    sig = Tuple{typeof(queried_a), Float64}
    w1 = Base.get_world_counter()
    @test !C.cached_has_frule(sig, w1, nothing)
    ci1 = only(rule_query_instances(C.has_frule_gen, sig))
    @test ci1.max_world == typemax(UInt)

    @eval unrelated_definition_2() = nothing
    w2 = Base.get_world_counter()
    @test !C.cached_has_frule(sig, w2, nothing)
    @test only(rule_query_instances(C.has_frule_gen, sig)) === ci1

    # A rule for another function does not touch the instance.
    @eval forward(config, ::Const{typeof(queried_b)}, ::Type{<:Duplicated}, x::Duplicated) =
        Duplicated(queried_b(x.val), x.dval)
    w3 = Base.get_world_counter()
    @test !C.cached_has_frule(sig, w3, nothing)
    @test only(rule_query_instances(C.has_frule_gen, sig)) === ci1

    # A rule for the signature closes it; the old world keeps its answer.
    @eval forward(config, ::Const{typeof(queried_a)}, ::Type{<:Duplicated}, x::Duplicated) =
        Duplicated(queried_a(x.val), x.dval)
    w4 = Base.get_world_counter()
    @test C.cached_has_frule(sig, w4, nothing)
    @test ci1.max_world == w4 - 1
    @test length(rule_query_instances(C.has_frule_gen, sig)) == 2
    @test !C.cached_has_frule(sig, w3, nothing)
end

end # module
