# Which rules apply to a call, cached by Enzyme itself and verified by Enzyme itself.
#
# Whether a call of signature `specTypes` has a rule of some kind (`forward`,
# `augmented_primal`, `inactive`, ...) is a method lookup, but not one Julia's own
# invalidation can be trusted to track: the lookup consults the back-end's overlay method
# tables as well as the internal one, it is made out of band of inference, and an answer
# of "no rule" depends on the absence of methods. Native backedges model none of that: a
# missing-method edge on an overlay table is not verified when a package image loads
# (JuliaLang/julia#63959), and an edge emitted only so that a method definition
# invalidates something carries the native meaning of a call that was never made. So Enzyme
# keeps no native edges for these answers. It records, per answer, what it looked up
# (`RuleLookup`: the signature, the table, the methods that matched or that none did) and
# re-checks those lookups itself, modelled on `verify_call` in `Compiler/src/reinfer.jl`,
# whenever an answer is used in a world it has not been checked in.
#
# The answers are used in three places, each of which keeps the answers it depends on:
#
# - the rule queries themselves (`rule_answer`), cached in `RULE_ANSWERS`;
# - the inference results of the `EnzymeInterpreter`, which decide inlining from them
#   (`enzyme_call_kind`): each result carries an `EnzymeRuleDeps` among its
#   `analysis_results`, so it is serialized with the `CodeInstance` into a package image,
#   and the interpreter's code cache verifies it, and every Enzyme-owned `CodeInstance` it
#   reaches through its edges, before handing it out (`ci_rule_valid!`);
# - compiled thunks, whose generated code checks a `ThunkGuard` before returning the thunk
#   and recompiles when an answer it baked in no longer holds. A compilation that
#   differentiates such code (nested differentiation) settles the check in the world it
#   compiles for, and takes the thunk's answers as its own (`thunk_guard_holds!`).
#
# Before Julia 1.12 (`HAS_RULE_DEPS`), inference results carry no rule answers, so they rely
# on the cache token as before, and thunks keep the rule signature edges in addition to their
# guard. The guard then still verifies that a thunk is this session's (a thunk restored from
# a package image holds an address of the session that wrote the image) and the answers its
# compilation consulted directly.
#
# An answer, inference result or thunk made by another session (one restored from a
# package image) is never taken on trust: its session differs from `RULE_SESSION[]`, so it
# is verified in full, against the methods of this session, before its first use.

# Inference results carry their rule answers from this version on: the frame's edges are a
# list Enzyme can walk to the `CodeInstance`s it reached (see `ci_rule_valid!`), and
# `jl_update_codeinst` stores the `analysis_results` the frame stacked. Before it, the cache
# token carries the set of rule signatures instead (see `EnzymeCacheToken`).
const HAS_RULE_DEPS = VERSION >= v"1.12.0-DEV.1531"

# `Base.ReinferUtils.verify_call` (Compiler/src/reinfer.jl, Julia 1.13+) verifies a lookup
# in the internal method table with the fast paths of package-image verification.
const HAS_REINFER_VERIFY_CALL = isdefined(Base, :ReinferUtils) && isdefined(Base.ReinferUtils, :verify_call)

# Identifies this session. Everything that records a verification records it with the
# session it was made in, so that a record restored from a package image, whose worlds
# count those of the session that wrote the image, is not mistaken for one of ours.
const RULE_SESSION = Ref{Any}(nothing)
function init_rule_session!()
    RULE_SESSION[] = Ref{UInt}(0)
    reset_rule_stats!()
    return nothing
end

"""
    RuleLookup

One method lookup a rule answer depends on: the methods of `table` matching `sig` were
`methods` (empty for a lookup that found none). `table` is `nothing` for the internal
method table and a `Core.MethodTable` for an overlay table. Any other method table view
cannot be split into tables; its lookup holds only in the `world` it was made in.
"""
struct RuleLookup
    table::Any
    sig::Any
    methods::Core.SimpleVector
    fully_covers::Bool
    world::UInt
end

"""
    RuleAnswer

The answer `value` to a rule query, with the lookups it was computed from. It holds from
`min_world` and is known to hold through `validated`; `max_world` closes it once a
verification failed. All worlds are those of `session`.
"""
mutable struct RuleAnswer
    const value::Bool
    const lookups::Vector{RuleLookup}
    @atomic min_world::UInt
    @atomic validated::UInt
    @atomic max_world::UInt
    @atomic session::Any
end

# The analysis result an inference result of the `EnzymeInterpreter` carries: the rule
# answers its frame (and the constant-propagation frames under it) consulted.
struct EnzymeRuleDeps
    answers::Vector{RuleAnswer}
end

# Statistics, for measuring the cost of the scheme.
mutable struct RuleDepStats
    queries::Int
    computed::Int
    verified::Int
    verify_failed::Int
    ci_walks::Int
    ci_closed::Int
    guard_slow::Int
    guard_recompiled::Int
end
const RULE_STATS = RuleDepStats(0, 0, 0, 0, 0, 0, 0, 0)
function reset_rule_stats!()
    for f in fieldnames(RuleDepStats)
        setfield!(RULE_STATS, f, 0)
    end
    return RULE_STATS
end

const RULE_LOCK = ReentrantLock()

# The newest world. `Base.get_world_counter` answers `typemax(UInt)` inside a generator,
# where much of this runs, so read the counter itself.
newest_world() = unsafe_load(cglobal(:jl_world_counter, UInt))

# The rule answers of this session, by query: `(rule, specTypes, method tables)`.
const RULE_ANSWERS = Dict{Any, Vector{RuleAnswer}}()

# The world through which each Enzyme-owned `CodeInstance` was verified in this session.
const CI_VALIDATED = IdDict{Core.CodeInstance, UInt}()

## lookups and their verification

function lookup_table(@nospecialize(sig), @nospecialize(table), world::UInt)
    min_valid = Ref{UInt}(1)
    max_valid = Ref{UInt}(typemax(UInt))
    has_ambig = Ref{Int32}(0)
    result = Base._methods_by_ftype(sig, table, -1, world, false, min_valid, max_valid, has_ambig)
    if !(result isa Vector)
        # Not a lookup Julia can answer in full; hold it to this world.
        return RuleLookup(table, sig, Core.svec(), false, world), world, world
    end
    methods = Vector{Any}(undef, length(result))
    fully_covers = false
    for i in 1:length(result)
        match = result[i]::Core.MethodMatch
        methods[i] = match.method
        fully_covers = match.fully_covers
    end
    return RuleLookup(table, sig, Core.svec(methods...), fully_covers, typemax(UInt)), min_valid[], max_valid[]
end

"""
    verify_lookup(l::RuleLookup, world) -> (ok, min_world)

Whether the lookup `l` finds the same methods in `world`, and from which world it has
found them. Modelled on `verify_call` of `Compiler/src/reinfer.jl`, which it calls for the
internal table where available; unlike package-image verification, a lookup in an overlay
table is verified as well, including one that found nothing.
"""
function verify_lookup(l::RuleLookup, world::UInt)::Tuple{Bool, UInt}
    if l.world != typemax(UInt)
        return l.world == world, l.world
    end
    @static if HAS_REINFER_VERIFY_CALL
        if l.table === nothing
            n = length(l.methods)
            min_valid, max_valid = Base.ReinferUtils.verify_call(l.sig, l.methods, 1, n, world, l.fully_covers || n == 0, Any[])
            return max_valid >= world, min_valid
        end
    end
    min_valid = Ref{UInt}(1)
    max_valid = Ref{UInt}(typemax(UInt))
    has_ambig = Ref{Int32}(0)
    result = Base._methods_by_ftype(l.sig, l.table, -1, world, false, min_valid, max_valid, has_ambig)
    result isa Vector || return false, min_valid[]
    length(result) == length(l.methods) || return false, min_valid[]
    for match in result
        found = false
        for m in l.methods
            if (match::Core.MethodMatch).method === m
                found = true
                break
            end
        end
        found || return false, min_valid[]
    end
    return true, min_valid[]
end

"""
    answer_valid!(a::RuleAnswer, world) -> Bool

Whether the rule answer `a` holds in `world`, verifying its lookups if it has not been
verified in that world (or never in this session). A failed verification closes it.
"""
function answer_valid!(a::RuleAnswer, world::UInt)::Bool
    session = RULE_SESSION[]
    ours = (@atomic :acquire a.session) === session
    if ours
        world < (@atomic :monotonic a.min_world) && return false
        world <= (@atomic :acquire a.validated) && return true
        world > (@atomic :monotonic a.max_world) && return false
    end
    counter = newest_world()
    RULE_STATS.verified += 1
    min_valid = UInt(1)
    for l in a.lookups
        ok, lmin = verify_lookup(l, world)
        if !ok
            RULE_STATS.verify_failed += 1
            close_answer!(a, ours)
            return false
        end
        min_valid = max(min_valid, lmin)
    end
    if ours
        # The methods found are the same, but they may have changed in between: then the
        # answer did not hold all along.
        if min_valid > (@atomic :acquire a.validated) + 1
            close_answer!(a, ours)
            return false
        end
    else
        @atomic :monotonic a.min_world = min_valid
        @atomic :monotonic a.max_world = typemax(UInt)
        @atomic :release a.session = session
    end
    # Verified in `world`, and none of the lookups changes before the newest world we saw.
    newest = max(world, counter)
    old = @atomic :acquire a.validated
    if !ours || newest > old
        @atomic :release a.validated = newest
    end
    return true
end

function close_answer!(a::RuleAnswer, ours::Bool)
    if ours
        @atomic :release a.max_world = @atomic :acquire a.validated
    else
        @atomic :monotonic a.min_world = typemax(UInt)
        @atomic :release a.max_world = UInt(0)
        @atomic :release a.session = RULE_SESSION[]
    end
    return nothing
end

## the rule queries

# The signature `EnzymeCore.EnzymeRules.isapplicable` looks up for `rule` on a call of
# signature `specTypes`: the rule function applied to the annotated arguments for the
# `forward`/`augmented_primal` rules (as `has_frule_from_sig` and `has_rrule_from_sig`
# build it), and to the call's own arguments for the other rules.
function rule_query_tt(@nospecialize(rule), annotate::Bool, @nospecialize(specTypes::Type))
    annotate || return specTypes
    ft, tt = EnzymeRules._annotate_tt(specTypes)
    Config = rule === EnzymeRules.forward ? EnzymeRules.FwdConfig : EnzymeRules.RevConfig
    return Tuple{<:Config, <:Annotation{ft}, Type{<:Annotation}, tt...}
end

function rule_query_sig(@nospecialize(rule), annotate::Bool, @nospecialize(specTypes::Type))
    return Base.signature_type(rule, Base.to_tuple_type(rule_query_tt(rule, annotate, specTypes)))
end

# Answer the query in `world`, recording every lookup made: the overlay tables of `view`
# in their order, and the internal table unless an overlay match fully covers the
# signature, as `Core.Compiler.findall` on an `OverlayMethodTable` does.
function compute_rule_answer(@nospecialize(rule), annotate::Bool, @nospecialize(specTypes::Type), world::UInt, @nospecialize(view))::RuleAnswer
    session = RULE_SESSION[]
    sig = rule_query_sig(rule, annotate, specTypes)
    if !(ccall(:jl_method_table_for, Any, (Any,), sig) isa Core.MethodTable)
        # No method table can hold a rule for this signature, in any world.
        return RuleAnswer(false, RuleLookup[], UInt(1), typemax(UInt), typemax(UInt), session)
    end
    counter = newest_world()
    lookups = RuleLookup[]
    min_valid = UInt(1)
    max_valid = typemax(UInt)
    value = false
    covered = false
    tables = view === nothing ? () : method_tables(view)
    for table in tables
        if !(table isa Core.MethodTable)
            # A view Enzyme cannot split into tables: ask it as a whole, for this world only.
            value = EnzymeRules.isapplicable(rule, rule_query_tt(rule, annotate, specTypes); world, method_table = view)
            return RuleAnswer(value, RuleLookup[RuleLookup(view, sig, Core.svec(), false, world)], world, world, world, session)
        end
        l, lmin, lmax = lookup_table(sig, table, world)
        push!(lookups, l)
        min_valid = max(min_valid, lmin)
        max_valid = min(max_valid, lmax)
        value |= !isempty(l.methods)
        if l.fully_covers
            covered = true
            break
        end
    end
    if !covered
        l, lmin, lmax = lookup_table(sig, nothing, world)
        push!(lookups, l)
        min_valid = max(min_valid, lmin)
        max_valid = min(max_valid, lmax)
        value |= !isempty(l.methods)
    end
    validated = max_valid == typemax(UInt) ? max(world, counter) : max_valid
    return RuleAnswer(value, lookups, min_valid, validated, max_valid, session)
end

"""
    rule_answer(rule, annotate, specTypes, world, view) -> RuleAnswer

The answer to whether a call of signature `specTypes` has a method of `rule` in `world`,
looked up in the method table view `view` (`nothing` for the internal table). The answer is
recorded with the compilation collecting rule dependencies, if any (`RuleDepsCollector`).
"""
function rule_answer(@nospecialize(rule), annotate::Bool, @nospecialize(specTypes::Type), world::UInt, @nospecialize(view))::RuleAnswer
    key = (rule, specTypes, view === nothing ? () : method_tables(view))
    answer = nothing
    lock(RULE_LOCK)
    try
        RULE_STATS.queries += 1
        answers = get(RULE_ANSWERS, key, nothing)
        if answers !== nothing
            for a in answers
                if answer_valid!(a, world)
                    answer = a
                    break
                end
            end
        end
        if answer === nothing
            RULE_STATS.computed += 1
            answer = compute_rule_answer(rule, annotate, specTypes, world, view)
            if answers === nothing
                RULE_ANSWERS[key] = RuleAnswer[answer]
            else
                # Keep the answers of other worlds, but not the closed ones.
                j = 0
                for a in answers
                    if (@atomic :monotonic a.max_world) == typemax(UInt)
                        j += 1
                        answers[j] = a
                    end
                end
                resize!(answers, j)
                push!(answers, answer)
            end
        end
    finally
        unlock(RULE_LOCK)
    end
    record_answer!(answer)
    return answer
end

function rule_applies(@nospecialize(rule), annotate::Bool, @nospecialize(specTypes::Type), world::UInt, @nospecialize(view))::Bool
    func = specTypes.parameters[1]
    func isa Core.Builtin && return false
    return rule_answer(rule, annotate, specTypes, world, view).value
end

cached_has_frule(specTypes::Type, world::UInt, method_table)::Bool =
    rule_applies(EnzymeRules.forward, true, specTypes, world, method_table)
cached_has_rrule(specTypes::Type, world::UInt, method_table)::Bool =
    rule_applies(EnzymeRules.augmented_primal, true, specTypes, world, method_table)
cached_is_inactive(specTypes::Type, world::UInt, method_table)::Bool =
    rule_applies(EnzymeRules.inactive, false, specTypes, world, method_table)
cached_is_inactive_noinl(specTypes::Type, world::UInt, method_table)::Bool =
    rule_applies(EnzymeRules.inactive_noinl, false, specTypes, world, method_table)
cached_is_inactive_kwarg(specTypes::Type, world::UInt, method_table = nothing)::Bool =
    rule_applies(EnzymeRules.inactive_kwarg, false, specTypes, world, method_table)
cached_is_inactive_arg(specTypes::Type, world::UInt, method_table = nothing)::Bool =
    rule_applies(EnzymeRules.inactive_arg, false, specTypes, world, method_table)
cached_noalias(specTypes::Type, world::UInt, method_table)::Bool =
    rule_applies(EnzymeRules.noalias, false, specTypes, world, method_table)
cached_has_easy_rule(specTypes::Type, world::UInt)::Bool =
    rule_applies(EnzymeRules.has_easy_rule, false, specTypes, world, nothing)

## collecting what a compilation depends on

"""
    RuleDepsCollector

What a compilation in progress on this task depended on: the rule answers it consulted,
and the Enzyme-owned `CodeInstance`s it took from the cache or put into it, whose own
answers it depends on in turn (see `flatten_rule_deps`).
"""
struct RuleDepsCollector
    answers::Vector{RuleAnswer}
    cis::Vector{Core.CodeInstance}
    seen::IdDict{Any, Nothing}
end
RuleDepsCollector() = RuleDepsCollector(RuleAnswer[], Core.CodeInstance[], IdDict{Any, Nothing}())

const COLLECTOR_KEY = :EnzymeRuleDepsCollector

@inline function current_collector()
    tls = current_task().storage
    tls === nothing && return nothing
    return get(tls::IdDict{Any, Any}, COLLECTOR_KEY, nothing)::Union{Nothing, RuleDepsCollector}
end

function record_answer!(a::RuleAnswer)
    c = current_collector()
    c === nothing && return nothing
    if !haskey(c.seen, a)
        c.seen[a] = nothing
        push!(c.answers, a)
    end
    return nothing
end

function record_ci!(ci::Core.CodeInstance)
    c = current_collector()
    c === nothing && return nothing
    if !haskey(c.seen, ci)
        c.seen[ci] = nothing
        push!(c.cis, ci)
    end
    return nothing
end

function push_collector!()
    tls = task_local_storage()
    prev = get(tls, COLLECTOR_KEY, nothing)
    c = RuleDepsCollector()
    tls[COLLECTOR_KEY] = c
    return c, prev
end

# End the collection `c`; what it collected is also what the enclosing collection depends on.
function pop_collector!(c::RuleDepsCollector, @nospecialize(prev))
    tls = task_local_storage()
    if prev === nothing
        delete!(tls, COLLECTOR_KEY)
    else
        tls[COLLECTOR_KEY] = prev
        prev = prev::RuleDepsCollector
        for a in c.answers
            if !haskey(prev.seen, a)
                prev.seen[a] = nothing
                push!(prev.answers, a)
            end
        end
        for ci in c.cis
            if !haskey(prev.seen, ci)
                prev.seen[ci] = nothing
                push!(prev.cis, ci)
            end
        end
    end
    return nothing
end

function merge_answers!(answers::Vector{RuleAnswer})
    c = current_collector()
    c === nothing && return nothing
    for a in answers
        if !haskey(c.seen, a)
            c.seen[a] = nothing
            push!(c.answers, a)
        end
    end
    return nothing
end

## inference results

# The rule answers an Enzyme-owned `CodeInstance` (or an inference result) carries.
function rule_deps_of(@nospecialize(x))::Union{Nothing, EnzymeRuleDeps}
    @static if HAS_RULE_DEPS
        results = x.analysis_results
        while results isa Core.Compiler.AnalysisResults && isdefined(results, :next)
            r = results.result
            r isa EnzymeRuleDeps && return r
            results = results.next
        end
    end
    return nothing
end

is_enzyme_ci(ci::Core.CodeInstance) = ci.owner isa EnzymeCacheToken

# The Enzyme-owned `CodeInstance`s `ci` has edges to: its code was inferred with theirs,
# which may have been inlined into it, so it depends on their rule answers too.
function enzyme_ci_children(ci::Core.CodeInstance)
    children = Core.CodeInstance[]
    @static if HAS_RULE_DEPS
        isdefined(ci, :edges) || return children
        for e in ci.edges
            if e isa Core.CodeInstance && e !== ci && is_enzyme_ci(e)
                push!(children, e)
            end
        end
    end
    return children
end

"""
    flatten_rule_deps(c::RuleDepsCollector) -> Vector{RuleAnswer}

Every rule answer the compilation `c` collected depends on: its own, and those of the
`CodeInstance`s it used and of all Enzyme-owned instances they reach through their edges.
"""
function flatten_rule_deps(c::RuleDepsCollector)::Vector{RuleAnswer}
    answers = copy(c.answers)
    seen = IdDict{Any, Nothing}()
    for a in answers
        seen[a] = nothing
    end
    stack = copy(c.cis)
    while !isempty(stack)
        ci = pop!(stack)
        haskey(seen, ci) && continue
        seen[ci] = nothing
        deps = rule_deps_of(ci)
        if deps !== nothing
            for a in deps.answers
                if !haskey(seen, a)
                    seen[a] = nothing
                    push!(answers, a)
                end
            end
        end
        append!(stack, enzyme_ci_children(ci))
    end
    return answers
end

# Whether `ci` is known to be valid (`:valid`) or invalid (`:invalid`) in `world` without
# walking its edges.
function ci_quick(ci::Core.CodeInstance, world::UInt)::Symbol
    is_enzyme_ci(ci) || return :valid
    world > (@atomic :acquire ci.max_world) && return :invalid
    validated = get(CI_VALIDATED, ci, UInt(0))
    world <= validated && return :valid
    return :unknown
end

function own_answers_valid!(ci::Core.CodeInstance, world::UInt)::Tuple{Bool, UInt}
    deps = rule_deps_of(ci)
    min_valid = UInt(1)
    deps === nothing && return true, min_valid
    for a in deps.answers
        answer_valid!(a, world) || return false, min_valid
        min_valid = max(min_valid, @atomic :monotonic a.min_world)
    end
    return true, min_valid
end

"""
    ci_rule_valid!(ci, world) -> Bool

Whether the Enzyme-owned `ci` holds in `world` as far as rules are concerned: its own rule
answers and those of every Enzyme-owned `CodeInstance` it reaches through its edges hold
there. Like `verify_method` of `Compiler/src/reinfer.jl`, the walk settles every instance
it visits: those that hold are recorded as verified through `world`, and those that do not
are closed at the last world they were known to hold in, so that no lookup hands them out
again, Julia's own included.
"""
function ci_rule_valid!(ci::Core.CodeInstance, world::UInt)::Bool
    lock(RULE_LOCK)
    try
        q = ci_quick(ci, world)
        q === :unknown || return q === :valid
        RULE_STATS.ci_walks += 1
        nodes = Core.CodeInstance[]
        index = IdDict{Core.CodeInstance, Int}()
        valid = Bool[]
        min_worlds = UInt[]
        children_of = Vector{Vector{Core.CodeInstance}}()
        stack = Core.CodeInstance[ci]
        while !isempty(stack)
            n = pop!(stack)
            haskey(index, n) && continue
            push!(nodes, n)
            index[n] = length(nodes)
            ok, min_valid = own_answers_valid!(n, world)
            children = enzyme_ci_children(n)
            for child in children
                q = ci_quick(child, world)
                if q === :unknown
                    push!(stack, child)
                elseif q === :invalid
                    ok = false
                end
            end
            push!(valid, ok)
            push!(min_worlds, min_valid)
            push!(children_of, children)
        end
        # An instance is invalid if it reaches an invalid one.
        changed = true
        while changed
            changed = false
            for i in 1:length(nodes)
                valid[i] || continue
                for child in children_of[i]
                    j = get(index, child, 0)
                    if j != 0 && !valid[j]
                        valid[i] = false
                        changed = true
                        break
                    end
                end
            end
        end
        counter = newest_world()
        for i in 1:length(nodes)
            n = nodes[i]
            if valid[i]
                CI_VALIDATED[n] = max(world, get(CI_VALIDATED, n, UInt(0)))
                # The answers may hold from a later world than the instance's own edges.
                if min_worlds[i] > (@atomic :monotonic n.min_world) && min_worlds[i] <= world
                    @atomic :monotonic n.min_world = min_worlds[i]
                end
            else
                RULE_STATS.ci_closed += 1
                last_good = get(CI_VALIDATED, n, UInt(0))
                if last_good < (@atomic :monotonic n.min_world)
                    last_good = UInt(0)
                end
                if last_good < (@atomic :acquire n.max_world)
                    @atomic :release n.max_world = last_good
                end
                delete!(CI_VALIDATED, n)
            end
        end
        return valid[1]
    finally
        unlock(RULE_LOCK)
    end
end

# A `CodeInstance` the `EnzymeInterpreter` just made in `world`: its answers were all
# looked up in that world.
function ci_inserted!(ci::Core.CodeInstance, world::UInt)
    lock(RULE_LOCK)
    try
        CI_VALIDATED[ci] = world
    finally
        unlock(RULE_LOCK)
    end
    record_ci!(ci)
    return nothing
end

## compiled thunks

"""
    ThunkGuard{T}

The thunk `result` the generated `thunk` bakes into its code, with the rule answers its
compilation depended on. The generated code returns `result` only while those hold in the
world it runs in (`thunk_guard_ok`); otherwise it recompiles in that world, and later calls
of the same, still natively valid, code find the recompiled thunk in `replacement`
(`guarded_thunk_slow`).
"""
mutable struct ThunkGuard{T}
    const result::T
    const answers::Vector{RuleAnswer}
    const params::Any
    @atomic min_world::UInt
    # The world through which the guard is known to hold, kept as a `Ptr` so that a package
    # image does not carry it: Julia writes the `Ptr` fields of mutable objects into an image
    # as `C_NULL` (see `guard_validated`).
    @atomic validated::Ptr{Cvoid}
    @atomic session::Any
    @atomic replacement::Any
end

# The world through which `g` is known to hold: 0 for a guard restored from a package image.
@inline guard_validated(g::ThunkGuard) = UInt((@atomic :acquire g.validated))

# A world as `validated` keeps it. `-1` would survive an image, so the newest world it can
# record is one before.
validated_ptr(world::UInt) = Ptr{Cvoid}(min(world, typemax(UInt) - 1))

function ThunkGuard(@nospecialize(result), answers::Vector{RuleAnswer}, @nospecialize(params), world::UInt)
    validated = typemax(UInt)
    for a in answers
        validated = min(validated, @atomic :acquire a.validated)
    end
    validated = max(validated, world)
    return ThunkGuard{typeof(result)}(result, answers, params, world, validated_ptr(validated), RULE_SESSION[], nothing)
end

# Whether the thunk baked into the code holds in the world the code runs in. The code
# returns the baked thunk itself, a constant, when it does: a differentiation of that code
# (nested differentiation) must see the constant to find the thunk's IR (`autodiff_cache`).
#
# This runs on every call of the code, so it decides with two plain loads. The code runs in
# no world before the one it was generated for (`g.min_world`), and in none after the newest
# world, so the guard holds where it runs if it is known to hold through the newest world.
# When it is not (a method was defined anywhere since it was last verified),
# `guarded_thunk_slow` verifies it in the world of the call, which extends `validated` to
# the newest world again. A guard restored from a package image has `validated` 0, so it
# always takes that path, which checks that the guard is this session's.
@inline function thunk_guard_ok(g::ThunkGuard)::Bool
    return newest_world() <= guard_validated(g)
end

function guard_holds!(g::ThunkGuard, world::UInt)::Bool
    (@atomic :acquire g.session) === RULE_SESSION[] || return false
    world < (@atomic :monotonic g.min_world) && return false
    # Each answer that holds in `world` is known to hold through some newer world as well
    # (`answer_valid!`), and so is the guard, through the first of them to end.
    validated = max(world, newest_world())
    for a in g.answers
        answer_valid!(a, world) || return false
        validated = min(validated, @atomic :acquire a.validated)
    end
    validated = max(validated, world)
    if validated > guard_validated(g)
        @atomic :release g.validated = validated_ptr(validated)
    end
    return true
end

@noinline function guarded_thunk_slow(g::ThunkGuard{T})::T where {T}
    world = GPUCompiler.tls_world_age()
    RULE_STATS.guard_slow += 1
    guard_holds!(g, world) && return g.result
    r = @atomic :acquire g.replacement
    while r !== nothing
        r = r::ThunkGuard{T}
        guard_holds!(r, world) && return r.result
        r = @atomic :acquire r.replacement
    end
    RULE_STATS.guard_recompiled += 1
    new = recompile_thunk(g.params, world)::ThunkGuard
    if !(new isa ThunkGuard{T})
        error(
            """Enzyme: a change of the rules since this code was compiled changes the type of its thunk
            from `$T` to `$(typeof(new.result))`; call it again from a newer world (e.g. `Base.invokelatest`)."""
        )
    end
    # Chain it, newest first, so later calls of this code find it.
    @atomic :release new.replacement = @atomic :acquire g.replacement
    @atomic :release g.replacement = new
    return new.result
end

"""
    clear_rule_caches!()

Forget the rule answers and verifications of this session (see `clear_caches!`).
"""
function clear_rule_caches!()
    lock(RULE_LOCK)
    try
        empty!(RULE_ANSWERS)
        empty!(CI_VALIDATED)
    finally
        unlock(RULE_LOCK)
    end
    return nothing
end
