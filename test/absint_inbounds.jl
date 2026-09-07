using Enzyme, Test

# These reproducers only reach the paths they are meant to cover when
# `@inbounds` is honored, so runtests.jl runs this file on a worker with
# `--check-bounds=auto` rather than `Pkg.test`'s default `--check-bounds=yes`.
@assert Base.JLOptions().check_bounds == 0

struct AbsintNode
    neighbors::Vector{AbsintNode}
end

mutable struct AbsintArchetype
    tables::Vector{UInt32}
    node::AbsintNode
end

struct AbsintTable
    entities::Vector{Int64}
    id::UInt32
end

struct AbsintQuery{S <: Tuple}
    archetypes::Vector{AbsintArchetype}
    tables::Vector{AbsintTable}
    storages::S
end

function absint_iterate(q::AbsintQuery, state::Tuple{Int, Int})
    arch, tab = state
    while arch <= length(q.archetypes)
        @inbounds archetype = q.archetypes[arch]
        if tab == 0
            if isempty(archetype.tables)
                arch += 1
                continue
            end
            tab = 1
        end
        tables = archetype.tables
        while tab <= length(tables)
            @inbounds table = q.tables[Int(tables[tab])]
            @inbounds positions = q.storages[1][table.id]
            return (table.entities, positions), (arch, tab + 1)
        end
        arch += 1
        tab = 0
    end
    return nothing
end

Base.iterate(q::AbsintQuery, state::Tuple{Int, Int}) = absint_iterate(q, state)
Base.iterate(q::AbsintQuery) = Base.iterate(q, (1, 0))

function absint_run_world(args::Vector{Float64})
    @inbounds alpha = args[1]
    archetypes = AbsintArchetype[AbsintArchetype(UInt32[1], AbsintNode(Vector{AbsintNode}(undef, 1)))]
    tables = AbsintTable[AbsintTable(Int64[1], UInt32(1))]
    push!(archetypes, AbsintArchetype(UInt32[2], AbsintNode(Vector{AbsintNode}(undef, 1))))
    push!(tables, AbsintTable(Int64[101], UInt32(2)))
    columns = Vector{Float64}[[alpha], [2 * alpha]]
    q = AbsintQuery(archetypes, tables, (columns,))
    total = zero(alpha)
    for (entities, positions) in q
        @inbounds for pos in positions
            total += pos
        end
    end
    return total
end

# The `node` field makes `AbsintNode` a recursive type, whose typetree is
# necessarily incomplete. Without absint deducing the type of the dynamically
# indexed `q.archetypes[arch]` load, the `UInt32` table index loaded from
# `archetype.tables` is conservatively assumed active, and Enzyme errors out
# trying to differentiate the `shl` computing the byte offset into `q.tables`.
@testset "Absint dynamic index of vector of mutable structs" begin
    @test absint_run_world([1.0]) ≈ 3.0
    # On 1.10 this hits an "undef value upon lcssa" in lookupM, which reproduces
    # without the absint change this test was added alongside. Fixed by
    # https://github.com/EnzymeAD/Enzyme/pull/3114; unskip once that is in a
    # released Enzyme_jll.
    @static if VERSION < v"1.11-"
        @test_skip Enzyme.gradient(set_runtime_activity(Reverse), absint_run_world, [1.0])[1] ≈ [3.0]
    else
        @test Enzyme.gradient(set_runtime_activity(Reverse), absint_run_world, [1.0])[1] ≈ [3.0]
    end
end


struct AbsintInlineTable
    entities::Vector{Float64}
    id::UInt32
end

struct AbsintInlineQuery
    tables::Vector{AbsintInlineTable}
    archetypes::Vector{UInt32}
    storages::Vector{Float64}
end

Base.iterate(q::AbsintInlineQuery) = Base.iterate(q, 1)

function Base.iterate(q::AbsintInlineQuery, state::Int)
    arch = state
    arch > 1 && return nothing
    @inbounds archetype = q.archetypes[1]
    table = @inbounds q.tables[archetype]
    @inbounds col = q.storages[table.id]
    return col, arch + 1
end

function absint_make_query(args)
    tables = [AbsintInlineTable([1.0, 2.0, 3.0], UInt32(2))]
    arches = UInt32[1]
    storages = [args[1], 2 * args[1], 3 * args[1]]
    return AbsintInlineQuery(tables, arches, storages)
end

function absint_inline_field(args)
    q = absint_make_query(args)
    total = zero(eltype(args))
    for positions in q
        total += positions
    end
    return total
end

# `AbsintInlineTable` is stored inline in the memory, so the load of its second
# field is a gep of -sizeof(field) off of the memoryref's data pointer. Reducing
# that negative offset with rem rather than mod left absint unable to type the
# `UInt32`, which was then assumed active, and Enzyme errored out trying to
# differentiate the `shl` computing the byte offset into `q.storages`.
@testset "Absint trailing field of inline memory element" begin
    @test absint_inline_field([0.1]) ≈ 0.2
    @test Enzyme.gradient(Reverse, absint_inline_field, [0.1])[1] ≈ [2.0]
end
