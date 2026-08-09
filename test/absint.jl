using Enzyme, Test

struct BufferedMap!{X}
    x_buffer::Vector{X}
end

function (bc::BufferedMap!)()
    return @inbounds bc.x_buffer[1][1]
end

@testset "Absint struct vector of vector" begin
    f = BufferedMap!([[2.7]])
    df = BufferedMap!([[3.1]])

    @test autodiff(Forward, Duplicated(f, df))[1] ≈ 3.1
end

@testset "Absint sum vector of vector" begin
    a = [[2.7]]
    da = [[3.1]]
    @test autodiff(Forward, sum, Duplicated(a, da))[1] ≈ [3.1]
end

struct MyStruct
    a::Float64
    b::Int
    c::Float64
    d::Int
end

function f_absint_memcpy!(dest, src)
    if length(src) > 0
        dest[1] = src[1]
        for i in 2:length(src)
            dest[i] = src[i]
        end
    end
    nothing
end

@testset "Absint Ptr/GEP memcpy translation" begin
    dest = [MyStruct(0.0, 0, 0.0, 0) for _ in 1:3]
    ddest = [MyStruct(0.0, 0, 0.0, 0) for _ in 1:3]
    src = [MyStruct(1.0, 2, 3.0, 4) for _ in 1:3]
    dsrc = [MyStruct(0.0, 0, 0.0, 0) for _ in 1:3]

    autodiff(Reverse, f_absint_memcpy!, Duplicated(dest, ddest), Duplicated(src, dsrc))
    @test ddest[1].a == 0.0 # Just verifying it runs without EnzymeNoTypeError
end

struct PeriodicTorsion{N, T}
    phases::NTuple{N, T}
    proper::Bool
end
function inject_interaction(inter::PeriodicTorsion{N, T}, params_dic) where {N, T}
    return PeriodicTorsion{N, T}(
        Base.inferencebarrier(ntuple(Returns(params_dic[]), N)),
        inter.proper,
    )
end
function loss(params_dic, inters)
    # Broadcast inject_interaction
    new_inters = inject_interaction.(inters, (params_dic,))
    inter = first(new_inters)
    # Use phases and ks
    return first(inter.phases)
end
@testset "Absint Ptr/GEP of select" begin
    T = Float64
    params_dic = Ref(1.5)
    
    inters = [
        PeriodicTorsion{2, Float64}(
            (2.7, 3.1),
            true
        )
    ]
    types = ["type1"]
    grads_enzyme = make_zero(params_dic)
    
    autodiff(
        set_runtime_activity(Reverse), loss, Active,
        Duplicated(params_dic, grads_enzyme), Const(inters),
    )
    @test grads_enzyme[] ≈ 1.0
end

@testset "Absint load of constantexpr gep HVP" begin
    @inline function mydouble(x)
        y = similar(x)
        for i in eachindex(x, y)
            y[i] = 2 * x[i]
        end
        return y
    end

    @inline function myouterproduct(x, y)
        z = similar(x, length(x), length(y))
        for i in eachindex(x)
            for j in eachindex(y)
                z[i, j] = x[i] * y[j]
            end
        end
        return z
    end

    function arr_to_num(x::AbstractArray)
        a = mydouble(x)
        b = myouterproduct(a, x)
        return b[1]
    end

    function f(x, c)
        c[1] = arr_to_num(x)
        return c[1]
    end

    function g(x, c)
        dx = zero(x)
        dc = zero(c)
        autodiff(Reverse, f, Duplicated(x, dx), Duplicated(c, dc))
        return dx
    end

    function h(x, c, dx_batch)
        dc_batch = map(dx -> zero(c), dx_batch)
        result = autodiff(Forward, g, BatchDuplicated(x, dx_batch), BatchDuplicated(c, dc_batch))
        return result
    end

    x = [3.0, 5.0]
    dx_batch = ([1.0, 0.0], [0.0, 1.0])
    c = [0.0]
    res = h(x, c, dx_batch)
    @test res[1][1] ≈ [4.0, 0.0]
    @test res[1][2] ≈ [0.0, 0.0]
end

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
    # On 1.10 this hits a pre-existing "undef value upon lcssa" in lookupM, which
    # reproduces without the absint change this test was added alongside.
    @static if VERSION < v"1.11-"
        @test_skip Enzyme.gradient(set_runtime_activity(Reverse), absint_run_world, [1.0])[1] ≈ [3.0]
    else
        @test Enzyme.gradient(set_runtime_activity(Reverse), absint_run_world, [1.0])[1] ≈ [3.0]
    end
end
