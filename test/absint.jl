using Enzyme, Test
using LLVM
import GPUCompiler

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

struct PaddedMixed
    data::Vector{Float64}
    sp::Tuple{Int, Bool}
end

@inline function absint_copy_zip(A)
    dst = similar(A)
    @inbounds for (i, a) in zip(eachindex(dst), A)
        dst[i] = a
    end
    return dst
end

function absint_padded_copies!(out, v)
    src = [PaddedMixed(v .* i, (i, isodd(i))) for i in 1:3]
    out[1] = absint_copy_zip(src)
    out[2] = absint_copy_zip(src)
    s = 0.0
    for m in out[1]
        s += sum(m.data)
    end
    for m in out[2]
        s += sum(m.data)
    end
    return s
end

@testset "Absint negative offset memcpy into array of padded structs" begin
    v = [1.0, 2.0, 3.0]
    dv = zero(v)
    out = Vector{Vector{PaddedMixed}}(undef, 2)
    dout = Vector{Vector{PaddedMixed}}(undef, 2)
    autodiff(Reverse, absint_padded_copies!, Active, Duplicated(out, dout), Duplicated(v, dv))
    @test dv ≈ [12.0, 12.0, 12.0]
    for k in 1:2
        @test [m.sp for m in dout[k]] == [m.sp for m in out[k]]
    end
end

struct AbsintInlineField
    data::Vector{Float64}
    pad::NTuple{22, Float64}
end

mutable struct AbsintWideModel
    pad::NTuple{70, Float64}
    vel::@NamedTuple{a::AbsintInlineField, b::AbsintInlineField, c::AbsintInlineField, d::AbsintInlineField}
    x::Float64
end

@noinline function absint_consume_wide(vel, n)
    s = 0.0
    for i in 1:n
        s += vel.d.pad[1] * vel.a.data[i] * vel.d.data[i]
    end
    return s
end

@noinline function absint_wide_tendencies(m::AbsintWideModel, callbacks)
    vel = m.vel
    s = absint_consume_wide(vel, length(vel.a.data))
    for cb in callbacks
        cb(m)
    end
    return s
end

absint_wide_loss(m) = absint_wide_tendencies(m, Any[])

absint_wide_field(v, p) = AbsintInlineField(v, ntuple(Returns(p), 22))

@testset "Absint memcpy out of an object past the type analysis offset limit" begin
    # `m.vel` is 736 bytes at offset 560, copied out piecewise: each tracked
    # pointer is loaded on its own and the runs of floats between them are
    # memcpy'd. Both the source offset and the fresh stack slot it lands in lie
    # beyond what type analysis keeps for either object, so the copy must be
    # typed from the Julia layout of the source.
    m = AbsintWideModel(
        ntuple(Returns(0.0), 70),
        (
            a = absint_wide_field([1.0, 2.0], 0.0), b = absint_wide_field([0.0, 0.0], 0.0),
            c = absint_wide_field([0.0, 0.0], 0.0), d = absint_wide_field([3.0, 4.0], 2.0),
        ),
        0.0,
    )
    dm = Enzyme.make_zero(m)
    autodiff(Reverse, absint_wide_loss, Active, Duplicated(m, dm))
    @test dm.vel.a.data ≈ [6.0, 8.0]
    @test dm.vel.d.data ≈ [2.0, 4.0]
    @test dm.vel.d.pad[1] ≈ 11.0
end

@noinline absint_closure_call(f) = f()
# Re-capturing `f` in a new closure copies it by value; on Julia 1.13 that copy
# is a memcpy spanning `f`'s leading reference field and the bits after it.
@noinline absint_closure_wrap(f) = absint_closure_call(() -> f())

function absint_closure_recapture(x, t)
    r = Ref(0)
    return absint_closure_wrap(() -> x[1] * t + r[])
end

@testset "Absint memcpy of a closure re-captured by value" begin
    x = [2.0]
    dx = zero(x)
    autodiff(Reverse, absint_closure_recapture, Active, Duplicated(x, dx), Const(0.5))
    @test dx ≈ [0.5]
end                                         

Base.@noinline absint_undef_barrier(f, args...; kwargs...) = f(args...; kwargs...)

struct AbsintAnyBox
    fn::Any
end
(box::AbsintAnyBox)(a, b) = box.fn(a, b)

struct AbsintTwoPtrs{T, C}
    tunable::T
    caches::C
end

struct AbsintPtrAndInt{U}
    u::U
    i::Int
end

function absint_undef_loss(p, box, s)
    absint_undef_barrier(box, p, (s,))
    return sum(p.tunable)
end

@testset "Absint memcpy from a stack slot that is never written" begin
    # On 1.13 the tracked slots of `p` travel in a separate roots object, so SROA
    # leaves the slice of the argument tuple's data half that would hold them
    # unwritten and unread -- but still memcpys it into the varargs tuple. That
    # copy reads only undefined bytes and has no type to find.
    p = AbsintTwoPtrs([2.0], [0.0])
    dp = Enzyme.make_zero(p)
    s = AbsintPtrAndInt([1.0, 1.0], 0)
    box = AbsintAnyBox((a, b) -> nothing)
    autodiff(
        set_runtime_activity(Reverse), Const(absint_undef_loss), Active,
        Duplicated(p, dp), Const(box), Const(s),
    )
    @test dp.tunable == [1.0]
    @test dp.caches == [0.0]
end

abstract type SlotAbs end
struct SlotConst{T} <: SlotAbs
    σ::T
end

const SLOT_KEEP = Any[]

function slot_stores_constant(x)
    dists = SlotAbs[SlotConst{Float64}(1.0)]
    push!(SLOT_KEEP, dists)
    return @inbounds dists[1].σ
end

# `f` returns the value loaded from the global slot `gv`, in the shape Julia's codegen gives a
# reference to a Julia value: an untracked load, cast into the tracked address space.
function slot_module(name::String)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 10)

    mod = LLVM.Module("slots")
    gv = LLVM.GlobalVariable(mod, T_pjlvalue, name)
    fn = LLVM.Function(mod, "f", LLVM.FunctionType(T_prjlvalue))
    value = LLVM.IRBuilder() do builder
        LLVM.position!(builder, LLVM.BasicBlock(fn, "entry"))
        loaded = LLVM.load!(builder, T_pjlvalue, gv)
        tracked = LLVM.addrspacecast!(builder, loaded, T_prjlvalue)
        LLVM.ret!(builder, tracked)
        tracked
    end
    return mod, gv, value
end

# A slot without an initializer has no address of the value it refers to in the IR. The
# compilation's table of slots is what tells `absint` which Julia value a load of it yields.
@testset "absint resolves a symbolic slot from the context" begin
    LLVM.Context() do ctx
        mod, gv, value = slot_module("slot")

        # Nothing to go by: no initializer, and no compilation in flight.
        @test Enzyme.Compiler.absint(value) == (false, nothing)
        @test !Enzyme.Compiler.abs_typeof(value)[1]

        enzyme_ctx = Enzyme.Compiler.EnzymeContext(Base.get_world_counter())
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
            # A compilation that has no record of the slot does not know either.
            @test Enzyme.Compiler.absint(value) == (false, nothing)

            enzyme_ctx.julia_values["slot"] = SlotConst{Float64}
            @test Enzyme.Compiler.absint(value) == (true, SlotConst{Float64})
            @test Enzyme.Compiler.abs_typeof(value) ==
                (true, Type{SlotConst{Float64}}, GPUCompiler.BITS_REF)

            # `nothing` is a value like any other, not a miss.
            enzyme_ctx.julia_values["slot"] = nothing
            @test Enzyme.Compiler.absint(value) == (true, nothing)
            @test Enzyme.Compiler.abs_typeof(value) ==
                (true, Nothing, GPUCompiler.BITS_REF)
        end
        LLVM.dispose(mod)
    end
end

# A slot whose initializer holds the address stays readable, with or without a table.
@testset "absint still decodes a baked slot" begin
    LLVM.Context() do ctx
        mod, gv, value = slot_module("slot")
        addr = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), SlotConst{Float64}))
        T_pjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]))
        word = LLVM.ConstantInt(LLVM.IntType(8 * sizeof(UInt)), addr)
        LLVM.initializer!(gv, LLVM.const_inttoptr(word, T_pjlvalue))

        @test Enzyme.Compiler.absint(value) == (true, SlotConst{Float64})
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => Enzyme.Compiler.EnzymeContext(Base.get_world_counter()) begin
            @test Enzyme.Compiler.absint(value) == (true, SlotConst{Float64})
            @test Enzyme.Compiler.abs_typeof(value) ==
                (true, Type{SlotConst{Float64}}, GPUCompiler.BITS_REF)
        end
        LLVM.dispose(mod)
    end
end

function count_slot_loads(mod::LLVM.Module, julia_values::Dict{String, Any})
    nslots = 0
    nresolved = 0
    for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
        inst isa LLVM.LoadInst || continue
        gv = LLVM.operands(inst)[1]
        gv isa LLVM.GlobalVariable || continue
        haskey(julia_values, LLVM.name(gv)) || continue
        nslots += 1
        legal, val = Enzyme.Compiler.absint(inst, false, true)
        if legal && val === julia_values[LLVM.name(gv)]
            nresolved += 1
        end
    end
    return nslots, nresolved
end

# GPUCompiler reports the object behind each slot of the module it emits as `gv_to_value`; the
# table is filled from that, and resolves every load of a slot.
@testset "the table is filled from gv_to_value" begin
    world = Base.get_world_counter()
    mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(slot_stores_constant), Tuple{Float64}, world)
    config = GPUCompiler.CompilerConfig(
        Enzyme.Compiler.DefaultCompilerTarget(),
        Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode);
        kernel = false, libraries = true, toplevel = false, optimize = false,
        cleanup = false, only_entry = false, validate = false, entry_abi = :specfunc,
    )
    job = GPUCompiler.CompilerJob(mi, config, world)
    GPUCompiler.JuliaContext() do _
        GPUCompiler.prepare_job!(job)
        mod, meta = GPUCompiler.emit_llvm(job)

        enzyme_ctx = Enzyme.Compiler.EnzymeContext(world)
        Enzyme.Compiler.record_julia_values!(enzyme_ctx, meta)
        julia_values = enzyme_ctx.julia_values
        @static if VERSION < v"1.11-"
            # Julia 1.10's codegen writes the address of an object into the IR, not a slot.
            @test isempty(julia_values)
        else
            @test !isempty(julia_values)
            @test SlotConst{Float64}(1.0) in values(julia_values)
            Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
                nslots, nresolved = count_slot_loads(mod, julia_values)
                @test nslots > 0
                @test nresolved == nslots
            end
        end
    end
end

# Folding a constant load takes the object from the table too, so it holds when the address in
# the IR does not say the same: here the initializer points at one tuple, the table at another.
# Read straight out of an `Any` container, so that the address taken is that of a box that stays
# alive; an immutable passed to `jl_value_ptr` from a concretely typed variable is boxed afresh.
const SLOT_BAKED = Ref{Any}(("slot-baked",))
const SLOT_RECORDED = Ref{Any}(("slot-recorded",))

@testset "constant-load folding takes the slot's object from the table" begin
    LLVM.Context() do ctx
        mod, gv, value = slot_module("slot")
        LLVM.metadata(gv)["julia.constgv"] = LLVM.MDNode(LLVM.Metadata[])
        addr = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), SLOT_BAKED[]))
        T_pjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]))
        word = LLVM.ConstantInt(LLVM.IntType(8 * sizeof(UInt)), addr)
        LLVM.initializer!(gv, LLVM.const_inttoptr(word, T_pjlvalue))

        folded = Enzyme.Compiler.try_replace_constant_load!(value; do_replace = false)
        @test folded !== value
        @test Enzyme.Compiler.absint(folded) == (true, SLOT_BAKED[])

        enzyme_ctx = Enzyme.Compiler.EnzymeContext(Base.get_world_counter())
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
            enzyme_ctx.julia_values["slot"] = SLOT_RECORDED[]
            folded = Enzyme.Compiler.try_replace_constant_load!(value; do_replace = false)
            @test Enzyme.Compiler.absint(folded) == (true, SLOT_RECORDED[])

            # The table holds an `isbits` value unboxed, without the address of its box: the
            # initializer is what is left to go by.
            enzyme_ctx.julia_values["slot"] = 2.5
            folded = Enzyme.Compiler.try_replace_constant_load!(value; do_replace = false)
            @test Enzyme.Compiler.absint(folded) == (true, SLOT_BAKED[])

            # Without an initializer, as Enzyme keeps slots until the module is linked, the
            # record is all there is: the load still folds, into a global named after the object.
            enzyme_ctx.julia_values["slot"] = SLOT_RECORDED[]
            LLVM.initializer!(gv, nothing)
            folded = Enzyme.Compiler.try_replace_constant_load!(value; do_replace = false)
            @test Enzyme.Compiler.absint(folded) == (true, SLOT_RECORDED[])
            @test !occursin(string(UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), SLOT_RECORDED[]))), LLVM.name(folded))
        end
        # Nor is a slot without an initializer folded when nothing records it.
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => Enzyme.Compiler.EnzymeContext(Base.get_world_counter()) begin
            @test Enzyme.Compiler.try_replace_constant_load!(value; do_replace = false) === value
        end
        LLVM.dispose(mod)
    end
end

# Enzyme names a Julia value it inserts by the value's `objectid`, not by its address, and the
# JIT resolves the name from `JuliaEnzymeNameMap`: nothing session-specific is in the name.
@testset "inserted Julia values are named, not addressed" begin
    val = SlotConst{Float64}(2.5)
    key = Enzyme.insert_julia_value!("hint", val)
    @test startswith(key, "inserted\$hint\$")
    @test !occursin(string(UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), val))), key)
    @test Enzyme.Compiler.JuliaEnzymeNameMap[key] === val
    # The same value is the same name; another value is another one.
    @test Enzyme.insert_julia_value!("hint", val) == key
    @test Enzyme.insert_julia_value!("hint", SlotConst{Float64}(3.5)) != key
end

# Enzyme works on a module with its slots as declarations, the way GPUCompiler 2.x hands over a
# job compiled on behalf of another, and writes the addresses back in when it is linked.
# Julia 1.10's codegen writes the address of an object into the IR, not a slot.
@static if VERSION >= v"1.11-"
@testset "slots stay symbolic until the module is linked" begin
    world = Base.get_world_counter()
    mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(slot_stores_constant), Tuple{Float64}, world)
    config = GPUCompiler.CompilerConfig(
        Enzyme.Compiler.DefaultCompilerTarget(),
        Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode);
        kernel = false, libraries = true, toplevel = false, optimize = false,
        cleanup = false, only_entry = false, validate = false, entry_abi = :specfunc,
    )
    job = GPUCompiler.CompilerJob(mi, config, world)
    GPUCompiler.JuliaContext() do _
        GPUCompiler.prepare_job!(job)
        mod, meta = GPUCompiler.emit_llvm(job)
        enzyme_ctx = Enzyme.Compiler.EnzymeContext(world)
        Enzyme.Compiler.record_julia_values!(enzyme_ctx, meta)
        julia_values = enzyme_ctx.julia_values
        baked = Dict(
            LLVM.name(gv) => string(LLVM.initializer(gv)) for gv in LLVM.globals(mod)
                if haskey(julia_values, LLVM.name(gv))
        )
        @test !isempty(baked)

        Enzyme.Compiler.make_slots_symbolic!(mod)
        for name in keys(baked)
            gv = LLVM.globals(mod)[name]
            @test LLVM.isdeclaration(gv)
            @test LLVM.isconstant(gv)
        end
        # The addresses of the values are gone from the module.
        str = string(mod)
        for name in keys(baked)
            addr = Enzyme.Compiler.JuliaSlotMap[name][2]
            @test !occursin("i64 $(reinterpret(UInt, addr)) to", str)
        end
        # Analysis sees through the slots all the same, and so would a later compilation.
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
            nslots, nresolved = count_slot_loads(mod, julia_values)
            @test nslots > 0
            @test nresolved == nslots
        end
        nslots, nresolved = count_slot_loads(mod, julia_values)
        @test nresolved == nslots

        Enzyme.Compiler.resolve_slots!(mod)
        for (name, init) in baked
            gv = LLVM.globals(mod)[name]
            @test !LLVM.isdeclaration(gv)
            @test string(LLVM.initializer(gv)) == init
        end
        @test LLVM.verify(mod) === nothing
    end
end
end

# GPUCompiler 1.x resolves nothing in device code, so the Julia values Enzyme refers to there by
# name get their address when the derivative is handed over.
@testset "Julia-value globals of a device module are baked on GPUCompiler 1.x" begin
    LLVM.Context() do ctx
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Enzyme.Compiler.Tracked)
        mod = LLVM.Module("device")
        key = Enzyme.insert_julia_value!("device", SlotConst{Float64})
        gv = LLVM.GlobalVariable(mod, T_jlvalue, "ejl_" * key, Enzyme.Compiler.Tracked)
        fn = LLVM.Function(mod, "f", LLVM.FunctionType(T_prjlvalue))
        LLVM.IRBuilder() do B
            LLVM.position!(B, LLVM.BasicBlock(fn, "entry"))
            LLVM.ret!(B, gv)
        end
        Enzyme.Compiler.bake_julia_value_globals!(mod)
        @test !haskey(LLVM.globals(mod), "ejl_" * key)
        @test occursin("inttoptr", string(mod))
        @test LLVM.verify(mod) === nothing
        LLVM.dispose(mod)
    end
end
