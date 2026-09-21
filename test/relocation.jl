using Enzyme, Test

# How a Julia object referenced by generated code reaches Enzyme differs between GPUCompiler
# majors: 1.x bakes the host address into the `julia.constgv` slot, 2.x keeps the slot symbolic
# until it is resolved. Enzyme must end up with the host address either way.

abstract type RelocAbs end
struct RelocConst{T} <: RelocAbs
    σ::T
end

const RELOC_KEEP = Any[]

function reloc_stores_constant(x)
    dists = RelocAbs[RelocConst{Float64}(1.0)]
    push!(RELOC_KEEP, dists)
    return @inbounds dists[1].σ
end

# A constant that differentiated host code stores into GC-tracked memory must be the real heap
# object (a module-resident replica would make the collector fault on its mark bits).
@testset "constant global survives GC" begin
    empty!(RELOC_KEEP)
    res = autodiff(ForwardWithPrimal, Const(reloc_stores_constant), Duplicated{Float64}, Duplicated(2.7, 3.1))
    @test res[1] == 0.0
    @test res[2] == 1.0

    stored = RELOC_KEEP[1][1]
    @test stored === RelocConst{Float64}(1.0)
    GC.gc(true)
    @test RELOC_KEEP[1][1] === RelocConst{Float64}(1.0)
    empty!(RELOC_KEEP)
end

using LLVM
import GPUCompiler

# `f` returns the value loaded from the global slot `gv`, in the shape Julia's codegen gives a
# reference to a Julia value: an untracked load, cast into the tracked address space.
function reloc_slot_module(name::String)
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

# A GPUCompiler 2.x slot is a declaration: no address of the value it refers to is in the IR
# unless the back-end bakes one in. The compilation's table of slots is what tells `absint`
# which Julia value a load of it yields.
@testset "absint resolves a symbolic slot from the context" begin
    LLVM.Context() do ctx
        mod, gv, value = reloc_slot_module("slot")

        # Nothing to go by: no initializer, and no compilation in flight.
        @test Enzyme.Compiler.absint(value) == (false, nothing)
        @test !Enzyme.Compiler.abs_typeof(value)[1]

        enzyme_ctx = Enzyme.Compiler.EnzymeContext()
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
            # A compilation that has no record of the slot does not know either.
            @test Enzyme.Compiler.absint(value) == (false, nothing)

            enzyme_ctx.julia_values["slot"] = RelocConst{Float64}
            @test Enzyme.Compiler.absint(value) == (true, RelocConst{Float64})
            @test Enzyme.Compiler.abs_typeof(value) ==
                (true, Type{RelocConst{Float64}}, GPUCompiler.BITS_REF)

            # `nothing` is a value like any other, not a miss.
            enzyme_ctx.julia_values["slot"] = nothing
            @test Enzyme.Compiler.absint(value) == (true, nothing)
            @test Enzyme.Compiler.abs_typeof(value) ==
                (true, Nothing, GPUCompiler.BITS_REF)
        end
        LLVM.dispose(mod)
    end
end

# GPUCompiler 1.x, and a `:bake` back-end of 2.x, write the address into the initializer. That
# stays readable, with or without a table.
@testset "absint still decodes a baked slot" begin
    LLVM.Context() do ctx
        mod, gv, value = reloc_slot_module("slot")
        addr = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), RelocConst{Float64}))
        T_pjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]))
        word = LLVM.ConstantInt(LLVM.IntType(8 * sizeof(UInt)), addr)
        LLVM.initializer!(gv, LLVM.const_inttoptr(word, T_pjlvalue))

        @test Enzyme.Compiler.absint(value) == (true, RelocConst{Float64})
        Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => Enzyme.Compiler.EnzymeContext() begin
            @test Enzyme.Compiler.absint(value) == (true, RelocConst{Float64})
            @test Enzyme.Compiler.abs_typeof(value) ==
                (true, Type{RelocConst{Float64}}, GPUCompiler.BITS_REF)
        end
        LLVM.dispose(mod)
    end
end

function reloc_count_slot_loads(mod::LLVM.Module, julia_values::Dict{String, Any})
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

# The same, on the module GPUCompiler 2.x hands Enzyme for a job compiled on behalf of another:
# the slots are still symbolic, and the relocation records are all that names their values.
@static if Enzyme.Compiler.HAS_GPUCOMPILER_2
    @testset "absint resolves the slots of an unresolved primal module" begin
        world = Base.get_world_counter()
        mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(reloc_stores_constant), Tuple{Float64}, world)
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

            enzyme_ctx = Enzyme.Compiler.EnzymeContext()
            Enzyme.Compiler.record_julia_values!(enzyme_ctx, meta)
            julia_values = enzyme_ctx.julia_values
            @test !isempty(julia_values)
            # The constant `reloc_stores_constant` stores, which no name map knows of.
            @test RelocConst{Float64}(1.0) in values(julia_values)

            # Without the table there is nothing in the IR to read the value from.
            nslots, nresolved = reloc_count_slot_loads(mod, julia_values)
            @test nslots > 0
            @test nresolved == 0

            Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
                nslots, nresolved = reloc_count_slot_loads(mod, julia_values)
                @test nresolved == nslots
            end
        end
    end
end
