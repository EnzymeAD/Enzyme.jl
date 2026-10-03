using Enzyme, Test
using LLVM
import GPUCompiler

# How a Julia object referenced by generated code reaches Enzyme differs between GPUCompiler
# majors: 1.x bakes the host address into the `julia.constgv` slot, 2.x keeps the slot symbolic
# until it is resolved. Enzyme must end up with the host address either way.
#
# These tests cover a toplevel job, whose references `emit_llvm` already resolves. The non-toplevel
# case, where Enzyme resolves a deferred job's references itself, is the "Nested Type Error"
# testset in test/basic.jl.

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

# The module GPUCompiler 2.x hands Enzyme for a job compiled on behalf of another keeps its slots
# symbolic until the requesting job lowers them, so they are declarations when Enzyme validates
# the module. Folding a load of one takes the object from the recorded values, and inserts a
# global named after it: no address goes into the module.
@static if Enzyme.Compiler.HAS_GPUCOMPILER_2
    @testset "constant-load folding of a symbolic slot" begin
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
            slots = Set(rec.name for rec in meta.relocations.records)
            enzyme_ctx = Enzyme.Compiler.EnzymeContext(world)
            Enzyme.Compiler.record_julia_values!(enzyme_ctx, meta)
            nchecked = 0
            for f in LLVM.functions(mod), bb in LLVM.blocks(f), inst in LLVM.instructions(bb)
                isa(inst, LLVM.LoadInst) || continue
                gv = LLVM.operands(inst)[1]
                isa(gv, LLVM.GlobalVariable) && LLVM.name(gv) in slots || continue
                LLVM.initializer(gv) === nothing || continue
                for u in LLVM.uses(inst)
                    user = LLVM.user(u)
                    isa(user, LLVM.Instruction) || continue
                    T = LLVM.value_type(user)
                    isa(T, LLVM.PointerType) && LLVM.addrspace(T) == Enzyme.Compiler.Tracked || continue
                    Enzyme.@with Enzyme.Compiler.ENZYME_CONTEXT => enzyme_ctx begin
                        folded = Enzyme.Compiler.try_replace_constant_load!(user; do_replace = false)
                        folded === user && continue # e.g. a mutable object, which is not folded
                        @test startswith(LLVM.name(folded), "ejl_inserted")
                        legal, val = Enzyme.Compiler.absint(folded)
                        @test legal
                        addr = UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), val))
                        @test !occursin(string(addr), LLVM.name(folded))
                    end
                    nchecked += 1
                end
            end
            # Only Julia 1.11+ emits the slots as declarations.
            GPUCompiler.supports_relocatable_ir() && @test nchecked > 0
        end
    end
end
