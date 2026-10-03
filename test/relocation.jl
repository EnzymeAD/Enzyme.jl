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

# The device module of a derivative refers to Julia values through `ejl_` globals, which only the
# host JIT resolves. GPUCompiler 2.x is handed a slot and a relocation record for each instead.
# (GPUCompiler 1.x, which resolves nothing, gets the address; see test/absint.jl.)
@testset "Julia-value globals in a device module" begin
    LLVM.Context() do ctx
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Enzyme.Compiler.Tracked)
        T_int8 = LLVM.Int8Type()
        mod = LLVM.Module("device")
        key = Enzyme.insert_julia_value!("device", RelocConst{Float64})
        gv = LLVM.GlobalVariable(mod, T_jlvalue, "ejl_" * key, Enzyme.Compiler.Tracked)

        fn = LLVM.Function(mod, "f", LLVM.FunctionType(LLVM.Int1Type(), [T_prjlvalue, LLVM.Int1Type()]))
        LLVM.IRBuilder() do B
            entry, other, join = (LLVM.BasicBlock(fn, n) for n in ("entry", "other", "join"))
            LLVM.position!(B, entry)
            LLVM.br!(B, LLVM.parameters(fn)[2], join, other)
            LLVM.position!(B, other)
            LLVM.br!(B, join)
            LLVM.position!(B, join)
            # Used directly, through a phi, and through constant expressions.
            phi = LLVM.phi!(B, T_prjlvalue)
            append!(LLVM.incoming(phi), [(gv, entry), (LLVM.parameters(fn)[1], other)])
            field = LLVM.const_inbounds_gep(T_int8, gv, [LLVM.ConstantInt(Int64(8))])
            loaded = LLVM.load!(B, T_prjlvalue, LLVM.addrspacecast!(B, field, LLVM.PointerType(T_jlvalue, Enzyme.Compiler.Derived)))
            same = LLVM.icmp!(B, LLVM.API.LLVMIntEQ, phi, loaded)
            LLVM.ret!(B, LLVM.and!(B, same, LLVM.icmp!(B, LLVM.API.LLVMIntEQ, LLVM.parameters(fn)[1], gv)))
        end
        @test LLVM.verify(mod) === nothing

        @static if Enzyme.Compiler.HAS_GPUCOMPILER_2
            relocs = GPUCompiler.Relocations()
            Enzyme.Compiler.relocate_julia_value_globals!(mod, relocs)
            @test LLVM.verify(mod) === nothing
            @test !haskey(LLVM.globals(mod), "ejl_" * key)
            slot_name = "ejl_slot_" * GPUCompiler.safe_name(key)
            @test haskey(LLVM.globals(mod), slot_name)
            @test LLVM.initializer(LLVM.globals(mod)[slot_name]) === nothing
            rec = only(relocs.records)
            @test rec.name == slot_name
            @test rec.target isa GPUCompiler.JuliaValueRef
            @test rec.target.value === RelocConst{Float64}
            # No address of the value is left in the module.
            addr = string(UInt(ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), RelocConst{Float64})))
            @test !occursin(addr, string(mod))
        end
        LLVM.dispose(mod)
    end
end
