using Enzyme, LLVM, Test

# How a Julia object referenced by generated code is resolved differs between GPUCompiler
# majors: v1 bakes the host address into the `julia.constgv` slot, v2 strips the initializer
# for session portability and resolves it at the link step, possibly to a module-resident
# replica of the object. These test that Enzyme handles either form.

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

# A materialized replica lives in the module's read-only data rather than on the heap, so a
# collector that reaches one faults while setting its mark bits. Differentiating through a
# constant that is stored into GC-tracked memory must not produce such a pointer.
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

# Device code keeps the replica -- there is no Julia GC to upset and a host address would be
# meaningless -- so abstract interpretation has to be able to read one.
@testset "materialized box abstract interpretation" begin
    LLVM.Context() do ctx
        tag = UInt(pointer_from_objref(RelocConst{Float64}))
        mod = parse(
            LLVM.Module, """
            @jl_global_1_box = private unnamed_addr constant { i64, [8 x i8] } { i64 $tag, [8 x i8] c"\\00\\00\\00\\00\\00\\00\\F0?" }, align 16
            """
        )
        gv = LLVM.globals(mod)["jl_global_1_box"]
        boxty = LLVM.global_value_type(gv)
        T_int32 = LLVM.Int32Type()
        T_prjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Enzyme.Compiler.Tracked)
        idx(i) = LLVM.ConstantInt(T_int32, i)

        payload = LLVM.const_gep(boxty, gv, LLVM.Constant[idx(0), idx(1)])
        val = LLVM.const_addrspacecast(payload, T_prjlvalue)

        @test Enzyme.Compiler.absint_materialized_box(val) == (true, RelocConst{Float64}(1.0))
        @test Enzyme.Compiler.absint(val) == (true, RelocConst{Float64}(1.0))
        legal, typ, byref = Enzyme.Compiler.abs_typeof(val)
        @test legal
        @test typ == RelocConst{Float64}

        # The tag word is not the object.
        hdr = LLVM.const_addrspacecast(LLVM.const_gep(boxty, gv, LLVM.Constant[idx(0), idx(0)]), T_prjlvalue)
        @test Enzyme.Compiler.absint_materialized_box(hdr) == (false, nothing)
    end
end
