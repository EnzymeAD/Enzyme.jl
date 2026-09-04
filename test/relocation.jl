using Enzyme, LLVM, Test

# How a Julia object referenced by generated code reaches Enzyme differs between GPUCompiler
# majors: 1.x bakes the host address into the `julia.constgv` slot, 2.x keeps the slot symbolic
# until it is resolved, for a device job possibly to a module-resident replica of the object.
# Enzyme must handle either form.

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

# Device code keeps the replica, so abstract interpretation has to be able to read one.
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

# A box whose payload happens to be all zeros: LLVM stores that array as `zeroinitializer`
# (a `ConstantAggregateZero`, not a `ConstantDataArray`), which the byte-wise decode must not
# hand to `LLVMGetElementAsConstant`.
@testset "materialized box with a zero payload" begin
    LLVM.Context() do ctx
        tag = UInt(pointer_from_objref(RelocConst{Float64}))
        mod = parse(
            LLVM.Module, """
            @jl_global_2_box = private unnamed_addr constant { i64, [8 x i8] } { i64 $tag, [8 x i8] zeroinitializer }, align 16
            """
        )
        gv = LLVM.globals(mod)["jl_global_2_box"]
        boxty = LLVM.global_value_type(gv)
        T_int32 = LLVM.Int32Type()
        T_prjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Enzyme.Compiler.Tracked)
        idx(i) = LLVM.ConstantInt(T_int32, i)

        payload = LLVM.const_gep(boxty, gv, LLVM.Constant[idx(0), idx(1)])
        val = LLVM.const_addrspacecast(payload, T_prjlvalue)

        @test Enzyme.Compiler.absint_materialized_box(val) == (true, RelocConst{Float64}(0.0))
    end
end

# A derivative compiled on behalf of another job — `autodiff_deferred` reached from an outer
# `autodiff` — is not a toplevel job, and GPUCompiler 2.x resolves a job's Julia-value
# references only for toplevel ones. Enzyme resolves the rest itself; without that the type
# argument of an allocation is a symbolic slot rather than a `DataType`, and the constant
# globals reached from the deferred code have no shadow.
@testset "deferred job sees resolved constants" begin
    reloc_f(x) = sum(tanh, x)

    function reloc_df!(dx, x)
        make_zero!(dx)
        autodiff_deferred(Reverse, Const(reloc_f), Active, Duplicated(x, dx))
        return nothing
    end

    function reloc_hvp!(hv, v, x)
        make_zero!(hv)
        autodiff(Forward, reloc_df!, Const, Duplicated(make_zero(x), hv), Duplicated(x, v))
        return nothing
    end

    x = [0.5]
    v = [1.0]
    hv = make_zero(v)
    reloc_hvp!(hv, v, x)
    @test hv ≈ [-2 * tanh(0.5) * sech(0.5)^2]
end
