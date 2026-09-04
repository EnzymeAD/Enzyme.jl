using Enzyme, LLVM, Test

# Julia's codegen refers to a Julia object through a word-sized slot. Enzyme asks
# GPUCompiler to leave those slots symbolic (`:patch`) and rewrites each one into a
# reference to the object by name, so the differentiated module carries no address of this
# process and its artifact can be reused by another session.

const C = Enzyme.Compiler
const THUNK_CACHE = C.THUNK_CACHE

rp_plain(x) = sin(x) * x
rp_alloc(x) = sum(abs2, [x, 2x])
const RP_GLOBAL = [2.0, 3.0]
rp_global(x) = RP_GLOBAL[1] * x * RP_GLOBAL[2]

function rp_artifact(f)
    # A fresh session, so that differentiating `f` compiles a thunk rather than reusing the
    # one the generated method already holds.
    C.reset_session!()
    grad = autodiff(Reverse, f, Active(3.0))[1][1]
    h = only(
        hh for r in values(THUNK_CACHE.thunks) for hh in (r.adjoint, r.primal)
            if hh isa C.ThunkHandle && hh.mi.specTypes.parameters[1] === typeof(f)
    )
    job = C.CompilerJob(h.mi, h.config, Base.get_world_counter())
    res = C.enzyme_ci_results(job)
    art = res === nothing ? nothing : C.find_artifact(res, C.config_key(job))
    return grad, art
end

baked_addresses(ir) = unique([m.captures[1] for m in eachmatch(r"inttoptr \(i64 (\d{9,})", ir)])

@testset "derivatives are unchanged" begin
    @test rp_artifact(rp_plain)[1] ≈ cos(3.0) * 3.0 + sin(3.0)
    @test rp_artifact(rp_alloc)[1] ≈ 2 * 3.0 + 8 * 3.0
    @test rp_artifact(rp_global)[1] ≈ 6.0
end

# The symbolic primal is off by default (see `SYMBOLIC_PRIMAL`); these run it explicitly.
function with_symbolic_primal(f)
    old = C.SYMBOLIC_PRIMAL[]
    C.SYMBOLIC_PRIMAL[] = true
    return try
        f()
    finally
        C.SYMBOLIC_PRIMAL[] = old
        C.reset_session!()
    end
end

@static if C.HAS_CI_RESULTS && isdefined(Enzyme.Compiler.GPUCompiler, :Relocations)
    @testset "the module carries no address of this process" begin
        with_symbolic_primal() do
            for f in (rp_plain, rp_alloc)
                _, art = rp_artifact(f)
                @test art !== nothing
                @test art.persistable
                LLVM.Context() do ctx
                    mod = parse(LLVM.Module, LLVM.MemoryBuffer(art.bitcode))
                    ir = string(mod)
                    @test isempty(baked_addresses(ir))
                    # Whatever Julia objects it refers to are named instead of addressed.
                    @test f !== rp_plain || occursin("ejl_v_", ir)
                end
            end
        end
    end

    @testset "mutable data is named, not bound, but is not persisted" begin
        with_symbolic_primal() do
            # The reference is symbolic like any other, so the module holds no address; the
            # object itself cannot be reconstructed in another process, so the artifact is kept
            # out of the CodeInstance rather than silently reused.
            _, art = rp_artifact(rp_global)
            @test art === nothing
            h = only(
                hh for r in values(THUNK_CACHE.thunks) for hh in (r.adjoint, r.primal)
                    if hh isa C.ThunkHandle && hh.mi.specTypes.parameters[1] === typeof(rp_global)
            )
            ir = C.current_link(h).modstr
            @test isempty(baked_addresses(ir))
            @test occursin("ejl_v_", ir)
        end
    end

    @testset "baked addresses are detected" begin
        LLVM.Context() do ctx
            mod = LLVM.Module("baked")
            @test !C.bakes_addresses(mod)
            gv = LLVM.GlobalVariable(mod, LLVM.Int64Type(), "g")
            LLVM.initializer!(gv, LLVM.ConstantInt(Int64(7)))
            @test !C.bakes_addresses(mod)
            T_ptr = LLVM.PointerType(LLVM.Int8Type())
            pv = LLVM.GlobalVariable(mod, T_ptr, "p")
            LLVM.initializer!(pv, LLVM.const_inttoptr(LLVM.ConstantInt(UInt64(0x00007f0000000000)), T_ptr))
            @test C.bakes_addresses(mod)
        end
    end
end
