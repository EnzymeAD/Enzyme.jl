using Enzyme, LLVM, Test

# A compiled thunk's session-portable artifact (bitcode plus entry names) is attached to its
# primal-partition CodeInstance on Julia 1.11+. A fresh session, or one that dropped its
# ThunkCache, links the thunk from that artifact instead of running enzyme-core again.

const C = Enzyme.Compiler
const THUNK_CACHE = C.THUNK_CACHE

cir_sq(x) = x * x
cir_cube(x) = x * x * x

@testset "artifact is attached and reused" begin
    C.reset_session!()
    C.EMIT_COUNT[] = 0
    @test autodiff(Reverse, cir_sq, Active(3.0))[1][1] == 6.0
    @test C.EMIT_COUNT[] == 1

    # Same signature, same session: served from the ThunkCache, no emit.
    @test autodiff(Reverse, cir_sq, Active(4.0))[1][1] == 8.0
    @test C.EMIT_COUNT[] == 1

    if C.HAS_CI_RESULTS
        # The exact job of the compiled thunk (its handle records the method instance and
        # config) has a persistable artifact attached to its CodeInstance.
        h = only(
            hh for r in values(THUNK_CACHE.thunks) for hh in (r.adjoint, r.primal)
                if hh isa C.ThunkHandle && hh.mi.specTypes.parameters[1] === typeof(cir_sq)
        )
        job = C.CompilerJob(h.mi, h.config, Base.get_world_counter())
        res = C.enzyme_ci_results(job)
        @test res !== nothing
        @test C.find_artifact(res, C.config_key(job)) !== nothing

        # A session reset drops the ThunkCache but keeps the CodeInstance results, so the
        # thunk relinks from its artifact without a further emit.
        C.reset_session!()
        @test isempty(THUNK_CACHE.thunks)
        before = C.EMIT_COUNT[]
        @test autodiff(Reverse, cir_sq, Active(5.0))[1][1] == 10.0
        @test C.EMIT_COUNT[] == before
    end
end


# A package that differentiates during its own precompilation ships the thunk artifact in
# its image; loading the image in a fresh process links the thunk without an emit.
@testset "artifacts ride into a package image" begin
    C.HAS_CI_RESULTS || return
    load_path = mktempdir()
    depot = mktempdir()
    pkg = "EnzymeCIResultsTest"
    write(
        joinpath(load_path, "$pkg.jl"),
        """
        module $pkg
        using Enzyme
        # Refer to Julia objects by name, so the compiled thunk carries no address of the
        # precompiling process and its artifact can be reused (see `SYMBOLIC_PRIMAL`).
        Enzyme.Compiler.SYMBOLIC_PRIMAL[] = true
        f(x) = sin(x) * x
        grad(x) = Enzyme.autodiff(Reverse, f, Active(x))[1][1]
        const PRECOMPILED = grad(1.5)
        end
        """,
    )
    code = """
    pushfirst!(LOAD_PATH, $(repr(load_path)))
    using Enzyme, $pkg
    Enzyme.Compiler.SYMBOLIC_PRIMAL[] = true
    Enzyme.Compiler.EMIT_COUNT[] = 0
    v = $pkg.grad(2.0)
    print(v, " ", Enzyme.Compiler.EMIT_COUNT[])
    """
    cmd = addenv(
        `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ";" : ":"),
    )
    read(cmd, String)                    # precompile the package
    out = read(cmd, String)              # load the image
    val, emit = split(out)
    expected = cos(2.0) * 2.0 + sin(2.0)
    @test parse(Float64, val) ≈ expected
    # The reloaded thunk linked from its artifact: no enzyme-core run for it.
    @test parse(Int, emit) == 0
end

# A natively called custom rule (a `@noinline` rule, Julia 1.12+) is reached through its
# CodeInstance, referenced as a Julia value and read for its entry point at run time, so a
# thunk calling it is persistable too, and a fresh process starts it without an emit.
@static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
    import Enzyme.EnzymeRules: forward, FwdConfig

    @noinline cir_native(x) = x * x
    @noinline function forward(config::FwdConfig, ::Const{typeof(cir_native)}, ::Type{RT}, x::Duplicated) where {RT <: Annotation}
        dv = 2 * x.val * x.dval
        return RT <: Const ? cir_native(x.val) : (RT <: DuplicatedNoNeed ? dv : Duplicated(cir_native(x.val), dv))
    end
    cir_calls_native(x) = cir_native(x) + x

    @testset "native rules keep the artifact persistable" begin
        C.reset_session!()
        @test autodiff(Forward, cir_calls_native, Duplicated(3.0, 1.0))[1] == 7.0
        if C.HAS_CI_RESULTS
            h = only(
                hh for r in values(THUNK_CACHE.thunks) for hh in (r.adjoint, r.primal)
                    if hh isa C.ThunkHandle && hh.mi.specTypes.parameters[1] === typeof(cir_calls_native)
            )
            job = C.CompilerJob(h.mi, h.config, Base.get_world_counter())
            art = C.find_artifact(C.enzyme_ci_results(job), C.config_key(job))
            @test art !== nothing
            @test art.persistable
            @test any(t -> last(t) isa Core.CodeInstance, art.manifest)
            # No address of this session in the bitcode: the rule is called through its CI.
            LLVM.Context() do ctx
                mod = parse(LLVM.Module, LLVM.MemoryBuffer(art.bitcode))
                ir = string(mod)
                @test !occursin(r"call[^\n]*inttoptr \(i64 \d{9,}", ir)
            end
            C.reset_session!()
            before = C.EMIT_COUNT[]
            @test autodiff(Forward, cir_calls_native, Duplicated(4.0, 1.0))[1] == 9.0
            @test C.EMIT_COUNT[] == before
        end
        # Differentiating through the rule (nested) still sees its code.
        cir_grad(x) = autodiff(Forward, cir_calls_native, Duplicated(x, 1.0))[1]
        @test autodiff(Forward, cir_grad, Duplicated(3.0, 1.0))[1] == 2.0
    end

    @testset "native rules ride into a package image" begin
        C.HAS_CI_RESULTS || return
        load_path = mktempdir()
        depot = mktempdir()
        pkg = "EnzymeCINativeRuleTest"
        write(
            joinpath(load_path, "$pkg.jl"),
            """
            module $pkg
            using Enzyme
            import Enzyme.EnzymeRules: forward, FwdConfig
            @noinline inner(x) = x * x
            @noinline function forward(config::FwdConfig, ::Const{typeof(inner)}, ::Type{RT}, x::Duplicated) where {RT <: Annotation}
                dv = 2 * x.val * x.dval
                return RT <: Const ? inner(x.val) : (RT <: DuplicatedNoNeed ? dv : Duplicated(inner(x.val), dv))
            end
            outer(x) = inner(x) + x
            grad(x) = Enzyme.autodiff(Forward, outer, Duplicated(x, 1.0))[1]
            const PRECOMPILED = grad(3.0)
            end
            """,
        )
        code = """
        pushfirst!(LOAD_PATH, $(repr(load_path)))
        using Enzyme, $pkg
        Enzyme.Compiler.EMIT_COUNT[] = 0
        print($pkg.grad(4.0), " ", Enzyme.Compiler.EMIT_COUNT[])
        """
        cmd = addenv(
            `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
            "JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ";" : ":"),
        )
        read(cmd, String)
        val, emit = split(read(cmd, String))
        @test parse(Float64, val) == 9.0
        @test parse(Int, emit) == 0
    end
end
