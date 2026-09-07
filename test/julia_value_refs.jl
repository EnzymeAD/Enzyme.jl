using Enzyme, LLVM, Test

# Julia objects that Enzyme's generated code refers to (rule configurations, types, error
# payloads, folded global constants) are named globals `ejl_v_*` bound to the objects'
# addresses only when the module is linked, so the module itself carries no address of this
# session. `Enzyme.Compiler.manifest` lists them.

const C = Enzyme.Compiler
const THUNK_CACHE = C.THUNK_CACHE

# The IR of every thunk compiled by differentiating `f` (the runtime rules compile thunks
# for dynamic callees as the derivative runs, so there can be several), as kept for nested
# differentiation. Only a first differentiation compiles anything.
function thunk_irs(f, args...)
    empty!(THUNK_CACHE.by_ptr)
    autodiff(Reverse, f, args...)
    return unique(map(last, values(THUNK_CACHE.by_ptr)))
end

value_globals(ir) = unique([m.captures[1] for m in eachmatch(r"@\"?(ejl_v_[0-9a-f_]+)", ir)])

# A dynamic call goes through the runtime rules, whose emitted code refers to the
# activity configuration, types and the called functions.
vr_dynamic(x) = (
    v = Any[x[1]]; s = 0.0; for e in v
        s += e * 2
    end; s
)

const VR_KEEP = Any[[1.0, 2.0]]
vr_global(x) = VR_KEEP[1][1] * x[1] + x[2]

x = [1.0, 2.0]

@testset "references are named and resolvable" begin
    irs = thunk_irs(vr_dynamic, Active, Duplicated(x, zero(x)))
    @test !isempty(irs)
    @test !any(Base.Fix1(occursin, "ejl_inserted"), irs)
    # The module with the most references is the one differentiating the dynamic call.
    ir = argmax(ir -> length(value_globals(ir)), irs)
    names = value_globals(ir)
    @test !isempty(names)
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, ir)
        man = C.manifest(mod)
        @test Set(first.(man)) == Set(names)
        @test C.persistable(man)
        for (name, target) in man
            gv = LLVM.globals(mod)[name]
            val = C.unbind(target)
            legal, v = C.absint(gv)
            @test legal && v === val
            legal, T, _ = C.abs_typeof(gv)
            @test legal && T == Core.Typeof(val)
            @test C.relocation_pointer(name) == C.unsafe_to_ptr(val)
        end
    end
end

@testset "equal values share a name, distinct objects do not" begin
    n1 = C.relocation_name(Val(3))
    @test C.relocation_name(Val(3)) == n1
    @test C.relocation_name((1, 2.0)) == C.relocation_name((1, 2.0))
    @test C.relocation_name(Val(3)) != C.relocation_name(Val(4))
    a = [1.0]
    b = [1.0]
    @test C.relocation_name(a) != C.relocation_name(b)
    @test C.relocation_name(a) == C.relocation_name(a)
    @test C.relocation_value(n1) == (true, Val(3))
    @test C.relocation_value("ejl_v_nonexistent") == (false, nothing)
    @test !C.persistable(["ejl_v_x" => a])
    @test C.persistable(["ejl_v_x" => Val(3), "ejl_v_y" => :sym, "ejl_v_z" => Int])
end

@testset "emitting a reference" begin
    LLVM.Context() do ctx
        mod = LLVM.Module("refs")
        ft = LLVM.FunctionType(LLVM.VoidType())
        fn = LLVM.Function(mod, "f", ft)
        bb = LLVM.BasicBlock(fn, "entry")
        LLVM.@dispose builder = LLVM.IRBuilder() begin
            LLVM.position!(builder, bb)
            g1 = C.unsafe_to_llvm(builder, Val(7))
            g2 = C.unsafe_to_llvm(builder, Val(7))
            g3 = C.unsafe_to_llvm(builder, nothing)
            LLVM.ret!(builder)
            @test g1 === g2
            @test g1 isa LLVM.GlobalVariable
            @test startswith(LLVM.name(g1), C.RELOC_PREFIX)
            @test LLVM.name(g3) == "ejl_jl_nothing"
            @test C.absint(g1) == (true, Val(7))
        end
        @test !occursin("inttoptr", string(mod))
    end
end

@testset "derivatives are unchanged" begin
    irs = thunk_irs(vr_global, Active, Duplicated(x, zero(x)))
    @test !any(Base.Fix1(occursin, "ejl_inserted"), irs)
    dx = zero(x)
    autodiff(Reverse, vr_global, Active, Duplicated(x, dx))
    @test dx == [1.0, 1.0]
    dx = zero(x)
    autodiff(Reverse, vr_dynamic, Active, Duplicated(x, dx))
    @test dx == [2.0, 0.0]
end
