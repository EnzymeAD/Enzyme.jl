using Enzyme, LLVM, Test
using FileCheck
import Libdl
import GPUCompiler


@testset "Partial return preservation" begin
    @test @filecheck begin
        # Both stores into the freshly allocated struct must be preserved.
        # Match without spelling out the pointer type so this works under both
        # opaque (`ptr addrspace(10)`) and typed (`{} addrspace(10)*`) pointers.
        @check_label "@inner"
        @check "store atomic"
        @check_same "%v1"
        @check "store atomic"
        @check_same "%v2"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}**, i64, {} addrspace(10)*) local_unnamed_addr #5

                define internal fastcc nonnull {} addrspace(10)* @inner({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %newstruct = call noalias nonnull dereferenceable(16) {} addrspace(10)* @julia.gc_alloc_obj({}** null, i64 16, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 129778359735376 to {}*) to {} addrspace(10)*)) #30
                  %a31 = addrspacecast {} addrspace(10)* %newstruct to {} addrspace(10)* addrspace(11)*
                  %a32 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %a31, i64 1
                  store atomic {} addrspace(10)* %v1, {} addrspace(10)* addrspace(11)* %a31 release, align 8
                  %a33 = addrspacecast {} addrspace(10)* %newstruct to i8 addrspace(11)*
                  %a34 = getelementptr inbounds i8, i8 addrspace(11)* %a33, i64 8
                  %a35 = bitcast i8 addrspace(11)* %a34 to {} addrspace(10)* addrspace(11)*
                  store atomic {} addrspace(10)* %v2, {} addrspace(10)* addrspace(11)* %a35 release, align 8
                  ret {} addrspace(10)* %newstruct
                }

                define {} addrspace(10)* @caller({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %ac = call fastcc nonnull {} addrspace(10)* @inner({} addrspace(10)* %v1, {} addrspace(10)* %v2)
                  %b = addrspacecast {} addrspace(10)* %ac to {} addrspace(10)* addrspace(11)*
                  %c = load atomic {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %b unordered, align 8
                  ret {} addrspace(10)* %c
                }

                attributes #5 = { inaccessiblememonly mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) "enzyme_no_escaping_allocation" "enzymejl_world"="31504" }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end


@testset "Dead return removal" begin
    @test @filecheck begin
        # The call to the (now dead) callee is removed, leaving only `ret void`,
        # and the callee itself is deleted from the module.
        @check_label "define void @caller"
        @check_next "top:"
        @check_next "ret void"
        @check_not "@julia_MyPrognosticVars_161"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                declare noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}**, i64, {} addrspace(10)*) local_unnamed_addr #5

                define internal fastcc nonnull {} addrspace(10)* @julia_MyPrognosticVars_161({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %newstruct = call noalias nonnull dereferenceable(16) {} addrspace(10)* @julia.gc_alloc_obj({}** null, i64 16, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 129778359735376 to {}*) to {} addrspace(10)*)) #30
                  %a31 = addrspacecast {} addrspace(10)* %newstruct to {} addrspace(10)* addrspace(11)*
                  %a32 = getelementptr inbounds {} addrspace(10)*, {} addrspace(10)* addrspace(11)* %a31, i64 1
                  store atomic {} addrspace(10)* %v1, {} addrspace(10)* addrspace(11)* %a31 release, align 8
                  %a33 = addrspacecast {} addrspace(10)* %newstruct to i8 addrspace(11)*
                  %a34 = getelementptr inbounds i8, i8 addrspace(11)* %a33, i64 8
                  %a35 = bitcast i8 addrspace(11)* %a34 to {} addrspace(10)* addrspace(11)*
                  store atomic {} addrspace(10)* %v2, {} addrspace(10)* addrspace(11)* %a35 release, align 8
                  ret {} addrspace(10)* %newstruct
                }

                define void @caller({} addrspace(10)* %v1, {} addrspace(10)* %v2) {
                top:
                  %ac = call fastcc nonnull {} addrspace(10)* @julia_MyPrognosticVars_161({} addrspace(10)* %v1, {} addrspace(10)* %v2)
                  ret void
                }

                attributes #5 = { inaccessiblememonly mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(1) "enzyme_no_escaping_allocation" "enzymejl_world"="31504" }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end

@testset "Return roots preservation" begin
    @test @filecheck begin
        # The dead `%data` argument is dropped from the call, but the `sret` and
        # `enzymejl_returnRoots` arguments must be preserved (roots last).
        @check_label "define void @caller"
        @check "call void @julia_dims_4189"
        @check_same "sret"
        @check_same "enzymejl_returnRoots"
        @check_same "%roots)"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                define private void @julia_dims_4189({ double, {} addrspace(10)*, {} addrspace(10)* }* sret({ double, {} addrspace(10)*, {} addrspace(10)* }) %res, [2 x {} addrspace(10)*]* "enzymejl_returnRoots"="2", double addrspace(11)* %data) #0 {
                top:
                  %val = load double, double addrspace(11)* %data, align 8
                  store { double, {} addrspace(10)*, {} addrspace(10)* } zeroinitializer, { double, {} addrspace(10)*, {} addrspace(10)* }* %res
                  ret void
                }

                define void @caller({} addrspace(10)* %v1, {} addrspace(10)* %v2, double addrspace(11)* %data) {
                top:
                  %sret = alloca { double, {} addrspace(10)*, {} addrspace(10)* }
                  %roots = alloca [2 x {} addrspace(10)*]
                  call void @julia_dims_4189({ double, {} addrspace(10)*, {} addrspace(10)* }* sret({ double, {} addrspace(10)*, {} addrspace(10)* }) %sret, [2 x {} addrspace(10)*]* "enzymejl_returnRoots"="2" %roots, double addrspace(11)* %data)
                  ret void
                }

                attributes #0 = { nofree nosync nounwind willreturn noinline "enzyme_inactive" }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), true)
            string(mod)
        end
    end
end

@testset "Recursively dead function removal" begin
    @test @filecheck begin
        # Both the recursive function and its callee are dead and must be removed.
        @check_not "@dead_recursive_fn"
        @check_not "@dead_callee"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define internal fastcc void @dead_callee(i32* nocapture %arg) {
                top:
                  %val = load i32, i32* %arg, align 4
                  ret void
                }

                define internal fastcc void @dead_recursive_fn(i32* %arg) {
                top:
                  call fastcc void @dead_recursive_fn(i32* %arg)
                  call fastcc void @dead_callee(i32* %arg)
                  ret void
                }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end

@testset "Mismatched calling convention/function type DAE safety" begin
    @test @filecheck begin
        # The call site bitcasts the callee to a mismatched function type, so DAE
        # must leave the callee untouched: both of its arguments are preserved.
        @check "define internal fastcc void @callee("
        @check_same "%arg1"
        @check_same "%arg2"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define internal fastcc void @callee(i32* %arg1, i32* %arg2) {
                top:
                  store i32 42, i32* %arg1, align 4
                  ret void
                }

                define void @caller(i32 addrspace(10)* %arg1, i32 addrspace(10)* %arg2) {
                top:
                  call fastcc void (i32 addrspace(10)*, i32 addrspace(10)*) bitcast (void (i32*, i32*)* @callee to void (i32 addrspace(10)*, i32 addrspace(10)*)*)(i32 addrspace(10)* %arg1, i32 addrspace(10)* %arg2)
                  ret void
                }
                """
            )

            Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
            string(mod)
        end
    end
end

@testset "link_split_existing!" begin
    LLVM.Context() do ctx
        # Test 1: Differing definitions are preserved as distinct internal specializations
        dst = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            define i64 @only_in_dst(i64 %x) {
              ret i64 %x
            }
            """,
        )
        src = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 2
              ret i64 %r
            }
            define i64 @uses_dup(i64 %x) {
              %r = call i64 @julia___dup(i64 %x)
              ret i64 %r
            }
            """,
        )

        Enzyme.Compiler.link_split_existing!(dst, src)

        fns = LLVM.functions(dst)
        @test haskey(fns, "julia___dup")
        @test !LLVM.isdeclaration(fns["julia___dup"])
        @test LLVM.linkage(fns["julia___dup"]) == LLVM.API.LLVMExternalLinkage
        @test haskey(fns, "only_in_dst")
        @test haskey(fns, "uses_dup")
        usesfn = fns["uses_dup"]
        callinst = first(
            filter(
                Base.Fix2(isa, LLVM.CallInst),
                collect(instructions(first(blocks(usesfn)))),
            ),
        )
        called_fn = last(collect(operands(callinst)))
        @test LLVM.linkage(called_fn) == LLVM.API.LLVMInternalLinkage

        # Test 2: Identical definitions are folded by MergeFunctionsPass
        dst2 = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            """,
        )
        src2 = parse(
            LLVM.Module,
            """
            define i64 @julia___dup(i64 %x) {
              %r = add i64 %x, 1
              ret i64 %r
            }
            define i64 @uses_dup2(i64 %x) {
              %r = call i64 @julia___dup(i64 %x)
              ret i64 %r
            }
            """,
        )
        Enzyme.Compiler.link_split_existing!(dst2, src2)
        fns2 = LLVM.functions(dst2)
        @test haskey(fns2, "julia___dup")
        @test haskey(fns2, "uses_dup2")
        usesfn2 = fns2["uses_dup2"]
        callinst2 = first(
            filter(
                Base.Fix2(isa, LLVM.CallInst),
                collect(instructions(first(blocks(usesfn2)))),
            ),
        )
        @test LLVM.name(last(collect(operands(callinst2)))) == "julia___dup"
    end
end

@testset "Literal-pointer symbol resolution" begin
    # `jl_lookup_code_address` can attribute two distinct pointers to the same
    # (nearest) symbol name on platforms with export-only symbol info. The
    # helpers below back the guards in `check_ir!` that keep such call sites
    # from being merged onto one restored address.
    libmpfr = Libdl.dlpath(Base.MPFR.libmpfr)
    hnd = Libdl.dlopen(libmpfr)
    p_add = Libdl.dlsym(hnd, :mpfr_add)
    p_sub = Libdl.dlsym(hnd, :mpfr_sub)
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", libmpfr, p_add) == :match
    # A nearest-symbol misattribution must be detected, not trusted.
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", libmpfr, p_sub) == :mismatch
    # The containing module is found via the loader, so an unloadable `file` (as
    # reported on Linux/macOS, where it is a source path) does not degrade the answer.
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", "/not/a/library/add.c", p_add) ==
        :match
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", "/not/a/library/add.c", p_sub) ==
        :mismatch
    @test Enzyme.Compiler.resolve_symbol_name("not_a_real_symbol_abcxyz", libmpfr, p_add) ==
        :unknown
    # A pointer outside any loaded module cannot be attributed at all.
    heapptr = Libc.malloc(8)
    @test Enzyme.Compiler.resolve_symbol_name("mpfr_add", "", heapptr) == :unknown
    Libc.free(heapptr)

    LLVM.Context() do ctx
        mod = parse(
            LLVM.Module,
            """
            declare void @foo() #0
            declare void @bar()
            attributes #0 = { "enzymejl_needs_restoration"="12345" }
            """,
        )
        @test Enzyme.Compiler.restoration_ptr(functions(mod)["foo"]) == UInt(12345)
        @test Enzyme.Compiler.restoration_ptr(functions(mod)["bar"]) === nothing
    end
end

# https://github.com/EnzymeAD/Enzyme.jl/issues/3284
# The `jl_get_abi_converter` lowering of `@cfunction` only exists on 1.12+, and
# older Julia's LLVM cannot parse the opaque-pointer IR below.
@static if VERSION >= v"1.12"

    @testset "Rewrite abi converter calls (1.12 pattern)" begin
        LLVM.Context() do ctx
            # Shape of the `@cfunction` dispatch site emitted by Julia 1.12's
            # codegen: `jl_get_abi_converter(ct, fptr, last_world, cfuncdata)` with
            # a six-slot cfuncdata whose third slot holds the in-module
            # `unspecialized` apply-generic thunk.
            mod = parse(
                LLVM.Module,
                """
                @jl_world_counter = external global i64
                @fptr = private global ptr @gfthunk
                @last_world = private global i64 0
                @declrt = private global ptr null
                @sigt = private global ptr null
                @cfuncdata = private global [6 x ptr] [ptr null, ptr null, ptr @gfthunk, ptr @declrt, ptr @sigt, ptr inttoptr (i64 1 to ptr)]

                declare ptr @jl_get_abi_converter(ptr, ptr, ptr, ptr)

                define internal double @gfthunk(double %0) {
                top:
                  ret double %0
                }

                define double @trampoline(ptr %ct, double %x) {
                top:
                  %last_world = load atomic i64, ptr @last_world acquire, align 8
                  %fptr = load atomic ptr, ptr @fptr monotonic, align 8
                  %world = load atomic i64, ptr @jl_world_counter acquire, align 8
                  %stale = icmp ne i64 %last_world, %world
                  br i1 %stale, label %guard_pass, label %guard_exit

                guard_pass:
                  %cw = call ptr @jl_get_abi_converter(ptr %ct, ptr @fptr, ptr @last_world, ptr @cfuncdata)
                  br label %guard_exit

                guard_exit:
                  %target = phi ptr [ %fptr, %top ], [ %cw, %guard_pass ]
                  %res = call double %target(double %x)
                  ret double %res
                }
                """,
            )
            Enzyme.Compiler.rewrite_abi_converter_calls!(mod)
            @test @filecheck begin
                @check_label "define double @trampoline"
                @check_not "call ptr @jl_get_abi_converter"
                @check "phi ptr [ %fptr, %top ], [ @gfthunk, %guard_pass ]"
                string(mod)
            end
        end
    end

    @testset "Rewrite abi converter calls (1.13 pattern)" begin
        LLVM.Context() do ctx
            # Shape of the dispatch site emitted by Julia 1.13+:
            # `jl_get_abi_converter(ct, cfuncdata)` with an eight-slot cfuncdata
            # ([fptr, last_world, plast_codeinst, last_codeinst, unspecialized,
            # declrt, sigt, flags]) whose fifth slot holds the thunk.
            mod = parse(
                LLVM.Module,
                """
                @jl_world_counter = external global i64
                @declrt = private global ptr null
                @sigt = private global ptr null
                @cfuncdata = private global [8 x ptr] [ptr @gfthunk, ptr null, ptr null, ptr null, ptr @gfthunk, ptr @declrt, ptr @sigt, ptr inttoptr (i64 1 to ptr)]

                declare ptr @ijl_get_abi_converter(ptr, ptr)

                define internal double @gfthunk(double %0) {
                top:
                  ret double %0
                }

                define double @trampoline(ptr %ct, double %x) {
                top:
                  %last_world_p = getelementptr inbounds i64, ptr @cfuncdata, i32 1
                  %last_world = load atomic i64, ptr %last_world_p acquire, align 8
                  %fptr = load atomic ptr, ptr @cfuncdata monotonic, align 8
                  %world = load atomic i64, ptr @jl_world_counter acquire, align 8
                  %stale = icmp ne i64 %last_world, %world
                  br i1 %stale, label %guard_pass, label %guard_exit

                guard_pass:
                  %cw = call ptr @ijl_get_abi_converter(ptr %ct, ptr @cfuncdata)
                  br label %guard_exit

                guard_exit:
                  %target = phi ptr [ %fptr, %top ], [ %cw, %guard_pass ]
                  %res = call double %target(double %x)
                  ret double %res
                }
                """,
            )
            Enzyme.Compiler.rewrite_abi_converter_calls!(mod)
            @test @filecheck begin
                @check_label "define double @trampoline"
                @check_not "call ptr @ijl_get_abi_converter"
                @check "phi ptr [ %fptr, %top ], [ @gfthunk, %guard_pass ]"
                string(mod)
            end
        end
    end

end # VERSION >= v"1.12"

# An addrspace(11) phi with argument-derived and poison incomings must be skipped by
# the `all_args` prefilter; the rewriter cannot handle a bare addrspace(11) argument.
@testset "nodecayed_phis! addrspace(11) phi with poison incoming" begin
    @test @filecheck begin
        # incomings: gep(arg, 8) and poison
        @check_label "@kernel_gep"
        @check_not "nodecayed"
        @check "phi"
        @check_same "addrspace(11)"
        @check_same "%gep"
        @check_same "poison"
        # zero-offset variant: the argument itself and poison
        @check_label "@kernel_direct"
        @check_not "nodecayed"
        @check "phi"
        @check_same "addrspace(11)"
        @check_same "%q"
        @check_same "poison"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define i8 @kernel_gep(i1 %cond, i8 addrspace(11)* %g) #0 {
                top:
                  br i1 %cond, label %ok, label %guard

                ok:
                  %gep = getelementptr inbounds i8, i8 addrspace(11)* %g, i64 8
                  br label %merge

                guard:
                  br label %merge

                merge:
                  %p = phi i8 addrspace(11)* [ %gep, %ok ], [ poison, %guard ]
                  %ld = load i8, i8 addrspace(11)* %p, align 1
                  ret i8 %ld
                }

                define i8 @kernel_direct(i1 %cond, i8 addrspace(11)* %q) #0 {
                top:
                  br i1 %cond, label %ok2, label %guard2

                ok2:
                  br label %merge2

                guard2:
                  br label %merge2

                merge2:
                  %p2 = phi i8 addrspace(11)* [ %q, %ok2 ], [ poison, %guard2 ]
                  %ld2 = load i8, i8 addrspace(11)* %p2, align 1
                  ret i8 %ld2
                }

                attributes #0 = { "enzymejl_world"="1" }
                """
            )

            Enzyme.Compiler.nodecayed_phis!(mod)
            string(mod)
        end
    end
end


# --- fix_decayaddr! -----------------------------------------------------------

struct DecayBig
    x::NTuple{100, Float64}
end

# `===` on a padding-free immutable this large lowers to `emit_bits_compare`,
# which decays both operands through `julia.pointer_from_objref` and compares
# them with `memcmp`; LLVM's `LibCallSimplifier::optimizeMemCmp` then rewrites
# that to `bcmp`, since the result feeds nothing but an `icmp eq ..., 0`.
# `@nospecialize` keeps the arguments boxed, so the operands are `addrspace(10)`
# and the decay is the one `fix_decayaddr!` has to deal with.
@noinline function decay_egal(@nospecialize(a), @nospecialize(b))
    return (a::DecayBig) === (b::DecayBig)
end

"""
    decay_egal_module()

The module Julia emits for [`decay_egal`](@ref), run through Enzyme's own
pre-AD optimization pipeline. Everything the test relies on -- the libcall, its
attributes, the `jl_roots` operand bundle -- comes from that emission rather
than from hand-written IR.

The module is generated straight into the active context, the same way
Enzyme's compile pipeline does it, rather than being round-tripped through
`code_llvm` text: Julia's own codegen context need not agree with a fresh
one on typed vs. opaque pointers (it does not on 1.11), and the text form
of the one cannot be parsed in the other.
"""
function decay_egal_module()
    target = Enzyme.Compiler.DefaultCompilerTarget()
    params = Enzyme.Compiler.PrimalCompilerParams(Enzyme.API.DEM_ForwardMode)
    mi = Enzyme.Compiler.my_methodinstance(nothing, typeof(decay_egal), Tuple{Any, Any})
    job = GPUCompiler.CompilerJob(
        mi,
        GPUCompiler.CompilerConfig(
            target, params;
            kernel = false, libraries = true, toplevel = true, optimize = false,
            cleanup = false, only_entry = false, validate = false,
        ),
    )
    GPUCompiler.prepare_job!(job)
    mod, _ = GPUCompiler.emit_llvm(job)
    Enzyme.Compiler.optimize!(mod, Enzyme.Compiler.JIT.get_tm())
    return mod
end

"The `memcmp` / `bcmp` call in `mod`, or `nothing` if there is none."
function find_bits_compare(mod::LLVM.Module)
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        isa(inst, LLVM.CallInst) || continue
        callee = LLVM.called_operand(inst)
        isa(callee, LLVM.Function) || continue
        if LLVM.name(callee) in ("bcmp", "memcmp")
            return inst
        end
    end
    return nothing
end

"""
    collapse_decay!(call)

Rewrite each `julia.pointer_from_objref(addrspacecast p10 -> p11)` feeding
`call` into the direct `addrspacecast p10 -> p0` it stands for, and return how
many were rewritten. Neither Julia nor Enzyme's pre-AD pipeline forms that cast
here -- it is what a later simplification of the two-step derivation leaves
behind, and it is the input `fix_decayaddr!` has to repair.
"""
function collapse_decay!(call::LLVM.CallInst)
    n = 0
    for (i, arg) in enumerate(Enzyme.Compiler.arg_operands_view(call))
        # With typed pointers the `{}*` result is bitcast to `i8*` first; look
        # through that to the derivation underneath.
        pfo = isa(arg, LLVM.BitCastInst) ? operands(arg)[1] : arg
        isa(pfo, LLVM.CallInst) || continue
        callee = LLVM.called_operand(pfo)
        (isa(callee, LLVM.Function) && LLVM.name(callee) == "julia.pointer_from_objref") ||
            continue
        src = operands(pfo)[1]
        isa(src, LLVM.AddrSpaceCastInst) || continue
        obj = operands(src)[1]
        LLVM.addrspace(value_type(obj)) == 10 || continue
        b = LLVM.IRBuilder()
        LLVM.position!(b, call)
        LLVM.API.LLVMSetOperand(
            call, i - 1, LLVM.addrspacecast!(b, obj, value_type(arg))
        )
        if arg != pfo && isempty(LLVM.uses(arg))
            LLVM.erase!(arg)
        end
        isempty(LLVM.uses(pfo)) && LLVM.erase!(pfo)
        n += 1
    end
    return n
end

"Strip every read-only marker from `call` and from the function it calls."
function drop_readonly!(call::LLVM.CallInst)
    callee = LLVM.called_operand(call)::LLVM.Function
    for attrs in (LLVM.function_attributes(callee), LLVM.function_attributes(call))
        for attr in collect(attrs)
            if Enzyme.Compiler.is_readonly(attr)
                delete!(attrs, attr)
            end
        end
    end
    return nothing
end

@testset "fix_decayaddr! read-only libcall" begin
    GPUCompiler.JuliaContext() do ctx
        mod = decay_egal_module()
        cmp = find_bits_compare(mod)
        @test cmp !== nothing
        # The callee has to be read-only for the rewrite below to apply at all;
        # that is how Julia and LLVM annotate `memcmp` / `bcmp`.
        @test Enzyme.Compiler.is_readonly(LLVM.called_operand(cmp)::LLVM.Function)
        @test collapse_decay!(cmp) == 2

        @test @filecheck begin
            # Each decayed argument becomes a gc-preserved
            # `julia.pointer_from_objref`, which late GC lowering turns back
            # into the cast that was there. The operand bundle is untouched --
            # only the argument operands get rewritten.
            @check_label "@julia_decay_egal"
            @check "gc_preserve_begin"
            @check "julia.pointer_from_objref"
            @check "gc_preserve_begin"
            @check "julia.pointer_from_objref"
            @check "@{{(bcmp|memcmp)}}("
            @check_same "jl_roots"
            @check "gc_preserve_end"
            @check "gc_preserve_end"
            Enzyme.Compiler.fix_decayaddr!(mod)
            string(mod)
        end

        # Nothing decays straight out of the tracked address space any more.
        for f in functions(mod), bb in blocks(f), inst in instructions(bb)
            if isa(inst, LLVM.AddrSpaceCastInst)
                @test !(
                    LLVM.addrspace(value_type(operands(inst)[1])) == 10 &&
                        LLVM.addrspace(value_type(inst)) == 0
                )
            end
        end
    end
end

@testset "fix_decayaddr! non-read-only libcall still rejected" begin
    # Guard against the read-only path swallowing calls that write through the
    # decayed pointer: those still need an sret to copy the object back.
    GPUCompiler.JuliaContext() do ctx
        mod = decay_egal_module()
        cmp = find_bits_compare(mod)
        @test cmp !== nothing
        @test collapse_decay!(cmp) == 2
        drop_readonly!(cmp)
        @test !Enzyme.Compiler.is_readonly(LLVM.called_operand(cmp)::LLVM.Function)

        @test_throws AssertionError Enzyme.Compiler.fix_decayaddr!(mod)
    end
end

@testset "Tracked GEPs are re-derived" begin
    # With typed pointers InstCombine turns `gep(addrspacecast(x))` into
    # `addrspacecast(gep(x))`, leaving an interior pointer in addrspace 10.
    # The pass must sink the cast back above the GEP chain (#3532).
    @test @filecheck begin
        @check_label "@load_field"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check "getelementptr inbounds i8"
        @check_same "addrspace(11)"
        @check_not "getelementptr inbounds i8, i8 addrspace(10)*"
        @check_not "getelementptr inbounds i8, ptr addrspace(10)"
        @check "load float"
        @check_same "addrspace(11)"
        @check_label "@store_field"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check "getelementptr inbounds i8"
        @check_same "addrspace(11)"
        @check "getelementptr inbounds i8"
        @check_same "addrspace(11)"
        @check_not "getelementptr inbounds i8, i8 addrspace(10)*"
        @check_not "getelementptr inbounds i8, ptr addrspace(10)"
        @check "store float %x"
        @check_same "addrspace(11)"
        LLVM.Context() do ctx
            mod = parse(
                LLVM.Module, """
                source_filename = "start"
                target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
                target triple = "x86_64-linux-gnu"

                define float @load_field({} addrspace(10)* %obj) {
                top:
                  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
                  %b = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
                  %c = bitcast i8 addrspace(10)* %b to float addrspace(10)*
                  %d = addrspacecast float addrspace(10)* %c to float addrspace(11)*
                  %v = load float, float addrspace(11)* %d, align 8
                  ret float %v
                }

                define void @store_field({} addrspace(10)* %obj, float %x) {
                top:
                  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
                  %b = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
                  %c = getelementptr inbounds i8, i8 addrspace(10)* %b, i64 4
                  %d = bitcast i8 addrspace(10)* %c to float addrspace(10)*
                  %e = addrspacecast float addrspace(10)* %d to float addrspace(11)*
                  store float %x, float addrspace(11)* %e, align 4
                  ret void
                }
                """
            )

            Enzyme.Compiler.rederive_tracked_geps!(mod)
            LLVM.verify(mod)
            string(mod)
        end
    end
end

# InstCombine also hoists the addrspacecast out of phis and selects, leaving a
# phi/select of Tracked interior pointers. The GEP pass has to move the join
# into addrspace 11 so that the decayed-phi pass can then root it (#3532).
const rederive_joins_ir = """
source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

define float @diamond({} addrspace(10)* %obj, i1 %c) {
top:
  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
  br i1 %c, label %l, label %r
l:
  %g1 = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
  br label %m
r:
  br label %m
m:
  %p = phi i8 addrspace(10)* [ %g1, %l ], [ %a, %r ]
  %pc = bitcast i8 addrspace(10)* %p to float addrspace(10)*
  %d = addrspacecast float addrspace(10)* %pc to float addrspace(11)*
  %v = load float, float addrspace(11)* %d, align 4
  ret float %v
}

define float @loop({} addrspace(10)* %obj, i64 %n) {
top:
  %a = bitcast {} addrspace(10)* %obj to float addrspace(10)*
  br label %body
body:
  %i = phi i64 [ 0, %top ], [ %i1, %body ]
  %p = phi float addrspace(10)* [ %a, %top ], [ %next, %body ]
  %acc = phi float [ 0.0, %top ], [ %acc1, %body ]
  %d = addrspacecast float addrspace(10)* %p to float addrspace(11)*
  %v = load float, float addrspace(11)* %d, align 4
  %acc1 = fadd float %acc, %v
  %next = getelementptr inbounds float, float addrspace(10)* %p, i64 1
  %i1 = add i64 %i, 1
  %cmp = icmp eq i64 %i1, %n
  br i1 %cmp, label %exit, label %body
exit:
  ret float %acc1
}

define float @sel({} addrspace(10)* %obj, i1 %c) {
top:
  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
  %g1 = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
  %g2 = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 16
  %p = select i1 %c, i8 addrspace(10)* %g1, i8 addrspace(10)* %g2
  %pc = bitcast i8 addrspace(10)* %p to float addrspace(10)*
  %d = addrspacecast float addrspace(10)* %pc to float addrspace(11)*
  %v = load float, float addrspace(11)* %d, align 4
  ret float %v
}
"""

@testset "Tracked phis and selects are re-derived" begin
    @test @filecheck begin
        @check_label "@diamond"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check "getelementptr inbounds i8"
        @check_same "addrspace(11)"
        @check "phi {{.*}}addrspace(11)"
        # Every replacement is built in front of the value it replaces, so a
        # Tracked leftover shows up *after* its Derived counterpart: the region
        # that has to stay clear is the one between the new phi and the load.
        @check_not "phi {{.*}}addrspace(10)"
        @check "load float"
        @check_same "addrspace(11)"
        @check_label "@loop"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check "phi {{.*}}addrspace(11)"
        @check_not "phi {{.*}}addrspace(10)"
        @check "load float"
        @check_same "addrspace(11)"
        @check "getelementptr inbounds float"
        @check_same "addrspace(11)"
        @check_not "getelementptr inbounds float"
        @check_label "@sel"
        @check "select i1 %c"
        @check_same "addrspace(11)"
        @check_not "select i1 %c, i8 addrspace(10)*"
        @check_not "select i1 %c, ptr addrspace(10)"
        @check "load float"
        @check_same "addrspace(11)"
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, rederive_joins_ir)
            Enzyme.Compiler.rederive_tracked_geps!(mod)
            LLVM.verify(mod)
            string(mod)
        end
    end

    # The Derived phis are then rooted by nodecayed_phis!: a phi of the whole
    # object plus a phi of the byte offset, and no Derived phi is left over.
    @test @filecheck begin
        @check_label "@diamond"
        @check_not "phi {{.*}}addrspace"
        @check "phi i$(8 * sizeof(Int)) [ 8, %l ], [ 0, %r ]"
        @check_not "phi {{.*}}addrspace"
        @check "load float"
        @check_same "addrspace(11)"
        @check_label "@loop"
        @check_not "phi {{.*}}addrspace(11)"
        @check "phi {{.*}}addrspace(10)"
        @check "phi i$(8 * sizeof(Int))"
        @check_not "phi {{.*}}addrspace"
        @check "load float"
        @check_same "addrspace(11)"
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, rederive_joins_ir)
            Enzyme.Compiler.rederive_tracked_geps!(mod)
            Enzyme.Compiler.nodecayed_phis!(mod)
            LLVM.verify(mod)
            string(mod)
        end
    end
end

# The rewrite is best effort, and tidy about what it leaves: one cast per base,
# placed at the base rather than at each use, nothing left behind where a field
# address turns out to have no user it can rewrite, and inactive functions
# untouched -- `nodecayed_phis!` skips those, so a Derived pointer created here
# would never be rooted.
const rederive_placement_ir = """
source_filename = "start"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-linux-gnu"

declare void @use(i8 addrspace(10)*)

define void @only_unhandled({} addrspace(10)* %obj) {
top:
  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
  %g = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
  call void @use(i8 addrspace(10)* %g)
  ret void
}

define float @many_fields({} addrspace(10)* %obj) {
top:
  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
  %g1 = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
  %c1 = bitcast i8 addrspace(10)* %g1 to float addrspace(10)*
  %d1 = addrspacecast float addrspace(10)* %c1 to float addrspace(11)*
  %v1 = load float, float addrspace(11)* %d1, align 4
  %g2 = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 16
  %c2 = bitcast i8 addrspace(10)* %g2 to float addrspace(10)*
  %d2 = addrspacecast float addrspace(10)* %c2 to float addrspace(11)*
  %v2 = load float, float addrspace(11)* %d2, align 4
  %s = fadd float %v1, %v2
  ret float %s
}

define float @in_loop({} addrspace(10)* %obj, i64 %n) {
top:
  %a = bitcast {} addrspace(10)* %obj to float addrspace(10)*
  br label %body
body:
  %i = phi i64 [ 0, %top ], [ %i1, %body ]
  %acc = phi float [ 0.0, %top ], [ %acc1, %body ]
  %g = getelementptr inbounds float, float addrspace(10)* %a, i64 %i
  %d = addrspacecast float addrspace(10)* %g to float addrspace(11)*
  %v = load float, float addrspace(11)* %d, align 4
  %acc1 = fadd float %acc, %v
  %i1 = add i64 %i, 1
  %cmp = icmp eq i64 %i1, %n
  br i1 %cmp, label %exit, label %body
exit:
  ret float %acc1
}

define void @inactive_fn({} addrspace(10)* %obj) #0 {
top:
  %a = bitcast {} addrspace(10)* %obj to i8 addrspace(10)*
  %g = getelementptr inbounds i8, i8 addrspace(10)* %a, i64 8
  %c = bitcast i8 addrspace(10)* %g to float addrspace(10)*
  %d = addrspacecast float addrspace(10)* %c to float addrspace(11)*
  store float 1.0, float addrspace(11)* %d, align 4
  ret void
}

attributes #0 = { "enzyme_inactive" }
"""

@testset "Re-derived pointers are placed and cleaned up" begin
    @test @filecheck begin
        # A field address whose only user is a call keeps its Tracked form, and
        # the Derived replacement built for it is erased again.
        @check_label "@only_unhandled"
        @check_not "addrspace(11)"
        @check "call void @use"
        # One cast serves both field addresses ...
        @check_label "@many_fields"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check_not "addrspacecast"
        @check "ret float"
        # ... and it is placed at the object, not inside the loop.
        @check_label "@in_loop"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check "br label %body"
        @check_not "addrspacecast"
        @check "ret float"
        # An inactive function is left exactly as it was.
        @check_label "@inactive_fn"
        @check "getelementptr inbounds i8"
        @check_same "addrspace(10)"
        @check "addrspacecast"
        @check_same "addrspace(11)"
        @check "store float"
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, rederive_placement_ir)
            Enzyme.Compiler.rederive_tracked_geps!(mod)
            LLVM.verify(mod)
            string(mod)
        end
    end
end
