using Enzyme, LLVM, Test
using FileCheck
import Libdl


@testset "Partial return preservation" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
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
        """)

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        
        callfn = LLVM.functions(mod)["inner"]
        @test length(collect(filter(Base.Fix2(isa, LLVM.StoreInst), collect(instructions(first(blocks(callfn))))))) == 2
    end
end


@testset "Dead return removal" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
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
        """)

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        callfn = LLVM.functions(mod)["caller"]
        @test length(collect(instructions(first(blocks(callfn))))) == 1
    end
end

@testset "Return roots preservation" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
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
        """)

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), true)
        
        caller = LLVM.functions(mod)["caller"]
        
        insts = collect(instructions(first(blocks(caller))))
        calls = collect(filter(i -> isa(i, LLVM.CallInst), insts))
        
        @test length(calls) == 1
        if length(calls) == 1
            call = calls[1]
            @test length(operands(call)) == 3 # 2 arg (sret + roots) + called function
        end
    end
end

@testset "Recursively dead function removal" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
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
        """)

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        
        @test !haskey(LLVM.functions(mod), "dead_recursive_fn")
        @test !haskey(LLVM.functions(mod), "dead_callee")
    end
end

@testset "Mismatched calling convention/function type DAE safety" begin
    LLVM.Context() do ctx
        mod = parse(LLVM.Module, """
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
        """)

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        
        @test haskey(LLVM.functions(mod), "callee")
        callee = LLVM.functions(mod)["callee"]
        @test length(LLVM.parameters(callee)) == 2
    end
end

@testset "link_split_existing!" begin
    LLVM.Context() do ctx
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
        # `dst`'s original definition is preserved.
        @test haskey(fns, "julia___dup")
        @test !LLVM.isdeclaration(fns["julia___dup"])
        # `src`'s colliding definition is preserved under a unique suffixed name.
        @test haskey(fns, "julia___dup_split")
        @test !LLVM.isdeclaration(fns["julia___dup_split"])
        # Non-colliding functions link in unchanged.
        @test haskey(fns, "only_in_dst")
        @test haskey(fns, "uses_dup")
        # `src`'s caller now targets its own (renamed) copy, not `dst`'s definition.
        usesfn = fns["uses_dup"]
        callinst = first(
            filter(
                Base.Fix2(isa, LLVM.CallInst),
                collect(instructions(first(blocks(usesfn)))),
            ),
        )
        @test LLVM.name(last(collect(operands(callinst)))) == "julia___dup_split"
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
