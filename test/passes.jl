using Enzyme, LLVM, Test, FileCheck


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
