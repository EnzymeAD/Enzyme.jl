using Enzyme, LLVM, Test


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
          call fastcc void (i32 addrspace(10)*, i32 addrspace(10)*)* bitcast (void (i32*, i32*)* @callee to void (i32 addrspace(10)*, i32 addrspace(10)*)*)(i32 addrspace(10)* %arg1, i32 addrspace(10)* %arg2)
          ret void
        }
        """)

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm(), false)
        
        @test haskey(LLVM.functions(mod), "callee")
        callee = LLVM.functions(mod)["callee"]
        @test length(LLVM.parameters(callee)) == 2
    end
end
