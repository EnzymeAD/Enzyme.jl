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

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm())
        
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

        Enzyme.Compiler.removeDeadArgs!(mod, Enzyme.Compiler.JIT.get_tm())
        callfn = LLVM.functions(mod)["caller"]
        @test length(collect(instructions(first(blocks(callfn))))) == 1
    end
end
