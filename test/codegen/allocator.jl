# RUN: julia --startup-file=no %s | FileCheck %s

using Enzyme
using LLVM

function create_ir!(ctx, mod, name, Type, Count, AlignedSize)

    T_int64 = LLVM.Int64Type(ctx)
    T_void = LLVM.VoidType(ctx)

    if Count === nothing
        FT = LLVM.FunctionType(T_void, [T_int64])
    else
        FT = LLVM.FunctionType(T_void, LLVM.LLVMType[])
    end

    fun = LLVM.Function(mod, name, FT)

    Builder(ctx) do builder
        entry = BasicBlock(fun, "entry"; ctx)
        position!(builder, entry)

        if Count === nothing
            Count = parameters(fun)[1]
        end

        obj = Enzyme.Compiler.julia_allocator(builder, Type, Count, AlignedSize)
        Enzyme.Compiler.julia_deallocator(builder, LLVM.Value(obj))
        ret!(builder)
    end

    return mod
end

LLVM.Context() do ctx
    LLVM.Module("test"; ctx) do mod

        create_ir!(ctx, mod, "simple_dynamic", LLVM.DoubleType(ctx), nothing, LLVM.ConstantInt(8; ctx))
# CHECK-LABEL: define void @simple_dynamic(i64 %0) {
# CHECK-NEXT: entry:
# CHECK-NEXT:   %1 = mul i64 %0, 8
# CHECK-NEXT:   %2 = call noalias nonnull i8* @malloc(i64 %1)
# CHECK-NEXT:   %3 = bitcast i8* %2 to double*
# CHECK-NEXT:   %4 = bitcast double* %3 to i8*
# CHECK-NEXT:   call void @free(i8* nonnull %4)
# CHECK-NEXT:   ret void
# CHECK-NEXT: }

        create_ir!(ctx, mod, "simple_static", LLVM.DoubleType(ctx), LLVM.ConstantInt(1; ctx), LLVM.ConstantInt(8; ctx))
# CHECK-LABEL: define void @simple_static() {
# CHECK-NEXT: entry:
# CHECK-NEXT:   %0 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) i8* @malloc(i64 8)
# CHECK-NEXT:   %1 = bitcast i8* %0 to double*
# CHECK-NEXT:   %2 = bitcast double* %1 to i8*
# CHECK-NEXT:   call void @free(i8* nonnull %2)
# CHECK-NEXT:   ret void
# CHECK-NEXT: }

        T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Enzyme.Compiler.Tracked)

        create_ir!(ctx, mod, "jltype", T_prjlvalue, LLVM.ConstantInt(1; ctx), LLVM.ConstantInt(8; ctx))
# CHECK-LABEL: define void @jltype() {
# CHECK-NEXT: entry:
# CHECK-NEXT:   %0 = call {}*** @julia.get_pgcstack()
# CHECK-NEXT:   %1 = bitcast {}*** %0 to {}**
# CHECK-NEXT:   %2 = getelementptr inbounds {}*, {}** %1, i64 -12
# CHECK-NEXT:   %3 = call noalias nonnull dereferenceable(8) dereferenceable_or_null(8) {} addrspace(10)* @julia.gc_alloc_obj({}** %2, i64 8, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 {{[0-9]+}} to {}*) to {} addrspace(10)*))
# CHECK-NEXT:   %4 = bitcast {} addrspace(10)* %3 to {} addrspace(10)* addrspace(10)*
# CHECK-NEXT:   ret void
# CHECK-NEXT: }

        create_ir!(ctx, mod, "jltype_dynamic", T_prjlvalue, nothing, LLVM.ConstantInt(8; ctx))
# CHECK-LABEL: define void @jltype_dynamic(i64 %0) {
# CHECK-NEXT: entry:
# CHECK-NEXT:   %1 = call {}*** @julia.get_pgcstack()
# CHECK-NEXT:   %2 = mul i64 %0, 8
# CHECK-NEXT:   %3 = call {} addrspace(10)* @ijl_box_int64(i64 %0)
# CHECK-NEXT:   %4 = call cc37 {} addrspace(10)* bitcast ({} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32)* @jl_f_apply_type to {} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*, {} addrspace(10)*)*)({} addrspace(10)* null, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 {{[0-9]+}} to {}*) to {} addrspace(10)*), {} addrspace(10)* %3, {} addrspace(10)* addrspacecast ({}* inttoptr (i64 {{[0-9]+}} to {}*) to {} addrspace(10)*))
# CHECK-NEXT:   %5 = bitcast {}*** %1 to {}**
# CHECK-NEXT:   %6 = getelementptr inbounds {}*, {}** %5, i64 -12
# CHECK-NEXT:   %7 = call noalias nonnull {} addrspace(10)* @julia.gc_alloc_obj({}** %6, i64 8, {} addrspace(10)* %4)
# CHECK-NEXT:   %8 = bitcast {} addrspace(10)* %7 to {} addrspace(10)* addrspace(10)*
# CHECK-NEXT:   ret void
# CHECK-NEXT: }

        write(stdout, string(mod))
    end
end


