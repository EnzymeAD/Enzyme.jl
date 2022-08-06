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

        write(stdout, string(mod))
    end
end


