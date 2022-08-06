# RUN: julia --project --startup-file=no %s | FileCheck %s 

using LLVM
using Enzyme

function create_ir(ctx, mod, name, Type, Count, AlignedSize)

    T_int64 = LLVM.Int64Type(ctx)
    T_void = LLVM.VoidType(ctx)

    if Count === nothing
        FT = LLVM.FunctionType(T_void, [T_int64])
    else
        FT = LLVM.FunctionType(T_void, LLVM.LLVMType[])
    end

    fun = LLVM.Function(mod, "test", FT)

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
    mod = LLVM.Module("test", ctx)
    create_ir!(ctx, mod, "simple_dynamic", LLVM.DoubleType(ctx), nothing, LLVM.ConstantInt(8; ctx))

    create_ir(ctx, mod, "simple_static" LLVM.DoubleType(ctx), LLVM.ConstantInt(1; ctx), LLVM.ConstantInt(8; ctx))
    write(stdout, string(mod))
end


