
function noop_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    return UInt8(false)
end

function alloc_obj_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    if API.HasFromStack(inst)
        return UInt8(false)
    end
    legal, typ = abs_typeof(inst)
    if !legal
        return UInt8(false)
        throw(AssertionError("Cannot deduce type of alloc obj, $(string(inst)) of $(string(LLVM.parent(LLVM.parent(inst))))"))
    end

    ctx = LLVM.context(LLVM.Value(val))
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))

    rest = typetree(typ, ctx, dl)
    only!(rest, -1)
    API.EnzymeMergeTypeTree(ret, rest)
    return UInt8(false)
end

function int_return_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Integer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end

function i64_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    val = LLVM.Instruction(val)
    TT = TypeTree(API.DT_Pointer, LLVM.context(val))
    if (direction & API.DOWN) != 0
        sub = TypeTree(unsafe_load(args))
        ctx = LLVM.context(val)
        dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(val)))))
        maxSize = div(width(value_type(operands(val)[1]))+7, 8)
        shift!(sub, dl, 0, maxSize, 0)
        API.EnzymeMergeTypeTree(TT, sub)
    end
    only!(TT, -1)
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end


function f32_box_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Float, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(unsafe_load(args), TT)

    API.EnzymeMergeTypeTree(TT, TypeTree(API.DT_Pointer,LLVM.context(LLVM.Value(val))))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end

function ptr_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    TT = TypeTree(API.DT_Pointer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeSetTypeTree(ret, TT)
    return UInt8(false)
end

function inout_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    if numArgs != 1
        return UInt8(false)
    end
    inst = LLVM.Instruction(val)

    legal, typ = abs_typeof(inst)

    if legal
        if (direction & API.DOWN) != 0
            ctx = LLVM.context(inst)
            dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))
            if GPUCompiler.deserves_retbox(typ)
                typ = Ptr{typ}
            end
            rest = typetree(typ, ctx, dl)
            changed, legal = API.EnzymeCheckedMergeTypeTree(ret, rest)
            @assert legal
        end
        return UInt8(false)
    end

    if (direction & API.UP) != 0
        changed, legal = API.EnzymeCheckedMergeTypeTree(unsafe_load(args), ret)
        @assert legal
    end
    if (direction & API.DOWN) != 0
        changed, legal = API.EnzymeCheckedMergeTypeTree(ret, unsafe_load(args))
        @assert legal
    end
    return UInt8(false)
end

function alloc_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)

    legal, typ = abs_typeof(inst)
    @assert legal

    ctx = LLVM.context(LLVM.Value(val))
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))

    rest = typetree(typ, ctx, dl)
    only!(rest, -1)
    API.EnzymeMergeTypeTree(ret, rest)

    for i = 1:numArgs
        API.EnzymeMergeTypeTree(unsafe_load(args, i), TypeTree(API.DT_Integer, -1, ctx))
    end
    return UInt8(false)
end
