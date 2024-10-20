
function int_return_rule(
    direction::Cint,
    ret::API.CTypeTreeRef,
    args::Ptr{API.CTypeTreeRef},
    known_values::Ptr{API.IntList},
    numArgs::Csize_t,
    val::LLVM.API.LLVMValueRef,
)::UInt8
    TT = TypeTree(API.DT_Integer, LLVM.context(LLVM.Value(val)))
    only!(TT, -1)
    API.EnzymeMergeTypeTree(ret, TT)
    return UInt8(false)
end

function inout_rule(
    direction::Cint,
    ret::API.CTypeTreeRef,
    args::Ptr{API.CTypeTreeRef},
    known_values::Ptr{API.IntList},
    numArgs::Csize_t,
    val::LLVM.API.LLVMValueRef,
)::UInt8
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

function inoutcopyslice_rule(
    direction::Cint,
    ret::API.CTypeTreeRef,
    args::Ptr{API.CTypeTreeRef},
    known_values::Ptr{API.IntList},
    numArgs::Csize_t,
    val::LLVM.API.LLVMValueRef,
)::UInt8
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

function inoutgcloaded_rule(
    direction::Cint,
    ret::API.CTypeTreeRef,
    args::Ptr{API.CTypeTreeRef},
    known_values::Ptr{API.IntList},
    numArgs::Csize_t,
    val::LLVM.API.LLVMValueRef,
)::UInt8
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
        changed, legal = API.EnzymeCheckedMergeTypeTree(unsafe_load(args, 2), ret)
        @assert legal
    end
    if (direction & API.DOWN) != 0
        changed, legal = API.EnzymeCheckedMergeTypeTree(ret, unsafe_load(args, 2))
        @assert legal
    end
    return UInt8(false)
end