
function noop_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    return UInt8(false)
end

function alloc_obj_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    if API.HasFromStack(inst)
        return UInt8(false)
    end
    legal, typ = abs_typeof(inst)
    @assert legal

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
            rest = typetree(typ, ctx, dl)
            if GPUCompiler.deserves_retbox(typ)
                merge!(rest, TypeTree(API.DT_Pointer, ctx))
                only!(rest, -1)
            end
            API.EnzymeMergeTypeTree(ret, rest)
        end
        return UInt8(false)
    end

    if (direction & API.UP) != 0
        API.EnzymeMergeTypeTree(unsafe_load(args), ret)
    end
    if (direction & API.DOWN) != 0
        API.EnzymeMergeTypeTree(ret, unsafe_load(args))
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

function julia_type_rule(direction::Cint, ret::API.CTypeTreeRef, args::Ptr{API.CTypeTreeRef}, known_values::Ptr{API.IntList}, numArgs::Csize_t, val::LLVM.API.LLVMValueRef)::UInt8
    inst = LLVM.Instruction(val)
    ctx = LLVM.context(inst)

    mi, RT = enzyme_custom_extract_mi(inst)

    ops = collect(operands(inst))[1:end-1]
    called = LLVM.called_operand(inst)


    llRT, sret, returnRoots =  get_return_info(RT)
    retRemoved, parmsRemoved = removed_ret_parms(inst)
    
    dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(inst)))))


    expectLen = (sret !== nothing) + (returnRoots !== nothing)
    for source_typ in mi.specTypes.parameters
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end
        expectLen+=1
    end
    expectLen -= length(parmsRemoved)
    
    # TODO fix the attributor inlining such that this can assert always true
    if expectLen == length(ops)

    f = LLVM.called_operand(inst)
    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(f, i)))) for i in 1:length(collect(parameters(f))))
    jlargs = classify_arguments(mi.specTypes, called_type(inst), sret !== nothing, returnRoots !== nothing, swiftself, parmsRemoved)


    for arg in jlargs
        if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
            continue
        end

        typ, byref = enzyme_extract_parm_type(f, arg.codegen.i)
        @assert typ == arg.typ
        
        op_idx = arg.codegen.i
        rest = typetree(arg.typ, ctx, dl)
        @assert (args.cc == GPUCompiler.BITS_REF) == byref
        if byref
            # adjust first path to size of type since if arg.typ is {[-1]:Int}, that doesn't mean the broader
            # object passing this in by ref isnt a {[-1]:Pointer, [-1,-1]:Int}
            # aka the next field after this in the bigger object isn't guaranteed to also be the same.
            if allocatedinline(arg.typ)
                shift!(rest, dl, 0, sizeof(arg.typ), 0)
            end
            merge!(rest, TypeTree(API.DT_Pointer, ctx))
            only!(rest, -1)
        else
            # canonicalize wrt size
        end
        PTT = unsafe_load(args, op_idx)
        changed, legal = API.EnzymeCheckedMergeTypeTree(PTT, rest)
        if !legal
            function c(io)
                println(io, "Illegal type analysis update from julia rule of method ", mi)
                println(io, "Found type ", arg.typ, " at index ", arg.codegen.i, " of ", string(rest))
                t = API.EnzymeTypeTreeToString(PTT)
                println(io, "Prior type ", Base.unsafe_string(t))
                println(io, inst)
                API.EnzymeStringFree(t)
            end
            msg = sprint(c)

            bt = GPUCompiler.backtrace(inst)
            ir = sprint(io->show(io, parent_scope(inst)))

            sval = ""
            # data = API.EnzymeTypeAnalyzerRef(data)
            # ip = API.EnzymeTypeAnalyzerToString(data)
            # sval = Base.unsafe_string(ip)
            # API.EnzymeStringFree(ip)
            throw(IllegalTypeAnalysisException(msg, sval, ir, bt))
        end
    end

    if sret !== nothing
        idx = 0
        if !in(0, parmsRemoved)
            API.EnzymeMergeTypeTree(unsafe_load(args, idx+1), typetree(sret, ctx, dl))
            idx+=1
        end
        if returnRoots !== nothing
            if !in(1, parmsRemoved)
                allpointer = TypeTree(API.DT_Pointer, -1, ctx)
                API.EnzymeMergeTypeTree(unsafe_load(args, idx+1), typetree(returnRoots, ctx, dl))
            end
        end
    end
    
    end

    if llRT !== nothing && value_type(inst) != LLVM.VoidType()
        @assert !retRemoved
        API.EnzymeMergeTypeTree(ret, typetree(llRT, ctx, dl))
    end

    return UInt8(false)
end
