
function enzyme_custom_setup_args(B, orig, gutils, mi, RT, reverse, isKWCall)
    ops = collect(operands(orig))
    called = ops[end]
    ops = ops[1:end-1]
    width = get_width(gutils)
    kwtup = nothing

    args = LLVM.Value[]
    activity = Type[]
    overwritten = Bool[]

    actives = LLVM.Value[]

    uncacheable = get_uncacheable(gutils, orig)
    mode = get_mode(gutils)
    
    retRemoved, parmsRemoved = removed_ret_parms(orig)

    @assert length(parmsRemoved) == 0

    _, sret, returnRoots = get_return_info(RT)
    sret = sret !== nothing
    returnRoots = returnRoots !== nothing

    cv = LLVM.called_operand(orig)
    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(cv, i)))) for i in 1:length(collect(parameters(cv))))
	jlargs = classify_arguments(mi.specTypes, called_type(orig), sret, returnRoots, swiftself, parmsRemoved)

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    world = enzyme_extract_world(LLVM.parent(LLVM.parent(orig)))

    for arg in jlargs
        @assert arg.cc != RemovedParam
        if arg.cc == GPUCompiler.GHOST
            @assert guaranteed_const_nongen(arg.typ, world)
            if isKWCall && arg.arg_i == 2
                Ty = arg.typ
                kwtup = Ty
                continue
            end
            push!(activity, Const{arg.typ})
            # Don't push overwritten for Core.kwcall
            if !(isKWCall && arg.arg_i == 1)
                push!(overwritten, false)
            end
            if Core.Compiler.isconstType(arg.typ) && !Core.Compiler.isconstType(Const{arg.typ})
                llty = convert(LLVMType, Const{arg.typ})
                al0 = al = emit_allocobj!(B, Const{arg.typ})
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

                ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
                val = unsafe_to_llvm(arg.typ.parameters[1])
                store!(B, val, ptr)

                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
                end
                push!(args, al)
            else
                @assert isghostty(Const{arg.typ}) || Core.Compiler.isconstType(Const{arg.typ})
            end
            continue
        end
        @assert !(isghostty(arg.typ) || Core.Compiler.isconstType(arg.typ))

        op = ops[arg.codegen.i]
        # Don't push the keyword args to uncacheable
        if !(isKWCall && arg.arg_i == 2)
            push!(overwritten, uncacheable[arg.codegen.i] != 0)
        end

        val = new_from_original(gutils, op)
        if reverse
            val = lookup_value(gutils, val, B)
        end

        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, op, #=isforeign=#false)

        if isKWCall && arg.arg_i == 2
            Ty = arg.typ

            push!(args, val)

            # Only constant kw arg tuple's are currently supported
            if activep == API.DFT_CONSTANT
                kwtup = Ty
            else
                @assert activep == API.DFT_DUP_ARG
                kwtup = Duplicated{Ty}
            end
            continue
        end

        # TODO type analysis deduce if duplicated vs active
        if activep == API.DFT_CONSTANT
            Ty = Const{arg.typ}
            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed=true)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            if value_type(val) != eltype(value_type(ptr))
                val = load!(B, arty, val)
            end
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            push!(args, al)

            push!(activity, Ty)

        elseif activep == API.DFT_OUT_DIFF || (mode != API.DEM_ForwardMode && active_reg(arg.typ, world) )
            Ty = Active{arg.typ}
            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed=true)
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            if value_type(val) != eltype(value_type(ptr))
                @assert !overwritten[end]
                val = load!(B, arty, val)
            end
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            push!(args, al)

            push!(activity, Ty)
            push!(actives, op)
        else
            ival = invert_pointer(gutils, op, B)
            if reverse
                ival = lookup_value(gutils, ival, B)
            end
            if width == 1
                if activep == API.DFT_DUP_ARG
                    Ty = Duplicated{arg.typ}
                else
                    @assert activep == API.DFT_DUP_NONEED
                    Ty = DuplicatedNoNeed{arg.typ}
                end
            else
                if activep == API.DFT_DUP_ARG
                    Ty = BatchDuplicated{arg.typ, Int(width)}
                else
                    @assert activep == API.DFT_DUP_NONEED
                    Ty = BatchDuplicatedNoNeed{arg.typ, Int(width)}
                end
            end

            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed=true)
            sarty = LLVM.LLVMType(API.EnzymeGetShadowType(width, arty))
            al0 = al = emit_allocobj!(B, Ty)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            if value_type(val) != eltype(value_type(ptr))
                val = load!(B, arty, val)
                ptr_val = ival
                ival = UndefValue(sarty)
                for idx in 1:width
                    ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx-1)
                    ld = load!(B, arty, ev)
                    ival = (width == 1 ) ? ld : insert_value!(B, ival, ld, idx-1)
                end
            end
            store!(B, val, ptr)

            iptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 1)])
            store!(B, ival, iptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val, ival))
            end

            push!(args, al)
            push!(activity, Ty)
        end

    end
    return args, activity, (overwritten...,), actives, kwtup
end

function enzyme_custom_setup_ret(gutils, orig, mi, RealRt)
    width = get_width(gutils)
    mode = get_mode(gutils)
    
    world = enzyme_extract_world(LLVM.parent(LLVM.parent(orig)))

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP)
    needsPrimal = needsPrimalP[] != 0
    origNeedsPrimal = needsPrimal
    _, sret, _ = get_return_info(RealRt)
    if sret !== nothing
        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, operands(orig)[1], #=isforeign=#false)
        needsPrimal = activep == API.DFT_DUP_ARG || activep == API.DFT_CONSTANT
        needsShadowP[] = activep == API.DFT_DUP_ARG || activep == API.DFT_DUP_NONEED
    end

    if !needsPrimal && activep == API.DFT_DUP_ARG
        activep = API.DFT_DUP_NONEED
    end

    if activep == API.DFT_CONSTANT
        RT = Const{RealRt}

    elseif activep == API.DFT_OUT_DIFF || (mode != API.DEM_ForwardMode && active_reg(RealRt, world) )
        RT = Active{RealRt}

    elseif activep == API.DFT_DUP_ARG
        if width == 1
            RT = Duplicated{RealRt}
        else
            RT = BatchDuplicated{RealRt, Int(width)}
        end
    else
        @assert activep == API.DFT_DUP_NONEED
        if width == 1
            RT = DuplicatedNoNeed{RealRt}
        else
            RT = BatchDuplicatedNoNeed{RealRt, Int(width)}
        end
    end
    return RT, needsPrimal, needsShadowP[] != 0, origNeedsPrimal
end

function enzyme_custom_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    if shadowR != C_NULL
        unsafe_store!(shadowR,UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))).ref)
    end

    # TODO: don't inject the code multiple times for multiple calls

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)

    kwfunc = nothing

    isKWCall = isKWCallSignature(mi.specTypes)
    if isKWCall
        kwfunc = Core.kwfunc(EnzymeRules.forward)
    end

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup = enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, #=reverse=#false, isKWCall)
    RT, needsPrimal, needsShadow, origNeedsPrimal = enzyme_custom_setup_ret(gutils, orig, mi, RealRt)

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    tt = copy(activity)
    if isKWCall
        popfirst!(tt)
        @assert kwtup !== nothing
        insert!(tt, 1, kwtup)
        insert!(tt, 2, Core.typeof(EnzymeRules.forward))
        insert!(tt, 4, Type{RT})
    else
        @assert kwtup === nothing
        insert!(tt, 2, Type{RT})
    end
    TT = Tuple{tt...}

    if kwtup !== nothing && kwtup <: Duplicated
        @safe_debug "Non-constant keyword argument found for " TT
        emit_error(B, orig, "Enzyme: Non-constant keyword argument found for " * string(TT))
        return false
    end

    # TODO get world
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)
    @safe_debug "Trying to apply custom forward rule" TT isKWCall
    llvmf = nothing
    if isKWCall
        if EnzymeRules.isapplicable(kwfunc, TT; world)
            @safe_debug "Applying custom forward rule (kwcall)" TT
            llvmf = nested_codegen!(mode, mod, kwfunc, TT, world)
            fwd_RT = Core.Compiler.return_type(kwfunc, TT, world)
        end
    else
        if EnzymeRules.isapplicable(EnzymeRules.forward, TT; world)
            @safe_debug "Applying custom forward rule" TT
            llvmf = nested_codegen!(mode, mod, EnzymeRules.forward, TT, world)
            fwd_RT = Core.Compiler.return_type(EnzymeRules.forward, TT, world)
        end
    end

    if llvmf === nothing
        @safe_debug "No custom forward rule is applicable for" TT
        emit_error(B, orig, "Enzyme: No custom rule was appliable for " * string(TT))
        return false
    end
    
    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))

    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(llvmf, i)))) for i in 1:length(collect(parameters(llvmf))))
    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end
    _, sret, returnRoots = get_return_info(enzyme_custom_extract_mi(llvmf)[2])
    if sret !== nothing
        sret = alloca!(alloctx, convert(LLVMType, eltype(sret)))
        pushfirst!(args, sret)
        if returnRoots !== nothing
            returnRoots = alloca!(alloctx, convert(LLVMType, eltype(returnRoots)))
            insert!(args, 2, returnRoots)
        else
            returnRoots = nothing
        end
    else
        sret = nothing
    end


    if length(args) != length(parameters(llvmf))
        GPUCompiler.@safe_error "Calling convention mismatch", args, llvmf, orig, isKWCall, kwtup, TT, sret, returnRoots
        return false
    end

    for i in eachindex(args)
        party = value_type(parameters(llvmf)[i])
        if value_type(args[i]) == party
            continue
        end
        # Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
        args[i] = calling_conv_fixup(B, args[i], party)
        # GPUCompiler.@safe_error "Calling convention mismatch", party, args[i], i, llvmf, fn, args, sret, returnRoots
        return false
    end

    res = LLVM.call!(B, LLVM.function_type(llvmf), llvmf, args)
    debug_from_orig!(gutils, res, orig)
    callconv!(res, callconv(llvmf))

    hasNoRet = any(map(k->kind(k)==kind(EnumAttribute("noreturn")), collect(function_attributes(llvmf))))

    if hasNoRet
        return false
    end

    if sret !== nothing
        if LLVM.version().major >= 12
            attr = TypeAttribute("sret", eltype(value_type(parameters(llvmf)[1])))
        else
            attr = EnumAttribute("sret")
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1), attr)
        res = load!(B, eltype(value_type(parameters(llvmf)[1])), sret)
    end
    if swiftself
        attr = EnumAttribute("swiftself")
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+(sret !== nothing)), attr)
    end

    shadowV = C_NULL
    normalV = C_NULL

    if RT <: Const
        # TODO introduce const-no-need
        if needsPrimal || true
            if RealRt != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of const primal-only forward custom rule - "*(string(RT))*" "*string(activity)*" want just return type "*string(RealRt)*" found "*string(fwd_RT))
                return false
            end
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, res, val)
            else
                normalV = res.ref
            end
        else
            if Nothing != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of const no-primal forward custom rule - "*(string(RT))*" "*string(activity)*" want just return type Nothing found "*string(fwd_RT))
                return false
            end
        end
    else
        if !needsPrimal
            ST = RealRt
            if width != 1
                ST = NTuple{Int(width), ST}
            end
            if ST != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of shadow-only forward custom rule - "*(string(RT))*" "*string(activity)*" want just shadow type "*string(ST)*" found "*string(fwd_RT))
                return false
            end
            if get_return_info(RealRt)[2] !== nothing
                dval_ptr = invert_pointer(gutils, operands(orig)[1], B)
                for idx in 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx-1)
                    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx-1)
                    store!(B, res, pev)
                end
            else
                shadowV = res.ref
            end
        else
            ST = if width == 1
                Duplicated{RealRt}
            else
                BatchDuplicated{RealRt, Int(width)}
            end
            if ST != fwd_RT
                emit_error(B, orig, "Enzyme: incorrect return type of prima/shadow forward custom rule - "*(string(RT))*" "*string(activity)*" want just shadow type "*string(ST)*" found "*string(fwd_RT))
                return false
            end
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, extract_value!(B, res, 0), val)
                
                dval_ptr = invert_pointer(gutils, operands(orig)[1], B)
                dval = extract_value!(B, res, 1)
                for idx in 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx-1)
                    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx-1)
                    store!(B, ev, pev)
                end
            else
                normalV = extract_value!(B, res, 0).ref
                shadowV = extract_value!(B, res, 1).ref
            end
        end
    end

    if shadowR != C_NULL
        unsafe_store!(shadowR, shadowV)
    end

    # Delete the primal code
    if origNeedsPrimal
        unsafe_store!(normalR, normalV)
    else
        ni = new_from_original(gutils, orig)
        if value_type(ni) != LLVM.VoidType()
            API.EnzymeGradientUtilsReplaceAWithB(gutils, ni, LLVM.UndefValue(value_type(ni)))
        end
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    return false
end

function enzyme_custom_common_rev(forward::Bool, B, orig::LLVM.CallInst, gutils, normalR, shadowR, tape)::LLVM.API.LLVMValueRef

    ctx = LLVM.context(orig)

    width = get_width(gutils)

    shadowType = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
    if shadowR != C_NULL
        unsafe_store!(shadowR,UndefValue(shadowType).ref)
    end

    # TODO: don't inject the code multiple times for multiple calls

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)
    isKWCall = isKWCallSignature(mi.specTypes)

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup = enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, #=reverse=#!forward, isKWCall)
    RT, needsPrimal, needsShadow, origNeedsPrimal = enzyme_custom_setup_ret(gutils, orig, mi, RealRt)

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)

    C = EnzymeRules.Config{Bool(needsPrimal), Bool(needsShadow), Int(width), overwritten}
    
    mode = get_mode(gutils)

    ami = nothing

    augprimal_tt = copy(activity)
    if isKWCall
        popfirst!(augprimal_tt)
        @assert kwtup !== nothing
        insert!(augprimal_tt, 1, kwtup)
        insert!(augprimal_tt, 2, Core.typeof(EnzymeRules.augmented_primal))
        insert!(augprimal_tt, 3, C)
        insert!(augprimal_tt, 5, Type{RT})

        augprimal_TT = Tuple{augprimal_tt...}
        kwfunc = Core.kwfunc(EnzymeRules.augmented_primal)
        try
            ami = GPUCompiler.methodinstance(Core.Typeof(kwfunc), augprimal_TT, world)
            @safe_debug "Applying custom augmented_primal rule (kwcall)" TT=augprimal_TT
        catch e
        end
    else
        @assert kwtup === nothing
        insert!(augprimal_tt, 1, C)
        insert!(augprimal_tt, 3, Type{RT})

        augprimal_TT = Tuple{augprimal_tt...}
        try
            ami = GPUCompiler.methodinstance(Core.Typeof(EnzymeRules.augmented_primal), augprimal_TT, world)
            @safe_debug "Applying custom augmented_primal rule" TT=augprimal_TT
        catch e
        end
    end
    
    if ami !== nothing
        target = DefaultCompilerTarget()
        params = PrimalCompilerParams(mode)
        job    = CompilerJob(ami, CompilerConfig(target, params; kernel=false), world)
        interp = GPUCompiler.get_interpreter(job)
        aug_RT = something(Core.Compiler.typeinf_type(interp, ami.def, ami.specTypes, ami.sparam_vals), Any)
    else
        @safe_debug "No custom augmented_primal rule is applicable for" augprimal_TT
        emit_error(B, orig, "Enzyme: No custom augmented_primal rule was appliable for " * string(augprimal_TT))
        return C_NULL
    end

    if kwtup !== nothing && kwtup <: Duplicated
        @safe_debug "Non-constant keyword argument found for " augprimal_TT
        emit_error(B, orig, "Enzyme: Non-constant keyword argument found for " * string(augprimal_TT))
        return C_NULL
    end

    rev_TT = nothing
    rev_RT = nothing

    TapeT = Nothing

    if (aug_RT <: EnzymeRules.AugmentedReturn || aug_RT <: EnzymeRules.AugmentedReturnFlexShadow) && !(aug_RT isa UnionAll) && !(aug_RT isa Union) && !(aug_RT === Union{})
        TapeT = EnzymeRules.tape_type(aug_RT)
    end

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    llvmf = nothing

    if forward
        llvmf = nested_codegen!(mode, mod, ami, world)
        @assert llvmf !== nothing
    else
        tt = copy(activity)
        if isKWCall
            popfirst!(tt)
            @assert kwtup !== nothing
            insert!(tt, 1, kwtup)
            insert!(tt, 2, Core.typeof(EnzymeRules.reverse))
            insert!(tt, 3, C)
            insert!(tt, 5, RT <: Active ? RT : Type{RT})
            insert!(tt, 6, TapeT)
        else
            @assert kwtup === nothing
            insert!(tt, 1, C)
            insert!(tt, 3, RT <: Active ? RT : Type{RT})
            insert!(tt, 4, TapeT)
        end
        rev_TT = Tuple{tt...}

        if isKWCall
            rkwfunc = Core.kwfunc(EnzymeRules.reverse)
            if EnzymeRules.isapplicable(rkwfunc, rev_TT; world)
                @safe_debug "Applying custom reverse rule (kwcall)" TT=rev_TT
                llvmf = nested_codegen!(mode, mod, rkwfunc, rev_TT, world)
                rev_RT = Core.Compiler.return_type(rkwfunc, rev_TT, world)
            end
        else
            if EnzymeRules.isapplicable(EnzymeRules.reverse, rev_TT; world)
                @safe_debug "Applying custom reverse rule" TT=rev_TT
                llvmf = nested_codegen!(mode, mod, EnzymeRules.reverse, rev_TT, world)
                rev_RT = Core.Compiler.return_type(EnzymeRules.reverse, rev_TT, world)
            end
        end

        if llvmf == nothing
            @safe_debug "No custom reverse rule is applicable for" rev_TT
            emit_error(B, orig, "Enzyme: No custom reverse rule was appliable for " * string(rev_TT))
            return C_NULL
        end
    end
    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))

    needsTape = !isghostty(TapeT) && !Core.Compiler.isconstType(TapeT)

    tapeV = C_NULL
    if forward && needsTape
        tapeV = LLVM.UndefValue(convert(LLVMType, TapeT; allow_boxed=true)).ref
    end

    # if !forward
    #     argTys = copy(activity)
    #     if RT <: Active
    #         if width == 1
    #             push!(argTys, RealRt)
    #         else
    #             push!(argTys, NTuple{RealRt, (Int)width})
    #         end
    #     end
    #     push!(argTys, tapeType)
    #     llvmf = nested_codegen!(mode, mod, rev_func, Tuple{argTys...}, world)
    # end

    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(llvmf, i)))) for i in 1:length(collect(parameters(llvmf))))

    _, sret, returnRoots = get_return_info(enzyme_custom_extract_mi(llvmf)[2])

    if !forward
        if needsTape
            @assert tape != C_NULL
            tape_idx = 1+(kwtup!==nothing && !isghostty(kwtup))+(isKWCall && !isghostty(rev_TT.parameters[4]))
            innerTy = value_type(parameters(llvmf)[tape_idx+(sret !== nothing)+(RT <: Active)])
            if innerTy != value_type(tape)
                llty = convert(LLVMType, TapeT; allow_boxed=true)
                al0 = al = emit_allocobj!(B, TapeT)
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                store!(B, tape, al)
                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, tape))
                end
                tape = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))
            end
            insert!(args, tape_idx, tape)
        end
        if RT <: Active

            llty = convert(LLVMType, RT)

            if API.EnzymeGradientUtilsGetDiffeType(gutils, orig, #=isforeign=#false) == API.DFT_OUT_DIFF
                val = LLVM.Value(API.EnzymeGradientUtilsDiffe(gutils, orig, B))
            else
                llety = convert(LLVMType, eltype(RT))
                ptr_val = invert_pointer(gutils, operands(orig)[1], B)
                val = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llety)))
                for idx in 1:width
                    ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx-1)
                    ld = load!(B, llety, ev)
                    store!(B, LLVM.null(llety), ev)
                    val = (width == 1 ) ? ld : insert_value!(B, val, ld, idx-1)
                end
            end

            al0 = al = emit_allocobj!(B, RT)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            ptr = inbounds_gep!(B, llty, al, [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)])
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            insert!(args, 1+(kwtup!==nothing && !isghostty(kwtup))+(isKWCall && !isghostty(rev_TT.parameters[4])), al)
        end
    end

    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end

    if sret !== nothing
        sret = alloca!(alloctx, convert(LLVMType, eltype(sret)))
        pushfirst!(args, sret)
        if returnRoots !== nothing
            returnRoots = alloca!(alloctx, convert(LLVMType, eltype(returnRoots)))
            insert!(args, 2, returnRoots)
        else
            returnRoots = nothing
        end
    else
        sret = nothing
    end

    if length(args) != length(parameters(llvmf))
        GPUCompiler.@safe_error "Calling convention mismatch", args, llvmf, orig, isKWCall, kwtup, augprimal_TT, rev_TT, fn, sret, returnRoots
        return tapeV
    end
    

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    for i in 1:length(args)
        party =  value_type(parameters(llvmf)[i])
        if value_type(args[i]) != party
            if party == T_prjlvalue
                while true
                    if isa(args[i], LLVM.BitCastInst)
                        args[i] = operands(args[i])[1]
                        continue
                    end
                    if isa(args[i], LLVM.AddrSpaceCastInst)
                        args[i] = operands(args[i])[1]
                        continue
                    end
                    break
                end
            end
        end

        if value_type(args[i]) == party
            continue
        end
        # Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
        function msg(io)
            println(io, string(llvmf))
            println(io, "args = ", args)
            println(io, "i = ", i)
            println(io, "args[i] = ", args[i])
            println(io, "party = ", party)
        end
        args[i] = calling_conv_fixup(B, args[i], party, LLVM.UndefValue(party), Cuint[], Cuint[], msg)
    end

    res = LLVM.call!(B, LLVM.function_type(llvmf), llvmf, args)
    ncall = res
    debug_from_orig!(gutils, res, orig)
    callconv!(res, callconv(llvmf))

    hasNoRet = any(map(k->kind(k)==kind(EnumAttribute("noreturn")), collect(function_attributes(llvmf))))

    if hasNoRet
        return tapeV
    end

    if sret !== nothing
        if LLVM.version().major >= 12
            attr = TypeAttribute("sret", eltype(value_type(parameters(llvmf)[1+swiftself])))
        else
            attr = EnumAttribute("sret")
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+swiftself), attr)
        res = load!(B, eltype(value_type(parameters(llvmf)[1+swiftself])), sret)
        API.SetMustCache!(res)
    end
    if swiftself
        attr = EnumAttribute("swiftself")
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1+(sret !== nothing)+(returnRoots !== nothing)), attr)
    end

    shadowV = C_NULL
    normalV = C_NULL


    if forward
        ShadT = RealRt
        if width != 1
            ShadT = NTuple{Int(width), RealRt}
        end
        ST = EnzymeRules.AugmentedReturn{needsPrimal ? RealRt : Nothing, needsShadow ? ShadT : Nothing, TapeT}
        if aug_RT != ST
            if aug_RT <: EnzymeRules.AugmentedReturnFlexShadow
                if convert(LLVMType, EnzymeRules.shadow_type(aug_RT); allow_boxed=true) !=
                    convert(LLVMType, EnzymeRules.shadow_type(ST)    ; allow_boxed=true)
                    emit_error(B, orig, "Enzyme: Augmented forward pass custom rule " * string(augprimal_TT) * " flex shadow ABI return type mismatch, expected "*string(ST)*" found "* string(aug_RT))
                    return tapeV
                end
                ST = EnzymeRules.AugmentedReturnFlexShadow{needsPrimal ? RealRt : Nothing, needsShadow ? EnzymeRules.shadow_type(aug_RT) : Nothing, TapeT}
            end
        end
        if aug_RT != ST
            ST = EnzymeRules.AugmentedReturn{needsPrimal ? RealRt : Nothing, needsShadow ? ShadT : Nothing, Any}
            emit_error(B, orig, "Enzyme: Augmented forward pass custom rule " * string(augprimal_TT) * " return type mismatch, expected "*string(ST)*" found "* string(aug_RT))
            return tapeV
        end

        idx = 0
        if needsPrimal
            @assert !isghostty(RealRt)
            normalV = extract_value!(B, res, idx)
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, normalV, val)
            else
                @assert value_type(normalV) == value_type(orig)
                normalV = normalV.ref
            end
            idx+=1
        end
        if needsShadow
            @assert !isghostty(RealRt)
            shadowV = extract_value!(B, res, idx)
            if get_return_info(RealRt)[2] !== nothing
                dval = invert_pointer(gutils, operands(orig)[1], B)

                for idx in 1:width
                    to_store = (width == 1) ? shadowV : extract_value!(B, shadowV, idx-1)

                    store_ptr = (width == 1) ? dval : extract_value!(B, dval, idx-1)

                    store!(B, to_store, store_ptr)
                end
                shadowV = C_NULL
            else
                @assert value_type(shadowV) == shadowType
                shadowV = shadowV.ref
            end
            idx+=1
        end
        if needsTape
            tapeV = extract_value!(B, res, idx).ref
            idx+=1
        end
    else
        Tys = (A <: Active ? eltype(A) : Nothing for A in activity[2+isKWCall:end])
        ST = Tuple{Tys...}
        if rev_RT != ST
            emit_error(B, orig, "Enzyme: Reverse pass custom rule " * string(rev_TT) * " return type mismatch, expected "*string(ST)*" found "* string(rev_RT))
            return tapeV
        end
        if length(actives) >= 1 && !isa(value_type(res), LLVM.StructType) && !isa(value_type(res), LLVM.ArrayType)
            GPUCompiler.@safe_error "Shadow arg calling convention mismatch found return ", res
            return tapeV
        end

        idx = 0
        dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(orig)))))
        Tys2 = (eltype(A) for A in activity[2+isKWCall:end] if A <: Active)
        for (v, Ty) in zip(actives, Tys2)
            TT = typetree(Ty, ctx, dl)
            Typ = C_NULL
            ext = extract_value!(B, res, idx)
            shadowVType = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(v)))
            if value_type(ext) != shadowVType
                size = sizeof(Ty)
                align = 0
                premask = C_NULL
                API.EnzymeGradientUtilsAddToInvertedPointerDiffeTT(gutils, orig, C_NULL, TT, size, v,           ext, B, align, premask)
            else
                @assert value_type(ext) == shadowVType
                API.EnzymeGradientUtilsAddToDiffe(gutils, v, ext, B, Typ)
            end
            idx+=1
        end
    end

    if forward
        if shadowR != C_NULL && shadowV != C_NULL
            unsafe_store!(shadowR, shadowV)
        end

        # Delete the primal code
        if origNeedsPrimal
            unsafe_store!(normalR, normalV)
        else
            ni = new_from_original(gutils, orig)
            erase_with_placeholder(gutils, ni, orig)
        end
    end

    return tapeV
end


function enzyme_custom_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    tape = enzyme_custom_common_rev(#=forward=#true, B, orig, gutils, normalR, shadowR, #=tape=#nothing)
    if tape != C_NULL
        unsafe_store!(tapeR, tape)
    end
    return false
end


function enzyme_custom_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return
    end
    enzyme_custom_common_rev(#=forward=#false, B, orig, gutils, #=normalR=#C_NULL, #=shadowR=#C_NULL, #=tape=#tape)
    return nothing
end