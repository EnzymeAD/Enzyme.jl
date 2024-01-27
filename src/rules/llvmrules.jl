include("customrules.jl")
include("jitrules.jl")
include("typeunstablerules.jl")
include("parallelrules.jl")

function jlcall_fwd(B, orig, gutils, normalR, shadowR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            return common_generic_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            return common_apply_latest_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            return common_newstructv_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            return common_f_tuple_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            return common_jl_getfield_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f_setfield", "jl_f_setfield"))
            return common_setfield_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            return common_apply_iterate_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(name, ("ijl_f__svec_ref", "jl_f__svec_ref"))
            return common_f_svec_ref_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    emit_error(B, orig, "Enzyme: jl_call calling convention not implemented in forward for "*string(orig))

    return false
end

function jlcall_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            return common_generic_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            return common_apply_latest_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            return common_newstructv_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            return common_f_tuple_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            return common_jl_getfield_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_s_getfield", "jl_s_getfield"))
            return common_setfield_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            return common_apply_iterate_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__svec_rev", "jl_f__svec_ref"))
            return common_f_svec_ref_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    emit_error(B, orig, "Enzyme: jl_call calling convention not implemented in aug_forward for "*string(orig))

    return false
end

function jlcall_rev(B, orig, gutils, tape)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            common_generic_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__apply_latest", "ijl_f__call_latest", "jl_f__apply_latest", "jl_f__call_latest"))
            common_apply_latest_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_new_structv", "jl_new_structv"))
            common_newstructv_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_tuple", "jl_f_tuple"))
            common_f_tuple_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_getfield", "jl_f_getfield"))
            common_jl_getfield_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f_setfield", "jl_f_setfield"))
            common_setfield_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            common_apply_iterate_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(name, ("ijl_f__svec_ref", "jl_f__svec_ref"))
            common_f_svec_ref_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return nothing
        end
    end

    emit_error(B, orig, "Enzyme: jl_call calling convention not implemented in reverse for "*string(orig))

    return nothing
end

function jlcall2_fwd(B, orig, gutils, normalR, shadowR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            return common_invoke_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return false
end

function jlcall2_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            return common_invoke_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return true
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return false
end

function jlcall2_rev(B, orig, gutils, tape)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if any(map(k->kind(k)==kind(StringAttribute("enzyme_inactive")), collect(function_attributes(F))))
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end


function noop_fwd(B, orig, gutils, normalR, shadowR)
    return true
end

function noop_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    return true
end

function duplicate_rev(B, orig, gutils, tape)
    newg = new_from_original(gutils, orig)

    real_ops = collect(operands(orig))[1:end-1]
    ops = [lookup_value(gutils, new_from_original(gutils, o), B) for o in real_ops]
    
    c = call_samefunc_with_inverted_bundles!(B, gutils, orig, ops, [API.VT_Primal for _ in ops], #=lookup=#false)
    callconv!(c, callconv(orig))

    return nothing
end

function arraycopy_fwd(B, orig, gutils, normalR, shadowR)
    ctx = LLVM.context(orig)

    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    origops = LLVM.operands(orig)

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)

    i8 = LLVM.IntType(8)
    algn = 0

    if width == 1
        shadowres = call_samefunc_with_inverted_bundles!(B, gutils, orig, [shadowin], [API.VT_Shadow], #=lookup=#false)

        # TODO zero based off runtime types, rather than presume floatlike?
        if is_constant_value(gutils, origops[1])
            elSize = get_array_elsz(B, shadowin)
            elSize = LLVM.zext!(B, elSize, LLVM.IntType(8*sizeof(Csize_t)))
            len = get_array_len(B, shadowin)
            length = LLVM.mul!(B, len, elSize)
            isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)
            GPUCompiler.@safe_warn "TODO forward zero-set of arraycopy used memset rather than runtime type"
            LLVM.memset!(B, get_array_data(B, shadowres), LLVM.ConstantInt(i8, 0, false), length, algn)
        end
        if API.runtimeActivity()
            prev = new_from_original(gutils, orig)
            shadowres = LLVM.select!(B, LLVM.icmp!(B, LLVM.API.LLVMIntNE, shadowin, new_from_original(gutils, origops[1])), shadowres, prev)
            API.moveBefore(prev, shadowres, B)
        end
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            ev = extract_value!(B, shadowin, idx-1)
            callv = call_samefunc_with_inverted_bundles!(B, gutils, orig, [ev], [API.VT_Shadow], #=lookup=#false)
            if is_constant_value(gutils, origops[1])
                elSize = get_array_elsz(B, shadowin)
                elSize = LLVM.zext!(B, elSize, LLVM.IntType(8*sizeof(Csize_t)))
                len = get_array_len(B, shadowin)
                length = LLVM.mul!(B, len, elSize)
                isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)
                GPUCompiler.@safe_warn "TODO forward zero-set of arraycopy used memset rather than runtime type"
                LLVM.memset!(B, get_array_data(callv), LLVM.ConstantInt(i8, 0, false), length, algn)
            end
            if API.runtimeActivity()
                prev = new_from_original(gutils, orig)
                callv = LLVM.select!(B, LLVM.icmp!(B, LLVM.API.LLVMIntNE, ev, new_from_original(gutils, origops[1])), callv, prev)
                if idx == 1
                    API.moveBefore(prev, callv, B)
                end
            end
            shadowres = insert_value!(B, shadowres, callv, idx-1)
        end
    end

    unsafe_store!(shadowR, shadowres.ref)
	return false
end

function arraycopy_common(fwd, B, orig, origArg, gutils, shadowdst)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, API.DEM_ReverseModePrimal)
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0
    if !needsShadow
        return nothing
    end

    if !fwd
        shadowdst = invert_pointer(gutils, orig, B)
    end

    # size_t len = jl_array_len(ary);
    # size_t elsz = ary->elsize;
    # memcpy(new_ary->data, ary->data, len * elsz);
	# JL_EXTENSION typedef struct {
	# 	JL_DATA_TYPE
	# 	void *data;
	# #ifdef STORE_ARRAY_LEN
	# 	size_t length;
	# #endif
	# 	jl_array_flags_t flags;
	# 	uint16_t elsize;  // element size including alignment (dim 1 memory stride)

	tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, orig))
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
	dl = string(LLVM.datalayout(mod))
	API.EnzymeTypeTreeLookupEq(tt, 1, dl)
	data0!(tt)
    ct = API.EnzymeTypeTreeInner0(tt)

    if ct == API.DT_Unknown
        # analyzer = API.EnzymeGradientUtilsTypeAnalyzer(gutils)
        # ip = API.EnzymeTypeAnalyzerToString(analyzer)
        # sval = Base.unsafe_string(ip)
        # API.EnzymeStringFree(ip)
        emit_error(B, orig, "Enzyme: Unknown concrete type in arraycopy_common. tt: " * string(tt))
        return nothing
    end

    @assert ct != API.DT_Unknown
    ctx = LLVM.context(orig)
    secretty = API.EnzymeConcreteTypeIsFloat(ct)

    off = sizeof(Cstring)
    if true # STORE_ARRAY_LEN
        off += sizeof(Csize_t)
    end
    #jl_array_flags_t
    off += 2

    actualOp = new_from_original(gutils, origArg)
    if fwd
        B0 = B
    elseif typeof(actualOp) <: LLVM.Argument
        B0 = LLVM.IRBuilder()
        position!(B0, first(instructions(new_from_original(gutils, LLVM.entry(LLVM.parent(LLVM.parent(orig)))))))
    else
        B0 = LLVM.IRBuilder()
        nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(actualOp))
        while isa(nextInst, LLVM.PHIInst)
            nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(nextInst))
        end
        position!(B0, nextInst)
    end

    elSize = get_array_elsz(B0, actualOp)
    elSize = LLVM.zext!(B0, elSize, LLVM.IntType(8*sizeof(Csize_t)))

    len = get_array_len(B0, actualOp)

    length = LLVM.mul!(B0, len, elSize)
    isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)

    # forward pass copy already done by underlying call
    allowForward = false
    intrinsic = LLVM.Intrinsic("llvm.memcpy").id

    if !fwd
        shadowdst = lookup_value(gutils, shadowdst, B)
    end
    shadowsrc = invert_pointer(gutils, origArg, B)
    if !fwd
        shadowsrc = lookup_value(gutils, shadowsrc, B)
    end

    width = get_width(gutils)

    # Zero the copy in the forward pass.
    #   initshadow = 2.0
    #   dres = copy(initshadow) # 2.0
    #
    #   This needs to be inserted
    #   memset(dres, 0, ...)
    #
    #   # removed return res[1]
    #   dres[1] += differeturn
    #   dmemcpy aka initshadow += dres
    algn = 0
    i8 = LLVM.IntType(8)

    if width == 1

    shadowsrc = get_array_data(B, shadowsrc)
    shadowdst = get_array_data(B, shadowdst)

    if fwd && secretty != nothing
        LLVM.memset!(B, shadowdst, LLVM.ConstantInt(i8, 0, false), length, algn)
    end

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst, false, shadowsrc, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)

    else
    for i in 1:width

    evsrc = extract_value!(B, shadowsrc, i-1)
    evdst = extract_value!(B, shadowdst, i-1)

    shadowsrc0 = get_array_data(B, evsrc)
    shadowdst0 = get_array_data(B, evdst)

    if fwd && secretty != nothing
        LLVM.memset!(B, shadowdst0, LLVM.ConstantInt(i8, 0, false), length, algn)
    end

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst0, false, shadowsrc0, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    end

    end

    return nothing
end

function arraycopy_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    arraycopy_fwd(B, orig, gutils, normalR, shadowR)

    origops = LLVM.operands(orig)

    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
      shadowres = LLVM.Value(unsafe_load(shadowR))

      arraycopy_common(#=fwd=#true, B, orig, origops[1], gutils, shadowres)
    end

	return false
end

function arraycopy_rev(B, orig, gutils, tape)
    origops = LLVM.operands(orig)
    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
        arraycopy_common(#=fwd=#false, B, orig, origops[1], gutils, nothing)
    end

    return nothing
end

function arrayreshape_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    origops = LLVM.operands(orig)
    if is_constant_value(gutils, origops[2])
        emit_error(B, orig, "Enzyme: reshape array has active return, but inactive input")
    end

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[2], B)
    if width == 1
        args = LLVM.Value[
                          new_from_original(gutils, origops[1])
                          shadowin
                          new_from_original(gutils, origops[3])
                          ]
        shadowres = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Primal, API.VT_Shadow, API.VT_Primal], #=lookup=#false)
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[new_from_original(gutils, origops[1])
                              extract_value!(B, shadowin, idx-1)
                              new_from_original(gutils, origops[3])
                              ]
            tmp = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Primal, API.VT_Shadow, API.VT_Primal], #=lookup=#false)
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)

	return false
end

function arrayreshape_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    arrayreshape_fwd(B, orig, gutils, normalR, shadowR)
end

function arrayreshape_rev(B, orig, gutils, tape)
    return nothing
end

function boxfloat_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if is_constant_value(gutils, orig)
        return true
    end

    flt = value_type(origops[1])
    shadowsin = LLVM.Value[invert_pointer(gutils, origops[1], B)]
    if width == 1
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), shadowsin)
        callconv!(shadowres, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, s, idx-1) for s in shadowsin
                              ]
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

function boxfloat_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if is_constant_value(gutils, orig)
        return true
    end

    flt = value_type(origops[1])
    TT = tape_type(flt)

    if width == 1
        obj = emit_allocobj!(B, Base.RefValue{TT})
        o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(value_type(obj))))
        store!(B, ConstantFP(flt, 0.0), o2)
        shadowres = obj
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, flt)))
        for idx in 1:width
            obj = emit_allocobj!(B, Base.RefValue{TT})
            o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(value_type(obj))))
            store!(B, ConstantFP(flt, 0.0), o2)
            shadowres = insert_value!(B, shadowres, obj, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

function boxfloat_rev(B, orig, gutils, tape)
    origops = collect(operands(orig))
    width = get_width(gutils)
    if !is_constant_value(gutils, orig)
        ip = lookup_value(gutils, invert_pointer(gutils, orig, B), B)
        flt = value_type(origops[1])
        if width == 1
            ipc = bitcast!(B, ip, LLVM.PointerType(flt, addrspace(value_type(orig))))
            ld = load!(B, flt, ipc)
            store!(B, ConstantFP(flt, 0.0), ipc)
            if !is_constant_value(gutils, origops[1])
                API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], ld, B, flt)
            end
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, flt)))
            for idx in 1:width
                ipc = extract_value!(B, ip, idx-1)
                ipc = bitcast!(B, ipc, LLVM.PointerType(flt, addrspace(value_type(orig))))
                ld = load!(B, flt, ipc)
                store!(B, ConstantFP(flt, 0.0), ipc)
                shadowres = insert_value!(B, shadowres, ld, idx-1)
            end
            if !is_constant_value(gutils, origops[1])
                API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], shadowret, B, flt)
            end
        end
    end
    return nothing
end

function eqtableget_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end

    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_eqtable_get")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function error_if_active(::Type{T}) where T
    seen = ()
    areg = active_reg_inner(T, seen, nothing, #=justActive=#Val(true))
    if areg == ActiveState
        throw(AssertionError("Found unhandled active variable in tuple splat, jl_eqtable $T"))
    end
    nothing
end

function eqtableget_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    width = get_width(gutils)

    origh, origkey, origdflt = operands(orig)[1:end-1]

    if is_constant_value(gutils, origh)
        emit_error(B, orig, "Enzyme: Not yet implemented constant table in jl_eqtable_get "*string(origh)*" "*string(orig)*" result: "*string(absint(orig))*" "*string(abs_typeof(orig, true))*" dict: "*string(absint(origh))*" "*string(abs_typeof(origh, true))*" key "*string(absint(origkey))*" "*string(abs_typeof(origkey, true))*" dflt "*string(absint(origdflt))*" "*string(abs_typeof(origdflt, true)))
    end
    
    shadowh = invert_pointer(gutils, origh, B)

    shadowdflt = if is_constant_value(gutils, origdflt)
        shadowdflt2 = julia_error(Base.unsafe_convert(Cstring, "Mixed activity for default of jl_eqtable_get "*string(orig)*" "*string(origdflt)),
                                 orig.ref, API.ET_MixedActivityError, gutils.ref, origdflt.ref, B.ref)
        if shadowdflt2 != C_NULL
            LLVM.Value(shadowdflt2)
        else
            nop = new_from_original(gutils, origdflt)
            if width == 1
                nop
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(nop)))
                shadowm = LLVM.UndefValue(ST)
                for j in 1:width
                    shadowm = insert_value!(B, shadowm, nop, j-1)
                end
                shadowm
            end
        end
    else
        invert_pointer(gutils, origdflt, B)
    end
        
    newvals = API.CValueType[API.VT_Shadow, API.VT_Primal, API.VT_Shadow]
    
    shadowres = if width == 1
        newops = LLVM.Value[shadowh, new_from_original(gutils, origkey), shadowdflt]
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
        callconv!(cal, callconv(orig))
        emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active), emit_jltypeof!(B, cal)])
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j in 1:width
            newops = LLVM.Value[extract_value!(B, shadowh, j-1), new_from_original(gutils, origkey), extract_value!(B, shadowdflt, j-1)]
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
            callconv!(cal, callconv(orig))
            emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active), emit_jltypeof!(B, cal)])
            shadow = insert_value!(B, shadow, cal, j-1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    return false
end

function eqtableget_rev(B, orig, gutils, tape)
    return nothing
end

function eqtableput_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_eqtable_put")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function eqtableput_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    origh, origkey, origval, originserted = operands(orig)[1:end-1]

    @assert !is_constant_value(gutils, origh)

    shadowh = invert_pointer(gutils, origh, B)
    shadowval = invert_pointer(gutils, origval, B)

    shadowval = if is_constant_value(gutils, origval)
        shadowdflt2 = julia_error(Base.unsafe_convert(Cstring, "Mixed activity for val of jl_eqtable_put "*string(orig)*" "*string(origval)),
                                 orig.ref, API.ET_MixedActivityError, gutils.ref, origval.ref, B.ref)
        if shadowdflt2 != C_NULL
            LLVM.Value(shadowdflt2)
        else
            nop = new_from_original(gutils, origval)
            if width == 1
                nop
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(nop)))
                shadowm = LLVM.UndefValue(ST)
                for j in 1:width
                    shadowm = insert_value!(B, shadowm, nop, j-1)
                end
                shadowm
            end
        end
    else
        invert_pointer(gutils, origval, B)
    end

    newvals = API.CValueType[API.VT_Shadow, API.VT_Primal, API.VT_Shadow, API.VT_None]
    
    shadowres = if width == 1
        emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active), emit_jltypeof!(B, shadowval)])
        newops = LLVM.Value[shadowh, new_from_original(gutils, origkey), shadowval, LLVM.null(value_type(originserted))]
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
        callconv!(cal, callconv(orig))
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j in 1:width
            sval2 = extract_value!(B, shadowval, j-1)
            emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active), emit_jltypeof!(B, sval2)])
            newops = LLVM.Value[extract_value!(B, shadowh, j-1), new_from_original(gutils, origkey), sval2, LLVM.null(value_type(originserted))]
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
            callconv!(cal, callconv(orig))
            shadow = insert_value!(B, shadow, cal, j-1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    return false
end

function eqtableput_rev(B, orig, gutils, tape)
    return nothing
end


function idtablerehash_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_idtable_rehash")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function idtablerehash_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_idtable_rehash")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function idtablerehash_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_idtable_rehash")
    return nothing
end

function jl_array_grow_end_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    if is_constant_value(gutils, origops[1])
        return true
    end

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    if width == 1
        args = LLVM.Value[
                          shadowin
                          new_from_original(gutils, origops[2])
                          ]
        call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
    else
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, shadowin, idx-1)
                              new_from_original(gutils, origops[2])
                              ]
            call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
        end
    end
    return false
end


function jl_array_grow_end_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    origops = collect(operands(orig))
    if is_constant_value(gutils, origops[1])
        return true
    end

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    ctx = LLVM.context(orig)
    i8 = LLVM.IntType(8)

    inc = new_from_original(gutils, origops[2])

    al = 0

    if width == 1
        anti = shadowin

        idx = get_array_nrows(B, anti)
        elsz = zext!(B, get_array_elsz(B, anti), value_type(idx))
        off = mul!(B, idx, elsz)
        tot = mul!(B, inc, elsz)

        args = LLVM.Value[anti, inc]
        call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)

        toset = get_array_data(B, anti)
        toset = gep!(B, i8, toset, LLVM.Value[off])
        mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
    else
        for idx in 1:width
            anti = extract_value!(B, shadowin, idx-1)

            idx = get_array_nrows(B, anti)
            elsz = zext!(B, get_array_elsz(B, anti), value_type(idx))
            off = mul!(B, idx, elsz)
            tot = mul!(B, inc, elsz)

            args = LLVM.Value[anti, inc]
            call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)

            toset = get_array_data(B, anti)
            toset = gep!(B, i8, toset, LLVM.Value[off])
            mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
        end
    end

    return false
end

function jl_array_grow_end_rev(B, orig, gutils, tape)
    origops = collect(operands(orig))
    if !is_constant_value(gutils, origops[1])

        width = get_width(gutils)

        called_value = origops[end]
        funcT = called_type(orig)
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delF, fty = get_function!(mod, "jl_array_del_end", funcT)

        shadowin = invert_pointer(gutils, origops[1], B)
        shadowin = lookup_value(gutils, shadowin, B)

        offset = new_from_original(gutils, origops[2])
        offset = lookup_value(gutils, offset, B)

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, fty, delF, args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, fty, delF, args)
            end
        end
    end
    return nothing
end

function jl_array_del_end_fwd(B, orig, gutils, normalR, shadowR)
    jl_array_grow_end_fwd(B, orig, gutils, normalR, shadowR)
end

function jl_array_del_end_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_del_end_fwd(B, orig, gutils, normalR, shadowR)
end

function jl_array_del_end_rev(B, orig, gutils, tape)
    origops = collect(operands(orig))
    if !is_constant_value(gutils, origops[1])
        width = get_width(gutils)

        called_value = origops[end]
        funcT = called_type(orig)
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delF, fty = get_function!(mod, "jl_array_grow_end", funcT)

        shadowin = invert_pointer(gutils, origops[1], B)
        shadowin = lookup_value(gutils, shadowin, B)

        offset = new_from_original(gutils, origops[2])
        offset = lookup_value(gutils, offset, B)

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, fty, delF, args)
        else
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, fty, delF, args)
            end
        end

        # GPUCompiler.@safe_warn "Not applying memsetUnknown concrete type" tt=string(tt)
        emit_error(B, orig, "Not applying memset on reverse of jl_array_del_end")
        # memset(data + idx * elsz, 0, inc * elsz);
    end
    return nothing
end

function jl_array_ptr_copy_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_inst(gutils, orig)
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)
    origops = collect(operands(orig))
    width = get_width(gutils)

    args = LLVM.Value[]
    for a in origops[1:end-2]
        v = invert_pointer(gutils, a, B)
        push!(args, v)
    end
    push!(args, new_from_original(gutils, origops[end-1]))
    valTys = API.CValueType[API.VT_Shadow, API.VT_Shadow, API.VT_Shadow, API.VT_Shadow, API.VT_Primal]

    if width == 1
        vargs = args
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, vargs, valTys, #=lookup=#false)
        debug_from_orig!(gutils, cal, orig)
        callconv!(cal, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            vargs = LLVM.Value[]
            for a in args[1:end-1]
                push!(vargs, extract_value!(B, a, idx-1))
            end
            push!(vargs, args[end])
            cal = call_samefunc_with_inverted_bundles!(b, gutils, orig, vargs, valTys, #=lookup=#false)
            debug_from_orig!(gutils, cal, orig)
            callconv!(cal, callconv(orig))
        end
    end

    return false
end
function jl_array_ptr_copy_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
  jl_array_ptr_copy_fwd(B, orig, gutils, normalR, shadowR)
end
function jl_array_ptr_copy_rev(B, orig, gutils, tape)
    return nothing
end

function jl_array_sizehint_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    if is_constant_value(gutils, origops[1])
        return true
    end
    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    if width == 1
        args = LLVM.Value[
                          shadowin
                          new_from_original(gutils, origops[2])
                          ]
        call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, shadowin, idx-1)
                              new_from_original(gutils, origops[2])
                              ]
            call_samefunc_with_inverted_bundles!(B, gutils, orig, args, [API.VT_Shadow, API.VT_Primal], #=lookup=#false)
        end
    end
    return false
end

function jl_array_sizehint_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_sizehint_fwd(B, orig, gutils, normalR, shadowR)
end

function jl_array_sizehint_rev(B, orig, gutils, tape)
    return nothing
end

function jl_unhandled_fwd(B, orig, gutils, normalR, shadowR)
    newo = new_from_original(gutils, orig)
    origops = collect(operands(orig))
    err = emit_error(B, orig, "Enzyme: unhandled forward for "*string(origops[end]))
    API.moveBefore(newo, err, C_NULL)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing

    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            position!(B, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(normal)))
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end
function jl_unhandled_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
  jl_unhandled_fwd(B, orig, gutils, normalR, shadowR)
end
function jl_unhandled_rev(B, orig, gutils, tape)
    return nothing
end

function get_binding_or_error_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled forward for jl_get_binding_or_error")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)

    if unsafe_load(shadowR) != C_NULL
    	valTys = API.CValueType[API.VT_Primal, API.VT_Primal]
		args = [new_from_original(gutils, operands(orig)[1]), new_from_original(gutils, operands(orig)[2])]
        normal = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, #=lookup=#false)
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

function get_binding_or_error_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_get_binding_or_error")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    if unsafe_load(shadowR) != C_NULL
    	valTys = API.CValueType[API.VT_Primal, API.VT_Primal]
		args = [new_from_original(gutils, operands(orig)[1]), new_from_original(gutils, operands(orig)[2])]
        normal = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, #=lookup=#false)
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

function get_binding_or_error_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: unhandled reverse for jl_get_binding_or_error")
    return nothing
end

function finalizer_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th or jl_gc_add_ptr_finalizer")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function finalizer_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    # err = emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
    # newo = new_from_original(gutils, orig)
    # API.moveBefore(newo, err, B)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end
    return false
end

function finalizer_rev(B, orig, gutils, tape)
    # emit_error(B, orig, "Enzyme: unhandled reverse for jl_gc_add_finalizer_th")
    return nothing
end


function register_handler!(variants, augfwd_handler, rev_handler, fwd_handler=nothing)
    for variant in variants
        if augfwd_handler !== nothing && rev_handler !== nothing
            API.EnzymeRegisterCallHandler(variant, augfwd_handler, rev_handler)
        end
        if fwd_handler !== nothing
            API.EnzymeRegisterFwdCallHandler(variant, fwd_handler)
        end
    end
end

macro augfunc(f)
   :(@cfunction((B, OrigCI, gutils, normalR, shadowR, tapeR) -> begin
     UInt8($f(LLVM.IRBuilder(B), LLVM.CallInst(OrigCI), GradientUtils(gutils), normalR, shadowR, tapeR)::Bool)
    end, UInt8, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})
    ))
end

macro revfunc(f)
   :(@cfunction((B, OrigCI, gutils, tape) -> begin
     $f(LLVM.IRBuilder(B), LLVM.CallInst(OrigCI), GradientUtils(gutils), tape == C_NULL ? nothing : LLVM.Value(tape))
    end,  Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)
    ))
end

macro fwdfunc(f)
   :(@cfunction((B, OrigCI, gutils, normalR, shadowR) -> begin
     UInt8($f(LLVM.IRBuilder(B), LLVM.CallInst(OrigCI), GradientUtils(gutils), normalR, shadowR)::Bool)
    end, UInt8, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})
    ))
end

@inline function register_llvm_rules()
    register_handler!(
        ("julia.call",),
        @augfunc(jlcall_augfwd),
        @revfunc(jlcall_rev),
        @fwdfunc(jlcall_fwd),
    )
    register_handler!(
        ("julia.call2",),
        @augfunc(jlcall2_augfwd),
        @revfunc(jlcall2_rev),
        @fwdfunc(jlcall2_fwd),
    )
    register_handler!(
        ("jl_apply_generic", "ijl_apply_generic"),
        @augfunc(generic_augfwd),
        @revfunc(generic_rev),
        @fwdfunc(generic_fwd),
    )
    register_handler!(
        ("jl_invoke", "ijl_invoke", "jl_f_invoke"),
        @augfunc(invoke_augfwd),
        @revfunc(invoke_rev),
        @fwdfunc(invoke_fwd),
    )
    register_handler!(
        ("jl_f__apply_latest", "jl_f__call_latest"),
        @augfunc(apply_latest_augfwd),
        @revfunc(apply_latest_rev),
        @fwdfunc(apply_latest_fwd),
    )
    register_handler!(
        ("jl_threadsfor",),
        @augfunc(threadsfor_augfwd),
        @revfunc(threadsfor_rev),
        @fwdfunc(threadsfor_fwd),
    )
    register_handler!(
        ("jl_pmap",),
        @augfunc(pmap_augfwd),
        @revfunc(pmap_rev),
        @fwdfunc(pmap_fwd),
    )
    register_handler!(
        ("jl_new_task", "ijl_new_task"),
        @augfunc(newtask_augfwd),
        @revfunc(newtask_rev),
        @fwdfunc(newtask_fwd),
    )
    register_handler!(
        ("jl_set_task_threadpoolid", "ijl_set_task_threadpoolid"),
        @augfunc(set_task_tid_augfwd),
        @revfunc(set_task_tid_rev),
        @fwdfunc(set_task_tid_fwd),
    )
    register_handler!(
        ("jl_enq_work",),
        @augfunc(enq_work_augfwd),
        @revfunc(enq_work_rev),
        @fwdfunc(enq_work_fwd)
    )
    register_handler!(
        ("enzyme_custom",),
        @augfunc(enzyme_custom_augfwd),
        @revfunc(enzyme_custom_rev),
        @fwdfunc(enzyme_custom_fwd)
    )
    register_handler!(
        ("jl_wait",),
        @augfunc(wait_augfwd),
        @revfunc(wait_rev),
        @fwdfunc(wait_fwd),
    )
    register_handler!(
        ("jl_","jl_breakpoint"),
        @augfunc(noop_augfwd),
        @revfunc(duplicate_rev),
        @fwdfunc(noop_fwd),
    )
    register_handler!(
        ("jl_array_copy","ijl_array_copy"),
        @augfunc(arraycopy_augfwd),
        @revfunc(arraycopy_rev),
        @fwdfunc(arraycopy_fwd),
    )
    register_handler!(
        ("jl_reshape_array","ijl_reshape_array"),
        @augfunc(arrayreshape_augfwd),
        @revfunc(arrayreshape_rev),
        @fwdfunc(arrayreshape_fwd),
    )
    register_handler!(
        ("jl_f_setfield","ijl_f_setfield"),
        @augfunc(setfield_augfwd),
        @revfunc(setfield_rev),
        @fwdfunc(setfield_fwd),
    )
    register_handler!(
        ("jl_box_float32","ijl_box_float32", "jl_box_float64", "ijl_box_float64"),
        @augfunc(boxfloat_augfwd),
        @revfunc(boxfloat_rev),
        @fwdfunc(boxfloat_fwd),
    )
    register_handler!(
        ("jl_f_tuple","ijl_f_tuple"),
        @augfunc(f_tuple_augfwd),
        @revfunc(f_tuple_rev),
        @fwdfunc(f_tuple_fwd),
    )
    register_handler!(
        ("jl_eqtable_get","ijl_eqtable_get"),
        @augfunc(eqtableget_augfwd),
        @revfunc(eqtableget_rev),
        @fwdfunc(eqtableget_fwd),
    )
    register_handler!(
        ("jl_eqtable_put","ijl_eqtable_put"),
        @augfunc(eqtableput_augfwd),
        @revfunc(eqtableput_rev),
        @fwdfunc(eqtableput_fwd),
    )
    register_handler!(
        ("jl_idtable_rehash","ijl_idtable_rehash"),
        @augfunc(idtablerehash_augfwd),
        @revfunc(idtablerehash_rev),
        @fwdfunc(idtablerehash_fwd),
    )
    register_handler!(
        ("jl_f__apply_iterate","ijl_f__apply_iterate"),
        @augfunc(apply_iterate_augfwd),
        @revfunc(apply_iterate_rev),
        @fwdfunc(apply_iterate_fwd),
    )
    register_handler!(
        ("jl_f__svec_ref","ijl_f__svec_ref"),
        @augfunc(f_svec_ref_augfwd),
        @revfunc(f_svec_ref_rev),
        @fwdfunc(f_svec_ref_fwd),
    )
    register_handler!(
        ("jl_new_structv","ijl_new_structv"),
        @augfunc(new_structv_augfwd),
        @revfunc(new_structv_rev),
        @fwdfunc(new_structv_fwd),
    )
    register_handler!(
        ("jl_new_structt","ijl_new_structt"),
        @augfunc(new_structt_augfwd),
        @revfunc(new_structt_rev),
        @fwdfunc(new_structt_fwd),
    )
    register_handler!(
        ("jl_get_binding_or_error", "ijl_get_binding_or_error"),
        @augfunc(get_binding_or_error_augfwd),
        @revfunc(get_binding_or_error_rev),
        @fwdfunc(get_binding_or_error_fwd),
    )
    register_handler!(
        ("jl_gc_add_finalizer_th","ijl_gc_add_finalizer_th", "jl_gc_add_ptr_finalizer","ijl_gc_add_ptr_finalizer"),
        @augfunc(finalizer_augfwd),
        @revfunc(finalizer_rev),
        @fwdfunc(finalizer_fwd),
    )
    register_handler!(
        ("jl_array_grow_end","ijl_array_grow_end"),
        @augfunc(jl_array_grow_end_augfwd),
        @revfunc(jl_array_grow_end_rev),
        @fwdfunc(jl_array_grow_end_fwd),
    )
    register_handler!(
        ("jl_array_del_end","ijl_array_del_end"),
        @augfunc(jl_array_del_end_augfwd),
        @revfunc(jl_array_del_end_rev),
        @fwdfunc(jl_array_del_end_fwd),
    )
    register_handler!(
        ("jl_f_getfield","ijl_f_getfield"),
        @augfunc(jl_getfield_augfwd),
        @revfunc(jl_getfield_rev),
        @fwdfunc(jl_getfield_fwd),
    )
    register_handler!(
        ("ijl_get_nth_field_checked","jl_get_nth_field_checked"),
        @augfunc(jl_nthfield_augfwd),
        @revfunc(jl_nthfield_rev),
        @fwdfunc(jl_nthfield_fwd),
    )
    register_handler!(
        ("jl_array_sizehint","ijl_array_sizehint"),
        @augfunc(jl_array_sizehint_augfwd),
        @revfunc(jl_array_sizehint_rev),
        @fwdfunc(jl_array_sizehint_fwd),
    )
    register_handler!(
        ("jl_array_ptr_copy","ijl_array_ptr_copy"),
        @augfunc(jl_array_ptr_copy_augfwd),
        @revfunc(jl_array_ptr_copy_rev),
        @fwdfunc(jl_array_ptr_copy_fwd),
    )
    register_handler!(
        (),
        @augfunc(jl_unhandled_augfwd),
        @revfunc(jl_unhandled_rev),
        @fwdfunc(jl_unhandled_fwd),
    )
end
