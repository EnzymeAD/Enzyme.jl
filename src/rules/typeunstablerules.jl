
function common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)

    world = enzyme_extract_world(LLVM.parent(position(B)))

    @assert is_constant_value(gutils, origops[offset])
    icvs = [is_constant_value(gutils, v) for v in origops[offset+1:end-1]]
    abs = [abs_typeof(v, true) for v in origops[offset+1:end-1]]

    legal = true
    for (icv, (found, typ)) in zip(icvs, abs)
        if icv
            if found
                if guaranteed_const_nongen(typ, world)
                    continue
                end
            end
            legal = false
        end
    end

    # if all(icvs)
    #     shadowres = new_from_original(gutils, orig)
    #     if width != 1
    #         shadowres2 = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(shadowres))))
    #         for idx in 1:width
    #             shadowres2 = insert_value!(B, shadowres2, shadowres, idx-1)
    #         end
    #         shadowres = shadowres2
    #     end
    #     unsafe_store!(shadowR, shadowres.ref)
    #     return false
    # end
    if !legal
        emit_error(B, orig, "Enzyme: Not yet implemented, mixed activity for jl_new_struct constants="*string(icvs)*" "*string(orig)*" "*string(abs)*" "*string([v for v in origops[offset+1:end-1]]))
    end

    shadowsin = LLVM.Value[invert_pointer(gutils, o, B) for o in origops[offset:end-1] ]
    if width == 1
        if offset != 1
            pushfirst!(shadowsin, origops[1])
        end
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), shadowsin)
        callconv!(shadowres, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              extract_value!(B, s, idx-1) for s in shadowsin
                              ]
            if offset != 1
                pushfirst!(args, origops[1])
            end
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end
function common_newstructv_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
end

function error_if_active_newstruct(::Type{T}, ::Type{Y}) where {T, Y}
    seen = ()
    areg = active_reg_inner(T, seen, nothing, #=justActive=#Val(true))
    if areg == ActiveState
        throw(AssertionError("Found unhandled active variable ($T) in reverse mode of jl_newstruct constructor for $Y"))
    end
    nothing
end

function common_newstructv_rev(offset, B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return true
    end
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, API.DEM_ReverseModePrimal)
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

	if !needsShadow
		return
	end
    
    origops = collect(operands(orig))
    width = get_width(gutils)

    world = enzyme_extract_world(LLVM.parent(position(B)))

    @assert is_constant_value(gutils, origops[offset])
    icvs = [is_constant_value(gutils, v) for v in origops[offset+1:end-1]]
    abs = [abs_typeof(v, true) for v in origops[offset+1:end-1]]


    ty = new_from_original(gutils, origops[offset])
    for v in origops[offset+1:end-1]
        emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active_newstruct), emit_jltypeof!(B, lookup_value(gutils, new_from_original(gutils, v), B)), ty])
    end

    return nothing
end

function common_f_tuple_fwd(offset, B, orig, gutils, normalR, shadowR)
    common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
end
function common_f_tuple_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    common_f_tuple_fwd(offset, B, orig, gutils, normalR, shadowR)
end

function common_f_tuple_rev(offset, B, orig, gutils, tape)
    # This function allocates a new return which returns a pointer, thus this instruction itself cannot transfer
    # derivative info, only create a shadow pointer, which is handled by the forward pass.
    return nothing
end


function f_tuple_fwd(B, orig, gutils, normalR, shadowR)
    common_f_tuple_fwd(1, B, orig, gutils, normalR, shadowR)
end

function f_tuple_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_f_tuple_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function f_tuple_rev(B, orig, gutils, tape)
    common_f_tuple_rev(1, B, orig, gutils, tape)
    return nothing
end

function new_structv_fwd(B, orig, gutils, normalR, shadowR)
    common_newstructv_fwd(1, B, orig, gutils, normalR, shadowR)
end

function new_structv_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_newstructv_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function new_structv_rev(B, orig, gutils, tape)
    common_apply_latest_rev(1, B, orig, gutils, tape)
    return nothing
end

function new_structt_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)

    @assert is_constant_value(gutils, origops[1])
    if is_constant_value(gutils, origops[2])
        emit_error(B, orig, "Enzyme: Not yet implemented, mixed activity for jl_new_struct_t"*string(orig))
    end

    shadowsin = invert_pointer(gutils, origops[2], B)
    if width == 1
        vals = [new_from_original(gutils, origops[1]), shadowsin]
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), vals)
        callconv!(shadowres, callconv(orig))
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx in 1:width
            vals = [new_from_original(gutils, origops[1]), extract_value!(B, shadowsin, idx-1)]
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end
function new_structt_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    new_structt_fwd(B, orig, gutils, normalR, shadowR)
end

function new_structt_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return true
    end
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, API.DEM_ReverseModePrimal)
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

	if !needsShadow
		return
	end
    emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_new_structt "*string(orig))
    return nothing
end

function common_jl_getfield_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    origops = collect(operands(orig))[offset:end]
    width = get_width(gutils)
    if !is_constant_value(gutils, origops[2])
        shadowin = invert_pointer(gutils, origops[2], B)
        if width == 1
            args = LLVM.Value[new_from_original(gutils, origops[1]), shadowin]
            for a in origops[3:end-1]
                push!(args, new_from_original(gutils, a))
            end
            if offset != 1
                pushfirst!(args, first(operands(orig)))
            end
            shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(shadowres, callconv(orig))
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx in 1:width
                args = LLVM.Value[new_from_original(gutils, origops[1]), extract_value!(B, shadowin, idx-1)]
                for a in origops[3:end-1]
                    push!(args, new_from_original(gutils, a))
                end
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
                end
                tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
                callconv!(tmp, callconv(orig))
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = new_from_original(gutils, orig)
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

function rt_jl_getfield_aug(dptr::T, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    res = if dptr isa Base.RefValue
	   Base.getfield(dptr[], symname)
    else
	   Base.getfield(dptr, symname)
    end
    RT = Core.Typeof(res)
    if active_reg(RT)
        if length(dptrs) == 0
            return Ref{RT}(make_zero(res))
        else
            return ( (Ref{RT}(make_zero(res)) for _ in 1:(1+length(dptrs)))..., )
        end
    else
        if length(dptrs) == 0
            return res
        else
            return (res, (getfield(dv, symname) for dv in dptrs)...)
        end
    end
end

function idx_jl_getfield_aug(dptr::T, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    res = if dptr isa Base.RefValue
	   Base.getfield(dptr[], symname+1)
    else
	   Base.getfield(dptr, symname+1)
    end
    RT = Core.Typeof(res)
    if active_reg(RT)
        if length(dptrs) == 0
            return Ref{RT}(make_zero(res))
        else
            return ( (Ref{RT}(make_zero(res)) for _ in 1:(1+length(dptrs)))..., )
        end
    else
        if length(dptrs) == 0
            return res
        else
            return (res, (getfield(dv, symname) for dv in dptrs)...)
        end
    end
end

function rt_jl_getfield_rev(dptr::T, dret, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    cur = if dptr isa Base.RefValue
	   getfield(dptr[], symname)
    else
	   getfield(dptr, symname)
    end

    RT = Core.Typeof(cur)
    if active_reg(RT) && !isconst
        if length(dptrs) == 0
            setfield!(dptr, symname, recursive_add(cur, dret[]))
        else
            setfield!(dptr, symname, recursive_add(cur, dret[1][]))
            for i in 1:length(dptrs)
                setfield!(dptrs[i], symname, recursive_add(cur, dret[1+i][]))
            end
        end
    end
    return nothing
end
function idx_jl_getfield_rev(dptr::T, dret, ::Type{Val{symname}}, ::Val{isconst}, dptrs...) where {T, symname, isconst}
    cur = if dptr isa Base.RefValue
	   Base.getfield(dptr[], symname+1)
    else
	   Base.getfield(dptr, symname+1)
    end

    RT = Core.Typeof(cur)
    if active_reg(RT) && !isconst
        if length(dptrs) == 0
            setfield!(dptr, symname+1, recursive_add(cur, dret[]))
        else
            setfield!(dptr, symname+1, recursive_add(cur, dret[1][]))
            for i in 1:length(dptrs)
                setfield!(dptrs[i], symname+1, recursive_add(cur, dret[1+i][]))
            end
        end
    end
    return nothing
end

function common_jl_getfield_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    ops = collect(operands(orig))[offset:end]
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if !is_constant_value(gutils, ops[2])
        inp = invert_pointer(gutils, ops[2], B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inps = [new_from_original(gutils, ops[2])]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    sym = new_from_original(gutils, ops[3])
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(rt_jl_getfield_aug))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if width == 1
        shadowres = cal
    else
        AT = LLVM.ArrayType(T_prjlvalue, Int(width))

        forgep = cal
        if !is_constant_value(gutils, ops[2])
            forgep = LLVM.addrspacecast!(B, forgep, LLVM.PointerType(T_jlvalue, Derived))
            forgep = LLVM.pointercast!(B, forgep, LLVM.PointerType(AT, Derived))
        end    

        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for i in 1:width
            if !is_constant_value(gutils, ops[2])
                gep = LLVM.inbounds_gep!(B, AT, forgep, [LLVM.ConstantInt(0), LLVM.ConstantInt(i-1)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
            else
                ld = forgep
            end
            shadow = insert_value!(B, shadow, ld, i-1)
        end
        shadowres = shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    unsafe_store!(tapeR, cal.ref)
    return false
end

function common_jl_getfield_rev(offset, B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return
    end
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, API.DEM_ReverseModePrimal)
    if needsShadowP[] == 0
        return
    end

    ops = collect(operands(orig))[offset:end]
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    if !is_constant_value(gutils, ops[2])
        inp = invert_pointer(gutils, ops[2], B)
        inp = lookup_value(gutils, inp, B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inp = new_from_original(gutils, ops[2])
        inp = lookup_value(gutils, inp, B)
        inps = [inp]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    push!(vals, tape)

    sym = new_from_original(gutils, ops[3])
    sym = lookup_value(gutils, sym, B)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(rt_jl_getfield_rev))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)
    return nothing
end

function jl_nthfield_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)
    if !is_constant_value(gutils, origops[1])
        shadowin = invert_pointer(gutils, origops[1], B)
        if width == 1
            args = LLVM.Value[
                              shadowin
                              new_from_original(gutils, origops[2])
                              ]
            shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(shadowres, callconv(orig))
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  new_from_original(gutils, origops[2])
                                  ]
                tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
                callconv!(tmp, callconv(orig))
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = new_from_original(gutils, orig)
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
function jl_nthfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    ops = collect(operands(orig))
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if !is_constant_value(gutils, ops[1])
        inp = invert_pointer(gutils, ops[1], B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inps = [new_from_original(gutils, ops[1])]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    sym = new_from_original(gutils, ops[2])
    sym = (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, sym)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(idx_jl_getfield_aug))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if width == 1
        shadowres = cal
    else
        AT = LLVM.ArrayType(T_prjlvalue, Int(width))
        forgep = cal
        if !is_constant_value(gutils, ops[1])
            forgep = LLVM.addrspacecast!(B, forgep, LLVM.PointerType(T_jlvalue, Derived))
            forgep = LLVM.pointercast!(B, forgep, LLVM.PointerType(AT, Derived))
        end    

        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for i in 1:width
            if !is_constant_value(gutils, ops[1])
                gep = LLVM.inbounds_gep!(B, AT, forgep, [LLVM.ConstantInt(0), LLVM.ConstantInt(i-1)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
            else
                ld = forgep
            end
            shadow = insert_value!(B, shadow, ld, i-1)
        end
        shadowres = shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    unsafe_store!(tapeR, cal.ref)
    return false
end
function jl_nthfield_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return
    end

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, API.DEM_ReverseModePrimal)
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

	if !needsShadow
		return
	end

    ops = collect(operands(orig))
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    
    if !is_constant_value(gutils, ops[1])
        inp = invert_pointer(gutils, ops[1], B)
        inp = lookup_value(gutils, inp, B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w in 1:width
                push!(inps, extract_value!(B, inp, w-1))
            end
        end
    else
        inp = new_from_original(gutils, ops[1])
        inp = lookup_value(gutils, inp, B)
        inps = [inp]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    push!(vals, tape)

    sym = new_from_original(gutils, ops[2])
    sym = lookup_value(gutils, sym, B)
    sym = (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, sym)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(Val(is_constant_value(gutils, orig))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(idx_jl_getfield_rev))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)
    return nothing
end

function jl_getfield_fwd(B, orig, gutils, normalR, shadowR)
    common_jl_getfield_fwd(1, B, orig, gutils, normalR, shadowR)
end
function jl_getfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_jl_getfield_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end
function jl_getfield_rev(B, orig, gutils, tape)
    common_jl_getfield_rev(1, B, orig, gutils, tape)
end

function common_setfield_fwd(offset, B, orig, gutils, normalR, shadowR)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    origops = collect(operands(orig))[offset:end]
    if !is_constant_value(gutils, origops[4])
        width = get_width(gutils)

        shadowin = if !is_constant_value(gutils, origops[2])
            invert_pointer(gutils, origops[2], B)
        else
            new_from_original(gutils, origops[2])
        end

        shadowout = invert_pointer(gutils, origops[4], B)
        if width == 1
            args = LLVM.Value[
                              new_from_original(gutils, origops[1])
                              shadowin
                              new_from_original(gutils, origops[3])
                              shadowout
                              ]
            valTys = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal, API.VT_Shadow]
            if offset != 1
                pushfirst!(args, first(operands(orig)))
                pushfirst!(valTys, API.VT_Primal)
            end

            shadowres = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, #=lookup=#false)
            callconv!(shadowres, callconv(orig))
        else
            for idx in 1:width
                args = LLVM.Value[
                                  new_from_original(gutils, origops[1])
                                  extract_value!(B, shadowin, idx-1)
                                  new_from_original(gutils, origops[3])
                                  extract_value!(B, shadowout, idx-1)
                                  ]
                valTys = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal, API.VT_Shadow]
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
                    pushfirst!(valTys, API.VT_Primal)
                end

                tmp = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, #=lookup=#false)

                callconv!(tmp, callconv(orig))
            end
        end
    end
    return false
end

function common_setfield_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_f_setfield")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_setfield_rev(offset, B, orig, gutils, tape)
  emit_error(B, orig, "Enzyme: unhandled reverse for jl_f_setfield")
  return nothing
end


function setfield_fwd(B, orig, gutils, normalR, shadowR)
    common_setfield_fwd(1, B, orig, gutils, normalR, shadowR)
end

function setfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_setfield_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function setfield_rev(B, orig, gutils, tape)
    common_setfield_rev(1, B, orig, gutils, tape)
end



function common_f_svec_ref_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_f__svec_ref")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function error_if_differentiable(::Type{T}) where T
    seen = ()
    areg = active_reg_inner(T, seen, nothing, #=justActive=#Val(true))
    if areg != AnyState
        throw(AssertionError("Found unhandled differentiable variable in jl_f_svec_ref $T"))
    end
    nothing
end

function common_f_svec_ref_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    width = get_width(gutils)

    origmi, origh, origkey = operands(orig)[offset:end-1]

    shadowh = invert_pointer(gutils, origh, B)
        
    newvals = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal]

    if offset != 1
        pushfirst!(newvals, API.VT_Primal)
    end
        
    errfn = if is_constant_value(gutils, origh)
        error_if_differentiable
    else
        error_if_active
    end
    
    mi = new_from_original(gutils, origmi)

    shadowres = if width == 1
        newops = LLVM.Value[mi, shadowh, new_from_original(gutils, origkey)]
        if offset != 1
            pushfirst!(newops, operands(orig)[1])
        end
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
        callconv!(cal, callconv(orig))
   
    
        emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(errfn), emit_jltypeof!(B, cal)])
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j in 1:width
            newops = LLVM.Value[mi, extract_value!(B, shadowh, j-1), new_from_original(gutils, origkey)]
            if offset != 1
                pushfirst!(newops, operands(orig)[1])
            end
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
            callconv!(cal, callconv(orig))
            emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(errfn), emit_jltypeof!(B, cal)])
            shadow = insert_value!(B, shadow, cal, j-1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)

    return false
end

function common_f_svec_ref_rev(offset, B, orig, gutils, tape)
    return nothing
end

function f_svec_ref_fwd(B, orig, gutils, normalR, shadowR)
    common_f_svec_ref_fwd(1, B, orig, gutils, normalR, shadowR)
    return nothing
end

function f_svec_ref_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_f_svec_ref_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
    return nothing
end

function f_svec_ref_rev(B, orig, gutils, tape)
    common_f_svec_ref_rev(1, B, orig, gutils, tape)
    return nothing
end
