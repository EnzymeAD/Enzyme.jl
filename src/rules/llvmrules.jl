macro register_aug(expr)
    decl = string(expr.args[1])
    name = decl[1:prevind(decl, findfirst('(', decl))]
    cname = name * "_cfunc"
    name = Symbol(name)
    cname = Symbol(cname)

    expr2 = :(@inline $expr)
    res = quote
        function $cname(
            B::LLVM.API.LLVMBuilderRef,
            OrigCI::LLVM.API.LLVMValueRef,
            gutils::API.EnzymeGradientUtilsRef,
            normalR::Ptr{LLVM.API.LLVMValueRef},
            shadowR::Ptr{LLVM.API.LLVMValueRef},
            tapeR::Ptr{LLVM.API.LLVMValueRef},
        )::UInt8
            return UInt8(
                $name(
                    LLVM.IRBuilder(B),
                    LLVM.CallInst(OrigCI),
                    GradientUtils(gutils),
                    normalR,
                    shadowR,
                    tapeR,
                )::Bool,
            )
        end
    end
    return Expr(:block, esc(expr2), esc(res))
end

macro register_rev(expr)
    decl = string(expr.args[1])
    name = decl[1:prevind(decl, findfirst('(', decl))]
    cname = name * "_cfunc"

    name = Symbol(name)
    cname = Symbol(cname)
    expr2 = :(@inline $expr)
    res = quote
        function $cname(
            B::LLVM.API.LLVMBuilderRef,
            OrigCI::LLVM.API.LLVMValueRef,
            gutils::API.EnzymeGradientUtilsRef,
            tape::LLVM.API.LLVMValueRef,
        )::Cvoid
            $name(
                LLVM.IRBuilder(B),
                LLVM.CallInst(OrigCI),
                GradientUtils(gutils),
                tape == C_NULL ? nothing : LLVM.Value(tape),
            )
            return
        end
    end
    return Expr(:block, esc(expr2), esc(res))
end

macro register_fwd(expr)
    decl = string(expr.args[1])
    name = decl[1:prevind(decl, findfirst('(', decl))]
    cname = name * "_cfunc"
    name = Symbol(name)
    cname = Symbol(cname)
    expr2 = :(@inline $expr)
    res = quote
        function $cname(
            B::LLVM.API.LLVMBuilderRef,
            OrigCI::LLVM.API.LLVMValueRef,
            gutils::API.EnzymeGradientUtilsRef,
            normalR::Ptr{LLVM.API.LLVMValueRef},
            shadowR::Ptr{LLVM.API.LLVMValueRef},
        )::UInt8
            return UInt8(
                $name(
                    LLVM.IRBuilder(B),
                    LLVM.CallInst(OrigCI),
                    GradientUtils(gutils),
                    normalR,
                    shadowR,
                )::Bool,
            )
        end
    end
    return Expr(:block, esc(expr2), esc(res))
end

macro register_diffuse(expr)
    decl = string(expr.args[1])
    name = decl[1:prevind(decl, findfirst('(', decl))]
    cname = name * "_cfunc"
    name = Symbol(name)
    cname = Symbol(cname)
    expr2 = :(@inline $expr)
    res = quote
        function $cname(
            OrigCI::LLVM.API.LLVMValueRef,
            gutils::API.EnzymeGradientUtilsRef,
            val::LLVM.API.LLVMValueRef,
            shadow::UInt8,
            mode::API.CDerivativeMode,
            useDefault::Ptr{UInt8},
        )::UInt8
            res = $name(
                LLVM.CallInst(OrigCI),
                GradientUtils(gutils),
                LLVM.Value(val),
                shadow != 0,
                mode,
            )::Tuple{Bool,Bool}
            unsafe_store!(useDefault, UInt8(res[2]))
            return UInt8(res[1])
        end
    end
    return Expr(:block, esc(expr2), esc(res))
end

include("customrules.jl")
include("jitrules.jl")
include("typeunstablerules.jl")
include("parallelrules.jl")

@register_fwd function jlcall_fwd(B, orig, gutils, normalR, shadowR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            return common_generic_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if in(
            name,
            (
                "ijl_f__apply_latest",
                "ijl_f__call_latest",
                "jl_f__apply_latest",
                "jl_f__call_latest",
            ),
        )
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
        if in(name, ("ijl_f_finalizer", "jl_f_finalizer"))
            return common_finalizer_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if any(
            map(
                k -> kind(k) == kind(StringAttribute("enzyme_inactive")),
                collect(function_attributes(F)),
            ),
        )
            return true
        end
    end

    err = emit_error(
        B,
        orig,
        "Enzyme: jl_call calling convention not implemented in forward for " * string(orig),
    )

    newo = new_from_original(gutils, orig)

    API.moveBefore(newo, err, B)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
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

@register_aug function jlcall_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            return common_generic_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(
            name,
            (
                "ijl_f__apply_latest",
                "ijl_f__call_latest",
                "jl_f__apply_latest",
                "jl_f__call_latest",
            ),
        )
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
        if in(name, ("ijl_f_setfield", "jl_f_setfield"))
            return common_setfield_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__apply_iterate", "jl_f__apply_iterate"))
            return common_apply_iterate_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f__svec_ref", "jl_f__svec_ref"))
            return common_f_svec_ref_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if in(name, ("ijl_f_finalizer", "jl_f_finalizer"))
            return common_finalizer_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if any(
            map(
                k -> kind(k) == kind(StringAttribute("enzyme_inactive")),
                collect(function_attributes(F)),
            ),
        )
            return true
        end
    end

    err = emit_error(
        B,
        orig,
        "Enzyme: jl_call calling convention not implemented in aug_forward for " *
        string(orig),
    )
    newo = new_from_original(gutils, orig)

    API.moveBefore(newo, err, B)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
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

@register_rev function jlcall_rev(B, orig, gutils, tape)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_apply_generic", "jl_apply_generic"))
            common_generic_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if in(
            name,
            (
                "ijl_f__apply_latest",
                "ijl_f__call_latest",
                "jl_f__apply_latest",
                "jl_f__call_latest",
            ),
        )
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
        if in(name, ("ijl_f_finalizer", "jl_f_finalizer"))
            common_finalizer_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if any(
            map(
                k -> kind(k) == kind(StringAttribute("enzyme_inactive")),
                collect(function_attributes(F)),
            ),
        )
            return nothing
        end
    end

    emit_error(
        B,
        orig,
        "Enzyme: jl_call calling convention not implemented in reverse for " * string(orig),
    )

    return nothing
end

@register_fwd function jlcall2_fwd(B, orig, gutils, normalR, shadowR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            return common_invoke_fwd(2, B, orig, gutils, normalR, shadowR)
        end
        if any(
            map(
                k -> kind(k) == kind(StringAttribute("enzyme_inactive")),
                collect(function_attributes(F)),
            ),
        )
            return true
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return false
end

@register_aug function jlcall2_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            return common_invoke_augfwd(2, B, orig, gutils, normalR, shadowR, tapeR)
        end
        if any(
            map(
                k -> kind(k) == kind(StringAttribute("enzyme_inactive")),
                collect(function_attributes(F)),
            ),
        )
            return true
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return false
end

@register_rev function jlcall2_rev(B, orig, gutils, tape)
    F = operands(orig)[1]
    if isa(F, LLVM.Function)
        name = LLVM.name(F)
        if in(name, ("ijl_invoke", "jl_invoke"))
            common_invoke_rev(2, B, orig, gutils, tape)
            return nothing
        end
        if any(
            map(
                k -> kind(k) == kind(StringAttribute("enzyme_inactive")),
                collect(function_attributes(F)),
            ),
        )
            return nothing
        end
    end

    @assert false "jl_call calling convention not implemented yet", orig

    return nothing
end


@register_fwd function noop_fwd(B, orig, gutils, normalR, shadowR)
    return true
end

@register_aug function noop_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    return true
end

@register_rev function duplicate_rev(B, orig, gutils, tape)
    newg = new_from_original(gutils, orig)

    real_ops = collect(operands(orig))[1:end-1]
    ops = [lookup_value(gutils, new_from_original(gutils, o), B) for o in real_ops]

    c = call_samefunc_with_inverted_bundles!(
        B,
        gutils,
        orig,
        ops,
        [API.VT_Primal for _ in ops],
        false,
    ) #=lookup=#
    callconv!(c, callconv(orig))

    return nothing
end

@register_fwd function arraycopy_fwd(B, orig, gutils, normalR, shadowR)
    ctx = LLVM.context(orig)

    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    origops = LLVM.operands(orig)

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)

    i8 = LLVM.IntType(8)
    algn = 0

    shadowres =
        UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
    for idx = 1:width
        ev = if width == 1
            shadowin
        else
            extract_value!(B, shadowin, idx - 1)
        end

        callv = call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            [ev],
            [API.VT_Shadow],
            false,
        ) #=lookup=#
        if is_constant_value(gutils, origops[1])
            elSize = get_array_elsz(B, ev)
            elSize = LLVM.zext!(B, elSize, LLVM.IntType(8 * sizeof(Csize_t)))
            len = get_array_len(B, ev)
            length = LLVM.mul!(B, len, elSize)
            bt = GPUCompiler.backtrace(orig)
            btstr = sprint() do io
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
            end
            GPUCompiler.@safe_warn "TODO forward zero-set of arraycopy used memset rather than runtime type $btstr"
            LLVM.memset!(
                B,
                get_array_data(B, callv),
                LLVM.ConstantInt(i8, 0, false),
                length,
                algn,
            )
        end
        if get_runtime_activity(gutils)
            prev = new_from_original(gutils, orig)
            callv = LLVM.select!(
                B,
                LLVM.icmp!(
                    B,
                    LLVM.API.LLVMIntNE,
                    ev,
                    new_from_original(gutils, origops[1]),
                ),
                callv,
                prev,
            )
            if idx == 1
                API.moveBefore(prev, callv, B)
            end
        end
        shadowres = if width == 1
            callv
        else
            insert_value!(B, shadowres, callv, idx - 1)
        end
    end

    unsafe_store!(shadowR, shadowres.ref)
    return false
end

# Optionally takes a length if requested
# If this is a memory, pass memoryptr=<underlying data>
function arraycopy_common(fwd, B, orig, shadowsrc, gutils, shadowdst; len=nothing, memoryptr=nothing)
	memory = memoryptr != nothing
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0
    if !needsShadow
        return nothing
    end

    if !fwd
        shadowdst = invert_pointer(gutils, orig, B)
    end

    tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, orig))
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    dl = string(LLVM.datalayout(mod))
	# memory stores the data pointer after a length
	if memory
    	API.EnzymeTypeTreeLookupEq(tt, 2*sizeof(Int), dl)
		API.EnzymeTypeTreeShiftIndiciesEq(tt, dl, sizeof(Int), sizeof(Int), 0)
	else
    	API.EnzymeTypeTreeLookupEq(tt, sizeof(Int), dl)
	end
    data0!(tt)
    ct = API.EnzymeTypeTreeInner0(tt)

    if ct == API.DT_Unknown
        # analyzer = API.EnzymeGradientUtilsTypeAnalyzer(gutils)
        # ip = API.EnzymeTypeAnalyzerToString(analyzer)
        # sval = Base.unsafe_string(ip)
        # API.EnzymeStringFree(ip)
        emit_error(
            B,
            orig,
            "Enzyme: Unknown concrete type in arraycopy_common. tt: " * string(tt)* " " * string(orig) * " " * string(abs_typeof(orig)),
        )
        return nothing
    end

    @assert ct != API.DT_Unknown
    ctx = LLVM.context(orig)
    secretty = API.EnzymeConcreteTypeIsFloat(ct)

    actualOp = new_from_original(gutils, shadowsrc)
    if fwd
        B0 = B
    elseif typeof(actualOp) <: LLVM.Argument
        B0 = LLVM.IRBuilder()
        position!(
            B0,
            first(
                instructions(
                    new_from_original(gutils, LLVM.entry(LLVM.parent(LLVM.parent(orig)))),
                ),
            ),
        )
    else
        B0 = LLVM.IRBuilder()
        nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(actualOp))
        while isa(nextInst, LLVM.PHIInst)
            nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(nextInst))
        end
		if len != nothing
			nextInst = new_from_original(gutils, orig)
		end
        position!(B0, nextInst)
    end

    elSize = if memory
		get_memory_elsz(B0, actualOp)
	else
		get_array_elsz(B0, actualOp)
	end

    elSize = LLVM.zext!(B0, elSize, LLVM.IntType(8 * sizeof(Csize_t)))

	if len == nothing
		if memory
			len = get_memory_len(B0, actualOp)
		else
			len = get_array_len(B0, actualOp)
		end
	elseif !fwd
        # len = lookup_value(gutils, len, B)
	end

	if memory
		length = LLVM.mul!(B0, len, elSize)
	else
		length = LLVM.mul!(B0, len, elSize)
	end

    isVolatile = LLVM.ConstantInt(LLVM.IntType(1), 0)

    # forward pass copy already done by underlying call
    allowForward = false
    intrinsic = LLVM.Intrinsic("llvm.memcpy").id

    if !fwd
        shadowdst = lookup_value(gutils, shadowdst, B)
    end


	lookup_src = true

	if memory
		if fwd
			shadowsrc = inttoptr!(B, memoryptr, LLVM.PointerType(LLVM.IntType(8)))
			lookup_src = false
		else
			shadowsrc = invert_pointer(gutils, shadowsrc, B)
			if !fwd
				shadowsrc = lookup_value(gutils, shadowsrc, B)
			end
		end
	else
		shadowsrc = invert_pointer(gutils, shadowsrc, B)
		if !fwd
			shadowsrc = lookup_value(gutils, shadowsrc, B)
		end
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

	for i = 1:width

		evsrc = if width == 1
			shadowsrc
		else
			extract_value!(B, shadowsrc, i - 1)
		end
		evdst = if width == 1
			shadowdst
		else
			extract_value!(B, shadowdst, i - 1)
		end

		# src already has done the lookup from the argument
		shadowsrc0 = if lookup_src
			if memory
				get_memory_data(B, evsrc)
			else
				get_array_data(B, evsrc)
			end
		else
			evsrc
		end

		shadowdst0 = if memory
			get_memory_data(B, evdst)
		else
			get_array_data(B, evdst)
		end

		if fwd && secretty != nothing
			LLVM.memset!(B, shadowdst0, LLVM.ConstantInt(i8, 0, false), length, algn)
		end

		API.sub_transfer(
			gutils,
			fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient,
			secretty,
			intrinsic,
			1,
			1,
			0,
			false,
			shadowdst0,
			false,
			shadowsrc0,
			length,
			isVolatile,
			orig,
			allowForward,
			!fwd,
		) #=shadowsLookedUp=#
	end

    return nothing
end

@register_aug function arraycopy_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    arraycopy_fwd(B, orig, gutils, normalR, shadowR)

    origops = LLVM.operands(orig)

    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
        shadowres = LLVM.Value(unsafe_load(shadowR))

        arraycopy_common(true, B, orig, origops[1], gutils, shadowres)
    end

    return false
end

@register_rev function arraycopy_rev(B, orig, gutils, tape)
    origops = LLVM.operands(orig)
    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
        arraycopy_common(false, B, orig, origops[1], gutils, nothing)
    end

    return nothing
end

@register_fwd function genericmemory_copy_slice_fwd(B, orig, gutils, normalR, shadowR)
    ctx = LLVM.context(orig)

    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    origops = LLVM.operands(orig)

    width = get_width(gutils)

    shadowin = invert_pointer(gutils, origops[1], B)
    shadowdata = invert_pointer(gutils, origops[2], B)
    len = new_from_original(gutils, origops[3])

    i8 = LLVM.IntType(8)
    algn = 0

    shadowres =
        UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
    for idx = 1:width
        ev = if width == 1
            shadowin
        else
            extract_value!(B, shadowin, idx - 1)
        end
        ev2 = if width == 1
            shadowdata
        else
            extract_value!(B, shadowdata, idx - 1)
        end
        callv = call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            [ev, ev2, len],
            [API.VT_Shadow, API.VT_Shadow, API.VT_Primal],
            false,
        ) #=lookup=#
        if is_constant_value(gutils, origops[1])
            elSize = get_array_elsz(B, ev)
            elSize = LLVM.zext!(B, elSize, LLVM.IntType(8 * sizeof(Csize_t)))
            length = LLVM.mul!(B, len, elSize)
            bt = GPUCompiler.backtrace(orig)
            btstr = sprint() do io
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
            end
            GPUCompiler.@safe_warn "TODO forward zero-set of memorycopy used memset rather than runtime type $btstr"
            LLVM.memset!(
                B,
                ev2,
                LLVM.ConstantInt(i8, 0, false),
                length,
                algn,
            )
        end
        if get_runtime_activity(gutils)
            prev = new_from_original(gutils, orig)
            callv = LLVM.select!(
                B,
                LLVM.icmp!(
                    B,
                    LLVM.API.LLVMIntNE,
                    ev,
                    new_from_original(gutils, origops[1]),
                ),
                callv,
                prev,
            )
            if idx == 1
                API.moveBefore(prev, callv, B)
            end
        end
        shadowres = if width == 1
            callv
        else
            insert_value!(B, shadowres, callv, idx - 1)
        end
    end

    unsafe_store!(shadowR, shadowres.ref)
    return false
end

@register_aug function genericmemory_copy_slice_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    genericmemory_copy_slice_fwd(B, orig, gutils, normalR, shadowR)

    origops = LLVM.operands(orig)

    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
        shadowres = LLVM.Value(unsafe_load(shadowR))

		len = new_from_original(gutils, origops[3])
		memoryptr = new_from_original(gutils, origops[2])
        arraycopy_common(true, B, orig, origops[1], gutils, shadowres; len, memoryptr)
    end

    return false
end

@register_rev function genericmemory_copy_slice_rev(B, orig, gutils, tape)
    origops = LLVM.operands(orig)
    if !is_constant_value(gutils, origops[1]) && !is_constant_value(gutils, orig)
		len = new_from_original(gutils, origops[3])
		memoryptr = new_from_original(gutils, origops[2])
        arraycopy_common(false, B, orig, origops[1], gutils, nothing; len, memoryptr)
    end

    return nothing
end

@register_fwd function arrayreshape_fwd(B, orig, gutils, normalR, shadowR)
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
        shadowres = call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            args,
            [API.VT_Primal, API.VT_Shadow, API.VT_Primal],
            false,
        ) #=lookup=#
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            args = LLVM.Value[
                new_from_original(gutils, origops[1])
                extract_value!(B, shadowin, idx - 1)
                new_from_original(gutils, origops[3])
            ]
            tmp = call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                args,
                [API.VT_Primal, API.VT_Shadow, API.VT_Primal],
                false,
            ) #=lookup=#
            shadowres = insert_value!(B, shadowres, tmp, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)

    return false
end

@register_aug function arrayreshape_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    arrayreshape_fwd(B, orig, gutils, normalR, shadowR)
end

@register_rev function arrayreshape_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function gcloaded_fwd(B, orig, gutils, normalR, shadowR)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0)
        return true
    end

    origops = LLVM.operands(orig)
    if is_constant_value(gutils, origops[1])
        emit_error(B, orig, "Enzyme: gcloaded has active return, but inactive input(1)")
    end
    if is_constant_value(gutils, origops[2])
        emit_error(B, orig, "Enzyme: gcloaded has active return, but inactive input(2)")
    end

    width = get_width(gutils)

    shadowin1 = invert_pointer(gutils, origops[1], B)
    shadowin2 = invert_pointer(gutils, origops[2], B)
    if width == 1
        args = LLVM.Value[shadowin1, shadowin2]
        shadowres = call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            args,
            [API.VT_Shadow, API.VT_Shadow],
            false,
        ) #=lookup=#
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            args = LLVM.Value[
                extract_value!(B, shadowin1, idx - 1)
                extract_value!(B, shadowin2, idx - 1)
            ]
            tmp = call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                args,
                [API.VT_Shadow, API.VT_Shadow],
                false,
            ) #=lookup=#
            shadowres = insert_value!(B, shadowres, tmp, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)

    return false
end

@register_aug function gcloaded_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    gcloaded_fwd(B, orig, gutils, normalR, shadowR)
end

@register_rev function gcloaded_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function boxfloat_fwd(B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    width = get_width(gutils)

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if is_constant_value(gutils, orig) || needsShadowP[] == 0
        return true
    end

    flt = value_type(origops[1])
    shadowsin = LLVM.Value[invert_pointer(gutils, origops[1], B)]
    if width == 1
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), shadowsin)
        callconv!(shadowres, callconv(orig))
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            args = LLVM.Value[extract_value!(B, s, idx - 1) for s in shadowsin]
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

@register_aug function boxfloat_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    origops = collect(operands(orig))
    width = get_width(gutils)

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if is_constant_value(gutils, orig) || needsShadowP[] == 0
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
        for idx = 1:width
            obj = emit_allocobj!(B, Base.RefValue{TT})
            o2 = bitcast!(B, obj, LLVM.PointerType(flt, addrspace(value_type(obj))))
            store!(B, ConstantFP(flt, 0.0), o2)
            shadowres = insert_value!(B, shadowres, obj, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

@register_rev function boxfloat_rev(B, orig, gutils, tape)

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )

    if is_constant_value(gutils, orig) || needsShadowP[] == 0
        return nothing
    end

    origops = collect(operands(orig))
    width = get_width(gutils)
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
        for idx = 1:width
            ipc = extract_value!(B, ip, idx - 1)
            ipc = bitcast!(B, ipc, LLVM.PointerType(flt, addrspace(value_type(orig))))
            ld = load!(B, flt, ipc)
            store!(B, ConstantFP(flt, 0.0), ipc)
            shadowres = insert_value!(B, shadowres, ld, idx - 1)
        end
        if !is_constant_value(gutils, origops[1])
            API.EnzymeGradientUtilsAddToDiffe(gutils, origops[1], shadowret, B, flt)
        end
    end
    return nothing
end

@register_fwd function eqtableget_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end

    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_eqtable_get")

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

function error_if_active(::Type{T}) where {T}
    seen = ()
    areg = active_reg_inner(T, seen, nothing, Val(true)) #=justActive=#
    if areg == ActiveState
        throw(
            AssertionError("Found unhandled active variable in tuple splat, jl_eqtable $T"),
        )
    end
    nothing
end

@register_aug function eqtableget_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    mode = get_mode(gutils)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        mode,
    )
    if needsShadowP[] == 0
        return false
    end

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    width = get_width(gutils)

    origh, origkey, origdflt = operands(orig)[1:end-1]

    if is_constant_value(gutils, origh)
        emit_error(
            B,
            orig,
            "Enzyme: Not yet implemented constant table in jl_eqtable_get " *
            string(origh) *
            " " *
            string(orig) *
            " result: " *
            string(absint(orig)) *
            " " *
            string(abs_typeof(orig, true)) *
            " dict: " *
            string(absint(origh)) *
            " " *
            string(abs_typeof(origh, true)) *
            " key " *
            string(absint(origkey)) *
            " " *
            string(abs_typeof(origkey, true)) *
            " dflt " *
            string(absint(origdflt)) *
            " " *
            string(abs_typeof(origdflt, true)),
        )
    end

    shadowh = invert_pointer(gutils, origh, B)

    shadowdflt = if is_constant_value(gutils, origdflt)
        shadowdflt2 = julia_error(
            Base.unsafe_convert(
                Cstring,
                "Mixed activity for default of jl_eqtable_get " *
                string(orig) *
                " " *
                string(origdflt),
            ),
            orig.ref,
            API.ET_MixedActivityError,
            gutils.ref,
            origdflt.ref,
            B.ref,
        )
        if shadowdflt2 != C_NULL
            LLVM.Value(shadowdflt2)
        else
            nop = new_from_original(gutils, origdflt)
            if width == 1
                nop
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(nop)))
                shadowm = LLVM.UndefValue(ST)
                for j = 1:width
                    shadowm = insert_value!(B, shadowm, nop, j - 1)
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
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, false) #=lookup=#
        callconv!(cal, callconv(orig))
        emit_apply_generic!(
            B,
            LLVM.Value[unsafe_to_llvm(B, error_if_active), emit_jltypeof!(B, cal)],
        )
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j = 1:width
            newops = LLVM.Value[
                extract_value!(B, shadowh, j - 1),
                new_from_original(gutils, origkey),
                extract_value!(B, shadowdflt, j - 1),
            ]
            cal = call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                newops,
                newvals,
                false,
            ) #=lookup=#
            callconv!(cal, callconv(orig))
            emit_apply_generic!(
                B,
                LLVM.Value[unsafe_to_llvm(B, error_if_active), emit_jltypeof!(B, cal)],
            )
            shadow = insert_value!(B, shadow, cal, j - 1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    return false
end

@register_rev function eqtableget_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function eqtableput_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_eqtable_put")

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

@register_aug function eqtableput_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    origh, origkey, origval, originserted = operands(orig)[1:end-1]

    @assert !is_constant_value(gutils, origh)

    shadowh = invert_pointer(gutils, origh, B)
    shadowval = invert_pointer(gutils, origval, B)

    shadowval = if is_constant_value(gutils, origval)
        shadowdflt2 = julia_error(
            Base.unsafe_convert(
                Cstring,
                "Mixed activity for val of jl_eqtable_put " *
                string(orig) *
                " " *
                string(origval),
            ),
            orig.ref,
            API.ET_MixedActivityError,
            gutils.ref,
            origval.ref,
            B.ref,
        )
        if shadowdflt2 != C_NULL
            LLVM.Value(shadowdflt2)
        else
            nop = new_from_original(gutils, origval)
            if width == 1
                nop
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(nop)))
                shadowm = LLVM.UndefValue(ST)
                for j = 1:width
                    shadowm = insert_value!(B, shadowm, nop, j - 1)
                end
                shadowm
            end
        end
    else
        invert_pointer(gutils, origval, B)
    end

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    newvals = API.CValueType[API.VT_Shadow, API.VT_Primal, API.VT_Shadow, API.VT_None]

    shadowres = if width == 1
        emit_apply_generic!(
            B,
            LLVM.Value[unsafe_to_llvm(B, error_if_active), emit_jltypeof!(B, shadowval)],
        )
        newops = LLVM.Value[
            shadowh,
            new_from_original(gutils, origkey),
            shadowval,
            LLVM.null(value_type(originserted)),
        ]
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, false) #=lookup=#
        callconv!(cal, callconv(orig))
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j = 1:width
            sval2 = extract_value!(B, shadowval, j - 1)
            emit_apply_generic!(
                B,
                LLVM.Value[unsafe_to_llvm(B, error_if_active), emit_jltypeof!(B, sval2)],
            )
            newops = LLVM.Value[
                extract_value!(B, shadowh, j - 1),
                new_from_original(gutils, origkey),
                sval2,
                LLVM.null(value_type(originserted)),
            ]
            cal = call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                newops,
                newvals,
                false,
            ) #=lookup=#
            callconv!(cal, callconv(orig))
            shadow = insert_value!(B, shadow, cal, j - 1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    return false
end

@register_rev function eqtableput_rev(B, orig, gutils, tape)
    return nothing
end


@register_fwd function idtablerehash_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented forward for jl_idtable_rehash")

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

@register_aug function idtablerehash_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(
        B,
        orig,
        "Enzyme: Not yet implemented augmented forward for jl_idtable_rehash",
    )

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return false
end

@register_rev function idtablerehash_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: Not yet implemented reverse for jl_idtable_rehash")
    return nothing
end

@register_fwd function jl_array_grow_end_fwd(B, orig, gutils, normalR, shadowR)
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
        call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            args,
            [API.VT_Shadow, API.VT_Primal],
            false,
        ) #=lookup=#
    else
        for idx = 1:width
            args = LLVM.Value[
                extract_value!(B, shadowin, idx - 1)
                new_from_original(gutils, origops[2])
            ]
            call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                args,
                [API.VT_Shadow, API.VT_Primal],
                false,
            ) #=lookup=#
        end
    end
    return false
end


@register_aug function jl_array_grow_end_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
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
        call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            args,
            [API.VT_Shadow, API.VT_Primal],
            false,
        ) #=lookup=#

        toset = get_array_data(B, anti)
        toset = gep!(B, i8, toset, LLVM.Value[off])
        mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
    else
        for idx = 1:width
            anti = extract_value!(B, shadowin, idx - 1)

            idx = get_array_nrows(B, anti)
            elsz = zext!(B, get_array_elsz(B, anti), value_type(idx))
            off = mul!(B, idx, elsz)
            tot = mul!(B, inc, elsz)

            args = LLVM.Value[anti, inc]
            call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                args,
                [API.VT_Shadow, API.VT_Primal],
                false,
            ) #=lookup=#

            toset = get_array_data(B, anti)
            toset = gep!(B, i8, toset, LLVM.Value[off])
            mcall = LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), tot, al)
        end
    end

    return false
end

@register_rev function jl_array_grow_end_rev(B, orig, gutils, tape)
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
            shadowres =
                UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx = 1:width
                args = LLVM.Value[
                    extract_value!(B, shadowin, idx - 1)
                    offset
                ]
                LLVM.call!(B, fty, delF, args)
            end
        end
    end
    return nothing
end

@register_fwd function jl_array_del_end_fwd(B, orig, gutils, normalR, shadowR)
    jl_array_grow_end_fwd(B, orig, gutils, normalR, shadowR)
end

@register_aug function jl_array_del_end_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_del_end_fwd(B, orig, gutils, normalR, shadowR)
end

@register_rev function jl_array_del_end_rev(B, orig, gutils, tape)
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

        # TODO get actual alignment
        algn = 0

        i8 = LLVM.IntType(8)
        for idx = 1:width
            anti = if width == 1
                shadowin
            else
                extract_value!(B, shadowin, idx - 1)
            end
            if get_runtime_activity(gutils)
                emit_error(
                    B,
                    orig,
                    "Enzyme: Not yet implemented runtime activity for reverse of jl_array_del_end",
                )
            end
            args = LLVM.Value[anti, offset]

            found, arty, byref = abs_typeof(origops[1])
            anti = shadowin
            elSize = if found
                LLVM.ConstantInt(Csize_t(actual_size(eltype(arty))))
            else
                elSize = LLVM.zext!(
                    B,
                    get_array_elsz(B, anti),
                    LLVM.IntType(8 * sizeof(Csize_t)),
                )
            end
            len = get_array_len(B, anti)

            LLVM.call!(B, fty, delF, args)

            length = LLVM.mul!(B, len, elSize)

            if !found && !(eltype(arty) <: Base.IEEEFloat)
		bt = GPUCompiler.backtrace(orig)
		btstr = sprint() do io
		    print(io, "\nCaused by:")
		    Base.show_backtrace(io, bt)
		end
                GPUCompiler.@safe_warn "TODO reverse jl_array_del_end zero-set used memset rather than runtime type of $((found, arty)) in $(string(origops[1])) $btstr"
            end
            toset = get_array_data(B, anti)
            toset = gep!(B, i8, toset, LLVM.Value[length])
            LLVM.memset!(B, toset, LLVM.ConstantInt(i8, 0, false), elSize, algn)
        end
    end
    return nothing
end

@register_fwd function jl_array_ptr_copy_fwd(B, orig, gutils, normalR, shadowR)
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
    valTys = API.CValueType[
        API.VT_Shadow,
        API.VT_Shadow,
        API.VT_Shadow,
        API.VT_Shadow,
        API.VT_Primal,
    ]

    if width == 1
        vargs = args
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, vargs, valTys, false) #=lookup=#
        debug_from_orig!(gutils, cal, orig)
        callconv!(cal, callconv(orig))
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            vargs = LLVM.Value[]
            for a in args[1:end-1]
                push!(vargs, extract_value!(B, a, idx - 1))
            end
            push!(vargs, args[end])
            cal =
                call_samefunc_with_inverted_bundles!(B, gutils, orig, vargs, valTys, false) #=lookup=#
            debug_from_orig!(gutils, cal, orig)
            callconv!(cal, callconv(orig))
        end
    end

    return false
end
@register_aug function jl_array_ptr_copy_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_ptr_copy_fwd(B, orig, gutils, normalR, shadowR)
end
@register_rev function jl_array_ptr_copy_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function jl_array_sizehint_fwd(B, orig, gutils, normalR, shadowR)
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
        call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            args,
            [API.VT_Shadow, API.VT_Primal],
            false,
        ) #=lookup=#
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            args = LLVM.Value[
                extract_value!(B, shadowin, idx - 1)
                new_from_original(gutils, origops[2])
            ]
            call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                args,
                [API.VT_Shadow, API.VT_Primal],
                false,
            ) #=lookup=#
        end
    end
    return false
end

@register_aug function jl_array_sizehint_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_array_sizehint_fwd(B, orig, gutils, normalR, shadowR)
end

@register_rev function jl_array_sizehint_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function jl_unhandled_fwd(B, orig, gutils, normalR, shadowR)
    newo = new_from_original(gutils, orig)
    origops = collect(operands(orig))
    err = emit_error(B, orig, "Enzyme: unhandled forward for " * string(origops[end]))
    API.moveBefore(newo, err, C_NULL)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing

    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            position!(B, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(normal)))
            shadowres = UndefValue(
                LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))),
            )
            for idx = 1:width
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end
@register_aug function jl_unhandled_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    jl_unhandled_fwd(B, orig, gutils, normalR, shadowR)
end
@register_rev function jl_unhandled_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function get_binding_or_error_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig)
        return true
    end
    err = emit_error(B, orig, "Enzyme: unhandled forward for jl_get_binding_or_error")
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)

    if unsafe_load(shadowR) != C_NULL
        valTys = API.CValueType[API.VT_Primal, API.VT_Primal]
        args = [
            new_from_original(gutils, operands(orig)[1]),
            new_from_original(gutils, operands(orig)[2]),
        ]
        normal = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, false) #=lookup=#
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(
                LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))),
            )
            for idx = 1:width
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@register_aug function get_binding_or_error_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end
    err = emit_error(
        B,
        orig,
        "Enzyme: unhandled augmented forward for jl_get_binding_or_error",
    )
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    if unsafe_load(shadowR) != C_NULL
        valTys = API.CValueType[API.VT_Primal, API.VT_Primal]
        args = [
            new_from_original(gutils, operands(orig)[1]),
            new_from_original(gutils, operands(orig)[2]),
        ]
        normal = call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, false) #=lookup=#
        width = get_width(gutils)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(
                LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))),
            )
            for idx = 1:width
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@register_rev function get_binding_or_error_rev(B, orig, gutils, tape)
    emit_error(B, orig, "Enzyme: unhandled reverse for jl_get_binding_or_error")
    return nothing
end

@register_fwd function finalizer_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    err = emit_error(
        B,
        orig,
        "Enzyme: unhandled forward for jl_gc_add_finalizer_th or jl_gc_add_ptr_finalizer",
    )
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

@register_aug function finalizer_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    err = emit_error(
        B,
        orig,
        "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th",
    )
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
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

@register_rev function finalizer_rev(B, orig, gutils, tape)
    # emit_error(B, orig, "Enzyme: unhandled reverse for jl_gc_add_finalizer_th")
    return nothing
end


@register_fwd function deferred_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    err = emit_error(
        B,
        orig,
        "There is a known issue in GPUCompiler.jl which is preventing higher-order AD of this code.\nPlease see https://github.com/JuliaGPU/GPUCompiler.jl/issues/629 for more information and to alert the GPUCompiler authors of your use case and need.",
    )
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

@register_aug function deferred_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    err = emit_error(
        B,
        orig,
        "There is a known issue in GPUCompiler.jl which is preventing higher-order AD of this code.\nPlease see https://github.com/JuliaGPU/GPUCompiler.jl/issues/629 for more information and to alert the GPUCompiler authors of your use case and need.",
    )
    newo = new_from_original(gutils, orig)
    API.moveBefore(newo, err, B)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
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

@register_rev function deferred_rev(B, orig, gutils, tape)
    return nothing
end


function register_handler!(variants, augfwd_handler, rev_handler, fwd_handler = nothing)
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
    cname = Symbol(string(f) * "_cfunc")
    :(@cfunction(
        $cname,
        UInt8,
        (
            LLVM.API.LLVMBuilderRef,
            LLVM.API.LLVMValueRef,
            API.EnzymeGradientUtilsRef,
            Ptr{LLVM.API.LLVMValueRef},
            Ptr{LLVM.API.LLVMValueRef},
            Ptr{LLVM.API.LLVMValueRef},
        )
    ))
end

macro revfunc(f)
    cname = Symbol(string(f) * "_cfunc")
    :(@cfunction(
        $cname,
        Cvoid,
        (
            LLVM.API.LLVMBuilderRef,
            LLVM.API.LLVMValueRef,
            API.EnzymeGradientUtilsRef,
            LLVM.API.LLVMValueRef,
        )
    ))
end

macro fwdfunc(f)
    cname = Symbol(string(f) * "_cfunc")
    :(@cfunction(
        $cname,
        UInt8,
        (
            LLVM.API.LLVMBuilderRef,
            LLVM.API.LLVMValueRef,
            API.EnzymeGradientUtilsRef,
            Ptr{LLVM.API.LLVMValueRef},
            Ptr{LLVM.API.LLVMValueRef},
        )
    ))
end

macro diffusefunc(f)
    cname = Symbol(string(f) * "_cfunc")
    :(@cfunction(
        Compiler.$cname,
        UInt8,
        (
            LLVM.API.LLVMValueRef,
            API.EnzymeGradientUtilsRef,
            LLVM.API.LLVMValueRef,
            UInt8,
            API.CDerivativeMode,
            Ptr{UInt8},
        )
    ))
end

@noinline function register_llvm_rules()
    API.EnzymeRegisterDiffUseCallHandler(
        "enzyme_custom",
        @diffusefunc(enzyme_custom_diffuse)
    )
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
        ("jl_", "jl_breakpoint"),
        @augfunc(noop_augfwd),
        @revfunc(duplicate_rev),
        @fwdfunc(noop_fwd),
    )
    register_handler!(
        ("jl_array_copy", "ijl_array_copy"),
        @augfunc(arraycopy_augfwd),
        @revfunc(arraycopy_rev),
        @fwdfunc(arraycopy_fwd),
    )
    register_handler!(
        ("jl_genericmemory_copy_slice", "ijl_genericmemory_copy_slice"),
        @augfunc(genericmemory_copy_slice_augfwd),
        @revfunc(genericmemory_copy_slice_rev),
        @fwdfunc(genericmemory_copy_slice_fwd),
    )
    register_handler!(
        ("jl_reshape_array", "ijl_reshape_array"),
        @augfunc(arrayreshape_augfwd),
        @revfunc(arrayreshape_rev),
        @fwdfunc(arrayreshape_fwd),
    )
    register_handler!(
        ("jl_f_setfield", "ijl_f_setfield"),
        @augfunc(setfield_augfwd),
        @revfunc(setfield_rev),
        @fwdfunc(setfield_fwd),
    )
    register_handler!(
        ("jl_box_float32", "ijl_box_float32", "jl_box_float64", "ijl_box_float64"),
        @augfunc(boxfloat_augfwd),
        @revfunc(boxfloat_rev),
        @fwdfunc(boxfloat_fwd),
    )
    register_handler!(
        ("jl_f_tuple", "ijl_f_tuple"),
        @augfunc(f_tuple_augfwd),
        @revfunc(f_tuple_rev),
        @fwdfunc(f_tuple_fwd),
    )
    register_handler!(
        ("jl_eqtable_get", "ijl_eqtable_get"),
        @augfunc(eqtableget_augfwd),
        @revfunc(eqtableget_rev),
        @fwdfunc(eqtableget_fwd),
    )
    register_handler!(
        ("jl_eqtable_put", "ijl_eqtable_put"),
        @augfunc(eqtableput_augfwd),
        @revfunc(eqtableput_rev),
        @fwdfunc(eqtableput_fwd),
    )
    register_handler!(
        ("jl_idtable_rehash", "ijl_idtable_rehash"),
        @augfunc(idtablerehash_augfwd),
        @revfunc(idtablerehash_rev),
        @fwdfunc(idtablerehash_fwd),
    )
    register_handler!(
        ("jl_f__apply_iterate", "ijl_f__apply_iterate"),
        @augfunc(apply_iterate_augfwd),
        @revfunc(apply_iterate_rev),
        @fwdfunc(apply_iterate_fwd),
    )
    register_handler!(
        ("jl_f__svec_ref", "ijl_f__svec_ref"),
        @augfunc(f_svec_ref_augfwd),
        @revfunc(f_svec_ref_rev),
        @fwdfunc(f_svec_ref_fwd),
    )
    register_handler!(
        ("jl_new_structv", "ijl_new_structv"),
        @augfunc(new_structv_augfwd),
        @revfunc(new_structv_rev),
        @fwdfunc(new_structv_fwd),
    )
    register_handler!(
        ("jl_new_structt", "ijl_new_structt"),
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
        (
            "jl_gc_add_finalizer_th",
            "ijl_gc_add_finalizer_th",
            "jl_gc_add_ptr_finalizer",
            "ijl_gc_add_ptr_finalizer",
        ),
        @augfunc(finalizer_augfwd),
        @revfunc(finalizer_rev),
        @fwdfunc(finalizer_fwd),
    )
    register_handler!(
        ("deferred_codegen",),
        @augfunc(deferred_augfwd),
        @revfunc(deferred_rev),
        @fwdfunc(deferred_fwd),
    )
    register_handler!(
        ("jl_array_grow_end", "ijl_array_grow_end"),
        @augfunc(jl_array_grow_end_augfwd),
        @revfunc(jl_array_grow_end_rev),
        @fwdfunc(jl_array_grow_end_fwd),
    )
    register_handler!(
        ("jl_array_del_end", "ijl_array_del_end"),
        @augfunc(jl_array_del_end_augfwd),
        @revfunc(jl_array_del_end_rev),
        @fwdfunc(jl_array_del_end_fwd),
    )
    register_handler!(
        ("jl_f_getfield", "ijl_f_getfield"),
        @augfunc(jl_getfield_augfwd),
        @revfunc(jl_getfield_rev),
        @fwdfunc(jl_getfield_fwd),
    )
    register_handler!(
        ("ijl_get_nth_field_checked", "jl_get_nth_field_checked"),
        @augfunc(jl_nthfield_augfwd),
        @revfunc(jl_nthfield_rev),
        @fwdfunc(jl_nthfield_fwd),
    )
    register_handler!(
        ("jl_array_sizehint", "ijl_array_sizehint"),
        @augfunc(jl_array_sizehint_augfwd),
        @revfunc(jl_array_sizehint_rev),
        @fwdfunc(jl_array_sizehint_fwd),
    )
    register_handler!(
        ("jl_array_ptr_copy", "ijl_array_ptr_copy"),
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

precompile(register_llvm_rules, ())
