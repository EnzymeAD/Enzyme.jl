function restore_alloca_type!(f::LLVM.Function)
    replaceAndErase = Tuple{LLVM.AllocaInst,Type, LLVMType, String}[]
    dl = datalayout(LLVM.parent(f))

    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.AllocaInst)
            if haskey(metadata(inst), "enzymejl_allocart") || haskey(metadata(inst), "enzymejl_gc_alloc_rt")
                mds = operands(metadata(inst)[haskey(metadata(inst), "enzymejl_allocart") ? "enzymejl_allocart" : "enzymejl_gc_alloc_rt"])[1]::MDString
                mds = Base.convert(String, mds)
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, mds))
                RT = Base.unsafe_pointer_to_objref(ptr)
                at = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(inst))
		lrt = struct_to_llvm(RT)
                if at == lrt
                    continue
                end
                cnt = operands(inst)[1]
                if !isa(cnt, LLVM.ConstantInt) || convert(UInt, cnt) != 1
                    continue
                end
                if LLVM.sizeof(dl, at) == LLVM.sizeof(dl, lrt) && CountTrackedPointers(at).count == 0
                    push!(replaceAndErase, (inst, RT, lrt, haskey(metadata(inst), "enzymejl_allocart") ? "enzymejl_allocart" : "enzymejl_gc_alloc_rt"))
                end
            end
        end
    end

    for (al, RT, lrt, mdname) in replaceAndErase
        at = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(al))
        if CountTrackedPointers(lrt).count != 0 && CountTrackedPointers(at).count == 0
            lrt2 = strip_tracked_pointers(lrt)
            @assert LLVM.sizeof(dl, lrt2) == LLVM.sizeof(dl, lrt)
            lrt = lrt2
        end
        b = IRBuilder()
        position!(b, al)
        pname = LLVM.name(al)
        LLVM.name!(al, "")
        al2 = alloca!(b, lrt, pname)
        cst = al2
        if value_type(cst) != value_type(al)
            cst = bitcast!(b, cst, value_type(al))
        end        
        LLVM.replace_uses!(al, cst)
        LLVM.API.LLVMInstructionEraseFromParent(al)
        metadata(al2)[mdname] = MDNode(LLVM.Metadata[MDString(string(convert(UInt, unsafe_to_pointer(RT))))])
    end
	return length(replaceAndErase) != 0
end

# Rewrite calls with "jl_roots" to only have the jl_value_t attached and not  { { {} addrspace(10)*, [1 x [2 x i64]], i64, i64 }, [2 x i64] } %unbox110183_replacementA
function rewrite_ccalls!(mod::LLVM.Module)
    for f in collect(functions(mod))
        replaceAndErase = Tuple{Instruction,Instruction}[]
        for bb in blocks(f), inst in instructions(bb)
            if isa(inst, LLVM.CallInst)
                fn = called_operand(inst)
                changed = false
                B = IRBuilder()
                position!(B, inst)
                if isa(fn, LLVM.Function) && LLVM.name(fn) == "llvm.julia.gc_preserve_begin"
                    uservals = LLVM.Value[]
                    for lval in collect(arguments(inst))
                        llty = value_type(lval)
                        if isa(llty, LLVM.PointerType)
                            push!(uservals, lval)
                            continue
                        end
                        vals = get_julia_inner_types(B, nothing, lval)
                        for v in vals
                            if isa(v, LLVM.PointerNull)
                                subchanged = true
                                continue
                            end
                            push!(uservals, v)
                        end
                        if length(vals) == 1 && vals[1] == lval
                            continue
                        end
                        changed = true
                    end
                    if changed
                        prevname = LLVM.name(inst)
                        LLVM.name!(inst, "")
                        if !isdefined(LLVM, :OperandBundleDef)
                            newinst = call!(
                                B,
                                called_type(inst),
                                called_operand(inst),
                                uservals,
                                collect(operand_bundles(inst)),
                                prevname,
                            )
                        else
                            newinst = call!(
                                B,
                                called_type(inst),
                                called_operand(inst),
                                uservals,
                                collect(map(LLVM.OperandBundleDef, operand_bundles(inst))),
                                prevname,
                            )
                        end
                        for idx in [
                            LLVM.API.LLVMAttributeFunctionIndex,
                            LLVM.API.LLVMAttributeReturnIndex,
                            [
                                LLVM.API.LLVMAttributeIndex(i) for
                                i = 1:(length(arguments(inst)))
                            ]...,
                        ]
                            idx = reinterpret(LLVM.API.LLVMAttributeIndex, idx)
                            count = LLVM.API.LLVMGetCallSiteAttributeCount(inst, idx)
                            Attrs = Base.unsafe_convert(
                                Ptr{LLVM.API.LLVMAttributeRef},
                                Libc.malloc(sizeof(LLVM.API.LLVMAttributeRef) * count),
                            )
                            LLVM.API.LLVMGetCallSiteAttributes(inst, idx, Attrs)
                            for j = 1:count
                                LLVM.API.LLVMAddCallSiteAttribute(
                                    newinst,
                                    idx,
                                    unsafe_load(Attrs, j),
                                )
                            end
                            Libc.free(Attrs)
                        end
                        API.EnzymeCopyMetadata(newinst, inst)
                        callconv!(newinst, callconv(inst))
                        push!(replaceAndErase, (inst, newinst))
                    end
                    continue
                end
                if !isdefined(LLVM, :OperandBundleDef)
                    newbundles = OperandBundle[]
                else
                    newbundles = OperandBundleDef[]
                end
                for bunduse in operand_bundles(inst)
                    if isdefined(LLVM, :OperandBundleDef)
                        bunduse = LLVM.OperandBundleDef(bunduse)
                    end

                    if !isdefined(LLVM, :OperandBundleDef)
                        if LLVM.tag(bunduse) != "jl_roots"
                            push!(newbundles, bunduse)
                            continue
                        end
                    else
                        if LLVM.tag_name(bunduse) != "jl_roots"
                            push!(newbundles, bunduse)
                            continue
                        end
                    end
                    uservals = LLVM.Value[]
                    subchanged = false
                    for lval in LLVM.inputs(bunduse)
                        llty = value_type(lval)
                        if isa(llty, LLVM.PointerType)
                            push!(uservals, lval)
                            continue
                        end
                        vals = get_julia_inner_types(B, nothing, lval)
                        for v in vals
                            if isa(v, LLVM.PointerNull)
                                subchanged = true
                                continue
                            end
                            push!(uservals, v)
                        end
                        if length(vals) == 1 && vals[1] == lval
                            continue
                        end
                        subchanged = true
                    end
                    if !subchanged
                        push!(newbundles, bunduse)
                        continue
                    end
                    changed = true
                    if !isdefined(LLVM, :OperandBundleDef)
                        push!(newbundles, OperandBundle(LLVM.tag(bunduse), uservals))
                    else
                        push!(
                            newbundles,
                            OperandBundleDef(LLVM.tag_name(bunduse), uservals),
                        )
                    end
                end
                changed = false
                if changed
                    prevname = LLVM.name(inst)
                    LLVM.name!(inst, "")
                    newinst = call!(
                        B,
                        called_type(inst),
                        called_operand(inst),
                        collect(arguments(inst)),
                        newbundles,
                        prevname,
                    )
                    for idx in [
                        LLVM.API.LLVMAttributeFunctionIndex,
                        LLVM.API.LLVMAttributeReturnIndex,
                        [
                            LLVM.API.LLVMAttributeIndex(i) for
                            i = 1:(length(arguments(inst)))
                        ]...,
                    ]
                        idx = reinterpret(LLVM.API.LLVMAttributeIndex, idx)
                        count = LLVM.API.LLVMGetCallSiteAttributeCount(inst, idx)
                        Attrs = Base.unsafe_convert(
                            Ptr{LLVM.API.LLVMAttributeRef},
                            Libc.malloc(sizeof(LLVM.API.LLVMAttributeRef) * count),
                        )
                        LLVM.API.LLVMGetCallSiteAttributes(inst, idx, Attrs)
                        for j = 1:count
                            LLVM.API.LLVMAddCallSiteAttribute(
                                newinst,
                                idx,
                                unsafe_load(Attrs, j),
                            )
                        end
                        Libc.free(Attrs)
                    end
                    API.EnzymeCopyMetadata(newinst, inst)
                    callconv!(newinst, callconv(inst))
                    push!(replaceAndErase, (inst, newinst))
                end
            end
        end
        for (inst, newinst) in replaceAndErase
            replace_uses!(inst, newinst)
            LLVM.API.LLVMInstructionEraseFromParent(inst)
        end
    end
end

function fixup_1p12_sret!(f::LLVM.Function)
    if VERSION < v"1.12"
        return
    end
    mi, RT = enzyme_custom_extract_mi(f, false)
    if mi === nothing
        return
    end

    _, sret, returnRoots = get_return_info(RT)

    if sret === nothing || returnRoots == nothing
        return
    end

    dl = datalayout(LLVM.parent(f))
    lltype = convert(LLVMType, RT)
    sz = LLVM.sizeof(dl, lltype)

    @assert VERSION < v"1.13"
    #TODO for 1.13 fixup this
    torep = LLVM.Instruction[]
    for u in LLVM.uses(parameters(f)[1])
        ci = LLVM.user(u)
        if isa(ci, LLVM.CallInst)
            intr = LLVM.API.LLVMGetIntrinsicID(LLVM.called_operand(ci))
            if intr == LLVM.Intrinsic("llvm.memcpy").id
                cst = operands(ci)[3]
                if cst isa LLVM.ConstantInt && convert(UInt, cst) == sz
                    push!(torep, ci)
                end
            end
        end
    end

    for ci in torep
        B = LLVM.IRBuilder()
        position!(B, ci)
        copy_struct_into!(B, lltype, operands(ci)[1], operands(ci)[2], false)
        LLVM.erase!(ci)
    end
    return
end

function force_recompute!(mod::LLVM.Module)
    for f in functions(mod), bb in blocks(f)
    iter = LLVM.API.LLVMGetFirstInstruction(bb)
    while iter != C_NULL
        inst = LLVM.Instruction(iter)
        iter = LLVM.API.LLVMGetNextInstruction(iter)
        if isa(inst, LLVM.LoadInst)
            has_loaded = false
            for u in LLVM.uses(inst)
                v = LLVM.user(u)
                if isa(v, LLVM.CallInst)
                    cf = LLVM.called_operand(v)
                    if isa(cf, LLVM.Function) && LLVM.name(cf) == "julia.gc_loaded" && operands(v)[2] == inst
                        has_loaded = true
                        break
                    end
                end
                if isa(v, LLVM.BitCastInst)
                    for u2 in LLVM.uses(v)
                        v2 = LLVM.user(u2)
                        if isa(v2, LLVM.CallInst)
                            cf = LLVM.called_operand(v2)
                            if isa(cf, LLVM.Function) && LLVM.name(cf) == "julia.gc_loaded" && operands(v2)[2] == v
                                has_loaded = true
                                break
                            end
                        end
                    end
                end
            end
            if has_loaded
                metadata(inst)["enzyme_nocache"] = MDNode(LLVM.Metadata[])
            end
        end
        if isa(inst, LLVM.CallInst)
            cf = LLVM.called_operand(inst)
            if isa(cf, LLVM.Function)
                if LLVM.name(cf) == "llvm.julia.gc_preserve_begin"
                    has_use = false
                    for u2 in LLVM.uses(inst)
                        has_use = true
                        break
                    end
                    if !has_use
                        eraseInst(bb, inst)
                    end
                end
            end
        end
    end
    end
end

function permit_inlining!(f::LLVM.Function)
    for bb in blocks(f), inst in instructions(bb)
        # remove illegal invariant.load and jtbaa_const invariants
        if isa(inst, LLVM.LoadInst)
            md = metadata(inst)
            if haskey(md, LLVM.MD_tbaa)
                modified = LLVM.Metadata(
                    ccall(
                        (:EnzymeMakeNonConstTBAA, API.libEnzyme),
                        LLVM.API.LLVMMetadataRef,
                        (LLVM.API.LLVMMetadataRef,),
                        md[LLVM.MD_tbaa],
                    ),
                )
                setindex!(md, modified, LLVM.MD_tbaa)
            end
            if haskey(md, LLVM.MD_invariant_load)
                delete!(md, LLVM.MD_invariant_load)
            end
        end
    end
end

function addNA(@nospecialize(inst::LLVM.Instruction), @nospecialize(node::LLVM.Metadata), MD::LLVM.MDKind)
    md = metadata(inst)
    next = nothing
    if haskey(md, MD)
        next = LLVM.MDNode(Metadata[node, operands(md[MD])...])
    else
        next = LLVM.MDNode(Metadata[node])
    end
    setindex!(md, next, MD)
end

function addr13NoAlias(mod::LLVM.Module)
    ctx = LLVM.context(mod)
    dom = API.EnzymeAnonymousAliasScopeDomain("addr13", ctx)
    scope = API.EnzymeAnonymousAliasScope(dom, "na_addr13")
    aliasscope = noalias = scope
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.StoreInst)
            addNA(inst, noalias, LLVM.MD_noalias)
        elseif isa(inst, LLVM.CallInst)
            fn = LLVM.called_operand(inst)
            if isa(fn, LLVM.Function)
                name = LLVM.name(fn)
                if startswith(name, "llvm.memcpy") || startswith(name, "llvm.memmove")
                    addNA(inst, noalias, LLVM.MD_noalias)
                end
            end
        elseif isa(inst, LLVM.LoadInst)
            ty = value_type(inst)
            if isa(ty, LLVM.PointerType)
                if addrspace(ty) == 13
                    addNA(inst, aliasscope, LLVM.MD_alias_scope)
                end
            end
        end
    end
    return true
end

## given code like
#  % a = alloca
#  ...
#  memref(cast(%a), %b, constant size == sizeof(a))
#   
#  turn this into load/store, as this is more
#  amenable to caching analysis infrastructure
function memcpy_alloca_to_loadstore(mod::LLVM.Module)
    dl = datalayout(mod)
    ctx = context(mod)
    seen = TypeTreeTable()
    for f in functions(mod)
        if length(blocks(f)) != 0
            bb = first(blocks(f))
            todel = Set{LLVM.Instruction}()
            for alloca in instructions(bb)
                if !isa(alloca, LLVM.AllocaInst)
                    continue
                end
                todo = Tuple{LLVM.Instruction,LLVM.Value}[(alloca, alloca)]
                copy = nothing
                legal = true
                elty = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(alloca))
                lifetimestarts = LLVM.Instruction[]
                while length(todo) > 0
                    cur, prev = pop!(todo)
                    if isa(cur, LLVM.AllocaInst) ||
                       isa(cur, LLVM.AddrSpaceCastInst) ||
                       isa(cur, LLVM.BitCastInst)
                        for u in LLVM.uses(cur)
                            u = LLVM.user(u)
                            push!(todo, (u, cur))
                        end
                        continue
                    end
                    if isa(cur, LLVM.CallInst) &&
                       isa(LLVM.called_operand(cur), LLVM.Function)
                        intr = LLVM.API.LLVMGetIntrinsicID(LLVM.called_operand(cur))
                        if intr == LLVM.Intrinsic("llvm.lifetime.start").id
                            push!(lifetimestarts, cur)
                            continue
                        end
                        if intr == LLVM.Intrinsic("llvm.lifetime.end").id
                            continue
                        end
                        if intr == LLVM.Intrinsic("llvm.memcpy").id
                            sz = operands(cur)[3]
                            if operands(cur)[1] == prev &&
                               isa(sz, LLVM.ConstantInt) &&
                               convert(Int, sz) == sizeof(dl, elty)
                                if copy === nothing || copy == cur
                                    copy = cur
                                    continue
                                end
                            end
                        end
                    end

                    # read only insts of arg, don't matter
                    if isa(cur, LLVM.LoadInst)
                        continue
                    end
                    if isa(cur, LLVM.CallInst) &&
                       isa(LLVM.called_operand(cur), LLVM.Function)
                        legalc = true
                        for (i, ci) in enumerate(operands(cur)[1:end-1])
                            if ci == prev
                                nocapture = false
                                readonly = false
                                for a in collect(
                                    parameter_attributes(LLVM.called_operand(cur), i),
                                )
                                    if kind(a) == kind(EnumAttribute("readonly"))
                                        readonly = true
                                    end
                                    if kind(a) == kind(EnumAttribute("readnone"))
                                        readonly = true
                                    end
                                    if kind(a) == kind(EnumAttribute("nocapture"))
                                        nocapture = true
                                    end
                                end
                                if !nocapture || !readonly
                                    legalc = false
                                    break
                                end
                            end
                        end
                        if legalc
                            continue
                        end
                    end

                    legal = false
                    break
                end

                if legal && copy !== nothing
                    B = LLVM.IRBuilder()
                    position!(B, copy)
                    dst = operands(copy)[1]
                    src = operands(copy)[2]
                    dst0 = bitcast!(
                        B,
                        dst,
                        LLVM.PointerType(LLVM.IntType(8), addrspace(value_type(dst))),
                    )

                    dst =
                        bitcast!(B, dst, LLVM.PointerType(elty, addrspace(value_type(dst))))
                    src =
                        bitcast!(B, src, LLVM.PointerType(elty, addrspace(value_type(src))))

                    src = load!(B, elty, src)
        
		    T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        
            	    legal, source_typ, byref = abs_typeof(src)
                    codegen_typ = value_type(src)
		    if legal
			if codegen_typ isa LLVM.PointerType || codegen_typ isa LLVM.IntegerType
			else
			    @assert byref == GPUCompiler.BITS_VALUE
			    source_typ
			end

			ec = typetree(source_typ, ctx, string(dl), seen)
			if byref == GPUCompiler.MUT_REF || byref == GPUCompiler.BITS_REF
			    ec = copy(ec)
			    merge!(ec, TypeTree(API.DT_Pointer, ctx))
			    only!(ec, -1)
			end
			    metadata(src)["enzyme_type"] = to_md(ec, ctx)
			    metadata(src)["enzymejl_source_type_$(source_typ)"] = MDNode(LLVM.Metadata[])
			    metadata(src)["enzymejl_byref_$(byref)"] = MDNode(LLVM.Metadata[])
		    
	@static if VERSION < v"1.11-"
	else    
			    legal2, obj = absint(src)
			    if legal2 && is_memory_instance(unbind(obj)) 
				metadata(src)["nonnull"] = MDNode(LLVM.Metadata[])
			    end
	end

		      elseif codegen_typ == T_prjlvalue
			    metadata(src)["enzyme_type"] =
				to_md(typetree(Ptr{Cvoid}, ctx, dl, seen), ctx)
		    end
                    FT = LLVM.FunctionType(
                        LLVM.VoidType(),
                        [LLVM.IntType(64), value_type(dst0)],
                    )
                    lifetimestart, _ = get_function!(mod, LLVM.name(LLVM.Intrinsic("llvm.lifetime.start"), [value_type(dst0)]), FT)
                    call!(
                        B,
                        FT,
                        lifetimestart,
                        LLVM.Value[LLVM.ConstantInt(Int64(sizeof(dl, elty))), dst0],
                    )
                    store!(B, src, dst)
                    push!(todel, copy)
                end
                for lt in lifetimestarts
                    push!(todel, lt)
                end
            end
            for inst in todel
                eraseInst(LLVM.parent(inst), inst)
            end
        end
    end
end

# Split a memcpy into an sret with jlvaluet into individual load/stores
function memcpy_sret_split!(mod::LLVM.Module)
    dl = datalayout(mod)
    ctx = context(mod)
	    sretkind = LLVM.kind(if LLVM.version().major >= 12
                LLVM.TypeAttribute("sret", LLVM.Int32Type())
            else
                LLVM.EnumAttribute("sret")
            end)
    for f in functions(mod)

        if length(blocks(f)) == 0
	    continue
	end
	if length(parameters(f)) == 0
	    continue
	end
	sty = nothing
	for attr in collect(LLVM.parameter_attributes(f, 1))
	    if LLVM.kind(attr) == sretkind
		 sty = LLVM.value(attr)
		 break
	    end
	end
	if sty === nothing
	    continue
	end
	tracked = CountTrackedPointers(sty)
	if tracked.all || tracked.count == 0
	    continue
	end
	todo = LLVM.CallInst[]
	for bb in blocks(f)
            for cur in instructions(bb)
                    if isa(cur, LLVM.CallInst) &&
                       isa(LLVM.called_operand(cur), LLVM.Function)
                        intr = LLVM.API.LLVMGetIntrinsicID(LLVM.called_operand(cur))
			if intr == LLVM.Intrinsic("llvm.memcpy").id
			    dst, _ = get_base_and_offset(operands(cur)[1]; offsetAllowed = false)
			    if isa(dst, LLVM.Argument) && parameters(f)[1] == dst
			    if isa(operands(cur)[3], LLVM.ConstantInt) && LLVM.sizeof(dl, sty) == convert(Int, operands(cur)[3])
				push!(todo, cur)
			    end
			    end
                        end
                    end
	    end
	end
	for cur in todo
	      B = IRBuilder()
	      position!(B, cur)
	      dst, _ = get_base_and_offset(operands(cur)[1]; offsetAllowed = false)
	      src, _ = get_base_and_offset(operands(cur)[2]; offsetAllowed = false)
	      if !LLVM.is_opaque(value_type(dst)) && eltype(value_type(dst)) != eltype(value_type(src))
	          src = pointercast!(B, src, LLVM.PointerType(eltype(value_type(dst)), addrspace(value_type(src))), "memcpy_sret_split_pointercast")
	      end
	      copy_struct_into!(B, sty, dst, src, VERSION < v"1.12")
	      LLVM.API.LLVMInstructionEraseFromParent(cur)
        end
    end
end

# If there is a phi node of a decayed value, Enzyme may need to cache it
# Here we force all decayed pointer phis to first addrspace from 10
function nodecayed_phis!(mod::LLVM.Module)
    # Simple handler to fix addrspace 11
    #complex handler for addrspace 13, which itself comes from a load of an
    # addrspace 10
    ctx = LLVM.context(mod)
    for f in functions(mod)

        guaranteedInactive = false

        for attr in collect(function_attributes(f))
            if !isa(attr, LLVM.StringAttribute)
                continue
            end
            if kind(attr) == "enzyme_inactive"
                guaranteedInactive = true
                break
            end
        end

        if guaranteedInactive
            continue
        end


        entry_ft = LLVM.function_type(f)

        RT = LLVM.return_type(entry_ft)
        inactiveRet = RT == LLVM.VoidType()

        for attr in collect(return_attributes(f))
            if !isa(attr, LLVM.StringAttribute)
                continue
            end
            if kind(attr) == "enzyme_inactive"
                inactiveRet = true
                break
            end
        end

        if inactiveRet
            for idx in length(collect(parameters(f)))
                inactiveParm = false
                for attr in collect(parameter_attributes(f, idx))
                    if !isa(attr, LLVM.StringAttribute)
                        continue
                    end
                    if kind(attr) == "enzyme_inactive"
                        inactiveParm = true
                        break
                    end
                end
                if !inactiveParm
                    inactiveRet = false
                    break
                end
            end
            if inactiveRet
                continue
            end
        end

        offty = LLVM.IntType(8 * sizeof(Int))
        i8 = LLVM.IntType(8)

        for addr in (11, 13)

            nextvs = Dict{LLVM.PHIInst,LLVM.PHIInst}()
            mtodo = Vector{LLVM.PHIInst}[]
            goffsets = Dict{LLVM.PHIInst,LLVM.PHIInst}()
            nonphis = LLVM.Instruction[]
            anyV = false
            for bb in blocks(f)
                todo = LLVM.PHIInst[]
                nonphi = nothing
                for inst in instructions(bb)
                    if !isa(inst, LLVM.PHIInst)
                        nonphi = inst
                        break
                    end
                    ty = value_type(inst)
                    if !isa(ty, LLVM.PointerType)
                        continue
                    end
                    if addrspace(ty) != addr
                        continue
                    end
                    if addr == 11
                        all_args = true
                        addrtodo = Value[inst]
                        seen = Set{LLVM.Value}()

                        while length(addrtodo) != 0
                            v = pop!(addrtodo)
                            base, _ = get_base_and_offset(v; offsetAllowed=false)
                            if in(base, seen)
                                continue
                            end
                            push!(seen, base)
                            if isa(base, LLVM.Argument) && addrspace(value_type(base)) == 11
                                continue
                            end
                            if isa(base, LLVM.PHIInst)
                                for (v, _) in LLVM.incoming(base)
                                    push!(addrtodo, v)
                                end
                                continue
                            end
                            all_args = false
                            break
                        end
                        if all_args
                            continue
                        end

                        all_args = true
                        addrtodo = Value[inst]
                        seen = Set{LLVM.Value}()

                        offset = nothing

                        while length(addrtodo) != 0
                            v = pop!(addrtodo)
                            base, toffset = get_base_and_offset(v)

                            if in(base, seen)
                                continue
                            end
                            push!(seen, base)		
                            if isa(base, LLVM.PHIInst)
                                for (v, _) in LLVM.incoming(base)
                                    push!(addrtodo, v)
                                end
                                continue
                            end
			    if offset === nothing
                                offset = toffset
                            else
                                if offset != toffset
                                    all_args = false
                                    break
                                end
                            end
                            if isa(base, LLVM.Argument) && addrspace(value_type(base)) == 11
                                continue
                            end
                            all_args = false
                            break
                        end
                        if all_args
                            continue
                        end
                    end

                    push!(todo, inst)
                    nb = IRBuilder()
                    position!(nb, inst)
                    el_ty = if addr == 11 && !LLVM.is_opaque(ty)
                        eltype(ty)
                    else
                        LLVM.StructType(LLVM.LLVMType[])
                    end
                    nphi = phi!(
                        nb,
                        LLVM.PointerType(el_ty, 10),
                        "nodecayed." * LLVM.name(inst),
                    )
                    nextvs[inst] = nphi
                    anyV = true

                    goffsets[inst] = phi!(nb, offty, "nodecayedoff." * LLVM.name(inst))
                end
                push!(mtodo, todo)
                push!(nonphis, nonphi)
            end
            for (bb, todo, nonphi) in zip(blocks(f), mtodo, nonphis)

                for inst in todo
                    ty = value_type(inst)
                    el_ty = if addr == 11 && !LLVM.is_opaque(ty)
                        eltype(ty)
                    else
                        LLVM.StructType(LLVM.LLVMType[])
                    end
                    nvs = Tuple{LLVM.Value,LLVM.BasicBlock}[]
                    offsets = Tuple{LLVM.Value,LLVM.BasicBlock}[]
                    for (v, pb) in LLVM.incoming(inst)
                        done = false
                        for ((nv, pb0), (offset, pb1)) in zip(nvs, offsets)
                            if pb0 == pb
                                push!(nvs, (nv, pb))
                                push!(offsets, (offset, pb))
                                done = true
                                break
                            end
                        end
                        if done
                            continue
                        end

                        v0 = v
			@inline function getparent(b::LLVM.IRBuilder, @nospecialize(v::LLVM.Value), @nospecialize(offset::LLVM.Value), hasload::Bool, phicache::Dict{LLVM.PHIInst, Tuple{LLVM.PHIInst, LLVM.PHIInst}})
                            if addr == 11 && addrspace(value_type(v)) == 10
                                return v, offset, hasload
                            end
                            if addr == 13 && hasload && addrspace(value_type(v)) == 10
                                return v, offset, hasload
                            end

                            if addr == 13  && !hasload
                                if isa(v, LLVM.LoadInst)
                                    v2, o2, hl2 = getparent(b, operands(v)[1], LLVM.ConstantInt(offty, 0), true, phicache)
                                    @static if VERSION < v"1.11-"
                                    else
                                        @assert offset == LLVM.ConstantInt(offty, 0)
                                        return v2, o2, true
                                    end

                                    rhs = LLVM.ConstantInt(offty, 0) 
                                    if o2 != rhs
                                        msg = sprint() do io::IO
                                            println(
                                                io,
                                                "Enzyme internal error addr13 load doesn't keep offset 0",
                                            )
                                            println(io, "v=", string(v))
                                            println(io, "v2=", string(v2))
                                            println(io, "o2=", string(o2))
                                            println(io, "hl2=", string(hl2))
                                            println(io, "offty=", string(offty))
                                            println(io, "rhs=", string(rhs))
                                        end
                                        throw(AssertionError(msg))
                                    end
                                    return v2, offset, true
                                end
                                if isa(v, LLVM.CallInst)
                                    cf = LLVM.called_operand(v)
                                    if isa(cf, LLVM.Function) && LLVM.name(cf) == "julia.gc_loaded"
                                        ld = operands(v)[2]
                                        ld0, o0, ol0 =  getparent(b, ld, LLVM.ConstantInt(offty, 0), hasload, phicache)
                                        v2 = ld0
                                        # v2, o2, hl2 = getparent(b, operands(ld)[1], LLVM.ConstantInt(offty, 0), true)

                                        rhs = LLVM.ConstantInt(offty, sizeof(Int))
                                        o2 = o0

                                            base_2, off_2 = get_base_and_offset(v2)
                                            base_1, off_1 = get_base_and_offset(operands(v)[1])

                                            if o2 == rhs && base_1 == base_2 && off_1 == off_2
                                                return operands(v)[1], offset, true
                                            end

                                            pty = TypeTree(API.DT_Pointer, LLVM.context(ld))
                                            only!(pty, -1)
                                            rhs = ptrtoint!(b, get_memory_data(b, operands(v)[1]), offty)
                                            metadata(rhs)["enzyme_type"] = to_md(pty, ctx)
                                            lhs = ptrtoint!(b, operands(v)[2], offty)
                                            metadata(rhs)["enzyme_type"] = to_md(pty, ctx)
                                            off2 = nuwsub!(b, lhs, rhs)
                                            ity = TypeTree(API.DT_Integer, LLVM.context(ld))
                                            only!(ity, -1)
                                            metadata(off2)["enzyme_type"] = to_md(ity, ctx)
                                            add = nuwadd!(b, offset, off2)
                                            metadata(add)["enzyme_type"] = to_md(ity, ctx)
                                            return operands(v)[1], add, true
                                    end
                                end
                            end

                            if addr == 13 && isa(v, LLVM.ConstantExpr)
                                if opcode(v) == LLVM.API.LLVMAddrSpaceCast
                                    v2 = operands(v)[1]
                                    if addrspace(value_type(v2)) == 0
                                        if addr == 13 && isa(v, LLVM.ConstantExpr)
					    PT = if LLVM.is_opaque(value_type(v))
						LLVM.PointerType(10)
					    else
						LLVM.PointerType(eltype(value_type(v)), 10)
					    end
                                            v2 = const_addrspacecast(
                                                operands(v)[1],
                                                PT
                                            )
                                            return v2, offset, hasload
                                        end
                                    end
                                end
                            end

                            if isa(v, LLVM.ConstantExpr)
                                if opcode(v) == LLVM.API.LLVMAddrSpaceCast
                                    v2 = operands(v)[1]
                                    if addrspace(value_type(v2)) == 10
                                        return v2, offset, hasload
                                    end
                                    if addrspace(value_type(v2)) == 0
                                        if addr == 11
					    PT = if LLVM.is_opaque(value_type(v))
						LLVM.PointerType(10)
					    else
						LLVM.PointerType(eltype(value_type(v)), 10)
					    end
                                            v2 = const_addrspacecast(
                                                v2,
                                                PT
                                            )
                                            return v2, offset, hasload
                                        end
                                    end
                                    if LLVM.isnull(v2)
					PT = if LLVM.is_opaque(value_type(v))
					   LLVM.PointerType(10)
				        else
					   LLVM.PointerType(eltype(value_type(v)), 10)
				        end
                                        v2 = const_addrspacecast(
                                            v2,
                                            PT
                                        )
                                        return v2, offset, hasload
                                    end
                                end
                                if opcode(v) == LLVM.API.LLVMBitCast
                                    preop = operands(v)[1]
                                    while isa(preop, LLVM.ConstantExpr) && opcode(preop) == LLVM.API.LLVMBitCast
                                        preop = operands(preop)[1]
                                    end
                                    v2, offset, skipload =
                                        getparent(b, preop, offset, hasload, phicache)
                                    v2 = const_bitcast(
                                        v2,
                                        LLVM.PointerType(
                                            eltype(value_type(v)),
                                            addrspace(value_type(v2)),
                                        ),
                                    )
                                    @assert eltype(value_type(v2)) == eltype(value_type(v))
                                    return v2, offset, skipload
                                end
                                
                                if opcode(v) == LLVM.API.LLVMGetElementPtr
                                    v2, offset, skipload =
                                        getparent(b, operands(v)[1], offset, hasload, phicache)
                                    offset = const_add(
                                        offset,
                                        API.EnzymeComputeByteOffsetOfGEP(b, v, offty),
                                    )
				    if !LLVM.is_opaque(value_type(v))
                                    v2 = const_bitcast(
                                        v2,
                                        LLVM.PointerType(
                                            eltype(value_type(v)),
                                            addrspace(value_type(v2)),
                                        ),
                                    )
                                    @assert eltype(value_type(v2)) == eltype(value_type(v))
				    end
                                    return v2, offset, skipload
                                end

                            end

                            if isa(v, LLVM.AddrSpaceCastInst)
                                if addrspace(value_type(operands(v)[1])) == 0
					PT = if LLVM.is_opaque(value_type(v))
					   LLVM.PointerType(10)
				        else
					   LLVM.PointerType(eltype(value_type(v)), 10)
				        end
                                    v2 = addrspacecast!(
                                        b,
                                        operands(v)[1],
                                        PT
                                    )
                                    return v2, offset, hasload
                                end
                                nv, noffset, nhasload =
                                    getparent(b, operands(v)[1], offset, hasload, phicache)
                                if !is_opaque(value_type(nv)) && eltype(value_type(nv)) != eltype(value_type(v))
                                    nv = bitcast!(
                                        b,
                                        nv,
                                        LLVM.PointerType(
                                            eltype(value_type(v)),
                                            addrspace(value_type(nv)),
                                        ),
                                    )
                                end
                                return nv, noffset, nhasload
                            end

                            if isa(v, LLVM.BitCastInst)
                                preop = operands(v)[1]
                                while isa(preop, LLVM.BitCastInst)
                                    preop = operands(preop)[1]
                                end
                                v2, offset, skipload =
                                    getparent(b, preop, offset, hasload, phicache)
                                v2 = bitcast!(
                                    b,
                                    v2,
                                    LLVM.PointerType(
                                        eltype(value_type(v)),
                                        addrspace(value_type(v2)),
                                    ),
                                )
                                @assert eltype(value_type(v2)) == eltype(value_type(v))
                                return v2, offset, skipload
                            end

                            if isa(v, LLVM.GetElementPtrInst) && all(
                                x -> (isa(x, LLVM.ConstantInt) && convert(Int, x) == 0),
                                operands(v)[2:end],
                            )
                                v2, offset, skipload =
                                    getparent(b, operands(v)[1], offset, hasload, phicache)
				    if !LLVM.is_opaque(value_type(v))
					    v2 = bitcast!(
					    b,
					    v2,
					    LLVM.PointerType(
						eltype(value_type(v)),
						addrspace(value_type(v2)),
					    ),
					)
				    end
                                @assert eltype(value_type(v2)) == eltype(value_type(v))
                                return v2, offset, skipload
                            end

                            if isa(v, LLVM.GetElementPtrInst)
                                v2, offset, skipload =
                                    getparent(b, operands(v)[1], offset, hasload, phicache)
                                offset = nuwadd!(
                                    b,
                                    offset,
                                    API.EnzymeComputeByteOffsetOfGEP(b, v, offty),
                                )
                                if !LLVM.is_opaque(value_type(v2))
                                    v2 = bitcast!(
                                        b,
                                        v2,
                                        LLVM.PointerType(
                                            eltype(value_type(v)),
                                            addrspace(value_type(v2)),
                                        ),
                                    )
                                    @assert eltype(value_type(v2)) == eltype(value_type(v))
                                end
                                return v2, offset, skipload
                            end

                            undeforpoison = isa(v, LLVM.UndefValue)
                            @static if LLVM.version() >= v"12"
                                undeforpoison |= isa(v, LLVM.PoisonValue)
                            end
                            if undeforpoison
				PT = if LLVM.is_opaque(value_type(v))
				   LLVM.PointerType(10)
				else
				   LLVM.PointerType(eltype(value_type(v)), 10)
				end
				return LLVM.UndefValue(PT), offset, addr == 13
                            end

                            if isa(v, LLVM.PHIInst) && !hasload && haskey(goffsets, v)
                                offset = nuwadd!(b, offset, goffsets[v])
                                nv = nextvs[v]
                                return nv, offset, addr == 13
                            end
                            
                            @static if VERSION < v"1.11-"
                            else
                            if addr == 13 && isa(v, LLVM.PHIInst)
				if haskey(phicache, v)
				   return (phicache[v]..., hasload)
				end
                                vs = Union{LLVM.Value, Nothing}[]
                                offs = Union{LLVM.Value, Nothing}[]
                                blks = LLVM.BasicBlock[]
                                
                                B = LLVM.IRBuilder()
                                position!(B, v)

                                sPT = if !LLVM.is_opaque(value_type(v))
                                    LLVM.PointerType(eltype(value_type(v)), 10)
                                else
                                    LLVM.PointerType(10)
                                end
                                vphi = phi!(B, sPT, "nondecay.vphi."*LLVM.name(v))
                                ophi = phi!(B, value_type(offset), "nondecay.ophi"*LLVM.name(v))
				phicache[v] = (vphi, ophi)

                                bbcache = Dict{BasicBlock, Value}()
                                for (vt, bb) in LLVM.incoming(v) 
                                    b2 = IRBuilder()
                                    position!(b2, terminator(bb))
                                    v2, o2, hl2 = getparent(b2, vt, offset, hasload, phicache)
                                    if value_type(v2) != sPT
                                        if haskey(bbcache, bb)
                                            v2 = bbcache[bb]
                                        else
                                            v2 = bitcast!(b2, v2, sPT)
                                            bbcache[bb] = v2
                                        end
                                    end

                                    @assert sPT == value_type(v2)
                                    push!(vs, v2)
                                    @assert value_type(offset) == value_type(o2)
                                    push!(offs, o2)
                                    push!(blks, bb)
                                end

                                append!(incoming(ophi), collect(zip(offs, blks)))
                                                    
                                append!(incoming(vphi), collect(zip(vs, blks)))

                                return vphi, ophi, hasload
                            end
                            end

                            if isa(v, LLVM.SelectInst)
                                lhs_v, lhs_offset, lhs_skipload =
                                    getparent(b, operands(v)[2], offset, hasload, phicache)
                                rhs_v, rhs_offset, rhs_skipload =
                                    getparent(b, operands(v)[3], offset, hasload, phicache)
                                if value_type(lhs_v) != value_type(rhs_v) ||
                                   value_type(lhs_offset) != value_type(rhs_offset) ||
                                   lhs_skipload != rhs_skipload
                                    msg = sprint() do io
                                        println(
                                            io,
                                            "Could not analyze [select] garbage collection behavior of",
                                        )
                                        println(io, " v0: ", string(v0))
                                        println(io, " v: ", string(v))
                                        println(io, " offset: ", string(offset))
                                        println(io, " hasload: ", string(hasload))
                                        println(io, " lhs_v", lhs_v)
                                        println(io, " rhs_v", rhs_v)
                                        println(io, " lhs_offset", lhs_offset)
                                        println(io, " rhs_offset", rhs_offset)
                                        println(io, " lhs_skipload", lhs_skipload)
                                        println(io, " rhs_skipload", rhs_skipload)
                                    end
                                    bt = GPUCompiler.backtrace(inst)
                                    throw(EnzymeInternalError(msg, string(f), bt))
                                end
                                return select!(b, operands(v)[1], lhs_v, rhs_v),
                                select!(b, operands(v)[1], lhs_offset, rhs_offset),
                                lhs_skipload
                            end

                            msg = sprint() do io
                                println(io, "Could not analyze garbage collection behavior of")
                                println(io, " inst: ", string(inst))
                                println(io, " v0: ", string(v0))
                                println(io, " v: ", string(v))
                                println(io, " offset: ", string(offset))
                                println(io, " hasload: ", string(hasload))
                            end
                            bt = GPUCompiler.backtrace(inst)
                            throw(EnzymeInternalError(msg, string(f), bt))
                        end
                    
                        b = IRBuilder()
                        position!(b, terminator(pb))

			phicache = Dict{LLVM.PHIInst, Tuple{LLVM.PHIInst, LLVM.PHIInst}}()
                        v, offset, hadload = getparent(b, v, LLVM.ConstantInt(offty, 0), false, phicache)

                        if addr == 13
                            @assert hadload
                        end

                        if !LLVM.is_opaque(value_type(v)) && eltype(value_type(v)) != el_ty
                            v = bitcast!(
                                b,
                                v,
                                LLVM.PointerType(el_ty, addrspace(value_type(v))),
                            )
                        end
                        push!(nvs, (v, pb))
                        push!(offsets, (offset, pb))
                    end
                        
                    nb = IRBuilder()
                    position!(nb, nonphi)

                    offset = goffsets[inst]
                    append!(LLVM.incoming(offset), offsets)
                    if all(x -> x[1] == offsets[1][1], offsets)
                        offset = offsets[1][1]
                    end

                    nphi = nextvs[inst]

                    function ogbc(@nospecialize(x::LLVM.Value))
                        while isa(x, LLVM.BitCastInst)
                            x = operands(x)[1]
                        end
                        return x
                    end

                    if all(x -> ogbc(x[1]) == ogbc(nvs[1][1]), nvs)
                        bc = ogbc(nvs[1][1])
                        if value_type(bc) != value_type(nphi)
                            bc = bitcast!(nb, bc, value_type(nphi))
                        end
                        replace_uses!(nphi, bc)
                        LLVM.API.LLVMInstructionEraseFromParent(nphi)
                        nphi = bc
                    else
                        append!(LLVM.incoming(nphi), nvs)
                    end

                    if addr == 13
                        @static if VERSION < v"1.11-"
                            nphi = bitcast!(nb, nphi, LLVM.PointerType(ty, 10))
                            nphi = addrspacecast!(nb, nphi, LLVM.PointerType(ty, 11))
                            nphi = load!(nb, ty, nphi)
                        else
                            base_obj = nphi

                            jlt = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), 10)
                            pjlt = LLVM.PointerType(jlt)

                            nphi = get_memory_data(nb, nphi)
                            nphi = bitcast!(nb, nphi, pjlt)

                            GTy = LLVM.FunctionType(LLVM.PointerType(jlt, 13), LLVM.LLVMType[jlt, pjlt])
                            gcloaded, _ = get_function!(
                                mod,
                                "julia.gc_loaded",
                                GTy
                            )
                            nphi = call!(nb, GTy, gcloaded, LLVM.Value[base_obj, nphi])
                            if value_type(nphi) != ty
                                nphi = bitcast!(nb, nphi, ty)
                            end
                        end
                    else
                        nphi = addrspacecast!(nb, nphi, ty)
                    end
                    if !isa(offset, LLVM.ConstantInt) || convert(Int64, offset) != 0
                        nphi = bitcast!(nb, nphi, LLVM.PointerType(i8, addrspace(ty)))
                        nphi = gep!(nb, i8, nphi, [offset])
                        nphi = bitcast!(nb, nphi, ty)
                    end
                    replace_uses!(inst, nphi)
                end
                for inst in todo
                    LLVM.API.LLVMInstructionEraseFromParent(inst)
                end
            end
        end
    end
    return nothing
end

function fix_decayaddr!(mod::LLVM.Module)
    for f in functions(mod)
        invalid = LLVM.Instruction[]
        for bb in blocks(f), inst in instructions(bb)
            if !isa(inst, LLVM.AddrSpaceCastInst)
                continue
            end
            prety = value_type(operands(inst)[1])
            postty = value_type(inst)
            if addrspace(prety) != 10
                continue
            end
            if addrspace(postty) != 0
                continue
            end
            push!(invalid, inst)
        end

        for inst in invalid
            temp = nothing
            for u in LLVM.uses(inst)
                st = LLVM.user(u)
                # Storing _into_ the decay addr is okay
                # we just cannot store the decayed addr into
                # somewhere
                if isa(st, LLVM.StoreInst)
                    if operands(st)[2] == inst
                        LLVM.API.LLVMSetOperand(st, 2 - 1, operands(inst)[1])
                    	nb = IRBuilder()
			position!(nb, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(st)))
                    	julia_post_cache_store(st.ref, nb.ref, reinterpret(Ptr{UInt64}, C_NULL))
                        continue
                    end
                end
                if isa(st, LLVM.LoadInst)
                    LLVM.API.LLVMSetOperand(st, 1 - 1, operands(inst)[1])
                    continue
                end

		if isa(st, LLVM.GetElementPtrInst)
		    legal = true
		    torem = LLVM.Instruction[]
		    for u in LLVM.uses(st)
			st2 = LLVM.user(u)
			# Storing _into_ the decay addr is okay
			# we just cannot store the decayed addr into
			# somewhere
			if isa(st2, LLVM.StoreInst)
			    if operands(st2)[2] == st
				push!(torem, st2)
				continue
			    end
			end
			if isa(st2, LLVM.LoadInst)
			     push!(torem, st2)
			    continue
			end
			legal = false
		    end
		    if legal
			B = IRBuilder()
			position!(B, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(st)))
		       cst = addrspacecast!(B, operands(inst)[1], LLVM.PointerType(Derived))
		       gep2 = gep!(B, LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(st)), cst, operands(inst)[2:end])
		       for st2 in torem
                	    if isa(st2, LLVM.StoreInst)
			        LLVM.API.LLVMSetOperand(st2, 2 - 1, gep2)
				nb = IRBuilder()
				position!(nb, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(st2)))
				julia_post_cache_store(st2.ref, nb.ref, reinterpret(Ptr{UInt64}, C_NULL))
				continue
			    end
			    if isa(st2, LLVM.LoadInst)
				    LLVM.API.LLVMSetOperand(st2, 1 - 1, gep2)
				    continue
			    end

		       end
                       LLVM.API.LLVMInstructionEraseFromParent(st)
		       continue
		    end
		end

                # if isa(st, LLVM.InsertValueInst)
                #    if operands(st)[1] == inst
                #        push!(invalid, st)
                #        LLVM.API.LLVMSetOperand(st, 1-1, LLVM.UndefValue(value_type(inst)))
                #        continue
                #    end
                #    if operands(st)[2] == inst
                #        push!(invalid, st)
                #        LLVM.API.LLVMSetOperand(st, 2-1, LLVM.UndefValue(value_type(inst)))
                #        continue
                #    end
                # end
                if !isa(st, LLVM.CallInst)
                    bt = GPUCompiler.backtrace(st)
                    msg = sprint() do io::IO
                        println(io, string(f))
                        println(io, inst)
                        println(io, st)
                        print(io, "Illegal decay of nonnull\n")
                        if bt !== nothing
                            print(io, "\nCaused by:")
                            Base.show_backtrace(io, bt)
                            println(io)
                        end
                    end
                    throw(AssertionError(msg))
                end

                fop = operands(st)[end]

                intr = LLVM.API.LLVMGetIntrinsicID(fop)

                if intr == LLVM.Intrinsic("llvm.memcpy").id ||
                   intr == LLVM.Intrinsic("llvm.memmove").id ||
                   intr == LLVM.Intrinsic("llvm.memset").id
                    newvs = LLVM.Value[]
                    for (i, v) in enumerate(operands(st)[1:end-1])
                        if v == inst
                            LLVM.API.LLVMSetOperand(st, i - 1, operands(inst)[1])
                            push!(newvs, operands(inst)[1])
                            continue
                        end
                        push!(newvs, v)
                    end

                    nb = IRBuilder()
                    position!(nb, st)
                    if intr == LLVM.Intrinsic("llvm.memcpy").id
                        newi = memcpy!(nb, newvs[1], 0, newvs[2], 0, newvs[3])
                    elseif intr == LLVM.Intrinsic("llvm.memmove").id
                        newi = memmove!(nb, newvs[1], 0, newvs[2], 0, newvs[3])
                    else
                        newi = memset!(nb, newvs[1], newvs[2], newvs[3], 0)
                    end

                    for idx in [
                        LLVM.API.LLVMAttributeFunctionIndex,
                        LLVM.API.LLVMAttributeReturnIndex,
                        [
                            LLVM.API.LLVMAttributeIndex(i) for
                            i = 1:(length(operands(st))-1)
                        ]...,
                    ]
                        idx = reinterpret(LLVM.API.LLVMAttributeIndex, idx)
                        count = LLVM.API.LLVMGetCallSiteAttributeCount(st, idx)

                        Attrs = Base.unsafe_convert(
                            Ptr{LLVM.API.LLVMAttributeRef},
                            Libc.malloc(sizeof(LLVM.API.LLVMAttributeRef) * count),
                        )
                        LLVM.API.LLVMGetCallSiteAttributes(st, idx, Attrs)
                        for j = 1:count
                            LLVM.API.LLVMAddCallSiteAttribute(
                                newi,
                                idx,
                                unsafe_load(Attrs, j),
                            )
                        end
                        Libc.free(Attrs)
                    end

                    API.EnzymeCopyMetadata(newi, st)

                    LLVM.API.LLVMInstructionEraseFromParent(st)
                    continue
                end
                mayread = false
                maywrite = false
                sret = true
		sret_elty = nothing
                sretkind = kind(if LLVM.version().major >= 12
                    TypeAttribute("sret", LLVM.Int32Type())
                else
                    EnumAttribute("sret")
                end)
                for (i, v) in enumerate(operands(st)[1:end-1])
                    if v == inst
                        readnone = false
                        readonly = false
                        writeonly = false
                        t_sret = false
                        for a in collect(parameter_attributes(fop, i))
                            if kind(a) == sretkind
				sret_elty = sret_ty(fop, i)
                                t_sret = true
                            end
                            if kind(a) == kind(StringAttribute("enzyme_sret"))
				sret_elty = sret_ty(fop, i)
                                t_sret = true
                            end
                            if kind(a) == kind(StringAttribute("enzymejl_returnRoots"))
				sret_elty = sret_ty(fop, i)
                                t_sret = true
                            end
                            if kind(a) == kind(StringAttribute("enzymejl_rooted_typ"))
			        sret_elty = convert(LLVMType, AnyArray(Int(CountTrackedPointers(get_rooted_typ(fop, i)).count)))
                                t_sret = true
                            end
                            # if kind(a) == kind(StringAttribute("enzyme_sret_v"))
                            #     t_sret = true
                            # end
                            if kind(a) == kind(EnumAttribute("readonly"))
                                readonly = true
                            end
                            if kind(a) == kind(EnumAttribute("readnone"))
                                readnone = true
                            end
                            if kind(a) == kind(EnumAttribute("writeonly"))
                                writeonly = true
                            end
                        end
                        if !t_sret
                            sret = false
                        end
                        if readnone
                            continue
                        end
                        if !readonly
                            maywrite = true
                        end
                        if !writeonly
                            mayread = true
                        end
                    end
                end
                if !sret
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: did not have sret when expected")
                        println(io, "f=", string(f))
                        println(io, "inst=", string(inst))
                        println(io, "st=", string(st))
                        println(io, "fop=", string(fop))
                    end
    		    throw(AssertionError(msg))
                end

		@assert sret_elty !== nothing
                if temp === nothing
                    nb = IRBuilder()
                    position!(nb, first(instructions(first(blocks(f)))))
                    temp = alloca!(nb, sret_elty)
                end
                if mayread
                    nb = IRBuilder()
                    position!(nb, st)
                    ld = load!(nb, sret_elty, operands(inst)[1])
                    store!(nb, ld, temp)
                end
                if maywrite
                    nb = IRBuilder()
                    position!(nb, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(st)))
                    ld = load!(nb, sret_elty, temp)
                    si = store!(nb, ld, operands(inst)[1])
                    julia_post_cache_store(si.ref, nb.ref, reinterpret(Ptr{UInt64}, C_NULL))
                end
            end

            if temp !== nothing
                replace_uses!(inst, temp)
            end
            LLVM.API.LLVMInstructionEraseFromParent(inst)
        end
    end
    return nothing
end

function pre_attr!(mod::LLVM.Module, run_attr)
    if run_attr
	    for fn in functions(mod)
		if isempty(blocks(fn))
		    continue
		end
		attrs = collect(function_attributes(fn))
		prevent = any(
		    kind(attr) == kind(StringAttribute("enzyme_preserve_primal")) for attr in attrs
		)
		if !prevent
		    continue
		end
        
		if linkage(fn) == LLVM.API.LLVMInternalLinkage
		    push!(LLVM.function_attributes(fn), StringAttribute("restorelinkage_internal"))
		    linkage!(fn, LLVM.API.LLVMExternalLinkage)
		end
        
		if linkage(fn) == LLVM.API.LLVMPrivateLinkage
		    push!(LLVM.function_attributes(fn), StringAttribute("restorelinkage_private"))
		    linkage!(fn, LLVM.API.LLVMExternalLinkage)
		end
		continue

		if !has_fn_attr(fn, EnumAttribute("noinline"))
		    push!(LLVM.function_attributes(fn), EnumAttribute("noinline"))
		    push!(LLVM.function_attributes(fn), StringAttribute("remove_noinline"))
		end
		
		if !has_fn_attr(fn, EnumAttribute("optnone"))
		    push!(LLVM.function_attributes(fn), EnumAttribute("optnone"))
		    push!(LLVM.function_attributes(fn), StringAttribute("remove_optnone"))
		end
	    end
    end
    return nothing
    
    for fn in collect(functions(mod))
        if isempty(blocks(fn))
            continue
        end
        if linkage(fn) != LLVM.API.LLVMInternalLinkage &&
           linkage(fn) != LLVM.API.LLVMPrivateLinkage
            continue
        end

        fty = LLVM.FunctionType(fn)
        nfn = LLVM.Function(mod, "enzyme_attr_prev_" * LLVM.name(enzymefn), fty)
        LLVM.IRBuilder() do builder
            entry = BasicBlock(nfn, "entry")
            position!(builder, entry)
            cv = call!(fn, [LLVM.UndefValue(ty) for ty in parameters(fty)])
            LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1), attr)
            if LLVM.return_type(fty) == LLVM.VoidType()
                ret!(builder)
            else
                ret!(builder, cv)
            end
        end
    end
end

function post_attr!(mod::LLVM.Module, run_attr)
    if run_attr
	    for fn in functions(mod)
		if has_fn_attr(fn, StringAttribute("restorelinkage_internal"))
		    delete!(LLVM.function_attributes(fn), StringAttribute("restorelinkage_internal"))
		    linkage!(fn, LLVM.API.LLVMInternalLinkage)
		end
		
		if has_fn_attr(fn, StringAttribute("restorelinkage_private"))
		    delete!(LLVM.function_attributes(fn), StringAttribute("restorelinkage_private"))
		    linkage!(fn, LLVM.API.LLVMPrivateLinkage)
		end

		if has_fn_attr(fn, StringAttribute("remove_noinline"))
		    delete!(LLVM.function_attributes(fn), EnumAttribute("noinline"))
		    delete!(LLVM.function_attributes(fn), StringAttribute("remove_noinline"))
		end
		
		if has_fn_attr(fn, StringAttribute("remove_optnone"))
		    delete!(LLVM.function_attributes(fn), EnumAttribute("optnone"))
		    delete!(LLVM.function_attributes(fn), StringAttribute("remove_optnone"))
		end
	    end
    end
    return nothing
end

function prop_global!(g::LLVM.GlobalVariable)
    newfns = String[]
    changed = false
    todo = Tuple{Vector{Cuint},LLVM.Value}[]
    for u in LLVM.uses(g)
        u = LLVM.user(u)
        push!(todo, (Cuint[], u))
    end
    while length(todo) > 0
        path, var = pop!(todo)
        if isa(var, LLVM.LoadInst)
            B = IRBuilder()
            position!(B, var)
            res = LLVM.initializer(g)
            for p in path
                res = extract_value!(B, res, p)
            end
            changed = true
            for u in LLVM.uses(var)
                u = LLVM.user(u)
                if isa(u, LLVM.CallInst)
                    f2 = LLVM.called_operand(u)
                    if isa(f2, LLVM.Function)
                        push!(newfns, LLVM.name(f2))
                    end
                end
            end
	    if value_type(var) != value_type(res)
		al = alloca!(B, value_type(res))
		store!(B, res, al)
		res = load!(B, value_type(var), al)
	    end
            replace_uses!(var, res)
            eraseInst(LLVM.parent(var), var)
            continue
        end
        if isa(var, LLVM.AddrSpaceCastInst)
            for u in LLVM.uses(var)
                u = LLVM.user(u)
                push!(todo, (path, u))
            end
            continue
        end
        if isa(var, LLVM.ConstantExpr) && opcode(var) == LLVM.API.LLVMAddrSpaceCast
            for u in LLVM.uses(var)
                u = LLVM.user(u)
                push!(todo, (path, u))
            end
            continue
        end
        if isa(var, LLVM.GetElementPtrInst)
            if all(isa(v, LLVM.ConstantInt) for v in operands(var)[2:end])
                if LLVM.API.LLVMConstIntGetZExtValue(operands(var)[2]) == 0
                    for u in LLVM.uses(var)
                        u = LLVM.user(u)
                        push!(
                            todo,
                            (
                                vcat(
                                    path,
                                    collect((
                                        convert(Cuint, v) for v in operands(var)[3:end]
                                    )),
                                ),
                                u,
                            ),
                        )
                    end
                end
                continue
            end
        end
    end
    return changed, newfns
end

# From https://llvm.org/doxygen/IR_2Instruction_8cpp_source.html#l00959
function mayWriteToMemory(@nospecialize(inst::LLVM.Instruction); err_is_readonly::Bool = false)::Bool
    # we will ignore fense here
    if isa(inst, LLVM.StoreInst)
        return true
    end
    if isa(inst, LLVM.VAArgInst)
        return true
    end
    if isa(inst, LLVM.AtomicCmpXchgInst)
        return true
    end
    if isa(inst, LLVM.AtomicRMWInst)
        return true
    end
    if isa(inst, LLVM.CatchPadInst)
        return true
    end
    if isa(inst, LLVM.CatchRetInst)
        return true
    end
    if isa(inst, LLVM.CallInst) || isa(inst, LLVM.InvokeInst) || isa(inst, LLVM.CallBrInst)
        idx = reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex)
        count = LLVM.API.LLVMGetCallSiteAttributeCount(inst, idx)

        Attrs = Base.unsafe_convert(
            Ptr{LLVM.API.LLVMAttributeRef},
            Libc.malloc(sizeof(LLVM.API.LLVMAttributeRef) * count),
        )
        LLVM.API.LLVMGetCallSiteAttributes(inst, idx, Attrs)
        for j = 1:count
            attr = LLVM.Attribute(unsafe_load(Attrs, j))
            if kind(attr) == kind(EnumAttribute("readnone"))
                return false
            end
            if kind(attr) == kind(EnumAttribute("readonly"))
                return false
            end
            # Note out of spec, and only legal in context of removing unused calls
            if kind(attr) == kind(StringAttribute("enzyme_error")) && err_is_readonly
                return false
            end
            if kind(attr) == kind(StringAttribute("memory"))
                if is_readonly(MemoryEffect(value(attr)))
                    return false
                end
            end
        end
        Libc.free(Attrs)
        return true
    end
    # Ignoring load unordered case
    return false
end

function remove_readonly_unused_calls!(fn::LLVM.Function, next::Set{String})
    calls = LLVM.CallInst[]

    hasUser = false
    for u in LLVM.uses(fn)
        un = LLVM.user(u)

        # Only permit call users
        if !isa(un, LLVM.CallInst)
            return false
        end
        un = un::LLVM.CallInst

        # Passing the fn as an argument is not permitted
        for op in collect(operands(un))[1:end-1]
            if op == fn
                return false
            end
        end

        # Something with a user is not permitted
        for u2 in LLVM.uses(un)
            hasUser = true
            break
        end
        push!(calls, un)
    end

    done = Set{LLVM.Function}()
    todo = LLVM.Function[fn]

    while length(todo) != 0
        cur = pop!(todo)
        if cur in done
            continue
        end
        push!(done, cur)

        if is_readonly(cur)
            continue
        end

        if LLVM.name(cur) == "julia.safepoint"
            continue
        end

        if isempty(blocks(cur))
            return false
        end

        err_is_readonly = true

        for bb in blocks(cur)
            for inst in instructions(bb)
                if !mayWriteToMemory(inst; err_is_readonly)
                    continue
                end
                if isa(inst, LLVM.CallInst)

                    fn2 = LLVM.called_operand(inst)
                    if isa(fn2, LLVM.Function)
                        push!(todo, fn2)
                        continue
                    end
                end
                return false
            end
        end
    end

    changed = set_readonly!(fn)

    if length(calls) == 0 || hasUser || !is_nounwind(fn)
        return changed
    end

    for c in calls
        parentf = LLVM.parent(LLVM.parent(c))
        push!(next, LLVM.name(parentf))
        LLVM.API.LLVMInstructionEraseFromParent(c)
    end
    push!(next, LLVM.name(fn))
    return true
end

function propagate_returned!(mod::LLVM.Module)
    globs = LLVM.GlobalVariable[]
    for g in globals(mod)
        if linkage(g) == LLVM.API.LLVMInternalLinkage ||
           linkage(g) == LLVM.API.LLVMPrivateLinkage
            if !isconstant(g)
                continue
            end
            push!(globs, g)
        end
    end
    todo = collect(functions(mod))
    while true
        next = Set{String}()
        changed = false
        for g in globs
            tc, tn = prop_global!(g)
            changed |= tc
            for f in tn
                push!(next, f)
            end
        end
        tofinalize = Tuple{LLVM.Function,Bool,Vector{Int64}}[]
        for fn in functions(mod)
            if isempty(blocks(fn))
                continue
            end
            if remove_readonly_unused_calls!(fn, next)
                changed = true
            end
            has_user = false
	    for u in LLVM.uses(fn)
		has_user = true
		break
	    end
            attrs = collect(function_attributes(fn))
            prevent = any(
                kind(attr) == kind(StringAttribute("enzyme_preserve_primal")) for
                attr in attrs
            )
            # if any(kind(attr) == kind(EnumAttribute("noinline")) for attr in attrs) 
            #     continue
            # end
            argn = nothing
            toremove = Int64[]
	    # Don't bother with functions we're about to delete anyways
	    if has_user
            for (i, arg) in enumerate(parameters(fn))
                if any(
                    kind(attr) == kind(EnumAttribute("returned")) for
                    attr in collect(parameter_attributes(fn, i))
                )
                    argn = i
                end

                # remove unused sret-like
                if !prevent &&
                   (
                       linkage(fn) == LLVM.API.LLVMInternalLinkage ||
                       linkage(fn) == LLVM.API.LLVMPrivateLinkage
                   ) &&
                   any(
                       kind(attr) == kind(EnumAttribute("nocapture")) for
                       attr in collect(parameter_attributes(fn, i))
                   )
                    val = nothing
                    illegalUse = false
                    torem = LLVM.Instruction[]

                    for u in LLVM.uses(fn)
                        un = LLVM.user(u)
                        if !isa(un, LLVM.CallInst)
                            illegalUse = true
                            break
                        end
                        ops = collect(operands(un))[1:end-1]
                        bad = false
                        for op in ops
                            if op == fn
                                bad = true
                                break
                            end
                        end
                        if bad
                            illegalUse = true
                            break
                        end
                        if !isa(ops[i], LLVM.AllocaInst) && !isa(ops[i], LLVM.UndefValue) && !isa(ops[i], LLVM.PoisonValue)
                            illegalUse = true
                            break
                        end
                        seenfn = false
                        todo = LLVM.Instruction[]
                        if isa(ops[i], LLVM.AllocaInst)
			for u2 in LLVM.uses(ops[i])
                            un2 = LLVM.user(u2)
                            push!(todo, un2)
                        end
			end
                        while length(todo) > 0
                            un2 = pop!(todo)
                            if isa(un2, LLVM.BitCastInst)
                                push!(torem, un2)
                                for u3 in LLVM.uses(un2)
                                    un3 = LLVM.user(u3)
                                    push!(todo, un3)
                                end
                                continue
                            end
                            if isa(un2, LLVM.GetElementPtrInst)
                                push!(torem, un2)
                                for u3 in LLVM.uses(un2)
                                    un3 = LLVM.user(u3)
                                    push!(todo, un3)
                                end
                                continue
                            end
                            if !isa(un2, LLVM.CallInst)
                                illegalUse = true
                                break
                            end
                            ff = LLVM.called_operand(un2)
                            if !isa(ff, LLVM.Function)
                                illegalUse = true
                                break
                            end
                            if un2 == un && !seenfn
                                seenfn = true
                                continue
                            end
                            intr = LLVM.API.LLVMGetIntrinsicID(ff)
                            if intr == LLVM.Intrinsic("llvm.lifetime.start").id
                                push!(torem, un2)
                                continue
                            end
                            if intr == LLVM.Intrinsic("llvm.lifetime.end").id
                                push!(torem, un2)
                                continue
                            end
                            if LLVM.name(ff) != "llvm.enzyme.sret_use"
                                illegalUse = true
                                break
                            end
                            push!(torem, un2)
                        end
                        if illegalUse
                            break
                        end
                    end
                    if !illegalUse
                        for c in reverse(torem)
                            eraseInst(LLVM.parent(c), c)
                        end
                        B = IRBuilder()

                        position!(B, first(instructions(first(blocks(fn)))))

                        has_use = false
                        for _ in LLVM.uses(arg)
                            has_use = true
                            break
                        end

                        if has_use
                            argeltype = sret_ty(fn, i)
                            al = alloca!(B, argeltype)
                            if value_type(al) != value_type(arg)
                                al = addrspacecast!(B, al, value_type(arg))
                            end
                            LLVM.replace_uses!(arg, al)
                        end
                    end
                end

                # interprocedural const prop from callers of arg
                if !prevent && (
                    linkage(fn) == LLVM.API.LLVMInternalLinkage ||
                    linkage(fn) == LLVM.API.LLVMPrivateLinkage
                )
                    val = nothing
                    illegalUse = false
                    for u in LLVM.uses(fn)
                        un = LLVM.user(u)
                        if !isa(un, LLVM.CallInst)
                            illegalUse = true
                            break
                        end
                        ops = collect(operands(un))[1:end-1]
                        bad = false
                        for op in ops
                            if op == fn
                                bad = true
                                break
                            end
                        end
                        if bad
                            illegalUse = true
                            break
                        end
                        if isa(ops[i], LLVM.UndefValue) || isa(ops[i], LLVM.PoisonValue)
                            continue
                        end
                        if ops[i] == arg
                            continue
                        end
                        if isa(ops[i], LLVM.Constant)
                            if val === nothing
                                val = ops[i]
                            else
                                if val != ops[i]
                                    illegalUse = true
                                    break
                                end
                            end
                            continue
                        end
                        illegalUse = true
                        break
                    end
                    if !illegalUse
                        if val === nothing
                            val = LLVM.UndefValue(value_type(arg))
                        end
                        for u in LLVM.uses(arg)
                            u = LLVM.user(u)
                            if isa(u, LLVM.CallInst)
                                f2 = LLVM.called_operand(u)
                                if isa(f2, LLVM.Function)
                                    push!(next, LLVM.name(f2))
                                end
                            end
                            changed = true
                        end
                        LLVM.replace_uses!(arg, val)
                    end
                end
                
		# see if there are no users of the value (excluding recursive/return)
                if !prevent
			baduse = false
			for u in LLVM.uses(arg)
			    u = LLVM.user(u)
			    if argn == i && LLVM.API.LLVMIsAReturnInst(u) != C_NULL
				continue
			    end
			    if !isa(u, LLVM.CallInst)
				baduse = true
				break
			    end
			    if LLVM.called_operand(u) != fn
				baduse = true
				break
			    end
			    for (si, op) in enumerate(operands(u))
				if si == i
				    continue
				end
				if op == arg
				    baduse = true
				    break
				end
			    end
			    if baduse
				break
			    end
			end
			if !baduse
			    push!(toremove, i - 1)
			end
		end
            end
	    end
            illegalUse = !(
                linkage(fn) == LLVM.API.LLVMInternalLinkage ||
                linkage(fn) == LLVM.API.LLVMPrivateLinkage
            )
            hasAnyUse = false
            for u in LLVM.uses(fn)
                un = LLVM.user(u)
                if !isa(un, LLVM.CallInst)
                    illegalUse = true
                    continue
                end
                ops = collect(operands(un))[1:end-1]
                bad = false
                for op in ops
                    if op == fn
                        bad = true
                        break
                    end
                end
                if bad
                    illegalUse = true
                    continue
                end
                if argn !== nothing
                    hasUse = false
                    for u in LLVM.uses(un)
                        hasUse = true
                        break
                    end
                    if hasUse
                        changed = true
                        push!(next, LLVM.name(LLVM.parent(LLVM.parent(un))))
                        LLVM.replace_uses!(un, ops[argn])
                    end
                else
                    for u in LLVM.uses(un)
                        u = LLVM.user(u)
                        if u isa LLVM.CallInst
                            op = LLVM.called_operand(u)
                            if op isa LLVM.Function && LLVM.name(op) == "llvm.enzymefakeread"
                                continue
                            end
                        end
                        hasAnyUse = true
                        break
                    end
                end
            end
            #if the function return has no users whatsoever, remove it
            if argn === nothing &&
               !hasAnyUse &&
               LLVM.return_type(LLVM.function_type(fn)) != LLVM.VoidType()
                argn = -1
            end
            if argn === nothing && length(toremove) == 0
                continue
            end
            if !illegalUse
                push!(tofinalize, (fn, argn === nothing, toremove))
            end
        end
        for (fn, keepret, toremove) in tofinalize
            todo = LLVM.CallInst[]
            for u in LLVM.uses(fn)
                un = LLVM.user(u)
                push!(next, LLVM.name(LLVM.parent(LLVM.parent(un))))
            end
            delete_writes_into_removed_args(fn, toremove, keepret)
            nm = LLVM.name(fn)
            #try
                nfn = LLVM.Function(
                    API.EnzymeCloneFunctionWithoutReturnOrArgs(fn, keepret, toremove),
                )
                for u in LLVM.uses(fn)
                    un = LLVM.user(u)
                    push!(todo, un)
                end
                for un in todo
                    md = metadata(un)
                    if !keepret && haskey(md, LLVM.MD_range)
                        delete!(md, LLVM.MD_range)
                    end
                    API.EnzymeSetCalledFunction(un, nfn, toremove)
                end
                eraseInst(mod, fn)
                changed = true
            # catch e
            #    break
            #end
        end
        if !changed
            break
        else
            todo = LLVM.Function[]
            for name in next
                fn = functions(mod)[name]
                if linkage(fn) == LLVM.API.LLVMInternalLinkage ||
                   linkage(fn) == LLVM.API.LLVMPrivateLinkage
                    has_user = false
                    for u in LLVM.uses(fn)
                        has_user = true
                        break
                    end
                    if !has_user
                        LLVM.API.LLVMDeleteFunction(fn)
                    end
                end
                push!(todo, fn)
            end
        end
    end
end

function delete_writes_into_removed_args(fn::LLVM.Function, toremove::Vector{Int64}, keepret::Bool)
    args = collect(parameters(fn))
    if !keepret
        for u in LLVM.uses(fn)
            u = LLVM.user(u)
            replace_uses!(u, LLVM.UndefValue(value_type(u)))
        end
    end
    for tr in toremove
        tr = tr + 1
        todorep = Tuple{LLVM.Instruction, LLVM.Value}[]
        for opv in LLVM.uses(args[tr])
            u = LLVM.user(opv)
            push!(todorep, (u, args[tr]))
        end
        toerase = LLVM.Instruction[]
        while length(todorep) != 0
            cur, cval = pop!(todorep)
            if isa(cur, LLVM.StoreInst)
                if operands(cur)[2] == cval
                    LLVM.API.LLVMInstructionEraseFromParent(nphi)
                    continue
                end
            end
            if isa(cur, LLVM.GetElementPtrInst) ||
               isa(cur, LLVM.BitCastInst) ||
               isa(cur, LLVM.AddrSpaceCastInst)
                for opv in LLVM.uses(cur)
                    u = LLVM.user(opv)
                    push!(todorep, (u, cur))
                end
                continue
            end
            if isa(cur, LLVM.CallInst)
                cf = LLVM.called_operand(cur)
                if cf == fn
                    baduse = false
                    for (i, v) in enumerate(operands(cur))
                        if i-1 in toremove
                            continue
                        end
                        if v == cval
                            baduse = true
                        end
                    end
                    if !baduse
                        continue
                    end
                end
            end
            if !keepret && LLVM.API.LLVMIsAReturnInst(cur) != C_NULL
                LLVM.API.LLVMSetOperand(cur, 0, LLVM.UndefValue(value_type(cval)))
                continue
	        end
            throw(AssertionError("Deleting argument with an unknown dependency, $(string(cur)) uses $(string(cval))"))
        end
    end
end

function validate_return_roots!(mod::LLVM.Module)
    for f in functions(mod)
        srets = []
        enzyme_srets = Int[]
        enzyme_srets_v = Int[]
        rroots = Int[]
        rroots_v = Int[]
        sretkind = kind(if LLVM.version().major >= 12
            TypeAttribute("sret", LLVM.Int32Type())
        else
            EnumAttribute("sret")
        end)
        for (i, a) in enumerate(parameters(f))
            for attr in collect(parameter_attributes(f, i))
                if isa(attr, StringAttribute)
                    if kind(attr) == "enzymejl_returnRoots"
                        push!(rroots, i)
                    end
                    if kind(attr) == "enzymejl_returnRoots_v"
                        push!(rroots_v, i)
                    end
                    if kind(attr) == "enzyme_sret"
                        push!(enzyme_srets, i)
                    end
                    if kind(attr) == "enzyme_sret_v"
                        push!(enzyme_srets, i)
                    end
                end
                if kind(attr) == sretkind
                    push!(srets, (i, attr))
                end
            end
        end
        if length(enzyme_srets) >= 1 && length(srets) == 0
            @assert enzyme_srets[1] == 1
            VT = LLVM.VoidType()
            if length(enzyme_srets) == 1 &&
               LLVM.return_type(LLVM.function_type(f)) == VT &&
               length(enzyme_srets_v) == 0
                # Upgrading to sret requires writeonly
                if !any(
                    kind(attr) == kind(EnumAttribute("writeonly")) for
                    attr in collect(parameter_attributes(f, 1))
                )
                    msg = sprint() do io::IO
                        println(io, "Enzyme internal error (not writeonly sret)")
                        println(io, string(f))
                        println(
                            io,
                            "collect(parameter_attributes(f, 1))=",
                            collect(parameter_attributes(f, 1)),
                        )
                    end
                    throw(AssertionError(msg))
                end

                alty = nothing
                for u in LLVM.uses(f)
                    u = LLVM.user(u)
                    @assert isa(u, LLVM.CallInst)
                    @assert LLVM.called_operand(u) == f
                    alop = operands(u)[1]
                    if !isa(alop, LLVM.AllocaInst)
                        msg = sprint() do io::IO
                            println(io, "Enzyme internal error (!isa(alop, LLVM.AllocaInst))")
                            println(io, "alop=", alop)
                            println(io, "u=", u)
                            println(io, "f=", string(f))
                        end
                        throw(AssertionError(msg))

                    end
                    @assert isa(alop, LLVM.AllocaInst)
                    nty = API.EnzymeAllocaType(alop)
                    if alty === nothing
                        alty = nty
                    else
                        @assert alty == nty
                    end
                    attr = if LLVM.version().major >= 12
                        TypeAttribute("sret", alty)
                    else
                        EnumAttribute("sret")
                    end
                    LLVM.API.LLVMAddCallSiteAttribute(
                        u,
                        LLVM.API.LLVMAttributeIndex(1),
                        attr,
                    )
                    LLVM.API.LLVMRemoveCallSiteStringAttribute(
                        u,
                        LLVM.API.LLVMAttributeIndex(1),
                        "enzyme_sret",
                        length("enzyme_sret"),
                    )
                end
                @assert alty !== nothing
                attr = if LLVM.version().major >= 12
                    TypeAttribute("sret", alty)
                else
                    EnumAttribute("sret")
                end

                push!(parameter_attributes(f, 1), attr)
                delete!(parameter_attributes(f, 1), StringAttribute("enzyme_sret"))
                srets = [(1, attr)]
                enzyme_srets = Int[]
            else

                enzyme_srets2 = Int[]
                for idx in enzyme_srets
                    alty = nothing
                    bad = false
                    for u in LLVM.uses(f)
                        u = LLVM.user(u)
                        @assert isa(u, LLVM.CallInst)
                        @assert LLVM.called_operand(u) == f
                        alop = operands(u)[1]
                        @assert isa(alop, LLVM.AllocaInst)
                        nty = API.EnzymeAllocaType(alop)
                        if any_jltypes(nty)
                            bad = true
                        end
                        LLVM.API.LLVMRemoveCallSiteStringAttribute(
                            u,
                            LLVM.API.LLVMAttributeIndex(idx),
                            "enzyme_sret",
                            length("enzyme_sret"),
                        )
                    end
                    if !bad
                        delete!(
                            parameter_attributes(f, idx),
                            StringAttribute("enzyme_sret"),
                        )
                    else
                        push!(enzyme_srets2, idx)
                    end
                end
                enzyme_srets = enzyme_srets2

                if length(enzyme_srets) != 0
                    msg = sprint() do io::IO
                        println(io, "Enzyme internal error (length(enzyme_srets) != 0)")
                        println(io, "f=", string(f))
                        println(io, "enzyme_srets=", enzyme_srets)
                        println(io, "enzyme_srets_v=", enzyme_srets_v)
                        println(io, "srets=", srets)
                        println(io, "rroots=", rroots)
                        println(io, "rroots_v=", rroots_v)
                    end
                    throw(AssertionError(msg))
                end
            end
        end
        @assert length(enzyme_srets_v) == 0
        for (i, attr) in srets
            @assert i == 1
        end
        for i in rroots
            @assert length(srets) != 0
            @assert i == 2
        end
        # illegal
        for i in rroots_v
            @assert false
        end
    end
end

function checkNoAssumeFalse(mod::LLVM.Module, shouldshow::Bool = false)
    for f in functions(mod)
        for bb in blocks(f), inst in instructions(bb)
            if !isa(inst, LLVM.CallInst)
                continue
            end
            intr = LLVM.API.LLVMGetIntrinsicID(LLVM.called_operand(inst))
            if intr != LLVM.Intrinsic("llvm.assume").id
                continue
            end
            op = operands(inst)[1]
            if isa(op, LLVM.ConstantInt)
                op2 = convert(Bool, op)
                if !op2
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: non-constant assume condition")
                        println(io, "mod=", string(mod))
                        println(io, "f=", string(f))
                        println(io, "bb=", string(bb))
                        println(io, "op2=", string(op2))
                    end
                    throw(AssertionError(msg))
                end
            end
            if isa(op, LLVM.ICmpInst)
                if predicate_int(op) == LLVM.API.LLVMIntNE &&
                   operands(op)[1] == operands(op)[2]
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: non-icmp assume condition")
                        println(io, "mod=", string(mod))
                        println(io, "f=", string(f))
                        println(io, "bb=", string(bb))
                        println(io, "op=", string(op))
                    end
                    throw(AssertionError(msg))
                end
            end
        end
    end
end

function removeDeadArgs!(mod::LLVM.Module, tm::LLVM.TargetMachine, post_gc_fixup::Bool)
    # We need to run globalopt first. This is because remove dead args will otherwise
    # take internal functions and replace their args with undef. Then on LLVM up to 
    # and including 12 (but fixed 13+), Attributor will incorrectly change functions that
    # call code with undef to become unreachable, even when there exist other valid
    # callsites. See: https://godbolt.org/z/9Y3Gv6q5M
    run!(GlobalDCEPass(), mod)

    # Prevent dead-arg-elimination of functions which we may require args for in the derivative
    funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[], vararg = true)
    if LLVM.version().major <= 15
        func, _ = get_function!(
            mod,
            "llvm.enzymefakeuse",
            funcT,
            LLVM.Attribute[EnumAttribute("readnone"), EnumAttribute("nofree")],
        )
        rfunc, _ = get_function!(
            mod,
            "llvm.enzymefakeread",
            funcT,
            LLVM.Attribute[
                EnumAttribute("readonly"),
                EnumAttribute("nofree"),
                EnumAttribute("argmemonly"),
            ],
        )
        sfunc, _ = get_function!(
            mod,
            "llvm.enzyme.sret_use",
            funcT,
            LLVM.Attribute[
                EnumAttribute("readonly"),
                EnumAttribute("nofree"),
                EnumAttribute("argmemonly"),
            ],
        )
    else
        func, _ = get_function!(
            mod,
            "llvm.enzymefakeuse",
            funcT,
            LLVM.Attribute[EnumAttribute("memory", NoEffects.data), EnumAttribute("nofree")],
        )
        rfunc, _ = get_function!(
            mod,
            "llvm.enzymefakeread",
            funcT,
            LLVM.Attribute[EnumAttribute("memory", ReadOnlyArgMemEffects.data), EnumAttribute("nofree")],
        )
        sfunc, _ = get_function!(
            mod,
            "llvm.enzyme.sret_use",
            funcT,
            LLVM.Attribute[EnumAttribute("memory", ReadOnlyArgMemEffects.data), EnumAttribute("nofree")],
        )
    end

    for fn in functions(mod)
        if isempty(blocks(fn))
            continue
        end

        rt = LLVM.return_type(LLVM.function_type(fn))
        if rt isa LLVM.PointerType && addrspace(rt) == 10
            for u in LLVM.uses(fn)
                u = LLVM.user(u)
                if isa(u, LLVM.CallInst)
                    B = IRBuilder()
                    nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(u))
                    position!(B, nextInst)
                    cl = call!(B, funcT, rfunc, LLVM.Value[u])
                    LLVM.API.LLVMAddCallSiteAttribute(
                        cl,
                        LLVM.API.LLVMAttributeIndex(1),
                        EnumAttribute("nocapture"),
                    )
                end
            end 
        end

        # Ensure that interprocedural optimizations do not delete the use of returnRoots (or shadows)
        # if inactive sret, this will only occur on 2. If active sret, inactive retRoot, can on 3, and
        # active both can occur on 4. If the original sret is removed (at index 1) we no longer need
        # to preserve this.
        if post_gc_fixup
        for idx in (2, 3, 4)
            if length(collect(parameters(fn))) >= idx && any(
                (
                    kind(attr) == kind(StringAttribute("enzymejl_returnRoots")) ||
                    kind(attr) == kind(StringAttribute("enzymejl_returnRoots_v"))
                ) for attr in collect(parameter_attributes(fn, idx))
            )
                for u in LLVM.uses(fn)
                    u = LLVM.user(u)
                    @assert isa(u, LLVM.CallInst)
                    B = IRBuilder()
                    nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(u))
                    position!(B, nextInst)
                    inp = operands(u)[idx]
                    cl = call!(B, funcT, rfunc, LLVM.Value[inp])
                    if isa(value_type(inp), LLVM.PointerType)
                        LLVM.API.LLVMAddCallSiteAttribute(
                            cl,
                            LLVM.API.LLVMAttributeIndex(1),
                            EnumAttribute("nocapture"),
                        )
                    end
                end
            end
        end
        end
        sretkind = kind(if LLVM.version().major >= 12
            TypeAttribute("sret", LLVM.Int32Type())
        else
            EnumAttribute("sret")
        end)
        for idx in (1, 2)
            if length(collect(parameters(fn))) < idx
                continue
            end
            attrs = collect(parameter_attributes(fn, idx))
            if any(
                (
                    kind(attr) == sretkind ||
                    kind(attr) == kind(StringAttribute("enzyme_sret")) ||
                    kind(attr) == kind(StringAttribute("enzyme_sret_v"))
                ) for attr in attrs
               ) && any_jltypes(sret_ty(fn, idx))
                for u in LLVM.uses(fn)
                    u = LLVM.user(u)
                    if isa(u, LLVM.ConstantExpr)
                        u = LLVM.user(only(LLVM.uses(u)))
                    end
                    if !isa(u, LLVM.CallInst)
                        continue
                    end
                    @assert isa(u, LLVM.CallInst)
                    B = IRBuilder()
                    nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(u))
                    position!(B, nextInst)
                    inp = operands(u)[idx]
                    cl = call!(B, funcT, sfunc, LLVM.Value[inp])
                    if isa(value_type(inp), LLVM.PointerType)
                        LLVM.API.LLVMAddCallSiteAttribute(
                            cl,
                            LLVM.API.LLVMAttributeIndex(1),
                            EnumAttribute("nocapture"),
                        )
                    end
                end
            end
        end
        attrs = collect(function_attributes(fn))
        prevent = any(
            kind(attr) == kind(StringAttribute("enzyme_preserve_primal")) for attr in attrs
        )
        # && any(kind(attr) == kind(StringAttribute("enzyme_math")) for attr in attrs)
        if prevent
            B = IRBuilder()
            position!(B, first(instructions(first(blocks(fn)))))
            call!(B, funcT, func, LLVM.Value[p for p in parameters(fn)])
        end
    end
    propagate_returned!(mod)
    LLVM.@dispose pb = NewPMPassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
		register!(pb, RestoreAllocaType())
        add!(pb, NewPMModulePassManager()) do mpm
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, AllocOptPass())
                add!(fpm, RestoreAllocaType())
                add!(fpm, SROAPass())
                add!(fpm, EarlyCSEPass())
            end
        end
        LLVM.run!(pb, mod)
    end
    propagate_returned!(mod)
    pre_attr!(mod, RunAttributor[])
    if RunAttributor[]
        API.EnzymeDetectReadonlyOrThrow(mod)
        LLVM.@dispose pb = NewPMPassBuilder() begin
            register!(pb, EnzymeAttributorPass())
            add!(pb, NewPMModulePassManager()) do mpm
                add!(mpm, EnzymeAttributorPass())
            end
            LLVM.run!(pb, mod)
        end
    end
    propagate_returned!(mod)
    LLVM.@dispose pb = NewPMPassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, EnzymeAttributorPass())
		register!(pb, RestoreAllocaType())
        add!(pb, NewPMModulePassManager()) do mpm
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, AllocOptPass())
                add!(fpm, RestoreAllocaType())
                add!(fpm, SROAPass())
            end
            if RunAttributor[]
                add!(mpm, EnzymeAttributorPass())
            end
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, EarlyCSEPass())
            end
        end
        LLVM.run!(pb, mod)
    end
    API.EnzymeDetectReadonlyOrThrow(mod)
    post_attr!(mod, RunAttributor[])
    propagate_returned!(mod)
    

    for u in LLVM.uses(rfunc)
        u = LLVM.user(u)
        eraseInst(LLVM.parent(u), u)
    end
    eraseInst(mod, rfunc)
    for u in LLVM.uses(sfunc)
        u = LLVM.user(u)
        eraseInst(LLVM.parent(u), u)
    end
    eraseInst(mod, sfunc)
    for fn in functions(mod)
        for b in blocks(fn)
            inst = first(LLVM.instructions(b))
            if isa(inst, LLVM.CallInst)
                fn = LLVM.called_operand(inst)
                if fn == func
                    eraseInst(b, inst)
                end
            end
        end
    end
    eraseInst(mod, func)
end

function safe_atomic_to_regular_store!(f::LLVM.Function)
    changed = false
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.StoreInst)
            continue
        end
        if !haskey(metadata(inst), "enzymejl_atomicgc")
            continue
        end
        Base.delete!(metadata(inst), "enzymejl_atomicgc")
        syncscope!(inst, LLVM.SyncScope("system"))
        ordering!(inst, LLVM.API.LLVMAtomicOrderingNotAtomic)
        changed = true
    end
    return changed
end


