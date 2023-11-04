function addNA(inst, node::LLVM.Metadata, MD)
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
            ty =value_type(inst)
            if isa(ty, LLVM.PointerType)
                if addrspace(ty) == 13
                    addNA(inst, aliasscope, LLVM.MD_alias_scope)
                end
            end
        end
    end
end

function source_elem(v)
    @static if LLVM.version() >= v"15"
        LLVM.LLVMType(LLVM.API.LLVMGetGEPSourceElementType(v))
    else
        eltype(value_type(operands(v)[1]))
    end
end

# If there is a phi node of a decayed value, Enzyme may need to cache it
# Here we force all decayed pointer phis to first addrspace from 10
function nodecayed_phis!(mod::LLVM.Module)
    # Simple handler to fix addrspace 11
    for f in functions(mod), bb in blocks(f)
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
            if addrspace(ty) != 11
                continue
            end
            push!(todo, inst)
        end

        for inst in todo
            ty = value_type(inst)
            nty = LLVM.PointerType(eltype(ty), 10)
            nvs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
            for (v, pb) in LLVM.incoming(inst)
                b = IRBuilder()
                position!(b, terminator(pb))
                while isa(v, LLVM.AddrSpaceCastInst)
                    v = operands(v)[1]
                end
                if value_type(v) != nty
                    v = addrspacecast!(b, v, nty)
                end
                push!(nvs, (v, pb))
            end
            nb = IRBuilder()
            position!(nb, inst)
            
            if !all(x->x[1]==nvs[1][1], nvs)
                nphi = phi!(nb, nty)
                append!(LLVM.incoming(nphi), nvs)
            else
                nphi = nvs[1][1]
            end
            
            position!(nb, nonphi)
            nphi = addrspacecast!(nb, nphi, ty)
            replace_uses!(inst, nphi)
            LLVM.API.LLVMInstructionEraseFromParent(inst)
        end
    end

    #complex handler for addrspace 13, which itself comes from a load of an
    # addrspace 10
    for f in functions(mod)
        offty = LLVM.IntType(8*sizeof(Int))
        i8 = LLVM.IntType(8)

        nty = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), 10)
        nextvs = Dict{LLVM.PHIInst, LLVM.PHIInst}()
        mtodo = Vector{LLVM.PHIInst}[]
        goffsets = Dict{LLVM.PHIInst, LLVM.PHIInst}()
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
                if addrspace(ty) != 13
                    continue
                end
                push!(todo, inst)
                nb = IRBuilder()
                position!(nb, inst)
                nphi = phi!(nb, nty, "nodecayed." * LLVM.name(inst))
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
            nvs = Tuple{LLVM.Value, LLVM.BasicBlock}[]
            offsets = Tuple{LLVM.Value, LLVM.BasicBlock}[]
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
                b = IRBuilder()
                position!(b, terminator(pb))

                offset = LLVM.ConstantInt(offty, 0)

                while true
                    if isa(v, LLVM.AddrSpaceCastInst) || isa(v, LLVM.BitCastInst)
                        v = operands(v)[1]
                        continue
                    end
                
                    if isa(v, LLVM.PHIInst)
                        push!(offsets, (nuwadd!(b, offset, goffsets[v]), pb))
                        push!(nvs, (nextvs[v], pb))
                        done = true
                        break
                    end

                    if isa(v, LLVM.GetElementPtrInst)
                        offset = nuwadd!(b, offset, API.EnzymeComputeByteOffsetOfGEP(b, v, offty))
                        v = operands(v)[1]
                        continue
                    end

                    undeforpoison = isa(v, LLVM.UndefValue)
                    @static if LLVM.version() >= v"12"
                        undeforpoison |= isa(v, LLVM.PoisonValue)
                    end
                    if undeforpoison
                        push!(offsets, (LLVM.ConstantInt(offty, 0), pb))
                        push!(nvs, (LLVM.UndefValue(nty), pb))
                        done = true
                        break
                    end

                    break
                end

                if done
                    continue
                end
                
                if !isa(v, LLVM.LoadInst)
                    println(string(f))
                    @show v, inst
                end
                @assert isa(v, LLVM.LoadInst)
                
                v = operands(v)[1]

                while isa(v, LLVM.AddrSpaceCastInst) || isa(v, LLVM.BitCastInst)
                    v = operands(v)[1]
                end
                if eltype(value_type(v)) != LLVM.StructType(LLVM.LLVMType[])
                    v = bitcast!(b, v, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), addrspace(value_type(v))))
                end
                if value_type(v) != nty
                    println(string(f))
                    @show v, inst, nty
                end
                @assert value_type(v) == nty
                push!(nvs, (v, pb))
                push!(offsets, (offset, pb))
            end

            nb = IRBuilder()
            position!(nb, inst)
            
            offset = goffsets[inst]
            append!(LLVM.incoming(offset), offsets)
            if all(x->x[1]==offsets[1][1], offsets)
                offset = offsets[1][1]
            end

            nphi = nextvs[inst]
            if !all(x->x[1]==nvs[1][1], nvs)
                append!(LLVM.incoming(nphi), nvs)
            else
                replace_uses!(nphi, nvs[1][1])
                LLVM.API.LLVMInstructionEraseFromParent(nphi)
                nphi = nvs[1][1]
            end

            position!(nb, nonphi)
            nphi = bitcast!(nb, nphi, LLVM.PointerType(ty, 10))
            nphi = addrspacecast!(nb, nphi, LLVM.PointerType(ty, 11))
            nphi = load!(nb, ty, nphi)
            if !isa(offset, LLVM.ConstantInt) || convert(Int64, offset) != 0
                nphi = bitcast!(nb, nphi, LLVM.PointerType(i8, 13))
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
    return nothing
end

function fix_decayaddr!(mod::LLVM.Module)
    for f in functions(mod)
        invalid = LLVM.AddrSpaceCastInst[]
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
							 LLVM.API.LLVMSetOperand(st, 2-1, operands(inst)[1])
							 continue
						 end
					 end
					 if !isa(st, LLVM.CallInst)
						  bt = GPUCompiler.backtrace(st)
						  msg = sprint() do io::IO
							  println(io, string(f))
							  println(io, inst)
							  println(io, st)
							  print(io, "Illegal decay of nonnull\n")
							  if bt !== nothing
								  print(io,"\nCaused by:")
								  Base.show_backtrace(io, bt)
								  println(io)
							  end
						  end
						  throw(AssertionError(msg))
                end
                
                fop = operands(st)[end]
                
                intr = LLVM.API.LLVMGetIntrinsicID(fop)

                if intr == LLVM.Intrinsic("llvm.memcpy").id || intr == LLVM.Intrinsic("llvm.memmove").id || intr == LLVM.Intrinsic("llvm.memset").id
                    newvs = LLVM.Value[]
                    for (i, v) in enumerate(operands(st)[1:end-1])
                        if v == inst
                            LLVM.API.LLVMSetOperand(st, i-1, operands(inst)[1])
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

                    for idx = [LLVM.API.LLVMAttributeFunctionIndex, LLVM.API.LLVMAttributeReturnIndex, [LLVM.API.LLVMAttributeIndex(i) for i in 1:(length(operands(st))-1)]...]
                        count = LLVM.API.LLVMGetCallSiteAttributeCount(st, idx);
                        
                        Attrs = Base.unsafe_convert(Ptr{LLVM.API.LLVMAttributeRef}, Libc.malloc(sizeof(LLVM.API.LLVMAttributeRef)*count))
                        LLVM.API.LLVMGetCallSiteAttributes(st, idx, Attrs)
                        for j in 1:count
                            LLVM.API.LLVMAddCallSiteAttribute(newi, idx, unsafe_load(Attrs, j))
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
                for (i, v) in enumerate(operands(st)[1:end-1])
                    if v == inst
                        readnone = false
                        readonly = false
                        writeonly = false
                        t_sret = false
                        for a in collect(parameter_attributes(fop, i))
                            if kind(a) == kind(EnumAttribute("sret"))
                                t_sret = true
                            end
                            if kind(a) == kind(StringAttribute("enzyme_sret"))
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
                    println(string(f))
                    @show inst, st, fop
                    flush(stdout)
                end
                
                @assert sret
               
                elt = eltype(value_type(inst))
                if temp === nothing
                    nb = IRBuilder()
                    position!(nb, first(instructions(first(blocks(f)))))
                    temp = alloca!(nb, elt)
                end
                if mayread
                    nb = IRBuilder()
                    position!(nb, st)
                    ld = load!(nb, elt, operands(inst)[1])
                    store!(nb, ld, temp)
                end
                if maywrite
                    nb = IRBuilder()
                    position!(nb, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(st)))
                    ld = load!(nb, elt, temp)
                    si = store!(nb, ld, operands(inst)[1])
                    julia_post_cache_store(si.ref, nb.ref, C_NULL)
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

function pre_attr!(mod::LLVM.Module)
    return nothing
    tofinalize = Tuple{LLVM.Function,Bool,Vector{Int64}}[]
    for fn in collect(functions(mod))
        if isempty(blocks(fn))
            continue
        end
        if linkage(fn) != LLVM.API.LLVMInternalLinkage && linkage(fn) != LLVM.API.LLVMPrivateLinkage
            continue
        end
   
        fty = LLVM.FunctionType(fn)
        nfn = LLVM.Function(mod, "enzyme_attr_prev_"*LLVM.name(enzymefn), fty)
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
    return nothing
end

function post_attr!(mod::LLVM.Module)
end

function prop_global!(g)
    newfns = String[]
    changed = false
        todo = Tuple{Vector{Cuint},LLVM.Value}[]
        for u in LLVM.uses(g)
            u = LLVM.user(u)
            push!(todo, (Cuint[],u))
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
                replace_uses!(var, res)
                unsafe_delete!(LLVM.parent(var), var)
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
                    if convert(Cuint, operands(var)[2]) == 0
                        for u in LLVM.uses(var)
                            u = LLVM.user(u)
                            push!(todo, (vcat(path,collect((convert(Cuint, v) for v in operands(var)[3:end]))), u))
                        end
                    end
                    continue
                end
            end
        end
    return changed, newfns
end

function propagate_returned!(mod::LLVM.Module)
    globs = LLVM.GlobalVariable[]
    for g in globals(mod)
        if linkage(g) == LLVM.API.LLVMInternalLinkage || linkage(g) == LLVM.API.LLVMPrivateLinkage
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
            attrs = collect(function_attributes(fn))
            prevent = any(kind(attr) == kind(StringAttribute("enzyme_preserve_primal")) for attr in attrs)
            # if any(kind(attr) == kind(EnumAttribute("noinline")) for attr in attrs) 
            #     continue
            # end
            argn = nothing
            toremove = Int64[]
            for (i, arg) in enumerate(parameters(fn))
                if any(kind(attr) == kind(EnumAttribute("returned")) for attr in collect(parameter_attributes(fn, i)))
                    argn = i
                end
                # interprocedural const prop from callers of arg
                if !prevent && (linkage(fn) == LLVM.API.LLVMInternalLinkage || linkage(fn) == LLVM.API.LLVMPrivateLinkage)
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
                        if isa(ops[i], LLVM.UndefValue)
                            continue
                        end
                        @static if LLVM.version() >= v"12"
                        if isa(ops[i], LLVM.PoisonValue)
                            continue
                        end
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
                # sese if there are no users of the value (excluding recursive/return)
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
                    push!(toremove, i-1)
                end
            end
            illegalUse = !(linkage(fn) == LLVM.API.LLVMInternalLinkage || linkage(fn) == LLVM.API.LLVMPrivateLinkage)
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
                        hasAnyUse = true
                        break
                    end
                end
            end
            #if the function return has no users whatsoever, remove it
            if argn === nothing && !hasAnyUse && LLVM.return_type(LLVM.function_type(fn)) != LLVM.VoidType()
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
            try
                todo = LLVM.CallInst[]
                for u in LLVM.uses(fn)
                    un = LLVM.user(u)
                    push!(next, LLVM.name(LLVM.parent(LLVM.parent(un))))
                end
                nfn = LLVM.Function(API.EnzymeCloneFunctionWithoutReturnOrArgs(fn, keepret, toremove))
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
                unsafe_delete!(mod, fn)
                changed = true
            catch
               break
            end
        end
        if !changed
            break
        else
            todo = collect(functions(mod)[name] for name in next)
        end
    end
end
function detect_writeonly!(mod::LLVM.Module)
    for f in functions(mod)
        if isempty(LLVM.blocks(f))
            continue
        end
        for (i, a) in enumerate(parameters(f))
            if isa(value_type(a), LLVM.PointerType)
                todo = LLVM.Value[a]
                seen = Set{LLVM.Value}()
                mayread = false
                maywrite = false
                while length(todo) > 0
                    cur = pop!(todo)
                    if in(cur, seen)
                        continue
                    end
                    push!(seen, cur)
                    
                    if isa(cur, LLVM.StoreInst)
                        maywrite = true
                        continue
                    end
                    
                    if isa(cur, LLVM.LoadInst)
                        mayread = true
                        continue
                    end

                    if isa(cur, LLVM.Argument) || isa(cur, LLVM.GetElementPtrInst) || isa(cur, LLVM.BitCastInst) || isa(cur, LLVM.AddrSpaceCastInst)
                        for u in LLVM.uses(cur)
                            push!(todo, LLVM.user(u))
                        end
                        continue
                    end
                    mayread = true
                    maywrite = true
                end
                if any(map(k->kind(k)==kind(EnumAttribute("readnone")), collect(parameter_attributes(f, i))))
                    mayread = false
                    maywrite = false
                end
                if any(map(k->kind(k)==kind(EnumAttribute("readonly")), collect(parameter_attributes(f, i))))
                    maywrite = false
                end
                if any(map(k->kind(k)==kind(EnumAttribute("writeonly")), collect(parameter_attributes(f, i))))
                    mayread = false
                end
        
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, LLVM.API.LLVMAttributeIndex(i), kind(EnumAttribute("readnone")))
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, LLVM.API.LLVMAttributeIndex(i), kind(EnumAttribute("readonly")))
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, LLVM.API.LLVMAttributeIndex(i), kind(EnumAttribute("writeonly")))

                if !mayread && !maywrite
                    push!(parameter_attributes(f, i), LLVM.EnumAttribute("readnone", 0))
                elseif !mayread
                    push!(parameter_attributes(f, i), LLVM.EnumAttribute("writeonly", 0))
                elseif !maywrite
                    push!(parameter_attributes(f, i), LLVM.EnumAttribute("readonly", 0))
                end

            end
        end
    end
    return nothing
end

function validate_return_roots!(mod)
    for f in functions(mod)
        srets = []
        enzyme_srets = Int[]
        enzyme_srets_v = Int[]
        rroots = Int[]
        rroots_v = Int[]
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
                if kind(attr) == kind(EnumAttribute("sret"))
                    push!(srets, (i, attr))
                end
            end
        end
        if length(enzyme_srets) >= 1 && length(srets) == 0
            @assert enzyme_srets[1] == 1
            VT = LLVM.VoidType()
            if length(enzyme_srets) == 1 && LLVM.return_type(LLVM.function_type(f)) == VT && length(enzyme_srets_v) == 0
                # Upgrading to sret requires writeonly
                if !any(kind(attr) == kind(EnumAttribute("writeonly")) for attr in collect(parameter_attributes(f, 1)))
                    @show f
                    @show collect(parameter_attributes(f, 1))
                    @assert false
                end
                
                alty = nothing
                for u in LLVM.uses(f)
                    u = LLVM.user(u)
                    @assert isa(u, LLVM.CallInst)
                    @assert LLVM.called_operand(u) == f
                    alop = operands(u)[1]
                    if !isa(alop, LLVM.AllocaInst)
                        @show alop, u, f
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
                    LLVM.API.LLVMAddCallSiteAttribute(u, LLVM.API.LLVMAttributeIndex(1), attr)
                    LLVM.API.LLVMRemoveCallSiteStringAttribute(u, LLVM.API.LLVMAttributeIndex(1), "enzyme_sret", length("enzyme_sret"))
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
                        LLVM.API.LLVMRemoveCallSiteStringAttribute(u, LLVM.API.LLVMAttributeIndex(idx), "enzyme_sret", length("enzyme_sret"))
                    end
                    if !bad
                        delete!(parameter_attributes(f, idx), StringAttribute("enzyme_sret"))
                    else
                        push!(enzyme_srets2, idx)
                    end
                end
                enzyme_srets = enzyme_srets2

                if length(enzyme_srets) != 0
                    @show f
                    @show enzyme_srets, enzyme_srets_v, srets, rroots, rroots_v
                    @assert false
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

function checkNoAssumeFalse(mod, shouldshow=false)
    for f in functions(mod)
        for bb in blocks(f), inst in instructions(bb)
            if !isa(inst, LLVM.CallInst)
                continue
            end
            intr = LLVM.API.LLVMGetIntrinsicID(LLVM.called_operand(inst))
            if shouldshow
                @show intr, inst
            end
            if intr != LLVM.Intrinsic("llvm.assume").id
                continue
            end
            if shouldshow
                @show inst
            end
            op = operands(inst)[1]
            if shouldshow
                @show op
            end
            if isa(op, LLVM.ConstantInt)
                op2 = convert(Bool, op)
                if shouldshow
                    @show op2
                end
                if !op2
                    println(string(mod))
                    println(string(f))
                    println(string(bb))
                    flush(stdout)
                    @assert false
                end
            end
            if isa(op, LLVM.ICmpInst)
                if predicate_int(op) == LLVM.API.LLVMIntNE && operands(op)[1] == operands(op)[2]
                    println(string(mod))
                    println(string(f))
                    println(string(bb))
                    flush(stdout)
                    @assert false
                end
            end
        end
    end
end

function removeDeadArgs!(mod::LLVM.Module)
    # We need to run globalopt first. This is because remove dead args will otherwise
    # take internal functions and replace their args with undef. Then on LLVM up to 
    # and including 12 (but fixed 13+), Attributor will incorrectly change functions that
    # call code with undef to become unreachable, even when there exist other valid
    # callsites. See: https://godbolt.org/z/9Y3Gv6q5M
    ModulePassManager() do pm
        global_dce!(pm)
        run!(pm, mod)
    end
    # Prevent dead-arg-elimination of functions which we may require args for in the derivative
    funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[], vararg=true)
    func, _ = get_function!(mod, "llvm.enzymefakeuse", funcT, [EnumAttribute("readnone"), EnumAttribute("nofree")])
    rfunc, _ = get_function!(mod, "llvm.enzymefakeread", funcT, [EnumAttribute("readonly"), EnumAttribute("nofree"), EnumAttribute("argmemonly")])

    for fn in functions(mod)
        if isempty(blocks(fn))
            continue
        end
        # Ensure that interprocedural optimizations do not delete the use of returnRoots (or shadows)
        # if inactive sret, this will only occur on 2. If active sret, inactive retRoot, can on 3, and
        # active both can occur on 4. If the original sret is removed (at index 1) we no longer need
        # to preserve this.
        for idx in (2, 3, 4)
            if length(collect(parameters(fn))) >= idx && any( ( kind(attr) == kind(StringAttribute("enzymejl_returnRoots")) || kind(attr) == StringAttribute("enzymejl_returnRoots_v")) for attr in collect(parameter_attributes(fn, idx)))
                for u in LLVM.uses(fn)
                    u = LLVM.user(u)
                    @assert isa(u, LLVM.CallInst)
                    B = IRBuilder()
                    nextInst = LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(u))
                    position!(B, nextInst)
                    cl = call!(B, funcT, rfunc, LLVM.Value[operands(u)[2]])
                    LLVM.API.LLVMAddCallSiteAttribute(cl, LLVM.API.LLVMAttributeIndex(1), EnumAttribute("nocapture"))
                end
            end
        end
        attrs = collect(function_attributes(fn))
        prevent = any(kind(attr) == kind(StringAttribute("enzyme_preserve_primal")) for attr in attrs)
        # && any(kind(attr) == kind(StringAttribute("enzyme_math")) for attr in attrs)
        if prevent
            B = IRBuilder()
            position!(B, first(instructions(first(blocks(fn)))))
            call!(B, funcT, func, LLVM.Value[p for p in parameters(fn)])
        end
    end
    propagate_returned!(mod)
    ModulePassManager() do pm
        instruction_combining!(pm)
        alloc_opt!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        run!(pm, mod)
    end
    propagate_returned!(mod)
    pre_attr!(mod)
    if LLVM.version().major >= 13
        ModulePassManager() do pm
            API.EnzymeAddAttributorLegacyPass(pm)
            run!(pm, mod)
        end
    end
    propagate_returned!(mod)
    ModulePassManager() do pm
        instruction_combining!(pm)
        alloc_opt!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        if LLVM.version().major >= 13
            API.EnzymeAddAttributorLegacyPass(pm)
        end
        run!(pm, mod)
    end
    post_attr!(mod)
    propagate_returned!(mod)

    for u in LLVM.uses(rfunc)
        u = LLVM.user(u)
        unsafe_delete!(LLVM.parent(u), u)
    end
    unsafe_delete!(mod, rfunc)
    for fn in functions(mod)
        for b in blocks(fn)
            inst = first(LLVM.instructions(b))
            if isa(inst, LLVM.CallInst)
                fn = LLVM.called_operand(inst)
                if fn == func
                    unsafe_delete!(b, inst)
                end
            end
        end
    end
    unsafe_delete!(mod, func)
end

function optimize!(mod::LLVM.Module, tm)
    addr13NoAlias(mod)
    # everying except unroll, slpvec, loop-vec
    # then finish Julia GC
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        propagate_julia_addrsp!(pm)
        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cfgsimplification!(pm)
        dce!(pm)
@static if isdefined(GPUCompiler, :cpu_features!)
        GPUCompiler.cpu_features!(pm)
end
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        mem_cpy_opt!(pm)
        always_inliner!(pm)
        alloc_opt!(pm)
        LLVM.API.LLVMAddGlobalOptimizerPass(pm) # Extra
        gvn!(pm) # Extra
        instruction_combining!(pm)
        cfgsimplification!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        instruction_combining!(pm)
        jump_threading!(pm)
        correlated_value_propagation!(pm)
        instruction_combining!(pm)
        reassociate!(pm)
        early_cse!(pm)
        alloc_opt!(pm)
        loop_idiom!(pm)
        loop_rotate!(pm)
        lower_simdloop!(pm)
        licm!(pm)
        if LLVM.version() >= v"15"                      
            simple_loop_unswitch_legacy!(pm)
        else
            loop_unswitch!(pm)
        end
        instruction_combining!(pm)
        ind_var_simplify!(pm)
        loop_deletion!(pm)
        loop_unroll!(pm)
        alloc_opt!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        gvn!(pm)
    
        # This InstCombine needs to be after GVN
        # Otherwise it will generate load chains in GPU code...
        instruction_combining!(pm)
        mem_cpy_opt!(pm)
        sccp!(pm)
        instruction_combining!(pm)
        jump_threading!(pm)
        dead_store_elimination!(pm)
        alloc_opt!(pm)
        cfgsimplification!(pm)
        loop_idiom!(pm)
        loop_deletion!(pm)
        jump_threading!(pm)
        correlated_value_propagation!(pm)
        # SLP_Vectorizer -- not for Enzyme
        
        run!(pm, mod)

        aggressive_dce!(pm)
        instruction_combining!(pm)
        # Loop Vectorize -- not for Enzyme
        # InstCombine

        # GC passes
        barrier_noop!(pm)
        gc_invariant_verifier!(pm, false)

        # FIXME: Currently crashes printing
        cfgsimplification!(pm)
        instruction_combining!(pm) # Extra for Enzyme
        LLVM.API.LLVMAddGlobalOptimizerPass(pm) # Exxtra
        gvn!(pm) # Exxtra
        run!(pm, mod)
    end

    removeDeadArgs!(mod)
    detect_writeonly!(mod)
    nodecayed_phis!(mod)
end

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L603
function addTargetPasses!(pm, tm, trip)
    add_library_info!(pm, trip)
    add_transform_info!(pm, tm)
end

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L620
function addOptimizationPasses!(pm)
    add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))

    constant_merge!(pm)

    propagate_julia_addrsp!(pm)
    scoped_no_alias_aa!(pm)
    type_based_alias_analysis!(pm)
    basic_alias_analysis!(pm)
    cfgsimplification!(pm)
    dce!(pm)
    scalar_repl_aggregates!(pm)

    # mem_cpy_opt!(pm)

    always_inliner!(pm) # Respect always_inline

    # Running `memcpyopt` between this and `sroa` seems to give `sroa` a hard time
    # merging the `alloca` for the unboxed data and the `alloca` created by the `alloc_opt`
    # pass.

    alloc_opt!(pm)
    # consider AggressiveInstCombinePass at optlevel > 2

    instruction_combining!(pm)
    cfgsimplification!(pm)
    scalar_repl_aggregates!(pm)
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy
    jump_threading!(pm)
    correlated_value_propagation!(pm)

    reassociate!(pm)

    early_cse!(pm)

    # Load forwarding above can expose allocations that aren't actually used
    # remove those before optimizing loops.
    alloc_opt!(pm)
    loop_rotate!(pm)
    # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
    loop_idiom!(pm)

    # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
    lower_simdloop!(pm) # Annotate loop marked with "loopinfo" as LLVM parallel loop
    licm!(pm)
    julia_licm!(pm)
    # Subsequent passes not stripping metadata from terminator
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy
    ind_var_simplify!(pm)
    loop_deletion!(pm)
    loop_unroll!(pm) # TODO: in Julia createSimpleLoopUnroll

    # Run our own SROA on heap objects before LLVM's
    alloc_opt!(pm)
    # Re-run SROA after loop-unrolling (useful for small loops that operate,
    # over the structure of an aggregate)
    scalar_repl_aggregates!(pm)
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy

    gvn!(pm)
    mem_cpy_opt!(pm)
    sccp!(pm)

    # Run instcombine after redundancy elimination to exploit opportunities
    # opened up by them.
    # This needs to be InstCombine instead of InstSimplify to allow
    # loops over Union-typed arrays to vectorize.
    instruction_combining!(pm)
    jump_threading!(pm)
    dead_store_elimination!(pm)

    # More dead allocation (store) deletion before loop optimization
    # consider removing this:
    alloc_opt!(pm)

    # see if all of the constant folding has exposed more loops
    # to simplification and deletion
    # this helps significantly with cleaning up iteration
    cfgsimplification!(pm)
    loop_deletion!(pm)
    instruction_combining!(pm)
    loop_vectorize!(pm)
    # TODO: createLoopLoadEliminationPass
    cfgsimplification!(pm)
    slpvectorize!(pm)
    # might need this after LLVM 11:
    # TODO: createVectorCombinePass()

    aggressive_dce!(pm)
end

function addMachinePasses!(pm)
    combine_mul_add!(pm)
    # TODO: createDivRemPairs[]

    demote_float16!(pm)
    gvn!(pm)
end

function addJuliaLegalizationPasses!(pm, lower_intrinsics=true)
    if lower_intrinsics
        # LowerPTLS removes an indirect call. As a result, it is likely to trigger
        # LLVM's devirtualization heuristics, which would result in the entire
        # pass pipeline being re-exectuted. Prevent this by inserting a barrier.
        barrier_noop!(pm)
        add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))
        lower_exc_handlers!(pm)
        # BUDE.jl demonstrates a bug here TODO
        gc_invariant_verifier!(pm, false)
        verifier!(pm)

        # Needed **before** LateLowerGCFrame on LLVM < 12
        # due to bug in `CreateAlignmentAssumption`.
        remove_ni!(pm)
        late_lower_gc_frame!(pm)
        final_lower_gc!(pm)
        # We need these two passes and the instcombine below
        # after GC lowering to let LLVM do some constant propagation on the tags.
        # and remove some unnecessary write barrier checks.
        gvn!(pm)
        sccp!(pm)
        # Remove dead use of ptls
        dce!(pm)
        lower_ptls!(pm, #=dump_native=# false)
        instruction_combining!(pm)
        # Clean up write barrier and ptls lowering
        cfgsimplification!(pm)
    else
        barrier_noop!(pm)
        remove_ni!(pm)
    end
end

function post_optimze!(mod, tm, machine=true)
    addr13NoAlias(mod)
    removeDeadArgs!(mod)
    for f in collect(functions(mod))
        API.EnzymeFixupJuliaCallingConvention(f)
    end
    out_error = Ref{Cstring}()
    if LLVM.API.LLVMVerifyModule(mod, LLVM.API.LLVMReturnStatusAction, out_error) != 0
        throw(LLVM.LLVMException("broken gc calling conv fix\n"*string(unsafe_string(out_error[]))*"\n"*string(mod)))
    end
    # @safe_show "pre_post", mod
    # flush(stdout)
    # flush(stderr)
    LLVM.ModulePassManager() do pm
        addTargetPasses!(pm, tm, LLVM.triple(mod))
        addOptimizationPasses!(pm)
        run!(pm, mod)
    end
    if machine
        # TODO enable validate_return_roots
        # validate_return_roots!(mod)
        LLVM.ModulePassManager() do pm
            addJuliaLegalizationPasses!(pm, true)
            addMachinePasses!(pm)
            run!(pm, mod)
        end
    end
    # @safe_show "post_mod", mod
    # flush(stdout)
    # flush(stderr)
end
