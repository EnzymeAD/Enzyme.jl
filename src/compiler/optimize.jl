function addNA(inst, node::LLVM.Metadata, MD)
    md = metadata(inst)
    next = nothing 
    ctx = LLVM.context(inst)
    if haskey(md, MD)
        next = LLVM.MDNode(Metadata[node, operands(md[MD])...]; ctx)
    else
        next = LLVM.MDNode(Metadata[node]; ctx)
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
            fn = LLVM.called_value(inst)
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

# If there is a phi node of a decayed value, Enzyme may need to cache it
# Here we force all decayed pointer phis to first addrspace from 10
function nodecayed_phis!(mod::LLVM.Module)
    ctx = LLVM.context(mod)
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
                b = IRBuilder(ctx)
                position!(b, terminator(pb))
                while isa(v, LLVM.AddrSpaceCastInst)
                    v = operands(v)[1]
                end
                if value_type(v) != nty
                    v = addrspacecast!(b, v, nty)
                end
                push!(nvs, (v, pb))
            end
            nb = IRBuilder(ctx)
            position!(nb, inst)
            nphi = phi!(nb, nty)
            append!(LLVM.incoming(nphi), nvs)
            
            position!(nb, nonphi)
            nphi = addrspacecast!(nb, nphi, ty)
            replace_uses!(inst, nphi)
            LLVM.API.LLVMInstructionEraseFromParent(inst)
        end
    end
    return nothing
end

function fix_decayaddr!(mod::LLVM.Module)
    ctx = LLVM.context(mod)
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
                if !isa(st, LLVM.CallInst)
                    @show f
                    @show inst
                    @show st
                    throw(AssertionError("illegal decay of noncall"))
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

                    nb = IRBuilder(ctx)
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
                            if kind(a) == kind(EnumAttribute("sret"; ctx))
                                t_sret = true
                            end
                            if kind(a) == kind(StringAttribute("enzyme_sret"; ctx))
                                t_sret = true
                            end
                            # if kind(a) == kind(StringAttribute("enzyme_sret_v"; ctx))
                            #     t_sret = true
                            # end
                            if kind(a) == kind(EnumAttribute("readonly"; ctx))
                                readonly = true
                            end
                            if kind(a) == kind(EnumAttribute("readnone"; ctx))
                                readnone = true
                            end
                            if kind(a) == kind(EnumAttribute("writeonly"; ctx))
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
                    @safe_show f
                    @safe_show inst, st, fop
                    flush(stdout)
                end
                
                @assert sret
               
                elt = eltype(value_type(inst))
                if temp === nothing
                    nb = IRBuilder(ctx)
                    position!(nb, first(instructions(first(blocks(f)))))
                    temp = alloca!(nb, elt)
                end
                if mayread
                    nb = IRBuilder(ctx)
                    position!(nb, st)
                    ld = load!(nb, elt, operands(inst)[1])
                    store!(nb, ld, temp)
                end
                if maywrite
                    nb = IRBuilder(ctx)
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

function propagate_returned!(mod::LLVM.Module)
    ctx = LLVM.context(mod)

    tofinalize = LLVM.Function[]
    for fn in functions(mod)
        if isempty(blocks(fn))
            continue
        end
        attrs = collect(function_attributes(fn))
        if any(kind(attr) == kind(EnumAttribute("noinline"; ctx)) for attr in attrs) 
            continue
        end
        argn = nothing
        for i in 1:length(parameters(fn))
            if any(kind(attr) == kind(EnumAttribute("returned"; ctx)) for attr in collect(parameter_attributes(fn, i)))
                argn = i
            end
        end
        if argn === nothing
            continue
        end
        illegalUse = linkage(fn) != LLVM.API.LLVMInternalLinkage
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
            LLVM.replace_uses!(un, ops[argn])
        end
        if !illegalUse
            push!(tofinalize, fn)
        end
    end
    for fn in tofinalize
        try
            nfn = LLVM.Function(API.EnzymeCloneFunctionWithoutReturn(fn))
            todo = LLVM.CallInst[]
            for u in LLVM.uses(fn)
                un = LLVM.user(u)
                push!(todo, un)
            end
            for un in todo
                API.EnzymeSetCalledFunction(un, nfn)
            end
            unsafe_delete!(mod, fn)
        catch
            break
        end
    end
end
function detect_writeonly!(mod::LLVM.Module)
    ctx = LLVM.context(mod)
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
                if any(map(k->kind(k)==kind(EnumAttribute("readnone"; ctx)), collect(parameter_attributes(f, i))))
                    mayread = false
                    maywrite = false
                end
                if any(map(k->kind(k)==kind(EnumAttribute("readonly"; ctx)), collect(parameter_attributes(f, i))))
                    maywrite = false
                end
                if any(map(k->kind(k)==kind(EnumAttribute("writeonly"; ctx)), collect(parameter_attributes(f, i))))
                    mayread = false
                end
        
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, LLVM.API.LLVMAttributeIndex(i), kind(EnumAttribute("readnone"; ctx)))
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, LLVM.API.LLVMAttributeIndex(i), kind(EnumAttribute("readonly"; ctx)))
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(f, LLVM.API.LLVMAttributeIndex(i), kind(EnumAttribute("writeonly"; ctx)))

                if !mayread && !maywrite
                    push!(parameter_attributes(f, i), LLVM.EnumAttribute("readnone", 0; ctx))
                elseif !mayread
                    push!(parameter_attributes(f, i), LLVM.EnumAttribute("writeonly", 0; ctx))
                elseif !maywrite
                    push!(parameter_attributes(f, i), LLVM.EnumAttribute("readonly", 0; ctx))
                end

            end
        end
    end
    return nothing
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
        loop_unswitch!(pm)
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
        
        run!(pm, mod)
    end
    
    # Prevent dead-arg-elimination of functions which we may require args for in the derivative
    ctx = LLVM.context(mod)
    funcT = LLVM.FunctionType(LLVM.VoidType(ctx), LLVMType[], vararg=true)
    func, _ = get_function!(mod, "llvm.enzymefakeuse", funcT, [EnumAttribute("readnone"; ctx)])
   
    for fn in functions(mod)
        if isempty(blocks(fn))
            continue
        end
        attrs = collect(function_attributes(fn))
        prevent = any(kind(attr) == kind(EnumAttribute("noinline"; ctx)) for attr in attrs) && any(kind(attr) == kind(StringAttribute("enzyme_math"; ctx)) for attr in attrs)
        if prevent
            B = IRBuilder(ctx)
            position!(B, first(instructions(first(blocks(fn)))))
            call!(B, funcT, func, LLVM.Value[p for p in parameters(fn)])
        end
    end
    ModulePassManager() do pm
        API.EnzymeAddAttributorLegacyPass(pm)
        run!(pm, mod)
    end
    propagate_returned!(mod)
    ModulePassManager() do pm
        instruction_combining!(pm)
        alloc_opt!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        API.EnzymeAddAttributorLegacyPass(pm)
        run!(pm, mod)
    end

    for fn in functions(mod)
        for b in blocks(fn)
            inst = first(LLVM.instructions(b))
            if isa(inst, LLVM.CallInst)
                fn = LLVM.called_value(inst)
                if fn == func
                    unsafe_delete!(b, inst)
                end
            end
        end
    end
    detect_writeonly!(mod)
    nodecayed_phis!(mod)
end

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L603
function addTargetPasses!(pm, tm)
    add_library_info!(pm, LLVM.triple(tm))
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
    # @safe_show "pre_post", mod
    # flush(stdout)
    # flush(stderr)
    LLVM.ModulePassManager() do pm
        addTargetPasses!(pm, tm)
        addOptimizationPasses!(pm)
        run!(pm, mod)
    end
    if machine
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
