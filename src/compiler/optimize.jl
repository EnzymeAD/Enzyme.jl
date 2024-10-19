struct PipelineConfig
    Speedup::Cint
    Size::Cint
    lower_intrinsics::Cint
    dump_native::Cint
    external_use::Cint
    llvm_only::Cint
    always_inline::Cint
    enable_early_simplifications::Cint
    enable_early_optimizations::Cint
    enable_scalar_optimizations::Cint
    enable_loop_optimizations::Cint
    enable_vector_pipeline::Cint
    remove_ni::Cint
    cleanup::Cint
end

const RunAttributor = Ref(true)

function pipeline_options(;
    lower_intrinsics = true,
    dump_native = false,
    external_use = false,
    llvm_only = false,
    always_inline = true,
    enable_early_simplifications = true,
    enable_early_optimizations = true,
    enable_scalar_optimizations = true,
    enable_loop_optimizations = true,
    enable_vector_pipeline = true,
    remove_ni = true,
    cleanup = true,
    Size = 0,
    Speedup = 3,
)
    return PipelineConfig(
        Speedup,
        Size,
        lower_intrinsics,
        dump_native,
        external_use,
        llvm_only,
        always_inline,
        enable_early_simplifications,
        enable_early_optimizations,
        enable_scalar_optimizations,
        enable_loop_optimizations,
        enable_vector_pipeline,
        remove_ni,
        cleanup,
    )
end

function run_jl_pipeline(pm, tm; kwargs...)
    config = Ref(pipeline_options(; kwargs...))
    function jl_pipeline(m)
        @dispose pb = NewPMPassBuilder() begin
            add!(pb, NewPMModulePassManager()) do mpm
                @ccall jl_build_newpm_pipeline(
                    mpm.ref::Ptr{Cvoid},
                    pb.ref::Ptr{Cvoid},
                    config::Ptr{PipelineConfig},
                )::Cvoid
            end
            LLVM.run!(mpm, m, tm)
        end
        return true
    end
    add!(pm, ModulePass("JLPipeline", jl_pipeline))
end

@static if VERSION < v"1.11.0-DEV.428"
else
    barrier_noop!(pm) = nothing
end

@static if VERSION < v"1.11-"
    function gc_invariant_verifier_tm!(pm, tm, cond)
        gc_invariant_verifier!(pm, cond)
    end
else
    function gc_invariant_verifier_tm!(pm, tm, cond)
        function gc_invariant_verifier(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, GCInvariantVerifierPass(; strong = cond))
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("GCInvariantVerifier", gc_invariant_verifier))
    end
end

@static if VERSION < v"1.11-"
    function propagate_julia_addrsp_tm!(pm, tm)
        propagate_julia_addrsp!(pm)
    end
else
    function propagate_julia_addrsp_tm!(pm, tm)
        function prop_julia_addr(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, PropagateJuliaAddrspacesPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("PropagateJuliaAddrSpace", prop_julia_addr))
    end
end

@static if VERSION < v"1.11-"
    function alloc_opt_tm!(pm, tm)
        alloc_opt!(pm)
    end
else
    function alloc_opt_tm!(pm, tm)
        function alloc_opt(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, AllocOptPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("AllocOpt", alloc_opt))
    end
end

@static if VERSION < v"1.11-"
    function remove_ni_tm!(pm, tm)
        remove_ni!(pm)
    end
else
    function remove_ni_tm!(pm, tm)
        function remove_ni(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, RemoveNIPass())
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("RemoveNI", remove_ni))
    end
end

@static if VERSION < v"1.11-"
    function julia_licm_tm!(pm, tm)
        julia_licm!(pm)
    end
else
    function julia_licm_tm!(pm, tm)
        function julia_licm(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, NewPMLoopPassManager()) do lpm
                            add!(lpm, JuliaLICMPass())
                        end
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        # really looppass
        add!(pm, ModulePass("JuliaLICM", julia_licm))
    end
end

@static if VERSION < v"1.11-"
    function lower_simdloop_tm!(pm, tm)
        lower_simdloop!(pm)
    end
else
    function lower_simdloop_tm!(pm, tm)
        function lower_simdloop(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, NewPMLoopPassManager()) do lpm
                            add!(lpm, LowerSIMDLoopPass())
                        end
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        # really looppass
        add!(pm, ModulePass("LowerSIMDLoop", lower_simdloop))
    end
end


function loop_optimizations_tm!(pm, tm)
    @static if true || VERSION < v"1.11-"
        lower_simdloop_tm!(pm, tm)
        licm!(pm)
        if LLVM.version() >= v"15"
            simple_loop_unswitch_legacy!(pm)
        else
            loop_unswitch!(pm)
        end
    else
        run_jl_pipeline(
            pm,
            tm;
            lower_intrinsics = false,
            dump_native = false,
            external_use = false,
            llvm_only = false,
            always_inline = false,
            enable_early_simplifications = false,
            enable_early_optimizations = false,
            enable_scalar_optimizations = false,
            enable_loop_optimizations = true,
            enable_vector_pipeline = false,
            remove_ni = false,
            cleanup = false,
        )
    end
end


function more_loop_optimizations_tm!(pm, tm)
    @static if true || VERSION < v"1.11-"
        loop_rotate!(pm)
        # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
        loop_idiom!(pm)

        # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
        lower_simdloop_tm!(pm, tm) # Annotate loop marked with "loopinfo" as LLVM parallel loop
        licm!(pm)
        julia_licm_tm!(pm, tm)
        # Subsequent passes not stripping metadata from terminator
        instruction_combining!(pm) # TODO: createInstSimplifyLegacy
        jl_inst_simplify!(pm)

        ind_var_simplify!(pm)
        loop_deletion!(pm)
        loop_unroll!(pm) # TODO: in Julia createSimpleLoopUnroll
    else
        # LowerSIMDLoopPass
        # LoopRotatePass [opt >= 2]
        # LICMPass
        # JuliaLICMPass
        # SimpleLoopUnswitchPass
        # LICMPass
        # JuliaLICMPass
        # IRCEPass
        # LoopInstSimplifyPass
        #   - in ours this is instcombine with jlinstsimplify
        # LoopIdiomRecognizePass
        # IndVarSimplifyPass
        # LoopDeletionPass
        # LoopFullUnrollPass
        run_jl_pipeline(
            pm,
            tm;
            lower_intrinsics = false,
            dump_native = false,
            external_use = false,
            llvm_only = false,
            always_inline = false,
            enable_early_simplifications = false,
            enable_early_optimizations = false,
            enable_scalar_optimizations = false,
            enable_loop_optimizations = true,
            enable_vector_pipeline = false,
            remove_ni = false,
            cleanup = false,
        )
    end
end

@static if VERSION < v"1.11-"
    function demote_float16_tm!(pm, tm)
        demote_float16!(pm)
    end
else
    function demote_float16_tm!(pm, tm)
        function demote_float16(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, DemoteFloat16Pass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("DemoteFloat16", demote_float16))
    end
end

@static if VERSION < v"1.11-"
    function lower_exc_handlers_tm!(pm, tm)
        lower_exc_handlers!(pm)
    end
else
    function lower_exc_handlers_tm!(pm, tm)
        function lower_exc_handlers(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, LowerExcHandlersPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("LowerExcHandlers", lower_exc_handlers))
    end
end

@static if VERSION < v"1.11-"
    function lower_ptls_tm!(pm, tm, dump_native)
        lower_ptls!(pm, dump_native)
    end
else
    function lower_ptls_tm!(pm, tm, dump_native)
        function lower_ptls(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, LowerPTLSPass())
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("LowerPTLS", lower_ptls))
    end
end

@static if VERSION < v"1.11-"
    function combine_mul_add_tm!(pm, tm)
        combine_mul_add!(pm)
    end
else
    function combine_mul_add_tm!(pm, tm)
        function combine_mul_add(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, CombineMulAddPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("CombineMulAdd", combine_mul_add))
    end
end

@static if VERSION < v"1.11-"
    function late_lower_gc_frame_tm!(pm, tm)
        late_lower_gc_frame!(pm)
    end
else
    function late_lower_gc_frame_tm!(pm, tm)
        function late_lower_gc_frame(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, LateLowerGCPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("LateLowerGCFrame", late_lower_gc_frame))
    end
end

@static if VERSION < v"1.11-"
    function final_lower_gc_tm!(pm, tm)
        final_lower_gc!(pm)
    end
else
    function final_lower_gc_tm!(pm, tm)
        function final_lower_gc(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, FinalLowerGCPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("FinalLowerGCFrame", final_lower_gc))
    end
end

@static if VERSION < v"1.11-"
    function cpu_features_tm!(pm, tm)
        @static if isdefined(LLVM.Interop, :cpu_features!)
            LLVM.Interop.cpu_features!(pm)
        else
            @static if isdefined(GPUCompiler, :cpu_features!)
                GPUCompiler.cpu_features!(pm)
            end
        end
    end
else
    function cpu_features_tm!(pm, tm)
        function cpu_features(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, CPUFeaturesPass())
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("CPUFeatures", cpu_features))
    end
end

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
            ty = value_type(inst)
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


## given code like
#  % a = alloca
#  ...
#  memref(cast(%a), %b, constant size == sizeof(a))
#   
#  turn this into load/store, as this is more
#  amenable to caching analysis infrastructure
function memcpy_alloca_to_loadstore(mod::LLVM.Module)
    dl = datalayout(mod)
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
                    FT = LLVM.FunctionType(
                        LLVM.VoidType(),
                        [LLVM.IntType(64), value_type(dst0)],
                    )
                    lifetimestart, _ = get_function!(mod, "llvm.lifetime.start.p0i8", FT)
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

# If there is a phi node of a decayed value, Enzyme may need to cache it
# Here we force all decayed pointer phis to first addrspace from 10
function nodecayed_phis!(mod::LLVM.Module)
    # Simple handler to fix addrspace 11
    #complex handler for addrspace 13, which itself comes from a load of an
    # addrspace 10
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
                            base = get_base_object(v)
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
                    end

                    push!(todo, inst)
                    nb = IRBuilder()
                    position!(nb, inst)
                    el_ty = if addr == 11
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
                    el_ty = if addr == 11
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
                        b = IRBuilder()
                        position!(b, terminator(pb))


                        v0 = v
                        @inline function getparent(v, offset, hasload)
                            if addr == 11 && addrspace(value_type(v)) == 10
                                return v, offset, hasload
                            end
                            if addr == 13 && hasload && addrspace(value_type(v)) == 10
                                return v, offset, hasload
                            end
                            if addr == 13  && !hasload
                                if isa(v, LLVM.LoadInst)
                                    v2, o2, hl2 = getparent(operands(v)[1], LLVM.ConstantInt(offty, 0), true)
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
                                        while isa(ld, LLVM.BitCastInst) || isa(ld, LLVM.AddrSpaceCastInst)
                                            ld = operands(ld)[1]
                                        end
                                        if isa(ld, LLVM.LoadInst)
                                            v2, o2, hl2 = getparent(operands(ld)[1], LLVM.ConstantInt(offty, 0), true)
                                            rhs = LLVM.ConstantInt(offty, sizeof(Int))

                                            base_2, off_2, _ = get_base_and_offset(v2)
                                            base_1, off_1, _ = get_base_and_offset(operands(v)[1])

                                            if o2 == rhs && base_1 == base_2 && off_1 == off_2
                                                return v2, offset, true
                                            end

                                            rhs = ptrtoint!(b, get_memory_data(b, operands(v)[1]), offty)
                                            lhs = ptrtoint!(b, operands(v)[2], offty)
                                            off2 = nuwsub!(b, rhs, lhs)
                                            return v2, nuwadd!(b, offset, off2), true
                                        end
                                    end
                                end
                            end

                            if addr == 13 && isa(v, LLVM.ConstantExpr)
                                if opcode(v) == LLVM.API.LLVMAddrSpaceCast
                                    v2 = operands(v)[1]
                                    if addrspace(value_type(v2)) == 0
                                        if addr == 13 && isa(v, LLVM.ConstantExpr)
                                            v2 = const_addrspacecast(
                                                operands(v)[1],
                                                LLVM.PointerType(eltype(value_type(v)), 10),
                                            )
                                            return v2, offset, hasload
                                        end
                                    end
                                end
                            end

                            if addr == 11 && isa(v, LLVM.ConstantExpr)
                                if opcode(v) == LLVM.API.LLVMAddrSpaceCast
                                    v2 = operands(v)[1]
                                    if addrspace(value_type(v2)) == 10
                                        return v2, offset, hasload
                                    end
                                    if addrspace(value_type(v2)) == 0
                                        if addr == 11
                                            v2 = const_addrspacecast(
                                                v2,
                                                LLVM.PointerType(eltype(value_type(v)), 10),
                                            )
                                            return v2, offset, hasload
                                        end
                                    end
                                    if LLVM.isnull(v2)
                                        v2 = const_addrspacecast(
                                            v2,
                                            LLVM.PointerType(eltype(value_type(v)), 10),
                                        )
                                        return v2, offset, hasload
                                    end
                                end
                            end

                            if isa(v, LLVM.AddrSpaceCastInst)
                                if addrspace(value_type(operands(v)[1])) == 0
                                    v2 = addrspacecast!(
                                        b,
                                        operands(v)[1],
                                        LLVM.PointerType(eltype(value_type(v)), 10),
                                    )
                                    return v2, offset, hasload
                                end
                                nv, noffset, nhasload =
                                    getparent(operands(v)[1], offset, hasload)
                                if eltype(value_type(nv)) != eltype(value_type(v))
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
                                v2, offset, skipload =
                                    getparent(operands(v)[1], offset, hasload)
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
                                    getparent(operands(v)[1], offset, hasload)
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

                            if isa(v, LLVM.GetElementPtrInst)
                                v2, offset, skipload =
                                    getparent(operands(v)[1], offset, hasload)
                                offset = nuwadd!(
                                    b,
                                    offset,
                                    API.EnzymeComputeByteOffsetOfGEP(b, v, offty),
                                )
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

                            if isa(v, LLVM.ConstantExpr) &&
                               opcode(v) == LLVM.API.LLVMGetElementPtr &&
                               !hasload
                                v2, offset, skipload =
                                    getparent(operands(v)[1], offset, hasload)
                                offset = nuwadd!(
                                    b,
                                    offset,
                                    API.EnzymeComputeByteOffsetOfGEP(b, v, offty),
                                )
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

                            undeforpoison = isa(v, LLVM.UndefValue)
                            @static if LLVM.version() >= v"12"
                                undeforpoison |= isa(v, LLVM.PoisonValue)
                            end
                            if undeforpoison
                                return LLVM.UndefValue(
                                    LLVM.PointerType(eltype(value_type(v)), 10),
                                ),
                                offset,
                                addr == 13
                            end

                            if isa(v, LLVM.PHIInst) && !hasload && haskey(goffsets, v)
                                offset = nuwadd!(b, offset, goffsets[v])
                                nv = nextvs[v]
                                return nv, offset, addr == 13
                            end

                            if isa(v, LLVM.SelectInst)
                                lhs_v, lhs_offset, lhs_skipload =
                                    getparent(operands(v)[2], offset, hasload)
                                rhs_v, rhs_offset, rhs_skipload =
                                    getparent(operands(v)[3], offset, hasload)
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

                        v, offset, hadload = getparent(v, LLVM.ConstantInt(offty, 0), false)

                        if addr == 13
                            @assert hadload
                        end

                        if eltype(value_type(v)) != el_ty
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
                    position!(nb, inst)

                    offset = goffsets[inst]
                    append!(LLVM.incoming(offset), offsets)
                    if all(x -> x[1] == offsets[1][1], offsets)
                        offset = offsets[1][1]
                    end

                    nphi = nextvs[inst]
                    if !all(x -> x[1] == nvs[1][1], nvs)
                        append!(LLVM.incoming(nphi), nvs)
                    else
                        replace_uses!(nphi, nvs[1][1])
                        LLVM.API.LLVMInstructionEraseFromParent(nphi)
                        nphi = nvs[1][1]
                    end

                    position!(nb, nonphi)
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
                        continue
                    end
                end
                if isa(st, LLVM.LoadInst)
                    LLVM.API.LLVMSetOperand(st, 1 - 1, operands(inst)[1])
                    continue
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
                    msg = sprint() do io
                        println(io, "Enzyme Internal Error: did not have sret when expected")
                        println(io, "f=", string(f))
                        println(io, "inst=", string(inst))
                        println(io, "st=", string(st))
                        println(io, "fop=", string(fop))
                    end
                    throw(AssertionError(msg))
                end

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
    return nothing
end

function jl_inst_simplify!(PM)
    ccall(
        (:LLVMAddJLInstSimplifyPass, API.libEnzyme),
        Cvoid,
        (LLVM.API.LLVMPassManagerRef,),
        PM,
    )
end

function post_attr!(mod::LLVM.Module) end

function prop_global!(g)
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
                if convert(Cuint, operands(var)[2]) == 0
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
function mayWriteToMemory(inst::LLVM.Instruction; err_is_readonly = false)::Bool
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

        err_is_readonly = !is_noreturn(cur)

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

    if length(calls) == 0 || hasUser
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
                    argeltype = if LLVM.version().major >= 12
                        # TODO try to get sret element type if possible
                        # note currently opaque pointers has this break [and we need to doa check if opaque
                        # and if so get inner piece]
                        eltype(value_type(arg))
                    else
                        eltype(value_type(arg))
                    end
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
                        if !isa(ops[i], LLVM.AllocaInst)
                            illegalUse = true
                            break
                        end
                        eltype = LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(ops[i]))
                        seenfn = false
                        todo = LLVM.Instruction[]
                        for u2 in LLVM.uses(ops[i])
                            un2 = LLVM.user(u2)
                            push!(todo, un2)
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
                        al = alloca!(B, argeltype)
                        if value_type(al) != value_type(arg)
                            al = addrspacecast!(B, al, value_type(arg))
                        end
                        LLVM.replace_uses!(arg, al)
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
                    push!(toremove, i - 1)
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
            try
                todo = LLVM.CallInst[]
                for u in LLVM.uses(fn)
                    un = LLVM.user(u)
                    push!(next, LLVM.name(LLVM.parent(LLVM.parent(un))))
                end
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
            catch
                break
            end
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
function detect_writeonly!(mod::LLVM.Module)
    for f in functions(mod)
        if isempty(LLVM.blocks(f))
            continue
        end
        for (i, a) in enumerate(parameters(f))
            if isa(value_type(a), LLVM.PointerType)
                todo = Tuple{LLVM.Value,LLVM.Instruction}[]
                for u in LLVM.uses(a)
                    push!(todo, (a, LLVM.user(u)))
                end
                seen = Set{Tuple{LLVM.Value,LLVM.Instruction}}()
                mayread = false
                maywrite = false
                while length(todo) > 0
                    cur = pop!(todo)
                    if in(cur, seen)
                        continue
                    end
                    push!(seen, cur)
                    curv, curi = cur

                    if isa(curi, LLVM.StoreInst)
                        if operands(curi)[1] != curv
                            maywrite = true
                            continue
                        end
                    end

                    if isa(curi, LLVM.LoadInst)
                        mayread = true
                        continue
                    end

                    if isa(curi, LLVM.GetElementPtrInst) ||
                       isa(curi, LLVM.BitCastInst) ||
                       isa(curi, LLVM.AddrSpaceCastInst)
                        for u in LLVM.uses(curi)
                            push!(todo, (curi, LLVM.user(u)))
                        end
                        continue
                    end
                    mayread = true
                    maywrite = true
                end
                if any(
                    map(
                        k -> kind(k) == kind(EnumAttribute("readnone")),
                        collect(parameter_attributes(f, i)),
                    ),
                )
                    mayread = false
                    maywrite = false
                end
                if any(
                    map(
                        k -> kind(k) == kind(EnumAttribute("readonly")),
                        collect(parameter_attributes(f, i)),
                    ),
                )
                    maywrite = false
                end
                if any(
                    map(
                        k -> kind(k) == kind(EnumAttribute("writeonly")),
                        collect(parameter_attributes(f, i)),
                    ),
                )
                    mayread = false
                end

                LLVM.API.LLVMRemoveEnumAttributeAtIndex(
                    f,
                    LLVM.API.LLVMAttributeIndex(i),
                    kind(EnumAttribute("readnone")),
                )
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(
                    f,
                    LLVM.API.LLVMAttributeIndex(i),
                    kind(EnumAttribute("readonly")),
                )
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(
                    f,
                    LLVM.API.LLVMAttributeIndex(i),
                    kind(EnumAttribute("writeonly")),
                )

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

function checkNoAssumeFalse(mod, shouldshow = false)
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

cse!(pm) = LLVM.API.LLVMAddEarlyCSEPass(pm)

function removeDeadArgs!(mod::LLVM.Module, tm)
    # We need to run globalopt first. This is because remove dead args will otherwise
    # take internal functions and replace their args with undef. Then on LLVM up to 
    # and including 12 (but fixed 13+), Attributor will incorrectly change functions that
    # call code with undef to become unreachable, even when there exist other valid
    # callsites. See: https://godbolt.org/z/9Y3Gv6q5M
    ModulePassManager() do pm
        global_dce!(pm)
        LLVM.run!(pm, mod)
    end
    # Prevent dead-arg-elimination of functions which we may require args for in the derivative
    funcT = LLVM.FunctionType(LLVM.VoidType(), LLVMType[], vararg = true)
    if LLVM.version().major <= 15
        func, _ = get_function!(
            mod,
            "llvm.enzymefakeuse",
            funcT,
            [EnumAttribute("readnone"), EnumAttribute("nofree")],
        )
        rfunc, _ = get_function!(
            mod,
            "llvm.enzymefakeread",
            funcT,
            [
                EnumAttribute("readonly"),
                EnumAttribute("nofree"),
                EnumAttribute("argmemonly"),
            ],
        )
        sfunc, _ = get_function!(
            mod,
            "llvm.enzyme.sret_use",
            funcT,
            [
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
            [EnumAttribute("memory", NoEffects.data), EnumAttribute("nofree")],
        )
        rfunc, _ = get_function!(
            mod,
            "llvm.enzymefakeread",
            funcT,
            [EnumAttribute("memory", ReadOnlyArgMemEffects.data), EnumAttribute("nofree")],
        )
        sfunc, _ = get_function!(
            mod,
            "llvm.enzyme.sret_use",
            funcT,
            [EnumAttribute("memory", ReadOnlyArgMemEffects.data), EnumAttribute("nofree")],
        )
    end

    for fn in functions(mod)
        if isempty(blocks(fn))
            continue
        end
        # Ensure that interprocedural optimizations do not delete the use of returnRoots (or shadows)
        # if inactive sret, this will only occur on 2. If active sret, inactive retRoot, can on 3, and
        # active both can occur on 4. If the original sret is removed (at index 1) we no longer need
        # to preserve this.
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
            )
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
    ModulePassManager() do pm
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        alloc_opt_tm!(pm, tm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        cse!(pm)
        LLVM.run!(pm, mod)
    end
    propagate_returned!(mod)
    pre_attr!(mod)
    if RunAttributor[]
        if LLVM.version().major >= 13
            ModulePassManager() do pm
                API.EnzymeAddAttributorLegacyPass(pm)
                LLVM.run!(pm, mod)
            end
        end
    end
    propagate_returned!(mod)
    ModulePassManager() do pm
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        alloc_opt_tm!(pm, tm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        if RunAttributor[]
            if LLVM.version().major >= 13
                API.EnzymeAddAttributorLegacyPass(pm)
            end
        end
        cse!(pm)
        LLVM.run!(pm, mod)
    end
    post_attr!(mod)
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

function optimize!(mod::LLVM.Module, tm)
    addr13NoAlias(mod)
    # everying except unroll, slpvec, loop-vec
    # then finish Julia GC
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        propagate_julia_addrsp_tm!(pm, tm)
        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cfgsimplification!(pm)
        dce!(pm)
        cpu_features_tm!(pm, tm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        mem_cpy_opt!(pm)
        always_inliner!(pm)
        alloc_opt_tm!(pm, tm)
        LLVM.run!(pm, mod)
    end

    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    ModulePassManager() do pm

        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cpu_features_tm!(pm, tm)

        LLVM.API.LLVMAddGlobalOptimizerPass(pm) # Extra
        gvn!(pm) # Extra
        LLVM.run!(pm, mod)
    end
    
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cpu_features_tm!(pm, tm)

        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        cfgsimplification!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        jump_threading!(pm)
        correlated_value_propagation!(pm)
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        reassociate!(pm)
        early_cse!(pm)
        alloc_opt_tm!(pm, tm)
        loop_idiom!(pm)
        loop_rotate!(pm)

        loop_optimizations_tm!(pm, tm)

        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        ind_var_simplify!(pm)
        loop_deletion!(pm)
        loop_unroll!(pm)
        alloc_opt_tm!(pm, tm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        gvn!(pm)

        # This InstCombine needs to be after GVN
        # Otherwise it will generate load chains in GPU code...
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        mem_cpy_opt!(pm)
        sccp!(pm)
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        jump_threading!(pm)
        dead_store_elimination!(pm)
        alloc_opt_tm!(pm, tm)
        cfgsimplification!(pm)
        loop_idiom!(pm)
        loop_deletion!(pm)
        jump_threading!(pm)
        correlated_value_propagation!(pm)
        # SLP_Vectorizer -- not for Enzyme

        LLVM.run!(pm, mod)

        aggressive_dce!(pm)
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        # Loop Vectorize -- not for Enzyme
        # InstCombine

        # GC passes
        barrier_noop!(pm)
        gc_invariant_verifier_tm!(pm, tm, false)

        # FIXME: Currently crashes printing
        cfgsimplification!(pm)
        instruction_combining!(pm) # Extra for Enzyme
        jl_inst_simplify!(pm)
        LLVM.run!(pm, mod)
    end
    
    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cpu_features_tm!(pm, tm)

        LLVM.API.LLVMAddGlobalOptimizerPass(pm) # Exxtra
        gvn!(pm) # Exxtra
        LLVM.run!(pm, mod)
    end
    removeDeadArgs!(mod, tm)
    detect_writeonly!(mod)
    nodecayed_phis!(mod)
end

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L603
function addTargetPasses!(pm, tm, trip)
    add_library_info!(pm, trip)
    add_transform_info!(pm, tm)
end

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L620
function addOptimizationPasses!(pm, tm)
    add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))

    constant_merge!(pm)

    propagate_julia_addrsp_tm!(pm, tm)
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

    alloc_opt_tm!(pm, tm)
    # consider AggressiveInstCombinePass at optlevel > 2

    instruction_combining!(pm)
    jl_inst_simplify!(pm)
    cfgsimplification!(pm)
    scalar_repl_aggregates!(pm)
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy
    jl_inst_simplify!(pm)
    jump_threading!(pm)
    correlated_value_propagation!(pm)

    reassociate!(pm)

    early_cse!(pm)

    # Load forwarding above can expose allocations that aren't actually used
    # remove those before optimizing loops.
    alloc_opt_tm!(pm, tm)

    more_loop_optimizations_tm!(pm, tm)

    # Run our own SROA on heap objects before LLVM's
    alloc_opt_tm!(pm, tm)
    # Re-run SROA after loop-unrolling (useful for small loops that operate,
    # over the structure of an aggregate)
    scalar_repl_aggregates!(pm)
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy
    jl_inst_simplify!(pm)

    gvn!(pm)
    mem_cpy_opt!(pm)
    sccp!(pm)

    # Run instcombine after redundancy elimination to exploit opportunities
    # opened up by them.
    # This needs to be InstCombine instead of InstSimplify to allow
    # loops over Union-typed arrays to vectorize.
    instruction_combining!(pm)
    jl_inst_simplify!(pm)
    jump_threading!(pm)
    dead_store_elimination!(pm)

    # More dead allocation (store) deletion before loop optimization
    # consider removing this:
    alloc_opt_tm!(pm, tm)

    # see if all of the constant folding has exposed more loops
    # to simplification and deletion
    # this helps significantly with cleaning up iteration
    cfgsimplification!(pm)
    loop_deletion!(pm)
    instruction_combining!(pm)
    jl_inst_simplify!(pm)
    loop_vectorize!(pm)
    # TODO: createLoopLoadEliminationPass
    cfgsimplification!(pm)
    slpvectorize!(pm)
    # might need this after LLVM 11:
    # TODO: createVectorCombinePass()

    aggressive_dce!(pm)
end

function addMachinePasses!(pm, tm)
    combine_mul_add_tm!(pm, tm)
    # TODO: createDivRemPairs[]

    demote_float16_tm!(pm, tm)
    gvn!(pm)
end

function addJuliaLegalizationPasses!(pm, tm, lower_intrinsics = true)
    if lower_intrinsics
        # LowerPTLS removes an indirect call. As a result, it is likely to trigger
        # LLVM's devirtualization heuristics, which would result in the entire
        # pass pipeline being re-exectuted. Prevent this by inserting a barrier.
        barrier_noop!(pm)
        add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))
        lower_exc_handlers_tm!(pm, tm)
        # BUDE.jl demonstrates a bug here TODO
        gc_invariant_verifier_tm!(pm, tm, false)
        verifier!(pm)

        # Needed **before** LateLowerGCFrame on LLVM < 12
        # due to bug in `CreateAlignmentAssumption`.
        remove_ni_tm!(pm, tm)
        late_lower_gc_frame_tm!(pm, tm)
        final_lower_gc_tm!(pm, tm)
        # We need these two passes and the instcombine below
        # after GC lowering to let LLVM do some constant propagation on the tags.
        # and remove some unnecessary write barrier checks.
        gvn!(pm)
        sccp!(pm)
        # Remove dead use of ptls
        dce!(pm)
        lower_ptls_tm!(pm, tm, false) #=dump_native=#
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        # Clean up write barrier and ptls lowering
        cfgsimplification!(pm)
    else
        barrier_noop!(pm)
        remove_ni_tm!(pm, tm)
    end
end

function post_optimze!(mod, tm, machine = true)
    addr13NoAlias(mod)
    removeDeadArgs!(mod, tm)
    for f in collect(functions(mod))
        API.EnzymeFixupJuliaCallingConvention(f)
    end
    for f in collect(functions(mod))
        API.EnzymeFixupBatchedJuliaCallingConvention(f)
    end
    out_error = Ref{Cstring}()
    if LLVM.API.LLVMVerifyModule(mod, LLVM.API.LLVMReturnStatusAction, out_error) != 0
        throw(
            LLVM.LLVMException(
                "broken gc calling conv fix\n" *
                string(unsafe_string(out_error[])) *
                "\n" *
                string(mod),
            ),
        )
    end
    LLVM.ModulePassManager() do pm
        addTargetPasses!(pm, tm, LLVM.triple(mod))
        addOptimizationPasses!(pm, tm)
        LLVM.run!(pm, mod)
    end
    if machine
        # TODO enable validate_return_roots
        # validate_return_roots!(mod)
        LLVM.ModulePassManager() do pm
            addJuliaLegalizationPasses!(pm, tm, true)
            addMachinePasses!(pm, tm)
            LLVM.run!(pm, mod)
        end
    end
    # @safe_show "post_mod", mod
    # flush(stdout)
    # flush(stderr)
end
