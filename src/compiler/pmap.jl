function pmap_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::UInt8
    orig = LLVM.Instruction(OrigCI)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) != 0 && API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) != 0
        return 1
    end
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)
    B = LLVM.IRBuilder(B)
    emit_error("fast pfor not implemented");
    return 0
end

function runtime_pmap_augfwd(count, ::Type{ThunkTy}, ::Val{AnyJL}, forward, args...) where {ThunkTy, AnyJL}
    TapeType = get_tape_type(ThunkTy)
    TT = if AnyJL
        Vector{TapeType}
    else
        Ptr{TapeType}
    end
    tapes = if AnyJL
        Vector{TapeType}(undef, count)
    else
        Base.unsafe_convert(Ptr{TapeType}, Libc.malloc(sizeof(TapeType)*count))
    end
    function fwd(idx, tapes, f_func, f, df, fargs...)
        st = raw_enzyme_call(ThunkTy(f_func), Const(f), idx, fargs...)[1]
        if !AnyJL
            unsafe_store!(tapes, st, idx)
        else
            @inbounds tapes[idx] = st
        end
    end
    Enzyme.pmap(count, fwd, tapes, forward, args...)
    return tapes
end

function runtime_pmap_rev(count, ::Type{ThunkTy}, ::Val{AnyJL}, adjoint, tapes, args...) where {ThunkTy, AnyJL}
    function adj(idx, tapes, r_func, f, df, rargs...)
        st = if !AnyJL
            unsafe_load(tapes, idx)
        else
            @inbounds tapes[idx]
        end
        raw_enzyme_call(ThunkTy(r_func), Const(f), idx, rargs..., st)
        # st = unsafe_load(tapes, idx)
        nothing
	end
	Enzyme.pmap(count, adj, tapes, adjoint, args...)
    if !AnyJL
        Libc.free(tapes)
    end
    return nothing
end

function julia_activity(orig, source_types, FTs, ops, gutils)
    source_types = source_types[2:end]
    # count, funcT, funcT
    args = Type[source_types[1]]
    dup_args = Type[Const{source_types[1]}]
    for T in FTs
      push!(args, T)
      push!(dup_args, T)
    end
    codegen_i = 2
    
    overwritten = Bool[]
    uncacheable = Vector{UInt8}(undef, length(ops))
    API.EnzymeGradientUtilsGetUncacheableArgs(gutils, orig, uncacheable, length(uncacheable))
    
    for source_typ in source_types[3:end]
        if isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            push!(overwritten, false)
            continue
        end
        push!(overwritten, uncacheable[codegen_i])

        codegen_typ = value_type(ops[codegen_i])
        if codegen_typ isa LLVM.PointerType && !issized(eltype(codegen_typ))
            push!(args, source_typ)
            if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[codegen_i]) == 0
              push!(args, source_typ)
              push!(dup_args, Duplicated{source_typ})
            else
              push!(dup_args, Const{source_typ})
            end
            #push!(args, (cc=GPUCompiler.MUT_REF, typ=source_typ,
            #             codegen=(typ=codegen_typ, i=codegen_i)))
        elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
               !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            push!(args, source_typ)
            if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[codegen_i]) == 0
              push!(args, source_typ)
              push!(dup_args, Duplicated{source_typ})
            else
              push!(dup_args, Const{source_typ})
            end
            # push!(args, (cc=GPUCompiler.BITS_REF, typ=source_typ,
            #             codegen=(typ=codegen_typ, i=codegen_i)))
        else
            push!(args, source_typ)
            if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[codegen_i]) == 0
              push!(args, source_typ)
              push!(dup_args, Duplicated{source_typ})
            else
              push!(dup_args, Const{source_typ})
            end
            # push!(args, (cc=GPUCompiler.BITS_VALUE, typ=source_typ,
            #              codegen=(typ=codegen_typ, i=codegen_i)))
        end
        codegen_i += 1
    end
    return args, dup_args, overwritten
end

function commonInnerCompile(runtime_fn, B, orig, gutils, tape, mode)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    llvmfn = LLVM.called_value(orig)
    mi, _ = enzyme_custom_extract_mi(orig)
    adjointnm = nothing
    augfwdnm = nothing
    TapeType = nothing
    for fattr in collect(function_attributes(llvmfn))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_tapetype"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                TapeType = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_augforward"
                augfwdnm = value(fattr)
            end
            if kind(fattr) == "enzymejl_adjoint"
                adjointnm = value(fattr)
            end
        end
    end

    countT = mi.specTypes.parameters[2]
	funcT = mi.specTypes.parameters[3]

    ops = collect(operands(orig))[1:end-1] 
    
    B = LLVM.IRBuilder(B)
    world = enzyme_extract_world(LLVM.parent(position(B)))
    
    @assert GPUCompiler.isghosttype(funcT) || Core.Compiler.isconstType(funcT) 

    _, dup, overwritten = julia_activity(orig, mi.specTypes.parameters, [], ops, gutils)
        e_tt = Tuple{dup...}
        @static if VERSION >= v"1.8" 
          RT = Core.Compiler.return_type(Tuple{funcT, map(eltype, dup)...}, world)
        else
          RT = Core.Compiler.return_type(Core.Compiler.singleton_type(funcT), Tuple{map(eltype, dup)...}, world)
        end
        eprimal = fspec(funcT, e_tt, world)
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        
    if augfwdnm === nothing
        # TODO: Clean this up and add to `nested_codegen!` asa feature
        etarget = Compiler.EnzymeTarget()
        funcOverwritten = true
        indexOverwritten = false
        eparams = Compiler.EnzymeCompilerParams(Tuple{Const{funcT}, dup...}, API.DEM_ReverseModePrimal, width, Const{RT}, true,
                                                #=abiwrap=#true, #=modifiedBetween=#(funcOverwritten, indexOverwritten, overwritten...,), #=returnPrimal=#false, #=shadowprimalInit=#false, Compiler.UnknownTapeType, FFIABI)
        ejob    = Compiler.CompilerJob(eprimal, CompilerConfig(etarget, eparams; kernel=false), world)
            
        jctx = ctx
@static if VERSION < v"1.9-"
else
        jctx = ctxToThreadSafe[jctx]
end
        
        cmod, adjointnm, augfwdnm, _, TapeType = _thunk(ejob, jctx)
        LLVM.link!(mod, cmod)
        attributes = function_attributes(llvmfn)
        push!(attributes, StringAttribute("enzymejl_augforward", augfwdnm; ctx))
        push!(attributes, StringAttribute("enzymejl_adjoint", adjointnm; ctx))
        attributes = function_attributes(llvmfn)
        push!(function_attributes(functions(mod)[augfwdnm]), EnumAttribute("alwaysinline"; ctx))
        push!(function_attributes(functions(mod)[adjointnm]), EnumAttribute("alwaysinline"; ctx))
        push!(attributes, StringAttribute("enzymejl_tapetype", string(convert(UInt, unsafe_to_pointer(TapeType))); ctx))
    end

        if mode == API.DEM_ReverseModePrimal
            thunkTy = AugmentedForwardThunk{Ptr{Cvoid}, Const{funcT}, Const{Nothing}, e_tt, Val{width},  #=returnPrimal=#Val(true), TapeType}
            subfunc = functions(mod)[augfwdnm]
       else
           thunkTy = AdjointThunk{Ptr{Cvoid}, Const{funcT}, Const{Nothing}, e_tt, Val{width}, TapeType}
            subfunc = functions(mod)[adjointnm]
        end

    STT = if !any_jltypes(TapeType)
        Ptr{TapeType}
    else
        Vector{TapeType}
    end

    splat, _, _ = julia_activity(orig, mi.specTypes.parameters, (mode != API.DEM_ReverseModeGradient) ? [Type{thunkTy}, Val{any_jltypes(TapeType)}, Int, funcT, funcT] : [Type{thunkTy}, Val{any_jltypes(TapeType)}, Int, STT, funcT, funcT], ops, gutils)
    tt = Tuple{splat...}
    entry = nested_codegen!(mode, mod, runtime_fn, tt, world)

    # 5) Call the function
    
    T_int64 = LLVM.Int64Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)


    # count
	vals = LLVM.Value[LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1]))]
    
    # function
    run_fn = functions(mod)[tape === nothing ? augfwdnm : adjointnm]
    push!(vals, ptrtoint!(B, run_fn, value_type(LLVM.ConstantInt(Int(0); ctx))))
 
    EB = LLVM.IRBuilder(ctx)
    position!(EB, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    

    # handle the accidental sret
    if isa(value_type(parameters(entry)[1]), LLVM.PointerType)
        a = alloca!(EB, eltype(value_type(parameters(entry)[1])))
        pushfirst!(vals, a)
    end
   
    if mode == API.DEM_ReverseModeGradient && STT != Nothing
		@assert tape != nothing
        push!(vals, tape)
    end

    i = 2
    for source_typ in mi.specTypes.parameters[3:end]
        if isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end


        primal = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[i]))
        shadow = if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[i]) == 0
          LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[i], B))
        else
          nothing
        end

        codegen_typ = value_type(parameters(entry)[length(vals)+1])

        if codegen_typ == value_type(primal)
            push!(vals, primal)
            if shadow !== nothing
              push!(vals, shadow)
            end
        elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
               !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            if !GPUCompiler.deserves_argbox(source_typ)
              primA = alloca!(EB, value_type(primal))
              store!(B, primal, primA)
              primal = addrspacecast!(B, primA, codegen_typ)
            end
            push!(vals, primal)
            if shadow !== nothing
              if !GPUCompiler.deserves_argbox(source_typ) 
                shadowA = alloca!(EB, value_type(shadow))
                store!(B, shadow, shadowA)
                shadow = addrspacecast!(B, shadowA, codegen_typ)
              end
              push!(vals, shadow)
            end
            # push!(args, (cc=GPUCompiler.BITS_REF, typ=source_typ,
            #             codegen=(typ=codegen_typ, i=codegen_i)))
        else
			@assert false
        end
        i += 1
    end

    res = LLVM.call!(B, LLVM.function_type(entry), entry, vals)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, res, orig)
    
    return res
end

function pmap_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::UInt8
    orig = LLVM.Instruction(OrigCI)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) != 0 && API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) != 0
        return 1
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    GPUCompiler.@safe_warn "active variables passed by value to jl_pmap not yet supported"
    tape = commonInnerCompile(runtime_pmap_augfwd, B, orig, gutils, nothing, API.DEM_ReverseModePrimal)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig))
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    unsafe_store!(tapeR, tape.ref)

    return 0
end

function pmap_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    commonInnerCompile(runtime_pmap_rev, B, orig, gutils, LLVM.Value(tape), API.DEM_ReverseModeGradient)
    return nothing
end
