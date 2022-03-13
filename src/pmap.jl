function pmap_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)
    B = LLVM.Builder(B)
    emit_error("fast pfor not implemented");
    return nothing
end

@generated function callfn(ptr, args...)
    ctx = LLVM.Context()
    Pvoid = convert(LLVMType, Ptr{Cvoid}; ctx)
    llvmtys = LLVMType[convert(LLVMType, Int; ctx)]
    realargs = Any[:ptr]
    realtypes = [ptr]
    for (i, a) in enumerate(args)
        if GPUCompiler.isghosttype(a) || Core.Compiler.isconstType(a)
            continue
        end
        push!(llvmtys, convert(LLVMType, a; ctx, allow_boxed=true))
        push!(realargs, :(args[$i]))
        push!(realtypes, a)
    end
    llvm_f, _ = LLVM.Interop.create_function(Pvoid, llvmtys)
    @show realargs, realtypes

	LLVM.Builder(ctx) do builder
		entry = BasicBlock(llvm_f, "entry"; ctx)
		position!(builder, entry)
		params = collect(LLVM.Value, parameters(llvm_f))
		lfn = @inbounds params[1]
		params = params[2:end]
		lfn = inttoptr!(builder, lfn, LLVM.PointerType(LLVM.FunctionType(Pvoid, [llvmtype(x) for x in params])))
		res = call!(builder, lfn, params)
        ret!(builder, res)
	end

    mod = LLVM.parent(llvm_f)

	ir = string(mod)
	fn = LLVM.name(llvm_f)

    mac = quote
        Base.@_inline_meta
		Base.llvmcall(($ir, $fn), Ptr{Cvoid}, $(Tuple{realtypes...}), $(realargs...))
	end
    @show mac
    mac
end

function runtime_pmap_augfwd(count, forward, args...)::Ptr{Ptr{Cvoid}}
    @warn "active variables passed by value to jl_pmap not yet supported"
    tapes = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, Libc.malloc(sizeof(Ptr{Cvoid})*count))
    res = callfn(forward, 0, args...)
    return tapes
   #  e_tt = Tuple{Const{typeof(count)}, map(typeof, args)...}
   #  RT = Core.Compiler.return_type(func, Tuple{typeof(count), map((x,)->eltype(typeof(x)), args)...})

   #  function fwd(idx, tapes, f_func, fargs...)
   #      @show f_func, idx, fargs...
   #      flush(stdout)
   #      f_func(Const(idx), fargs...)
   #      @show res
   #      flush(stdout)
   #  	tapes[idx] = res[1]
   #  end
   #  Enzyme.pmap(count, fwd, tapes, AugmentedForwardThunk{typeof(func), RT, e_tt, typeof(nothing)}(func, forward, nothing), args...)
   #  return tapes
end

function runtime_pmap_rev(count, tapes, adjoint, args...)
    # e_tt = Tuple{Const{typeof(count)}, map(typeof, args)...}
    # RT = Core.Compiler.return_type(func, Tuple{typeof(count), map((x,)->eltype(typeof(x)), args)...})

	# function adj(idx, tapes, r_func, rargs...)
    #     r_func(Const(idx), rargs..., tapes[idx])
    #     nothing
	# end
	# Enzyme.pmap(count, tapes, AdjointThunk{typeof(func), RT, e_tt, typeof(nothing)}(func, adjoint, nothing), args...)
    Libc.free(tapes)
    return nothing
end

function julia_activity(source_types, FTs, ops, gutils, tape::Bool)
    source_types = source_types[2:end]
    # count, funcT, funcT
    args = Type[source_types[1]]
    dup_args = Type[Const{source_types[1]}]
    if tape
      push!(args, Ptr{Ptr{Cvoid}})
      push!(dup_args, Ptr{Ptr{Cvoid}})
    end
    for T in FTs
      push!(args, T)
      push!(dup_args, T)
    end
    codegen_i = 2
    
    for source_typ in source_types[3:end]
        if isghosttype(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end

        codegen_typ = llvmtype(ops[codegen_i])
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
    return args, dup_args
end

function pmap_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    llvmfn = LLVM.called_value(orig)
    mi = nothing
    for fattr in collect(function_attributes(llvmfn))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
                break
            end
        end
    end

    countT = mi.specTypes.parameters[2]
	funcT = mi.specTypes.parameters[3]

    ops = collect(operands(orig))[1:end-1]

    _, dup = julia_activity(mi.specTypes.parameters, [], ops, gutils, #=tape=#false)
    e_tt = Tuple{dup...}
    RT = Core.Compiler.return_type(Core.Compiler.singleton_type(funcT), Tuple{map(eltype, dup)...})
    eprimal, eadjoint = fspec(Core.Compiler.singleton_type(funcT), e_tt)
    
    etarget = Compiler.EnzymeTarget()
    eparams = Compiler.EnzymeCompilerParams(eadjoint, API.DEM_ReverseModePrimal, Const{RT}, true, #=shadowfunc=#false, #=abiwrap=#false)
    ejob    = Compiler.CompilerJob(etarget, eprimal, eparams)
    
    cmod, adjointnm, forwardnm = _thunk(ejob)
    LLVM.link!(mod, cmod)
    splat, _ = julia_activity(mi.specTypes.parameters, [Int, funcT, funcT], ops, gutils, #=tape=#false)
    # splat[1] = eltype(splat[1])
    tt = Tuple{splat...}
   
    funcspec = FunctionSpec(runtime_pmap_augfwd, tt, #=kernel=# false, #=name=# nothing)

    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = GPUCompiler.NativeCompilerTarget()
    params = Compiler.PrimalCompilerParams()
    job    = CompilerJob(target, funcspec, params)  

    otherMod, meta = GPUCompiler.codegen(:llvm, job, optimize=false, validate=false)
    entry = name(meta.entry)

    # 4) Link the corresponding module
    LLVM.link!(mod, otherMod)
    

    # 5) Call the function
    entry = functions(mod)[entry]

    B = LLVM.Builder(B)
    
    T_int64 = LLVM.Int64Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)


    # count
	vals = LLVM.Value[LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1]))]

    # push!(vals, addrspacecast!(B, inttoptr!(B, LLVM.ConstantInt(convert(Int, pointer_from_objref(RT)); ctx), LLVM.PointerType(T_jlvalue)), T_prjlvalue))
    # push!(vals, addrspacecast!(B, inttoptr!(B, LLVM.ConstantInt(convert(Int, pointer_from_objref(e_tt)); ctx), LLVM.PointerType(T_jlvalue)), T_prjlvalue))
    # function
    push!(vals, ptrtoint!(B, functions(mod)[forwardnm], llvmtype(LLVM.ConstantInt(Int(0); ctx))))
    
    # TODO: Optimization by emitting liverange
    for i in 2:length(ops)
        primal = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[i]))
        push!(vals, primal)
        if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[i]) == 0
          shadow = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[i], B))
          push!(vals, shadow)
        end
    end
    
    tape = LLVM.call!(B, entry, vals)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end

    @show mod, entry

    unsafe_store!(tapeR, tape.ref)

    return nothing
end

function pmap_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    llvmfn = LLVM.called_value(orig)
    mi = nothing
    for fattr in collect(function_attributes(llvmfn))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
                break
            end
        end
    end

    countT = mi.specTypes.parameters[2]
	funcT = mi.specTypes.parameters[3]

    ops = collect(operands(orig))[1:end-1]
    
    _, dup = julia_activity(mi.specTypes.parameters, [], ops, gutils, #=tape=#false)
    e_tt = Tuple{dup...}
    RT = Core.Compiler.return_type(Core.Compiler.singleton_type(funcT), Tuple{map(eltype, dup)...})
    forward, adjoint = thunk(Core.Compiler.singleton_type(funcT), #=dfn=#nothing, Const{RT}, e_tt, Val(API.DEM_ReverseModePrimal))
    _, splat = julia_activity(mi.specTypes.parameters, [funcT, typeof(adjoint.adjoint)], ops, gutils, #=tape=#true)
    splat[1] = eltype(splat[1])
    tt = Tuple{splat...}
    
    funcspec = FunctionSpec(runtime_pmap_rev, tt, #=kernel=# false, #=name=# nothing)

    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = GPUCompiler.NativeCompilerTarget()
    params = Compiler.PrimalCompilerParams()
    job    = CompilerJob(target, funcspec, params)  

    otherMod, meta = GPUCompiler.codegen(:llvm, job, optimize=false, validate=false)
    entry = name(meta.entry)

    # 4) Link the corresponding module
    LLVM.link!(mod, otherMod)

    # 5) Call the function
    entry = functions(mod)[entry]

    B = LLVM.Builder(B)
    
    T_int64 = LLVM.Int64Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)

    ops = collect(operands(orig))[1:end-1]

    # count
	vals = LLVM.Value[LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1]), B))]
    
    # tape
    tape = LLVM.Value(tape)
    push!(vals, tape)
    
    
    # push!(vals, addrspacecast!(B, inttoptr!(B, LLVM.ConstantInt(convert(Int, pointer_from_objref(RT)); ctx), LLVM.PointerType(T_jlvalue)), T_prjlvalue))
    # push!(vals, addrspacecast!(B, inttoptr!(B, LLVM.ConstantInt(convert(Int, pointer_from_objref(e_tt)); ctx), LLVM.PointerType(T_jlvalue)), T_prjlvalue))
    
    # function
    push!(vals, LLVM.ConstantInt(convert(Int, adjoint.adjoint); ctx))
                                                                
    EB = LLVM.Builder(ctx)
    position!(EB, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    
    to_preserve = LLVM.Value[]
    for i in 2:length(ops)
        primal = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[i]))
        if API.EnzymeGradientUtilsIsConstantValue(gutils, ops[i]) == 0
          shadow = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[i], B))
          v = UndefValue(LLVM.ArrayType(llvmtype(primal), 2)) # llvmtype(shadow)]; ctx)) 
          v = insert_value!(B, v, primal, 0)
          v = insert_value!(B, v, shadow, 1)
          
          ret = LLVM.alloca!(EB, llvmtype(v))
          store!(B, v, ret)
          ret = addrspacecast!(B, ret, LLVM.PointerType(llvmtype(v), 11))
          push!(vals, ret)
        else
          push!(to_preserve, primal)
          push!(vals, primal)
        end
    end
    
    token = emit_gc_preserve_begin(B, to_preserve)
    LLVM.call!(B, entry, vals)
    emit_gc_preserve_end(B, token)
    
    return nothing
end

