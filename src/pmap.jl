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

	LLVM.Builder(ctx) do builder
		entry = BasicBlock(llvm_f, "entry"; ctx)
		position!(builder, entry)
		params = collect(LLVM.Value, parameters(llvm_f))
		lfn = @inbounds params[1]
		nparams = LLVM.Value[]

        for (parm, source_typ) in zip(params[2:end], realtypes[2:end])
            push!(nparams, parm)
            # codegen_typ = llvmtype(parm)
            # @show parm, source_typ
            # if codegen_typ isa LLVM.PointerType && !issized(eltype(codegen_typ))
            #     push!(nparams, parm)
            #     #push!(args, (cc=GPUCompiler.MUT_REF, typ=source_typ,
            #     #             codegen=(typ=codegen_typ, i=codegen_i)))
            # elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
            #        !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            #     if !GPUCompiler.deserves_argbox(source_typ) 
            #       push!(nparams, load!(builder, parm))
            #     else
            #       push!(nparams, parm)
            #     end
            #     # push!(args, (cc=GPUCompiler.BITS_REF, typ=source_typ,
            #     #             codegen=(typ=codegen_typ, i=codegen_i)))
            # else
            #     push!(nparams, parm)
            #     # push!(args, (cc=GPUCompiler.BITS_VALUE, typ=source_typ,
            #     #              codegen=(typ=codegen_typ, i=codegen_i)))
            # end
        end


		lfn = inttoptr!(builder, lfn, LLVM.PointerType(LLVM.FunctionType(Pvoid, [llvmtype(x) for x in nparams])))
		res = call!(builder, lfn, nparams)
        ret!(builder, res)
	end

    mod = LLVM.parent(llvm_f)

	ir = string(mod)
	fn = LLVM.name(llvm_f)

    quote
        Base.@_inline_meta
		Base.llvmcall(($ir, $fn), Ptr{Cvoid}, $(Tuple{realtypes...}), $(realargs...))
	end
end

function runtime_pmap_augfwd(count, forward, args...)::Ptr{Ptr{Cvoid}}
    # @warn "active variables passed by value to jl_pmap not yet supported"
    tapes = Base.unsafe_convert(Ptr{Ptr{Cvoid}}, Libc.malloc(sizeof(Ptr{Cvoid})*count))
    function fwd(idx, tapes, f_func, fargs...)
        st = callfn(f_func, idx, fargs...)
        Base.unsafe_store!(tapes, st, idx)
    end
    Enzyme.pmap(count, fwd, tapes, forward, args...)
    return tapes
end

function runtime_pmap_rev(count, adjoint, tapes, args...)
	function adj(idx, tapes, r_func, rargs...)
        st = unsafe_load(tapes, idx)
        callfn(r_func, idx, rargs..., st)
        nothing
	end
	Enzyme.pmap(count, adj, tapes, adjoint, args...)
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

function commonInnerCompile(runtime_fn, B, orig, gutils, tape)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    llvmfn = LLVM.called_value(orig)
    mi = nothing
    adjointnm = nothing
    forwardnm = nothing
    for fattr in collect(function_attributes(llvmfn))
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(Int, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
                break
            end
            if kind(fattr) == "enzymejl_forward"
                forwardnm = value(fattr)
            end
            if kind(fattr) == "enzymejl_adjoint"
                adjointnm = value(fattr)
            end
        end
    end

    countT = mi.specTypes.parameters[2]
	funcT = mi.specTypes.parameters[3]

    ops = collect(operands(orig))[1:end-1]

    if forwardnm === nothing
        @show mi.specTypes.parameters, ops
        flush(stdout)
        _, dup = julia_activity(mi.specTypes.parameters, [], ops, gutils, #=tape=#false)
        e_tt = Tuple{dup...}
        @static if VERSION >= v"1.8" 
          RT = Core.Compiler.return_type(Tuple{funcT, map(eltype, dup)...})
        else
          RT = Core.Compiler.return_type(Core.Compiler.singleton_type(funcT), Tuple{map(eltype, dup)...})
        end
        eprimal, eadjoint = fspec(Core.Compiler.singleton_type(funcT), e_tt)
        
        etarget = Compiler.EnzymeTarget()
        eparams = Compiler.EnzymeCompilerParams(eadjoint, API.DEM_ReverseModePrimal, Const{RT}, true, #=shadowfunc=#false, #=abiwrap=#false)
        ejob    = Compiler.CompilerJob(etarget, eprimal, eparams)
        
        cmod, adjointnm, forwardnm = _thunk(ejob)
        LLVM.link!(mod, cmod)
        attributes = function_attributes(llvmfn)
        push!(attributes, StringAttribute("enzymejl_forward", forwardnm; ctx))
        push!(attributes, StringAttribute("enzymejl_adjoint", adjointnm; ctx))
        attributes = function_attributes(llvmfn)
        push!(function_attributes(functions(mod)[forwardnm]), EnumAttribute("alwaysinline"; ctx))
        push!(function_attributes(functions(mod)[adjointnm]), EnumAttribute("alwaysinline"; ctx))
    end

    splat, _ = julia_activity(mi.specTypes.parameters, tape === nothing ? [Int, funcT, funcT] : [Int, Ptr{Ptr{Cvoid}}, funcT, funcT], ops, gutils, #=tape=#false)
    # splat[1] = eltype(splat[1])
    tt = Tuple{splat...}
   
    funcspec = FunctionSpec(runtime_fn, tt, #=kernel=# false, #=name=# nothing)

    # 3) Use the MI to create the correct augmented fwd/reverse
    # TODO:
    #  - GPU support
    #  - When OrcV2 only use a MaterializationUnit to avoid mutation of the module here

    target = GPUCompiler.NativeCompilerTarget()
    params = Compiler.PrimalCompilerParams()
    job    = CompilerJob(target, funcspec, params)  

    otherMod, meta = GPUCompiler.codegen(:llvm, job, optimize=false, validate=false)
    entry = name(meta.entry)
    optimize!(otherMod, JIT.get_tm())

    # 4) Link the corresponding module
    @show "prelink", mod
    @show "otherlink", otherMod
    LLVM.link!(mod, otherMod)
    @show "postlink", mod 

    # 5) Call the function
    entry = functions(mod)[entry]

    B = LLVM.Builder(B)
    
    T_int64 = LLVM.Int64Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)


    # count
	vals = LLVM.Value[LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1]))]
    
    # function
    run_fn = functions(mod)[tape === nothing ? forwardnm : adjointnm]
    push!(vals, ptrtoint!(B, run_fn, llvmtype(LLVM.ConstantInt(Int(0); ctx))))

    if tape !== nothing
        push!(vals, tape)
    end
    
    EB = LLVM.Builder(ctx)
    position!(EB, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    
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

        codegen_typ = llvmtype(parameters(entry)[i+1 + (tape !== nothing)])
        if codegen_typ isa LLVM.PointerType && !issized(eltype(codegen_typ))
            push!(vals, primal)
            if shadow !== nothing
              push!(vals, shadow)
            end
            #push!(args, (cc=GPUCompiler.MUT_REF, typ=source_typ,
            #             codegen=(typ=codegen_typ, i=codegen_i)))
        elseif codegen_typ isa LLVM.PointerType && issized(eltype(codegen_typ)) &&
               !(source_typ <: Ptr) && !(source_typ <: Core.LLVMPtr)
            if !GPUCompiler.deserves_argbox(source_typ) 
              primA = alloca!(EB, llvmtype(primal))
              store!(B, primal, primA)
              primal = addrspacecast!(B, primA, codegen_typ)
            end
            push!(vals, primal)
            if shadow !== nothing
              if !GPUCompiler.deserves_argbox(source_typ) 
                shadowA = alloca!(EB, llvmtype(shadow))
                store!(B, shadow, shadowA)
                shadow = addrspacecast!(B, shadowA, codegen_typ)
              end
              push!(vals, shadow)
            end
            # push!(args, (cc=GPUCompiler.BITS_REF, typ=source_typ,
            #             codegen=(typ=codegen_typ, i=codegen_i)))
        else
            push!(vals, primal)
            if shadow !== nothing
              push!(vals, shadow)
            end
            # push!(args, (cc=GPUCompiler.BITS_VALUE, typ=source_typ,
            #              codegen=(typ=codegen_typ, i=codegen_i)))
        end
        i += 1
    end

    return LLVM.call!(B, entry, vals)
end

function pmap_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    tape = commonInnerCompile(runtime_pmap_augfwd, B, orig, gutils, nothing)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end

    unsafe_store!(tapeR, tape.ref)

    return nothing
end

function pmap_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    commonInnerCompile(runtime_pmap_rev, B, orig, gutils, LLVM.Value(tape))
    return nothing
end

