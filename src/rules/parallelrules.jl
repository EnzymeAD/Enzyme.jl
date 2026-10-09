
function runtime_newtask_fwd(
    fn::FT1,
    dfn::FT2,
    post::Any,
    ssize::Int,
    runtimeActivity::Val{RuntimeActivity},
    strongZero::Val{StrongZero},
    ::Val{width},
) where {FT1,FT2,width,RuntimeActivity, StrongZero}
    FT = Core.Typeof(fn)
    ghos = guaranteed_const(FT)
    forward = thunk(
        Val(0),
        (ghos ? Const : Duplicated){FT},
        Const,
        Tuple{},
        Val(API.DEM_ForwardMode),
        Val(Int(width)),
        Val((false,)),
        Val(true),
        Val(false),
        FFIABI,
        Val(false),
        runtimeActivity,
        strongZero
    ) #=erriffuncwritten=#
    ft = ghos ? Const(fn) : Duplicated(fn, dfn)
    function fclosure()
        res = forward(ft)
        return res[1]
    end

    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)
end

struct Return2
    ret1::Any
    ret2::Any
end

function runtime_newtask_augfwd(
    fn::FT1,
    dfn::FT2,
    post::Any,
    ssize::Int,
    runtimeActivity::Val{RuntimeActivity},
    strongZero::Val{StrongZero},
    ::Val{width},
    ::Val{ModifiedBetween},
) where {FT1,FT2,width,ModifiedBetween,RuntimeActivity,StrongZero}
    # TODO make this AD subcall type stable
    FT = Core.Typeof(fn)
    ghos = guaranteed_const(FT)
    forward, adjoint = thunk(
        Val(0),
        (ghos ? Const : Duplicated){FT},
        Const,
        Tuple{},
        Val(API.DEM_ReverseModePrimal),
        Val(Int(width)),
        Val(ModifiedBetween),
        Val(true),
        Val(false),
        FFIABI,
        Val(false),
        runtimeActivity,
        strongZero
    ) #=erriffuncwritten=#
    ft = ghos ? Const(fn) : Duplicated(fn, dfn)
    taperef = Ref{Any}()

    function fclosure()
        res = forward(ft)
        taperef[] = res[1]
        return res[2]
    end

    ftask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)

    function rclosure()
        adjoint(ft, taperef[])
        return 0
    end

    rtask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), rclosure, post, ssize)

    return Return2(ftask, rtask)
end


function referenceCaller(fn::Ref{Clos}, args...) where {Clos}
    fval = fn[]
    fval = fval::Clos
    fval(args...)
end

function runtime_pfor_fwd(
    thunk::ThunkTy,
    ft::FT,
    threading_args...,
)::Cvoid where {ThunkTy,FT}
    function fwd(tid_args...)
        if length(tid_args) == 0
            thunk(ft)
        else
            thunk(ft, Const(tid_args[1]))
        end
    end
    Base.Threads.threading_run(fwd, threading_args...)
    return
end

"""
    runtime_pfor_erased(tramp::Ptr{Cvoid}, env::Ptr{Cvoid}, static::Bool)

Type-erased threading wrapper for the forward derivative of a `Threads.@threads` loop:
run `tramp(env, tid)` on every thread through `Base.Threads.threading_run`. `tramp` is
the C-ABI trampoline of the loop-body derivative (see [`pfor_erased_body!`](@ref)), and
`env` points to the annotated loop closure. Its specialization is shared by all loops
and compiled once by Julia, instead of emitting the threading machinery (task creation,
scheduling, waiting) into the differentiated module for every loop.
"""
@noinline function runtime_pfor_erased(tramp::Ptr{Cvoid}, env::Ptr{Cvoid}, static::Bool)::Cvoid
    fwd(tid) = ccall(tramp, Cvoid, (Ptr{Cvoid}, Int), env, tid)
    Base.Threads.threading_run(fwd, static)
    return
end

"""
    PFOR_ERASED

Use [`runtime_pfor_erased`](@ref) for forward-mode `Threads.@threads` loops where
available (Julia 1.12 and later, host code). Set to `false` to always emit the
specialized `runtime_pfor_fwd` into the differentiated module.
"""
const PFOR_ERASED = Ref(true)

pfor_erased_available(mod::LLVM.Module) =
    PFOR_ERASED[] && VERSION >= v"1.12-" && native_invoke_available(mod)

# Trampolines of loop-body derivatives and their edges, keyed by the hash of their
# compiler job.
const pfor_erased_cache = Dict{UInt, Tuple{Ptr{Cvoid}, Vector{Any}}}()

"""
    pfor_erased_body!(ejob, dFT) -> Union{Ptr{Cvoid}, Nothing}

Compile the derivative of a loop body, the forward-mode FFI-ABI job `ejob` taking the
annotated closure `dFT` and the thread id, as an ordinary thunk in its own JIT module.
Add to that module a trampoline `void(ptr env, i64 tid)` that loads the fields of `dFT`
from `env`, where they are laid out as the Julia value, and calls the derivative.
Return the address of the trampoline, or `nothing` when the derivative does not take
the fields of `dFT` by value followed by the thread id.
"""
function pfor_erased_body!(ejob, @nospecialize(dFT))
    key = hash(ejob)
    lock(cache_lock)
    try
        cached = get(pfor_erased_cache, key, nothing)
        if cached !== nothing
            append!(enzyme_context().edges, cached[2])
            return cached[1]
        end
        mod, edges, adjoint_name, _, TapeType, prepost, _ = _thunk(ejob)
        adj = functions(mod)[adjoint_name]
        fty = LLVM.function_type(adj)
        ps = parameters(fty)
        dl = LLVM.datalayout(mod)
        nf = isghostty(dFT) ? 0 : fieldcount(dFT)
        ok = LLVM.return_type(fty) isa LLVM.VoidType && length(ps) == nf + 1 &&
             ps[end] == LLVM.IntType(8 * sizeof(Int))
        for i in 1:nf
            ok || break
            ok = LLVM.storage_size(dl, ps[i]) == sizeof(fieldtype(dFT, i))
        end
        if !ok
            dispose(mod)
            return nothing
        end
        T_ptr = LLVM.PointerType(LLVM.Int8Type())
        trampf = LLVM.Function(mod, "pfor_tramp_" * adjoint_name,
                               LLVM.FunctionType(LLVM.VoidType(), [T_ptr, ps[end]]))
        B = LLVM.IRBuilder()
        position!(B, LLVM.BasicBlock(trampf, "entry"))
        env, tid = parameters(trampf)
        args = LLVM.Value[]
        for i in 1:nf
            ptr = inbounds_gep!(B, LLVM.Int8Type(), env,
                                LLVM.Value[LLVM.ConstantInt(Int64(fieldoffset(dFT, i)))])
            push!(args, load!(B, ps[i], ptr))
        end
        push!(args, tid)
        cal = call!(B, fty, adj, args)
        callconv!(cal, callconv(adj))
        ret!(B)
        dispose(B)
        obj = _link(ejob, mod, edges, LLVM.name(trampf), nothing, TapeType, prepost)
        tramp = obj.adjoint::Ptr{Cvoid}
        pfor_erased_cache[key] = (tramp, edges)
        append!(enzyme_context().edges, edges)
        return tramp
    finally
        unlock(cache_lock)
    end
end

"""
    pfor_erased_entry!(mod) -> LLVM.Function

Declaration in `mod` of the natively compiled [`runtime_pfor_erased`](@ref).
"""
function pfor_erased_entry!(mod::LLVM.Module)
    world = enzyme_context().world
    mi = my_methodinstance(Forward, typeof(runtime_pfor_erased), Tuple{Ptr{Cvoid}, Ptr{Cvoid}, Bool}, world)
    name = "ejl_runtime_pfor_erased"
    haskey(functions(mod), name) && return functions(mod)[name]
    ci = codeinst(mi, world)
    ci === nothing && return nothing
    specptr, _ = Interpreter.codeinst_entry(ci)
    specptr == C_NULL && return nothing
    push!(enzyme_context().edges, mi)
    return declare_native!(mod, mi, ci.rettype, specptr, name, world)
end

function runtime_pfor_augfwd(
    thunk::ThunkTy,
    ft::FT,
    ::Val{AnyJL},
    ::Val{byRef},
    threading_args...,
) where {ThunkTy,FT,AnyJL,byRef}
    TapeType = EnzymeRules.tape_type(ThunkTy)

    n = Base.Threads.threadpoolsize()
    tapes = if AnyJL
        Vector{TapeType}(undef, n)
    else
        Base.unsafe_convert(
            Ptr{TapeType},
            Libc.malloc(sizeof(TapeType) * n),
        )
    end

    function fwd(tid_args...)
        tid = tid_args[1]
        if byRef
            tres = thunk(Const(referenceCaller), ft, Const(tid))
        else
            tres = thunk(ft, Const(tid))
        end

        if !AnyJL
            unsafe_store!(tapes, tres[1], tid)
        else
            @inbounds tapes[tid] = tres[1]
        end
    end
    Base.Threads.threading_run(fwd, threading_args...)
    return tapes
end

struct ReversePFor{ThunkTy, FT, AnyJL, byRef, TT}
    thunk::ThunkTy
    ft::FT
    tapes::TT
end

function (st::ReversePFor{ThunkTy, FT, AnyJL, byRef, TT})(tid) where {ThunkTy, FT, AnyJL, byRef, TT}
    tres = if !AnyJL
        unsafe_load(st.tapes, tid)
    else
        @inbounds st.tapes[tid]
    end
    
    if byRef
        st.thunk(Const(referenceCaller), st.ft, Const(tid), tres)
    else
        st.thunk(st.ft, Const(tid), tres)
    end

    nothing
end

function runtime_pfor_rev(
    thunk::ThunkTy,
    ft::FT,
    ::Val{AnyJL},
    ::Val{byRef},
    tapes,
    threading_args...,
) where {ThunkTy,FT,AnyJL,byRef}
    Base.Threads.threading_run(ReversePFor{ThunkTy, FT, AnyJL, byRef, typeof(tapes)}(thunk, ft, tapes), threading_args...)
    if !AnyJL
        Libc.free(tapes)
    end
    return nothing
end

@inline function threadsfor_common(orig, gutils, B, mode, enzyme_ctx, tape = nothing)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    llvmfn = LLVM.called_operand(orig)
    mi = nothing
    fwdmodenm = nothing
    augfwdnm = nothing
    adjointnm = nothing
    TapeType = nothing
    attributes = function_attributes(llvmfn)
    for fattr in collect(attributes)
        if isa(fattr, LLVM.StringAttribute)
            if kind(fattr) == "enzymejl_mi"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                mi = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_tapetype"
                ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(fattr)))
                TapeType = Base.unsafe_pointer_to_objref(ptr)
            end
            if kind(fattr) == "enzymejl_forward"
                fwdmodenm = value(fattr)
            end
            if kind(fattr) == "enzymejl_augforward"
                augfwdnm = value(fattr)
            end
            if kind(fattr) == "enzymejl_adjoint"
                adjointnm = value(fattr)
            end
        end
    end

    funcT = mi.specTypes.parameters[2]


    # TODO actually do modifiedBetween
    e_tt = Tuple{Const{Int}}
    modifiedBetween = (mode != API.DEM_ForwardMode, false)

    world = enzyme_world()

    pfuncT = funcT

    mi2 = my_methodinstance(mode == API.DEM_ForwardMode ? Forward : Reverse, funcT, Tuple{map(eltype, e_tt.parameters)...}, world)
    @assert mi2 !== nothing

    refed = false

    # TODO: Clean this up and add to `nested_codegen!` asa feature
    width = Int(get_width(gutils))

    dupClosure = !guaranteed_const_nongen(funcT, world)
    if dupClosure
	if is_constant_value(gutils, operands(orig)[1])
	    dupClosure = false
	    if inline_roots_type(funcT) != 0
	        if !is_constant_value(gutils, operands(orig)[2])
		    dupClosure = true
		end
	    end
	end
    end
  
    pdupClosure = dupClosure

    subfunc = nothing

    dFT = (dupClosure ? (width == 1 ? Duplicated : (BatchDuplicated{T, Int(width)} where T)) : Const){funcT}

    if mode == API.DEM_ForwardMode
        if fwdmodenm === nothing
            etarget = Compiler.EnzymeTarget()
            eparams = Compiler.EnzymeCompilerParams(
                Tuple{dFT,e_tt.parameters...},
                API.DEM_ForwardMode,
                width,
                Const{Nothing},
                true,
                true,
                modifiedBetween,
                false,
                false,
                UnknownTapeType,
                FFIABI,
                false,
                get_runtime_activity(gutils),
                get_strong_zero(gutils),
            ) #=ErrIfFuncWritten=#
            ejob = Compiler.CompilerJob(
                mi2,
                CompilerConfig(etarget, eparams; kernel = false),
                world,
            )

            tramp = if pfor_erased_available(mod) && pfor_erased_entry!(mod) !== nothing
                pfor_erased_body!(ejob, dFT)
            end
            if tramp !== nothing
                subfunc = tramp
            else
                cmod, edges, fwdmodenm, _, _, _, value_table = _thunk(ejob, false) #=postopt=#

                LLVM.link!(mod, cmod)
                merge_julia_value_table!(enzyme_ctx, value_table)

                push!(attributes, StringAttribute("enzymejl_forward", fwdmodenm))
                push!(
                    function_attributes(functions(mod)[fwdmodenm]),
                    EnumAttribute("alwaysinline"),
                )
                permit_inlining!(functions(mod)[fwdmodenm])
            end
        end
        thunkTy = ForwardModeThunk{
            Ptr{Cvoid},
            dFT,
            Const{Nothing},
            e_tt,
            width,
            false,
        }  #=returnPrimal=#
        if !(subfunc isa Ptr{Cvoid})
            subfunc = functions(mod)[fwdmodenm]
        end

    elseif mode == API.DEM_ReverseModePrimal || mode == API.DEM_ReverseModeGradient
        if dupClosure
            if !guaranteed_nonactive(funcT, world)
                refed = true
		e_tt = Tuple{width == 1 ? Duplicated{Base.RefValue{funcT}} : BatchDuplicated{Base.RefValue{funcT}, Int(width)},e_tt.parameters...}
                funcT = Core.Typeof(referenceCaller)
                dupClosure = false
                modifiedBetween = (false, modifiedBetween...)
                mi2 = my_methodinstance(mode == API.DEM_ForwardMode ? Forward : Reverse, funcT, Tuple{map(eltype, e_tt.parameters)...}, world)
                @assert mi2 !== nothing
    		dFT = (dupClosure ? (width == 1 ? Duplicated : (BatchDuplicated{T, Int(width)} where T)) : Const){funcT}
            end
        end

        if augfwdnm === nothing || adjointnm === nothing
            etarget = Compiler.EnzymeTarget()
            # TODO modifiedBetween
            eparams = Compiler.EnzymeCompilerParams(
                Tuple{dFT,e_tt.parameters...},
                API.DEM_ReverseModePrimal,
                width,
                Const{Nothing},
                true,
                true,
                modifiedBetween,
                false,
                false,
                UnknownTapeType,
                FFIABI,
                false,
                get_runtime_activity(gutils),
                get_strong_zero(gutils),
            ) #=ErrIfFuncWritten=#
            ejob = Compiler.CompilerJob(
                mi2,
                CompilerConfig(etarget, eparams; kernel = false),
                world,
            )

            cmod, edges, adjointnm, augfwdnm, TapeType, _, value_table = _thunk(ejob, false) #=postopt=#

            LLVM.link!(mod, cmod)
            merge_julia_value_table!(enzyme_ctx, value_table)

            push!(attributes, StringAttribute("enzymejl_augforward", augfwdnm))
            push!(
                function_attributes(functions(mod)[augfwdnm]),
                EnumAttribute("alwaysinline"),
            )
            permit_inlining!(functions(mod)[augfwdnm])

            push!(attributes, StringAttribute("enzymejl_adjoint", adjointnm))
            push!(
                function_attributes(functions(mod)[adjointnm]),
                EnumAttribute("alwaysinline"),
            )
            permit_inlining!(functions(mod)[adjointnm])

            push!(
                attributes,
                StringAttribute(
                    "enzymejl_tapetype",
                    string(convert(UInt, unsafe_to_pointer(TapeType))),
                ),
            )

        end

        if mode == API.DEM_ReverseModePrimal
            thunkTy = AugmentedForwardThunk{
                Ptr{Cvoid},
                dFT,
                Const{Nothing},
                e_tt,
                width,
                true,
                TapeType,
            } #=returnPrimal=#
            subfunc = functions(mod)[augfwdnm]
        else
            thunkTy = AdjointThunk{
                Ptr{Cvoid},
                dFT,
                Const{Nothing},
                e_tt,
                width,
                TapeType,
            }
            subfunc = functions(mod)[adjointnm]
        end
    else
        @assert "Unknown mode"
    end

    ppfuncT = pfuncT
    dpfuncT = width == 1 ? pfuncT : NTuple{Int(width),pfuncT}

    if refed
        dpfuncT = Base.RefValue{dpfuncT}
        pfuncT = Base.RefValue{pfuncT}
    end

    dfuncT = pfuncT
    if pdupClosure
        if width == 1
            dfuncT = Duplicated{dfuncT}
        else
            dfuncT = BatchDuplicated{dfuncT,Int(width)}
        end
    else
        dfuncT = Const{dfuncT}
    end

    vals = LLVM.Value[]

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    ll_th = convert(LLVMType, thunkTy)
    al = alloca!(alloctx, ll_th)
    al = addrspacecast!(B, al, LLVM.PointerType(ll_th, Tracked))
    al = addrspacecast!(B, al, LLVM.PointerType(ll_th, Derived))
    push!(vals, al)
    @assert inline_roots_type(thunkTy) == 0

    copies = Tuple{LLVM.Value, LLVM.Value, LLVM.LLVMType}[]
    if !isghostty(dfuncT)

        llty = convert(LLVMType, dfuncT)

	num_arg_roots = inline_roots_type(llty)
        
	alloctx = LLVM.IRBuilder()
        position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
        
        llty_foralloca = if VERSION >= v"1.12" && num_arg_roots != 0
            strip_tracked_pointers(llty)
        else
            llty
        end

        al = alloca!(alloctx, llty_foralloca)
        al2 = if num_arg_roots != 0
            create_rooted_array(alloctx, num_arg_roots)
        end

        if !isghostty(ppfuncT)
            v = new_from_original(gutils, operands(orig)[1])
            pllty = convert(LLVMType, ppfuncT)
	    
            pv = nothing
                
	    fwdbuilder = if mode == API.DEM_ReverseModeGradient
	       B2 = LLVM.IRBuilder()
	       position!(B2, new_from_original(gutils, orig))
	       B2
	    else
	       B
	    end
	    
            if value_type(v) != pllty
                pv = v
                v = load!(fwdbuilder, pllty, v)
            end
	    
	    if inline_roots_type(ppfuncT) != 0
		v2 = new_from_original(gutils, operands(orig)[2])
		v = recombine_value!(fwdbuilder, v, v2)
	    end

            if mode == API.DEM_ReverseModeGradient
                v = lookup_value(gutils, v, B)
            end

        else
            v = makeInstanceOf(B, ppfuncT)
        end

        if refed
            val0 = val = emit_allocobj!(B, pfuncT)
            val = bitcast!(B, val, LLVM.PointerType(pllty, addrspace(value_type(val))))
            val = addrspacecast!(B, val, LLVM.PointerType(pllty, Derived)) 

	        if !(pv isa Nothing)
                push!(copies, (pv, val, pllty))
            end

            store!(B, v, val)

            if any_jltypes(pllty)
                emit_writebarrier!(B, get_julia_inner_types(B, val0, v))
            end
        else
            val0 = v
        end

        ptr = inbounds_gep!(
            B,
            llty,
            al,
            [LLVM.ConstantInt(LLVM.IntType(64), 0), LLVM.ConstantInt(LLVM.IntType(32), 0)],
        )
	    
        if al2 !== nothing
           extract_roots_from_value!(B, val0, al2)
           T_jlvalue = LLVM.StructType(LLVMType[])
           T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
           al3 = gep!(B, T_prjlvalue, al2, LLVM.Value[ConstantInt(CountTrackedPointers(value_type(val0)).count)])
        end
            
        store!(B, val0, ptr)

        if pdupClosure

            if !isghostty(ppfuncT)
                dv = invert_pointer(gutils, operands(orig)[1], B)
                   
		fwdbuilder = if mode == API.DEM_ReverseModeGradient
		     B2 = LLVM.IRBuilder()
		     position!(B2, new_from_original(gutils, orig))
		     B2
		   else
		     B
		   end
	        
                spllty = LLVM.LLVMType(API.EnzymeGetShadowType(width, pllty))
                pv = nothing
	        
                dv2 = if inline_roots_type(ppfuncT) != 0
                   invert_pointer(gutils, operands(orig)[2], B)
                end

                if value_type(dv) != spllty
                    pv = dv
                    if width == 1
                        dv = load!(fwdbuilder, spllty, dv)
			            if dv2 !== nothing
                           dv = recombine_value!(fwdbuilder, dv, dv2)
                        end
                    else
                        shadowres = UndefValue(spllty)
                        for idx = 1:width
                            arg = extract_value!(fwdbuilder, dv, idx - 1)
                            arg = load!(fwdbuilder, pllty, arg)
                            if dv2 !== nothing
                              arg2 = extract_value!(fwdbuilder, dv2, idx - 1)
                              arg = recombine_value!(fwdbuilder, arg, arg2)
                            end
                            shadowres = insert_value!(fwdbuilder, shadowres, arg, idx - 1)
                        end
                        dv = shadowres
                    end
                end
                
                if mode == API.DEM_ReverseModeGradient
                    dv = lookup_value(gutils, dv, B)
                end
            else
                @assert false
            end

            if refed
                dval0 = dval = emit_allocobj!(B, dpfuncT)
                dval =
                    bitcast!(B, dval, LLVM.PointerType(spllty, addrspace(value_type(dval))))
                dval = addrspacecast!(B, dval, LLVM.PointerType(spllty, Derived))
                store!(B, dv, dval)
                if any_jltypes(spllty)
                    emit_writebarrier!(B, get_julia_inner_types(B, dval0, dv))
                end
                pvl = lookup_value(gutils, pv, B)
                if mode == API.DEM_ReverseModeGradient
                    if width == 1
                       copy_floats_into!(B, spllty, dval, pvl)
                    else
                       for idx = 1:width
                           arg = extract_value!(B, pvl, idx - 1)
                           g0 = inbounds_gep!(B, spllty, dval, LLVM.Value[LLVM.ConstantInt(Int64(0)), LLVM.ConstantInt(Int32(idx-1))])
                           copy_floats_into!(B, pllty, g0, arg)
                        end
                    end
                end
                if pv !== nothing
                    push!(copies, (pv, dval, spllty))
                end
            else
                dval0 = dv
            end

            dptr = inbounds_gep!(
                B,
                llty,
                al,
                [
                    LLVM.ConstantInt(LLVM.IntType(64), 0),
                    LLVM.ConstantInt(LLVM.IntType(32), 1),
                ],
            )
	
	    if al2 !== nothing
	       extract_roots_from_value!(B, dval0, al3)
	    end
            store!(B, dval0, dptr)
        end

        al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

        push!(vals, al)
        
	if num_arg_roots != 0
	  push!(vals, al2)

	end
    end

    if tape !== nothing
        push!(vals, tape)
    end

    push!(vals, new_from_original(gutils, arg_operands_view(orig)[end]))

    return refed, subfunc isa Ptr{Cvoid} ? subfunc : LLVM.name(subfunc), dfuncT, vals, thunkTy, TapeType, copies
end

@register_fwd function threadsfor_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    _, sname, dfuncT, vals, thunkTy, _, _ =
        threadsfor_common(orig, gutils, B, API.DEM_ForwardMode, enzyme_context())

    if sname isa Ptr{Cvoid}
        cal = pfor_erased_call!(B, mod, sname, vals)
    else
        cal = pfor_emitted_call!(B, gutils, mod, sname, vals, thunkTy, dfuncT)
    end
    debug_from_orig!(gutils, cal, orig)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
	API.EnzymeReplaceOriginalToNew(gutils, orig, cal)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end
    return false
end

# Call `runtime_pfor_erased(tramp, env, static)`. `vals` are the arguments
# `threadsfor_common` prepared for `runtime_pfor_fwd`: the thunk buffer, the closure
# buffer and its roots (if the closure is not a ghost), and `static`.
function pfor_erased_call!(B::LLVM.IRBuilder, mod::LLVM.Module, tramp::Ptr{Cvoid}, vals::Vector{LLVM.Value})
    entry = pfor_erased_entry!(mod)::LLVM.Function
    params = parameters(LLVM.function_type(entry))
    off = has_gcstack_arg(entry) ? 1 : 0
    function cast(v, ty)
        vty = value_type(v)
        if ty isa LLVM.IntegerType
            return vty isa LLVM.IntegerType ? v : ptrtoint!(B, v, ty)
        elseif vty isa LLVM.IntegerType
            return inttoptr!(B, v, ty)
        else
            return addrspace(vty) == addrspace(ty) ? v : addrspacecast!(B, v, ty)
        end
    end
    trampv = cast(LLVM.ConstantInt(reinterpret(UInt, tramp)), params[off + 1])
    envty = params[off + 2]
    envv = if length(vals) >= 3
        cast(vals[2], envty)
    else
        envty isa LLVM.IntegerType ? LLVM.ConstantInt(envty, 0) : LLVM.null(envty)
    end
    # `env` is an untracked pointer: keep the roots of the closure alive across the call.
    preserve = LLVM.Value[]
    if length(vals) == 4
        roots = vals[3]
        nroots = convert(Int, operands(roots)[1])
        T_prjlvalue = LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[]), Tracked)
        for i in 1:nroots
            ptr = inbounds_gep!(B, T_prjlvalue, roots, LLVM.Value[LLVM.ConstantInt(Int64(i - 1))])
            push!(preserve, load!(B, T_prjlvalue, ptr))
        end
    end
    token = emit_gc_preserve_begin(B, preserve)
    args = LLVM.Value[trampv, envv, vals[end]]
    if off == 1
        pushfirst!(args, reinsert_gcmarker!(LLVM.parent(LLVM.position(B)), B))
    end
    cal = LLVM.call!(B, LLVM.function_type(entry), entry, args)
    # The ABI is lowered from the call site: it needs the calling convention and the
    # `swiftself` of `pgcstack` too.
    callconv!(cal, callconv(entry))
    if off == 1
        idx = gcstack_arg_index(entry)
        for attr in collect(parameter_attributes(entry, idx))
            attr isa LLVM.EnumAttribute && push!(argument_attributes(cal, idx), attr)
        end
    end
    emit_gc_preserve_end(B, token)
    return cal
end

# Call `runtime_pfor_fwd`, emitted into `mod` for this loop, with the derivative of the
# loop body `sname` linked into `mod`.
function pfor_emitted_call!(B::LLVM.IRBuilder, gutils, mod::LLVM.Module, sname::String, vals::Vector{LLVM.Value}, @nospecialize(thunkTy), @nospecialize(dfuncT))
    tt = Tuple{thunkTy,dfuncT,Bool}
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_fwd, tt)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"))

    pval = functions(mod)[sname]
    if VERSION < v"1.12"
        pval = const_ptrtoint(pval, convert(LLVMType, Ptr{Cvoid}))
    end
    pval = LLVM.ConstantArray(value_type(pval), [pval])
    store!(B, pval, vals[1])

    return LLVM.call!(B, LLVM.function_type(entry), entry, vals)
end

@register_aug function threadsfor_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    byRef, sname, dfuncT, vals, thunkTy, _, copies =
        threadsfor_common(orig, gutils, B, API.DEM_ReverseModePrimal, enzyme_context())

    tt = Tuple{
        thunkTy,
        dfuncT,
        Val{any_jltypes(EnzymeRules.tape_type(thunkTy))},
        Val{byRef},
        Bool,
    }
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_augfwd, tt)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"))

    pval = functions(mod)[sname]
    if VERSION < v"1.12"
       pval = const_ptrtoint(pval, convert(LLVMType, Ptr{Cvoid}))
    end
    pval = LLVM.ConstantArray(value_type(pval), [pval])
    store!(B, pval, vals[1])

    tape = LLVM.call!(B, LLVM.function_type(entry), entry, vals)
    debug_from_orig!(gutils, tape, orig)

    if !any_jltypes(EnzymeRules.tape_type(thunkTy))
        if value_type(tape) != convert(LLVMType, Ptr{Cvoid})
            tape = LLVM.ConstantInt(0)
            GPUCompiler.@safe_warn "Illegal calling convention for threadsfor augfwd"
        end
    end

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        ni = new_from_original(gutils, orig)
	API.EnzymeReplaceOriginalToNew(gutils, orig, tape)
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    unsafe_store!(tapeR, tape.ref)

    return false
end

@register_rev function threadsfor_rev(B, orig, gutils, tape)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return
    end

    byRef, sname, dfuncT, vals, thunkTy, TapeType, copies =
        threadsfor_common(orig, gutils, B, API.DEM_ReverseModeGradient, enzyme_context(), tape)

    STT = if !any_jltypes(TapeType)
        Ptr{TapeType}
    else
        Vector{TapeType}
    end

    tt = Tuple{
        thunkTy,
        dfuncT,
        Val{any_jltypes(EnzymeRules.tape_type(thunkTy))},
        Val{byRef},
        STT,
        Bool,
    }
    mode = get_mode(gutils)
    entry = nested_codegen!(mode, mod, runtime_pfor_rev, tt)
    push!(function_attributes(entry), EnumAttribute("alwaysinline"))

    pval = functions(mod)[sname]
    if VERSION < v"1.12"
	pval = const_ptrtoint(pval, convert(LLVMType, Ptr{Cvoid}))
    end
    pval = LLVM.ConstantArray(value_type(pval), [pval])
    store!(B, pval, vals[1])

    cal = LLVM.call!(B, LLVM.function_type(entry), entry, vals)
    debug_from_orig!(gutils, cal, orig)

    for (pv, val, pllty) in copies
        ld = load!(B, pllty, val)
	pv = lookup_value(gutils, pv, B)
        store!(B, ld, pv)
    end
    return nothing
end

@register_fwd function newtask_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)
    mode = get_mode(gutils)


    vals = LLVM.Value[
        unsafe_to_llvm(B, runtime_newtask_fwd),
        new_from_original(gutils, operands(orig)[1]),
        invert_pointer(gutils, operands(orig)[1], B),
        new_from_original(gutils, operands(orig)[2]),
        (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(
            B,
            new_from_original(gutils, operands(orig)[3]),
        ),
        unsafe_to_llvm(B, Val(get_runtime_activity(gutils))),
        unsafe_to_llvm(B, Val(get_strong_zero(gutils))),
        unsafe_to_llvm(B, Val(width)),
    ]

    ntask = emit_apply_generic!(B, vals)
    debug_from_orig!(gutils, ntask, orig)

    # TODO: GC, ret
    if shadowR != C_NULL
        unsafe_store!(shadowR, ntask.ref)
    end

    if normalR != C_NULL
        unsafe_store!(normalR, ntask.ref)
    end

    return false
end

@register_aug function newtask_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    # fn, dfn = augmentAndGradient(fn)
    # t = jl_new_task(fn)
    # # shadow t
    # dt = jl_new_task(dfn)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    GPUCompiler.@safe_warn "active variables passed by value to jl_new_task are not yet supported"
    width = get_width(gutils)
    mode = get_mode(gutils)

    uncacheable = get_uncacheable(gutils, orig)
    ModifiedBetween = (uncacheable[1] != 0,)


    vals = LLVM.Value[
        unsafe_to_llvm(B, runtime_newtask_augfwd),
        new_from_original(gutils, operands(orig)[1]),
        invert_pointer(gutils, operands(orig)[1], B),
        new_from_original(gutils, operands(orig)[2]),
        (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(
            B,
            new_from_original(gutils, operands(orig)[3]),
        ),
        unsafe_to_llvm(B, Val(get_runtime_activity(gutils))),
        unsafe_to_llvm(B, Val(get_strong_zero(gutils))),
        unsafe_to_llvm(B, Val(width)),
        unsafe_to_llvm(B, Val(ModifiedBetween)),
    ]

    ntask = emit_apply_generic!(B, vals)
    debug_from_orig!(gutils, ntask, orig)
    sret = ntask

    AT = LLVM.ArrayType(T_prjlvalue, 2)
    sret = LLVM.addrspacecast!(B, sret, LLVM.PointerType(T_jlvalue, Derived))
    sret = LLVM.pointercast!(B, sret, LLVM.PointerType(AT, Derived))

    if shadowR != C_NULL
        shadow = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)]),
        )
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

@register_rev function newtask_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function set_task_tid_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, operands(orig)[1])
        return true
    end

    inv = invert_pointer(gutils, operands(orig)[1], B)
    width = get_width(gutils)
    if width == 1
        nops = LLVM.Value[inv, new_from_original(gutils, operands(orig)[2])]
        valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, nops, valTys, false) #=lookup=#
        debug_from_orig!(gutils, cal, orig)
        callconv!(cal, callconv(orig))
    else
        for idx = 1:width
            nops = LLVM.Value[
                extract_value(B, inv, idx - 1),
                new_from_original(gutils, operands(orig)[2]),
            ]
            valTys = API.CValueType[API.VT_Shadow, API.VT_Primal]
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, nops, valTys, false) #=lookup=#

            debug_from_orig!(gutils, cal, orig)
            callconv!(cal, callconv(orig))
        end
    end

    return false
end

@register_aug function set_task_tid_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    set_task_tid_fwd(B, orig, gutils, normalR, shadowR)
end

@register_rev function set_task_tid_rev(B, orig, gutils, tape)
    return nothing
end

@register_fwd function enq_work_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            if width == 1
                shadowres = normal
            else
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end

    return false
end

@register_aug function enq_work_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    enq_work_fwd(B, orig, gutils, normalR, shadowR)
end

function find_match(mod, name)
    for f in functions(mod)
        iter = function_attributes(f)
        elems = Vector{LLVM.API.LLVMAttributeRef}(undef, length(iter))
        LLVM.API.LLVMGetAttributesAtIndex(iter.f, iter.idx, elems)
        for eattr in elems
            at = Attribute(eattr)
            if isa(at, LLVM.StringAttribute)
                if kind(at) == "enzyme_math"
                    if value(at) == name
                        return f
                    end
                end
            end
        end
    end
    return nothing
end

@register_rev function enq_work_rev(B, orig, gutils, tape)
    # jl_wait(shadow(t))
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    waitfn = find_match(mod, "jl_wait")
    if waitfn === nothing
        emit_error(
            B,
            orig,
            "Enzyme: could not find jl_wait fn to create shadow of jl_enq_work",
        )
        return nothing
    end
    @assert waitfn !== nothing
    shadowtask = lookup_value(gutils, invert_pointer(gutils, origops[1], B), B)
    cal = LLVM.call!(B, LLVM.function_type(waitfn), waitfn, [shadowtask])
    debug_from_orig!(gutils, cal, orig)
    callconv!(cal, callconv(orig))
    return nothing
end

@register_fwd function wait_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            if width == 1
                shadowres = normal
            else
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@register_aug function wait_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = get_width(gutils)
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            if width == 1
                shadowres = normal
            else
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@register_rev function wait_rev(B, orig, gutils, tape)
    # jl_enq_work(shadow(t))
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    enq_work_fn = find_match(mod, "jl_enq_work")
    if enq_work_fn === nothing
        emit_error(
            B,
            orig,
            "Enzyme: could not find jl_enq_work fn to create shadow of wait",
        )
        return nothing
    end
    @assert enq_work_fn !== nothing
    shadowtask = lookup_value(gutils, invert_pointer(gutils, origops[1], B), B)
    cal = LLVM.call!(B, LLVM.function_type(enq_work_fn), enq_work_fn, [shadowtask])
    debug_from_orig!(gutils, cal, orig)
    callconv!(cal, callconv(orig))
    return nothing
end
