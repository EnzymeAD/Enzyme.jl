# For julia runtime function emission
    
function emit_allocobj!(
    B::LLVM.IRBuilder,
    @nospecialize(tag::LLVM.Value),
    @nospecialize(Size::LLVM.Value),
    needs_workaround::Bool,
    name::String = "",
)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    T_int8 = LLVM.Int8Type()
    T_pint8 = LLVM.PointerType(T_int8)

    pgcstack = reinsert_gcmarker!(fn, B)
    ct = inbounds_gep!(
        B,
        T_pjlvalue,
        bitcast!(B, pgcstack, T_ppjlvalue),
        [LLVM.ConstantInt(current_task_offset())],
    )

    @static if VERSION < v"1.11.0-"    
        ptls_field = inbounds_gep!(B, T_pjlvalue, ct, [LLVM.ConstantInt(current_ptls_offset())])
        T_ppint8 = LLVM.PointerType(T_pint8)
        ptls = load!(B, T_pint8, bitcast!(B, ptls_field, T_ppint8))
    else
        ct = bitcast!(B, ct, T_pjlvalue)
    end

    if needs_workaround
        T_size_t = convert(LLVM.LLVMType, Int)
        # This doesn't allow for optimizations
        alty = LLVM.FunctionType(T_prjlvalue, [T_pint8, T_size_t, T_prjlvalue])
        alloc_obj, _ = get_function!(mod, "jl_gc_alloc_typed", alty)
        if value_type(Size) != T_size_t # Fix Int32/Int64 issues on 32bit systems
            Size = trunc!(B, Size, T_size_t)
        end
        return call!(B, alty, alloc_obj, [ptls, Size, tag])
    end

    T_size_t = convert(LLVM.LLVMType, Int)

    @static if VERSION < v"1.11.0-"
        alty = LLVM.FunctionType(T_prjlvalue, [T_ppjlvalue, T_size_t, T_prjlvalue])
    else
        alty = LLVM.FunctionType(T_prjlvalue, [T_pjlvalue, T_size_t, T_prjlvalue])
    end

    alloc_obj, _ = get_function!(mod, "julia.gc_alloc_obj", alty)

    return call!(B, alty, alloc_obj, [ct, Size, tag], name)
end
function emit_allocobj!(B::LLVM.IRBuilder, @nospecialize(T::DataType), name::String = "")
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    # Obtain tag
    tag = unsafe_to_llvm(B, T)

    T_size_t = convert(LLVM.LLVMType, UInt)
    Size = LLVM.ConstantInt(T_size_t, sizeof(T))
    emit_allocobj!(B, tag, Size, false, name) #=needs_workaround=#
end

declare_pointerfromobjref!(mod::LLVM.Module) =
    get_function!(mod, "julia.pointer_from_objref") do
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Derived)
        T_pjlvalue = LLVM.PointerType(T_jlvalue)
        LLVM.FunctionType(T_pjlvalue, [T_prjlvalue])
    end

function emit_pointerfromobjref!(B::LLVM.IRBuilder, @nospecialize(T::LLVM.Value))
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, fty = declare_pointerfromobjref!(mod)
    return call!(B, fty, func, [T])
end

declare_writebarrier!(mod) =
    get_function!(mod, "julia.write_barrier") do
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        LLVM.FunctionType(LLVM.VoidType(), [T_prjlvalue]; vararg = true)
    end
declare_apply_generic!(mod::LLVM.Module) =
    get_function!(mod, "ijl_apply_generic") do
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        LLVM.FunctionType(
            T_prjlvalue,
            [T_prjlvalue, LLVM.PointerType(T_prjlvalue), LLVM.Int32Type()],
        )
    end
declare_juliacall!(mod::LLVM.Module) =
    get_function!(mod, "julia.call") do
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg = true)
    end

function emit_jl!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue])
    fn, _ = get_function!(mod, "jl_", FT)
    call!(B, FT, fn, [val])
end

function emit_jl_isa!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value), @nospecialize(ty::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    ity = LLVM.IntType(8*sizeof(Int))
    FT = LLVM.FunctionType(ity, [T_prjlvalue, T_prjlvalue])
    fn, _ = get_function!(mod, "jl_isa", FT)
    call!(B, FT, fn, [val, val])
end

function emit_jl_isa!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value), @nospecialize(ty::Type))::LLVM.Value
    emit_jl_isa!(B, val, unsafe_to_llvm(B, ty))
end

function emit_getfield!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value), @nospecialize(fld::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    gen_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    inv, _ = get_function!(mod, "jl_f_getfield", gen_FT)

    args = [val, fld]

    julia_call, FT = get_function!(
        mod,
        "julia.call",
        LLVM.FunctionType(
            T_prjlvalue,
            [LLVM.PointerType(gen_FT), T_prjlvalue];
            vararg = true,
        ),
    )
    res = call!(B, FT, julia_call, LLVM.Value[inv, args...])
    return res
end


function emit_nthfield!(B::LLVM.IRBuilder, val::LLVM.Value, @nospecialize(fld::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_size_t = convert(LLVM.LLVMType, Int)

    gen_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_size_t])
    inv, _ = get_function!(mod, "jl_get_nth_field_checked", gen_FT)

    args = [val, fld]
    call!(B, gen_FT, inv, args)
end

function emit_nthfield!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value), fld::Integer)::LLVM.Value
	emit_nthfield!(B, val, LLVM.ConstantInt(Int(fld)))
end

function emit_jl_throw!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    T_void = LLVM.VoidType()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, 12)
    FT = LLVM.FunctionType(T_void, [T_prjlvalue])
    fn, _ = get_function!(mod, "jl_throw", FT)
    call!(B, FT, fn, [val])
end

function emit_box_int32!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_int32 = LLVM.Int32Type()

    FT = LLVM.FunctionType(T_prjlvalue, [T_int32])
    box_int32, _ = get_function!(mod, "ijl_box_int32", FT)
    call!(B, FT, box_int32, [val])
end

function emit_box_int64!(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_int64 = LLVM.Int64Type()

    FT = LLVM.FunctionType(T_prjlvalue, [T_int64])
    box_int64, _ = get_function!(mod, "ijl_box_int64", FT)
    call!(B, FT, box_int64, [val])
end

function emit_apply_generic!(B::LLVM.IRBuilder, @nospecialize(args))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    gen_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    inv, _ = get_function!(mod, "ijl_apply_generic", gen_FT)

    # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
    julia_call, FT = get_function!(
        mod,
        "julia.call",
        LLVM.FunctionType(
            T_prjlvalue,
            [LLVM.PointerType(gen_FT), T_prjlvalue];
            vararg = true,
        ),
    )
    res = call!(B, FT, julia_call, LLVM.Value[inv, args...])
    return res
end

function emit_invoke!(B::LLVM.IRBuilder, @nospecialize(args))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    # {} addrspace(10)* ({} addrspace(10)*, {} addrspace(10)**, i32, {} addrspace(10)*)* @ijl_invoke
    gen_FT =
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32, T_prjlvalue])
    inv = get_function!(mod, "ijl_invoke", gen_FT)

    # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
    julia_call, FT = get_function!(
        mod,
        "julia.call2",
        LLVM.FunctionType(
            T_prjlvalue,
            [LLVM.PointerType(generic_FT), T_prjlvalue];
            vararg = true,
        ),
    )
    res = call!(B, FT, julia_call, [inv, args...])
    return res
end

function emit_svec!(B::LLVM.IRBuilder, @nospecialize(args))::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    fn, fty = get_function!(mod, "jl_svec")
    sz = convert(LLVMType, Csize_t)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    LLVM.FunctionType(T_prjlvalue, [sz]; vararg = true)

    sz = convert(LLVMType, Csize_t)
    call!(B, fty, fn, [LLVM.ConstantInt(sz, length(args)), args...])
end


function load_if_mixed(oval::OT, val::VT) where {OT, VT}
    if !(oval isa Base.RefValue) && (val isa Base.RefValue)
        return val[]
    else
        return val
    end
end

function val_from_byref_if_mixed(B::LLVM.IRBuilder, gutils, @nospecialize(oval::LLVM.Value), @nospecialize(val::LLVM.Value))
    world = enzyme_extract_world(LLVM.parent(position(B)))
    legal, TT, _ = abs_typeof(oval)
    if !legal
        legal, TT, _ = abs_typeof(oval, true)
        if legal
            act = active_reg_inner(TT, (), world)
            if act == AnyState
                return val
            end
        end
        return emit_apply_generic!(B, [unsafe_to_llvm(B, load_if_mixed), new_from_original(gutils, oval), val]) 
    end
    act = active_reg_inner(TT, (), world)
    if act == ActiveState || act == MixedState
        legal2, TT2, _ = abs_typeof(val)
        if legal2
	        @assert TT2 <: Base.RefValue
	    else
	    	shadowpointer = false
	    	if isa(val, LLVM.PHIInst)
	    		if size(incoming(val))[1] == 0
	    			shadowpointer = true
	    		end
	    	elseif isa(val, LLVM.ExtractValueInst)
	    		m = operands(val)[1]
		    	if isa(m, LLVM.PHIInst)
		    		if size(incoming(m))[1] == 0
		    			shadowpointer = true
		    		end
		    	end
	    	end
	    	@assert shadowpointer
	    end
        return emit_nthfield!(B, val, 0)
    else
        return val
    end
end

function ref_if_mixed(val::VT) where {VT}
    if active_reg_inner(Core.Typeof(val), (), nothing, Val(true)) == ActiveState
        return Ref(val)
    else
        return val
    end
end

function byref_from_val_if_mixed(B::LLVM.IRBuilder, @nospecialize(val::LLVM.Value))
    world = enzyme_extract_world(LLVM.parent(position(B)))
    legal, TT, _ = abs_typeof(val)
    if !legal
        legal, TT, _ = abs_typeof(val, true)
        act = active_reg_inner(TT, (), world)
        if act == AnyState
            return val
        end
        if !legal
            return emit_apply_generic!(B, [unsafe_to_llvm(B, ref_if_mixed), val]) 
        end
    end
    act = active_reg_inner(TT, (), world)

    if act == ActiveState || act == MixedState
        obj = emit_allocobj!(B, Base.RefValue{TT})
        lty = convert(LLVMType, TT)
        ld = load!(B, lty, bitcast!(B, val, LLVM.PointerType(lty, addrspace(value_type(val)))))
        store!(B, ld, bitcast!(B, obj, LLVM.PointerType(lty, addrspace(value_type(val)))))
        emit_writebarrier!(B, get_julia_inner_types(B, obj, ld))
        return obj
    else
        return val
    end
end

function emit_apply_type!(B::LLVM.IRBuilder, Ty, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    legal = true
    found = []
    for arg in args
        slegal, foundv = absint(arg)
        if slegal
            push!(found, foundv)
        else
            legal = false
            break
        end
    end

    if legal
        return unsafe_to_llvm(B, Ty{found...})
    end

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    f_apply_type, _ = get_function!(mod, "jl_f_apply_type", generic_FT)
    Ty = unsafe_to_llvm(B, Ty)

    # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
    julia_call, FT = get_function!(
        mod,
        "julia.call",
        LLVM.FunctionType(
            T_prjlvalue,
            [LLVM.PointerType(generic_FT), T_prjlvalue];
            vararg = true,
        ),
    )
    tag = call!(
        B,
        FT,
        julia_call,
        LLVM.Value[f_apply_type, LLVM.PointerNull(T_prjlvalue), Ty, args...],
    )
    return tag
end

function emit_tuple!(B, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    legal = true
    found = []
    for arg in args
        slegal, foundv = absint(arg)
        if slegal
            push!(found, foundv)
        else
            legal = false
            break
        end
    end

    if legal
        return unsafe_to_llvm(B, (found...,))
    end

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)
    T_int32 = LLVM.Int32Type()

    generic_FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_pprjlvalue, T_int32])
    f_apply_type, _ = get_function!(mod, "jl_f_tuple", generic_FT)

    # %5 = call nonnull {}* ({}* ({}*, {}**, i32)*, {}*, ...) @julia.call({}* ({}*, {}**, i32)* @jl_f_apply_type, {}* null, {}* inttoptr (i64 139640605802128 to {}*), {}* %4, {}* inttoptr (i64 139640590432896 to {}*))
    julia_call, FT = get_function!(
        mod,
        "julia.call",
        LLVM.FunctionType(
            T_prjlvalue,
            [LLVM.PointerType(generic_FT), T_prjlvalue];
            vararg = true,
        ),
    )
    tag = call!(
        B,
        FT,
        julia_call,
        LLVM.Value[f_apply_type, LLVM.PointerNull(T_prjlvalue), args...],
    )
    return tag
end

function emit_jltypeof!(B::LLVM.IRBuilder, arg::LLVM.Value)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    legal, val, byref = abs_typeof(arg)
    if legal
        return unsafe_to_llvm(B, val)
    end

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg = true)
    fn, _ = get_function!(mod, "jl_typeof", FT)
    call!(B, FT, fn, [arg])
end

function emit_methodinstance!(B::LLVM.IRBuilder, func, args)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    world = enzyme_extract_world(fn)

    sizeT = convert(LLVMType, Csize_t)
    psizeT = LLVM.PointerType(sizeT)

    primalvaltys = LLVM.Value[unsafe_to_llvm(B, Core.Typeof(func))]
    for a in args
        push!(primalvaltys, emit_jltypeof!(B, a))
    end

    meth = only(methods(func))
    tag = emit_apply_type!(B, Tuple, primalvaltys)

    #    TT = meth.sig
    #    while TT isa UnionAll
    #        TT = TT.body
    #    end
    #    parms = TT.parameters
    #
    #    tosv = primalvaltys
    #    if length(parms) > 0 && typeof(parms[end]) == Core.TypeofVararg
    #        tosv = LLVM.Value[tosv[1:length(parms)-1]..., emit_apply_type!(B, Tuple, tosv[length(parms):end])]
    #    end
    #    sv = emit_svec!(B, tosv[2:end])
    #

    meth = unsafe_to_llvm(B, meth)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    worlds, FT = get_function!(
        mod,
        "jl_gf_invoke_lookup_worlds",
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, sizeT, psizeT, psizeT]),
    )
    EB = LLVM.IRBuilder()
    position!(EB, first(LLVM.instructions(LLVM.entry(fn))))
    minworld = alloca!(EB, sizeT)
    maxworld = alloca!(EB, sizeT)
    store!(B, LLVM.ConstantInt(sizeT, 0), minworld)
    store!(B, LLVM.ConstantInt(sizeT, -1), maxworld)
    methodmatch = call!(
        B,
        FT,
        worlds,
        LLVM.Value[
            tag,
            unsafe_to_llvm(B, nothing),
            LLVM.ConstantInt(sizeT, world),
            minworld,
            maxworld,
        ],
    )
    # emit_jl!(B, methodmatch)
    # emit_jl!(B, emit_jltypeof!(B, methodmatch))
    offset = 1
    AT = LLVM.ArrayType(T_prjlvalue, offset + 1)
    methodmatch = addrspacecast!(B, methodmatch, LLVM.PointerType(T_jlvalue, Derived))
    methodmatch = bitcast!(B, methodmatch, LLVM.PointerType(AT, Derived))
    gep = LLVM.inbounds_gep!(
        B,
        AT,
        methodmatch,
        LLVM.Value[LLVM.ConstantInt(0), LLVM.ConstantInt(offset)],
    )
    sv = LLVM.load!(B, T_prjlvalue, gep)

    fn, FT = get_function!(
        mod,
        "jl_specializations_get_linfo",
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue, T_prjlvalue, T_prjlvalue]),
    )

    mi = call!(B, FT, fn, [meth, tag, sv])

    return mi
end

function emit_writebarrier!(B, T)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, FT = declare_writebarrier!(mod)
    return call!(B, FT, func, T)
end


function get_array_struct()
    @static if VERSION < v"1.11-"
        # JL_EXTENSION typedef struct {
        #     JL_DATA_TYPE
        #     void *data;
        # #ifdef STORE_ARRAY_LEN (just true new newer versions)
        # 	size_t length;
        # #endif
        #     jl_array_flags_t flags;
        #     uint16_t elsize;  // element size including alignment (dim 1 memory stride)
        #     uint32_t offset;  // for 1-d only. does not need to get big.
        #     size_t nrows;
        #     union {
        #         // 1d
        #         size_t maxsize;
        #         // Nd
        #         size_t ncols;
        #     };
        #     // other dim sizes go here for ndims > 2
        #
        #     // followed by alignment padding and inline data, or owner pointer
        # } jl_array_t;

        i8 = LLVM.IntType(8)
        ptrty = LLVM.PointerType(i8, 13)
        sizeT = LLVM.IntType(8 * sizeof(Csize_t))
        arrayFlags = LLVM.IntType(16)
        elsz = LLVM.IntType(16)
        off = LLVM.IntType(32)
        nrows = LLVM.IntType(8 * sizeof(Csize_t))

        return LLVM.StructType([ptrty, sizeT, arrayFlags, elsz, off, nrows]; packed = true)
    else
        # JL_EXTENSION typedef struct {
        #     JL_DATA_TYPE
        #     size_t length;
        #     void *ptr;
        #     // followed by padding and inline data, or owner pointer
        # #ifdef _P64
        #     // union {
        #     //     jl_value_t *owner;
        #     //     T inl[];
        #     // };
        # #else
        #     //
        #     // jl_value_t *owner;
        #     // size_t padding[1];
        #     // T inl[];
        # #endif
        # } jl_genericmemory_t;
        # 
        # JL_EXTENSION typedef struct {
        #     JL_DATA_TYPE
        #     void *ptr_or_offset;
        #     jl_genericmemory_t *mem;
        # } jl_genericmemoryref_t;
        # 
        # JL_EXTENSION typedef struct {
        #     JL_DATA_TYPE
        #     jl_genericmemoryref_t ref;
        #     size_t dimsize[]; // length for 1-D, otherwise length is mem->length
        # } jl_array_t;
        i8 = LLVM.IntType(8)
        ptrty = LLVM.PointerType(i8, 10)
        sizeT = LLVM.IntType(8 * sizeof(Csize_t))
        return LLVM.StructType([ptrty, sizeT]; packed = true)
    end
end

function get_memory_struct()
	# JL_EXTENSION typedef struct {
	#     JL_DATA_TYPE
	#     size_t length;
	#     void *ptr;
	#     // followed by padding and inline data, or owner pointer
	# #ifdef _P64
	#     // union {
	#     //     jl_value_t *owner;
	#     //     T inl[];
	#     // };
	# #else
	#     //
	#     // jl_value_t *owner;
	#     // size_t padding[1];
	#     // T inl[];
	# #endif
	# } jl_genericmemory_t;

	i8 = LLVM.IntType(8)
	ptrty = LLVM.PointerType(i8)
	sizeT = LLVM.IntType(8 * sizeof(Csize_t))

	return LLVM.StructType([sizeT, ptrty]; packed = true)
end

function get_memory_data(B, array)
    mty = get_memory_struct()
    array = LLVM.pointercast!(
        B,
        array,
        LLVM.PointerType(mty, LLVM.addrspace(LLVM.value_type(array))),
    )
    v = inbounds_gep!(
        B,
        mty,
        array,
        LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(1))],
    )
	i8 = LLVM.IntType(8)
	ptrty = LLVM.PointerType(i8)
    return LLVM.load!(B, ptrty, v)
end

function get_layout_struct()
    # typedef struct {
    #     uint32_t size;
    #     uint32_t nfields;
    #     uint32_t npointers; // number of pointers embedded inside
    #     int32_t first_ptr; // index of the first pointer (or -1)
    #     uint16_t alignment; // strictest alignment over all fields
    #     struct { // combine these fields into a struct so that we can take addressof them
    #         uint16_t haspadding : 1; // has internal undefined bytes
    #         uint16_t fielddesc_type : 2; // 0 -> 8, 1 -> 16, 2 -> 32, 3 -> foreign type
    #         // metadata bit only for GenericMemory eltype layout
    #         uint16_t arrayelem_isboxed : 1;
    #         uint16_t arrayelem_isunion : 1;
    #         // If set, this type's egality can be determined entirely by comparing
    #         // the non-padding bits of this datatype.
    #         uint16_t isbitsegal : 1;
    #         uint16_t padding : 10;
    #     } flags;
    #     // union {
    #     //     jl_fielddesc8_t field8[nfields];
    #     //     jl_fielddesc16_t field16[nfields];
    #     //     jl_fielddesc32_t field32[nfields];
    #     // };
    #     // union { // offsets relative to data start in words
    #     //     uint8_t ptr8[npointers];
    #     //     uint16_t ptr16[npointers];
    #     //     uint32_t ptr32[npointers];
    #     // };
    # } jl_datatype_layout_t;
	i32 = LLVM.IntType(32)
	i16 = LLVM.IntType(16)
	return LLVM.StructType([i32, i32, i32, i32, i16, i16]; packed = true)
end

function get_datatype_struct()
    # typedef struct _jl_datatype_t {
    #     JL_DATA_TYPE
    #     jl_typename_t *name;
    #     struct _jl_datatype_t *super;
    #     jl_svec_t *parameters;
    #     jl_svec_t *types;
    #     jl_value_t *instance;  // for singletons
    #     const jl_datatype_layout_t *layout;
    #     // memoized properties (set on construction)
    #     uint32_t hash;
    #     uint16_t hasfreetypevars:1; // majority part of isconcrete computation
    #     uint16_t isconcretetype:1; // whether this type can have instances
    #     uint16_t isdispatchtuple:1; // aka isleaftupletype
    #     uint16_t isbitstype:1; // relevant query for C-api and type-parameters
    #     uint16_t zeroinit:1; // if one or more fields requires zero-initialization
    #     uint16_t has_concrete_subtype:1; // If clear, no value will have this datatype
    #     uint16_t maybe_subtype_of_cache:1; // Computational bit for has_concrete_supertype. See description in jltypes.c.
    #     uint16_t isprimitivetype:1; // whether this is declared with 'primitive type' keyword (sized, no fields, and immutable)
    #     uint16_t ismutationfree:1; // whether any mutable memory is reachable through this type (in the type or via fields)
    #     uint16_t isidentityfree:1; // whether this type or any object reachable through its fields has non-content-based identity
    #     uint16_t smalltag:6; // whether this type has a small-tag optimization
    # } jl_datatype_t;
	jlvaluet = LLVM.PointerType(LLVM.StructType(LLVMType[]), 10)
	i32 = LLVM.IntType(32)
	i16 = LLVM.IntType(16)
	return LLVM.StructType([jlvaluet, jlvaluet, jlvaluet, jlvaluet, jlvaluet, jlvaluet, i32, i16]; packed = true)
end

function get_array_data(B, array)
    i8 = LLVM.IntType(8)
    ptrty = LLVM.PointerType(i8, 13)
    array = LLVM.pointercast!(
        B,
        array,
        LLVM.PointerType(ptrty, LLVM.addrspace(LLVM.value_type(array))),
    )
    return LLVM.load!(B, ptrty, array)
end

function get_array_elsz(B, array)
    ST = get_array_struct()
    elsz = LLVM.IntType(16)
    array = LLVM.pointercast!(
        B,
        array,
        LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))),
    )
    v = inbounds_gep!(
        B,
        ST,
        array,
        LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(3))],
    )
    return LLVM.load!(B, elsz, v)
end

function emit_layout_of_type!(B, ty)
	legal, JTy = absint(ty)
	ls = get_layout_struct()
	lptr = LLVM.PointerType(ls, 10)
	if legal
		return LLVM.const_inttoptr(LLVM.ConstantInt(Base.reinterpret(UInt, JTy.layout)), lptr)
	end
	@assert !isa(ty, LLVM.ConstantExpr)
	@assert !isa(ty, LLVM.Constant)
	dt = get_datatype_struct()
	lty = bitcast!(B, ty, LLVM.PointerType(dt, addrspace(value_type(ty))))
	layoutp = inbounds_gep!(B, dt, lty, 
        LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(5))],
	)
	jlvaluet = LLVM.PointerType(LLVM.StructType(LLVMType[]), 10)
	layout = load!(B, jlvaluet, layoutp)
    layout = bitcast!(B, layout, lptr)
	return layout
end

function emit_memorytype_elsz!(B, ty)
	legal, JTy = absint(ty)
	if legal
		res = unsafe_load(reinterpret(Ptr{UInt32}, JTy.layout))
		return LLVM.ConstantInt(res)
	end
	ty = emit_layout_of_type!(B, ty)
	@assert !isa(ty, LLVM.ConstantExpr)
	@assert !isa(ty, LLVM.Constant)
	i32 = LLVM.IntType(32)
	lty = bitcast!(B, ty, LLVM.PointerType(i32, addrspace(value_type(ty))))
	return load!(B, i32, lty)
end

function get_memory_elsz(B, array)
    ty = emit_jltypeof!(B, array)
	return emit_memorytype_elsz!(B, ty)
end

function get_array_len(B, array)
    if isa(array, LLVM.CallInst)
        fn = LLVM.called_operand(array)
        nm = ""
        if isa(fn, LLVM.Function)
            nm = LLVM.name(fn)
        end

        for (fname, num) in (
            ("jl_alloc_array_1d", 1),
            ("ijl_alloc_array_1d", 1),
            ("jl_alloc_array_2d", 2),
            ("jl_alloc_array_2d", 2),
            ("jl_alloc_array_2d", 3),
            ("jl_alloc_array_2d", 3),
        )
            if nm == fname
                res = operands(array)[2]
                for i = 2:num
                    res = mul!(B, res, operands(array)[1+i])
                end
                return res
            end
        end
    end
    ST = get_array_struct()
    array = LLVM.pointercast!(
        B,
        array,
        LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))),
    )
    v = inbounds_gep!(
        B,
        ST,
        array,
        LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(1))],
    )
    sizeT = LLVM.IntType(8 * sizeof(Csize_t))
    return LLVM.load!(B, sizeT, v)
end

function get_memory_len(B, array)
    if isa(array, LLVM.CallInst)
        fn = LLVM.called_operand(array)
        nm = ""
        if isa(fn, LLVM.Function)
            nm = LLVM.name(fn)
        end

        for (fname, num) in (
            ("jl_alloc_genericmemory", 1),
            ("ijl_alloc_genericmemory", 1),
        )
            if nm == fname
                res = operands(array)[2]
                for i = 2:num
                    res = mul!(B, res, operands(array)[1+i])
                end
                return res
            end
        end
    end
    ST = get_memory_struct()
    array = LLVM.pointercast!(
        B,
        array,
        LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))),
    )
    v = inbounds_gep!(
        B,
        ST,
        array,
        LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(0))],
    )
    sizeT = LLVM.IntType(8 * sizeof(Csize_t))
    return LLVM.load!(B, sizeT, v)
end

function get_array_nrows(B, array)
    ST = get_array_struct()
    array = LLVM.pointercast!(
        B,
        array,
        LLVM.PointerType(ST, LLVM.addrspace(LLVM.value_type(array))),
    )
    v = inbounds_gep!(
        B,
        ST,
        array,
        LLVM.Value[LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32(5))],
    )
    nrows = LLVM.IntType(8 * sizeof(Csize_t))
    return LLVM.load!(B, nrows, v)
end

function emit_gc_preserve_begin(B::LLVM.IRBuilder, args = LLVM.Value[])
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, FT = get_function!(
        mod,
        "llvm.julia.gc_preserve_begin",
        LLVM.FunctionType(LLVM.TokenType(), vararg = true),
    )

    token = call!(B, FT, func, args)
    return token
end

function emit_gc_preserve_end(B::LLVM.IRBuilder, token)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    func, FT = get_function!(
        mod,
        "llvm.julia.gc_preserve_end",
        LLVM.FunctionType(LLVM.VoidType(), [LLVM.TokenType()]),
    )

    call!(B, FT, func, [token])
    return
end

function allocate_sret!(B::LLVM.IRBuilder, N)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    al = LLVM.alloca!(B, LLVM.ArrayType(T_prjlvalue, N))
    return al
end

function allocate_sret!(gutils::API.EnzymeGradientUtilsRef, N)
    B = LLVM.IRBuilder()
    position!(B, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    allocate_sret!(B, N)
end

function emit_error(B::LLVM.IRBuilder, orig, string, errty = EnzymeRuntimeException)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    if !isa(string, LLVM.Value)
        string = globalstring_ptr!(B, string, "enz_exception")
    end

    ct = if occursin("ptx", LLVM.triple(mod)) || occursin("amdgcn", LLVM.triple(mod))

        vt = LLVM.VoidType()
        ptr = convert(LLVMType, Ptr{Cvoid})

        exc, _ =
            get_function!(mod, "gpu_report_exception", LLVM.FunctionType(vt, [ptr]))

        string = ptrtoint!(B, string, ptr)

        call!(B, LLVM.function_type(exc), exc, [string])

        framefn, ft = get_function!(
            mod,
            "gpu_report_exception_frame",
            LLVM.FunctionType(vt, [LLVM.Int32Type(), ptr, ptr, LLVM.Int32Type()]),
        )

        if orig !== nothing
            bt = GPUCompiler.backtrace(orig)
            for (i, frame) in enumerate(bt)
                idx = ConstantInt(parameters(ft)[1], i)
                func = globalstring_ptr!(B, String(frame.func), "di_func")
                func = ptrtoint!(B, func, ptr)
                file = globalstring_ptr!(B, String(frame.file), "di_file")
                file = ptrtoint!(B, file, ptr)
                line = ConstantInt(parameters(ft)[4], frame.line)
                call!(B, ft, framefn, [idx, func, file, line])
            end
        end

        sigfn, sigft = get_function!(
            mod,
            "gpu_signal_exception",
            LLVM.FunctionType(vt, LLVM.LLVMType[]),
        )
        call!(B, sigft, sigfn)
        trap_ft = LLVM.FunctionType(LLVM.VoidType())
        trap = if haskey(functions(mod), "llvm.trap")
            functions(mod)["llvm.trap"]
        else
            LLVM.Function(mod, "llvm.trap", trap_ft)
        end
        call!(B, trap_ft, trap)
    else
        err = emit_allocobj!(B, errty)
        err2 = bitcast!(B, err, LLVM.PointerType(LLVM.PointerType(LLVM.Int8Type()), 10))
        store!(B, string, err2)
        emit_jl_throw!(
            B,
            addrspacecast!(B, err, LLVM.PointerType(LLVM.StructType(LLVMType[]), 12)),
        )
    end

    # 2. Call error function and insert unreachable
    LLVM.API.LLVMAddCallSiteAttribute(
        ct,
        reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
        EnumAttribute("noreturn"),
    )
    if EnzymeMutabilityException != errty
        LLVM.API.LLVMAddCallSiteAttribute(
            ct,
            reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
            StringAttribute("enzyme_error"),
        )
    end
    return ct
end

