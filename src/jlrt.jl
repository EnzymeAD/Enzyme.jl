# For julia runtime function emission

declare_allocobj!(mod) =
    get_function!(mod, "julia.gc_alloc_obj") do
        T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_ppjlvalue = LLVM.PointerType(LLVM.PointerType(T_jlvalue))
        T_size_t = convert(LLVM.LLVMType, Int)


        LLVM.FunctionType(T_prjlvalue, [T_ppjlvalue, T_size_t, T_prjlvalue])
    end
function emit_allocobj!(
    B::LLVM.IRBuilder,
    tag::LLVM.Value,
    Size::LLVM.Value,
    needs_workaround::Bool,
    name::String = "",
)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)

    T_int8 = LLVM.Int8Type()
    T_pint8 = LLVM.PointerType(T_int8)

    pgcstack = reinsert_gcmarker!(fn, B)
    ct = inbounds_gep!(
        B,
        T_pjlvalue,
        bitcast!(B, pgcstack, T_ppjlvalue),
        [LLVM.ConstantInt(current_task_offset())],
    )
    ptls_field = inbounds_gep!(B, T_pjlvalue, ct, [LLVM.ConstantInt(current_ptls_offset())])
    T_ppint8 = LLVM.PointerType(T_pint8)
    ptls = load!(B, T_pint8, bitcast!(B, ptls_field, T_ppint8))

    if needs_workaround
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_size_t = convert(LLVM.LLVMType, Int)
        # This doesn't allow for optimizations
        alty = LLVM.FunctionType(T_prjlvalue, [T_pint8, T_size_t, T_prjlvalue])
        alloc_obj, _ = get_function!(mod, "jl_gc_alloc_typed", alty)
        if value_type(Size) != T_size_t # Fix Int32/Int64 issues on 32bit systems
            Size = trunc!(B, Size, T_size_t)
        end
        return call!(B, alty, alloc_obj, [ptls, Size, tag])
    end


    alloc_obj, alty = declare_allocobj!(mod)

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

function emit_pointerfromobjref!(B::LLVM.IRBuilder, T::LLVM.Value)
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
declare_apply_generic!(mod) =
    get_function!(mod, "ijl_apply_generic") do
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        LLVM.FunctionType(
            T_prjlvalue,
            [T_prjlvalue, LLVM.PointerType(T_prjlvalue), LLVM.Int32Type()],
        )
    end
declare_juliacall!(mod) =
    get_function!(mod, "julia.call") do
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        LLVM.FunctionType(T_prjlvalue, [T_prjlvalue]; vararg = true)
    end

function emit_jl!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    FT = LLVM.FunctionType(T_prjlvalue, [T_prjlvalue])
    fn, _ = get_function!(mod, "jl_", FT)
    call!(B, FT, fn, [val])
end

function emit_getfield!(B::LLVM.IRBuilder, val::LLVM.Value, fld::LLVM.Value)::LLVM.Value
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


function emit_nthfield!(B::LLVM.IRBuilder, val::LLVM.Value, fld::LLVM.Value)::LLVM.Value
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

function emit_nthfield!(B::LLVM.IRBuilder, val::LLVM.Value, fld::Integer)::LLVM.Value
	emit_nthfield!(B, val, unsafe_to_llvm(B, Int(fld)))
end

function emit_jl_throw!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
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

function emit_box_int32!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
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

function emit_box_int64!(B::LLVM.IRBuilder, val::LLVM.Value)::LLVM.Value
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

function emit_apply_generic!(B::LLVM.IRBuilder, args)::LLVM.Value
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

function emit_invoke!(B::LLVM.IRBuilder, args)::LLVM.Value
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

function emit_svec!(B::LLVM.IRBuilder, args)::LLVM.Value
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


function val_from_byref_if_mixed(B::LLVM.IRBuilder, oval::LLVM.Value, val::LLVM.Value)
    legal, TT, _ = abs_typeof(oval)
    @assert legal
    world = enzyme_extract_world(LLVM.parent(position(IRBuilder(B))))
    act = active_reg_inner(TT, (), world)

    if act == ActiveState || act == MixedState
        legal2, TT2, _ = abs_typeof(val)
        @assert legal2
        @assert TT2 <: Base.RefValue
        return emit_nthfield!(B, val, 0)
    else
        return val
    end
end

function byref_from_val_if_mixed(B::LLVM.IRBuilder, val::LLVM.Value)
    legal, TT, _ = abs_typeof(oval)
    @assert legal
    world = enzyme_extract_world(LLVM.parent(position(IRBuilder(B))))
    act = active_reg_inner(TT, (), world)

    if act == ActiveState || act == MixedState
        obj = emit_allocobj!(B, Base.RefValue{TT})
        lty = convert(LLVMType, TT)
        ld = load!(lty, bitcast!(B, val, LLVM.PointerType(lty, addrspace(value_type(val)))))
        store!(B, ld, bitcast!(B, obj, LLVM.PointerType(lty, addrspace(value_type(val)))))
        emit_writebarrier!(B, get_julia_inner_types(B, obj, ld))
        return obj
    else
        return val
    end
end