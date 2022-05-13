module InternalRules

using LLVM
using GPUCompiler
using ..Compiler

import ..Compiler: AdjointThunk, API, allocatedinline, nested_codegen

struct Tape
    thunk::AdjointThunk
    internal_tape::Ptr{Cvoid}
    shadow_return::Any
    resT::DataType
end

declare_allocobj!(mod) = get_function!(mod, "julia.gc_alloc_obj") do mod, ctx, name
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    
	T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)

	#TODO make size_t > 32 => 64, else 32
	T_size = LLVM.IntType((sizeof(Csize_t)*8) > 32 ? 64 : 32; ctx)
    funcT = LLVM.FunctionType(T_prjlvalue, [LLVM.PointerType(T_ppjlvalue), T_size, T_prjlvalue])
    LLVM.Function(mod, name, funcT)
end
function emit_allocobj!(B, T, size)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
	ctx = context(mod)
    func = declare_allocobj!(mod)
	
	T_size = parameters(eltype(llvmtype(func)))[2]
    
	T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

	ty = inttoptr!(B, LLVM.ConstantInt(convert(Int, pointer_from_objref(T)); ctx), LLVM.PointerType(T_jlvalue))
	ty = addrspacecast!(B, ty, T_prjlvalue)
	size = LLVM.ConstantInt(T_size, size)
    args = [reinsert_gcmarker!(fn), size, ty]
	args[1] = bitcast!(B, args[1], parameters(eltype(llvmtype(func)))[1])
    return call!(B, func, args)
end
function emit_allocobj!(B, T)
    emit_allocobj!(B, T, sizeof(T))
end
declare_pointerfromobjref!(mod) = get_function!(mod, "julia.pointer_from_objref") do mod, ctx, name
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 11) 
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    funcT = LLVM.FunctionType(T_pjlvalue, [T_prjlvalue])
    LLVM.Function(mod, name, funcT)
end
function emit_pointerfromobjref!(B, T)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)
    func = declare_pointerfromobjref!(mod)
    return call!(B, func, [T])
end


function array_inner(::Type{<:Array{T}}) where T
    return T
end
function array_shadow_handler(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, numArgs::Csize_t, Args::Ptr{LLVM.API.LLVMValueRef})::LLVM.API.LLVMValueRef
    inst = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
    ctx = LLVM.context(LLVM.Value(OrigCI))

    ce = operands(inst)[1]
    while isa(ce, ConstantExpr)
        ce = operands(ce)[1]
    end
    ptr = reinterpret(Ptr{Cvoid}, convert(UInt64, ce))
    typ = array_inner(Base.unsafe_pointer_to_objref(ptr))

    b = LLVM.Builder(B)

    vals = LLVM.Value[]
    for i = 1:numArgs
        push!(vals, LLVM.Value(unsafe_load(Args, i)))
    end

    anti = LLVM.call!(b, LLVM.Value(LLVM.API.LLVMGetCalledValue(OrigCI)), vals)

    prod = LLVM.Value(unsafe_load(Args, 2))
    for i = 3:numArgs
        prod = LLVM.mul!(b, prod, LLVM.Value(unsafe_load(Args, i)))
    end

    isunboxed = allocatedinline(typ)
    elsz = sizeof(typ)

    isunion = typ <: Union

    LLT_ALIGN(x, sz) = (((x) + (sz)-1) & ~((sz)-1))

    if !isunboxed
        elsz = sizeof(Ptr{Cvoid})
        al = elsz;
    else
        al = 1 # check
        elsz = LLT_ALIGN(elsz, al)
    end

    tot = prod
    tot = LLVM.mul!(b, tot, LLVM.ConstantInt(LLVM.llvmtype(tot), elsz, false))

    if elsz == 1 && !isunion
        tot = LLVM.add!(b, t, LLVM.ConstantInt(LLVM.llvmtype(tot), 1, false))
    end
    if (isunion)
        # an extra byte for each isbits union array element, stored after a->maxsize
        tot = LLVM.add!(b, tot, prod)
    end

    i1 = LLVM.IntType(1; ctx)
    i8 = LLVM.IntType(8; ctx)
    ptrty = LLVM.PointerType(i8) #, LLVM.addrspace(LLVM.llvmtype(anti)))
    toset = LLVM.load!(b, LLVM.pointercast!(b, anti, LLVM.PointerType(ptrty, LLVM.addrspace(LLVM.llvmtype(anti)))))

    memtys = LLVM.LLVMType[ptrty, LLVM.llvmtype(tot)]
    memset = LLVM.Function(mod, LLVM.Intrinsic("llvm.memset"), memtys)
    memargs = LLVM.Value[toset, LLVM.ConstantInt(i8, 0, false), tot, LLVM.ConstantInt(i1, 0, false)]

    mcall = LLVM.call!(b, memset, memargs)
    ref::LLVM.API.LLVMValueRef = Base.unsafe_convert(LLVM.API.LLVMValueRef, anti)
    return ref
end

function null_free_handler(B::LLVM.API.LLVMBuilderRef, ToFree::LLVM.API.LLVMValueRef, Fn::LLVM.API.LLVMValueRef)::LLVM.API.LLVMValueRef
    return C_NULL
end

dedupargs() = ()
dedupargs(a, da, args...) = (a, dedupargs(args...)...)

# Force sret
struct Return2
    ret1::Any
    ret2::Any
end

struct Return3
    ret1::Any
    ret2::Any
    ret3::Any
end

function runtime_newtask_fwd(fn::Any, dfn::Any, post::Any, ssize::Int, width)
    tt′ = Tuple{}
    args = ()
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    forward = thunk(fn, dfn, Const, tt′, Val(API.DEM_ForwardMode), width)

    taperef = Ref{Ptr{Cvoid}}(C_NULL)

    function fclosure()
        res = forward()
        if length(res) > 1
            return res[1]
        else
            return nothing
        end
    end

    return ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)
end

function runtime_newtask_augfwd(fn::Any, dfn::Any, post::Any, ssize::Int, width)

    tt′ = Tuple{}
    args = ()
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    forward, adjoint = thunk(fn, dfn, Const, tt′, Val(API.DEM_ReverseModePrimal), width)

    taperef = Ref{Ptr{Cvoid}}(C_NULL)

    function fclosure()
        res = forward()
        taperef[] = res[1]
        if length(res) > 1
            return res[2]
        else
            return nothing
        end
    end

    ftask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), fclosure, post, ssize)

    function rclosure()
        adjoint(taperef[])
        return 0
    end
    
    rtask = ccall(:jl_new_task, Ref{Task}, (Any, Any, Int), rclosure, post, ssize)
    
    return Return2(ftask, rtask)
end

function runtime_generic_fwd(fn::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32,
                             width)
    if fn == Base.println || fn == Base.print || fn == Base.show || fn == Base.flush
        args = Any[]
        for i in 1:arg_size
            push!(args, Base.unsafe_load(arg_ptr, i))
        end

        res = fn(args...)
        return Return2(res, res)
    end

    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)
    args = Any[]
    
    for i in 1:arg_size
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            s = Base.unsafe_load(shadow_ptr, i)
            push!(args, Duplicated(p, s))
        else
            push!(args, Const(p))
        end
    end

    # TODO: Annotation of return value
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    if rt == Union{}
        annotation = Duplicated
    else
        annotation = guess_activity(rt, API.DEM_ForwardMode)
        if annotation <: DuplicatedNoNeed
            annotation = Duplicated
        end
    end

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ForwardMode), width)

    res = forward(args...)
    if annotation == Duplicated
        return Return2(res[1], res[2])
    else
        return Return2(nothing, nothing)
    end
end



function runtime_generic_augfwd(fn::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32,
                                width)

    if fn == Base.println || fn == Base.print || fn == Base.show || fn == Base.flush

        args = Any[]
        for i in 1:arg_size
            push!(args, Base.unsafe_load(arg_ptr, i))
        end

        res = fn(args...)
        return Return3(res, res, nothing)
    end

    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)
    args = Any[]
    
    for i in 1:arg_size
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            if typeof(p) <: AbstractFloat || typeof(p) <: Complex{<:AbstractFloat}
                push!(args, Active(p))
            else
                s = Base.unsafe_load(shadow_ptr, i)
                push!(args, Duplicated(p, s))
            end
        else
            push!(args, Const(p))
        end
    end

    # TODO: Annotation of return value
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt)

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward, adjoint = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ReverseModePrimal), width)

    res = forward(args...)
    if length(res) > 1
        origRet = res[2]
        resT = typeof(origRet)
    else
        origRet = nothing
        resT = Nothing
    end

    if annotation <: Active
        # This assumes that we have an Immutable object here that got passed as a boxed value
        shadow_return = Ref(zero(resT))
        ret2 = shadow_return
    elseif annotation <: Const
        shadow_return = nothing
        ret2 = origRet
    elseif annotation <: Duplicated ||  annotation <: DuplicatedNoNeed
        shadow_return = nothing
        ret2 = res[3]
    else
        error("Unknown annotation")
    end

    internal_tape = res[1]
    tape = Tape(adjoint, internal_tape, shadow_return, resT)

    return Return3(origRet, ret2, tape)
end

function runtime_generic_rev(fn::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32, tape::Any, width)
    if fn == Base.println || fn == Base.print || fn == Base.show || fn == Base.flush
        args = Any[]
        for i in 1:arg_size
            push!(args, Base.unsafe_load(arg_ptr, i))
        end

        res = fn(args...)
        return nothing
    end

    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)
    args = []
    actives = []
    for i in 1:arg_size
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            if typeof(p) <: AbstractFloat || typeof(p) <: Complex{<:AbstractFloat}
                push!(args, Active(p))
                push!(actives, (shadow_ptr, i))
            else
                s = Base.unsafe_load(shadow_ptr, i)
                push!(args, Duplicated(p, s))
            end
        else
            push!(args, Const(p))
        end
    end


    tape = tape::Tape

    if tape.shadow_return !== nothing
        val = tape.shadow_return
        if !(val isa tape.resT)
            val = tape.shadow_return[]
        end
        push!(args, val)
    end
    if tape.internal_tape !== nothing
        push!(args, tape.internal_tape)
    end

    tup = tape.thunk(args...)

    for (d, (s, i)) in zip(tup, actives)
        a = unsafe_load(s, i)
        # While `RefValue{T}` and boxed T for immutable are bitwise compatible
        # they are not idempotent on the Julia level. We could "force" `a` to be
        # a boxed T, but would lose the mutable memory semantics that Enzyme requires
        if a isa Base.RefValue
            @assert eltype(a) == typeof(d)
            a[] += d
        else
            ref = unsafe_load(reinterpret(Ptr{Ptr{typeof(a)}}, s), i)
            unsafe_store!(ref, d+a)
        end
    end
    return nothing
end

function runtime_invoke_fwd(mi::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32,
                            width)
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)

    fn = Base.unsafe_load(arg_ptr, 1)
    args = []
    for i in 2:arg_size
        val = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            # TODO use only for non mutable
            push!(args, Duplicated(val, Base.unsafe_load(shadow_ptr, i)))
        else
            push!(args, Const(val))
        end
    end

    specTypes = mi.specTypes.parameters
    F = specTypes[1]
    @assert F == typeof(fn)
    @assert in(mi.def, methods(fn))
    
    tt = Tuple{specTypes[2:end]...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt, API.DEM_ForwardMode)
    if annotation <: DuplicatedNoNeed
        annotation = Duplicated
    end

    tt′ = Tuple{map(Core.Typeof, args)...}

    forward = thunk(fn, #=dfn=#nothing, Duplicated, tt′, Val(API.DEM_ForwardMode), width)

    res = forward(args...)
    if annotation <: Duplicated
        return Return2(res[1], res[2])
    else
        return Return2(nothing, nothing)
    end
end

function runtime_invoke_augfwd(mi::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32, width)
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)

    fn = Base.unsafe_load(arg_ptr, 1)
    
    if fn == Base.println || fn == Base.print || fn == Base.show || fn == Base.flush
        res::Any = ccall(:jl_invoke, Any, (Any, Ptr{Any}, UInt32, Any), fn, args, length(args), mi)
        return Return3(res, res, nothing)
    end
    
    # TODO actually use the mi rather than fn
    @assert in(mi.def, methods(fn))

    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)
    args = Any[]
    for (i, typ) in zip(2:arg_size, mi.specTypes.parameters[2:end])
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            if typeof(p) <: AbstractFloat || typeof(p) <: Complex{<:AbstractFloat}
                push!(args, Active(p))
            else
                s = Base.unsafe_load(shadow_ptr, i)
                push!(args, Duplicated(p, s))
            end
        else
            push!(args, Const(p))
        end
    end

    # TODO: Annotation of return value
    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt)

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward, adjoint = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ReverseModePrimal), width)

    res = forward(args...)
    if length(res) > 1
        origRet = res[2]
        resT = typeof(origRet)
    else
        origRet = nothing
        resT = Nothing
    end

    if annotation <: Active
        # This assumes that we have an Immutable object here that got passed as a boxed value
        shadow_return = Ref(zero(resT))
        ret2 = shadow_return # due to this store we need typetag
    elseif annotation <: Const
        shadow_return = nothing
        ret2 = origRet
    elseif annotation <: Duplicated ||  annotation <: DuplicatedNoNeed
        shadow_return = nothing
        ret2 = res[3]
    else
        error("Unknown annotation")
    end
    internal_tape = res[1]

    tape = Tape(adjoint, internal_tape, shadow_return, resT)
    return Return3(origRet, ret2, tape)
end

function runtime_invoke_rev(mi::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32, tape::Any, width)
    fn = Base.unsafe_load(arg_ptr, 1)
    if fn == Base.println || fn == Base.print || fn == Base.show || fn == Base.flush
        args = Any[]
        for i in 1:arg_size
            push!(args, Base.unsafe_load(arg_ptr, i))
        end

        res = fn(args...)
        return nothing
    end
    
    # TODO actually use the mi rather than fn
    @assert in(mi.def, methods(fn))

    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)
    args = []
    actives = []
    for (i, typ) in zip(2:arg_size, mi.specTypes.parameters[2:end])
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            if typeof(p) <: AbstractFloat || typeof(p) <: Complex{<:AbstractFloat}
                push!(args, Active(p))
                push!(actives, (shadow_ptr, i))
            else
                s = Base.unsafe_load(shadow_ptr, i)
                push!(args, Duplicated(p, s))
            end
        else
            push!(args, Const(p))
        end
    end
    
    tape = tape::Tape

    if tape.shadow_return !== nothing
        val = tape.shadow_return
        if !(val isa tape.resT)
            val = tape.shadow_return[]
        end
        push!(args, val)
    end
    if tape.internal_tape !== nothing
        push!(args, tape.internal_tape)
    end

    tup = tape.thunk(args...)

    for (d, (s, i)) in zip(tup, actives)
        a = unsafe_load(s, i)
        # While `RefValue{T}` and boxed T for immutable are bitwise compatible
        # they are not idempotent on the Julia level. We could "force" `a` to be
        # a boxed T, but would lose the mutable memory semantics that Enzyme requires
        if a isa Base.RefValue
            @assert eltype(a) == typeof(d)
            a[] += d
        else
            ref = unsafe_load(reinterpret(Ptr{Ptr{typeof(a)}}, s), i)
            unsafe_store!(ref, d+a)
        end
    end
    
    return nothing
end

function runtime_apply_latest_fwd(fn::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32, width)
    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)

    args = Any[]
    for i in 1:arg_size
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            s = Base.unsafe_load(shadow_ptr, i)
            push!(args, Duplicated(p, s))
        else
            push!(args, Const(p))
        end
    end

    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt, API.DEM_ForwardMode)
    if annotation <: DuplicatedNoNeed
        annotation = Duplicated
    end

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward = thunk(fn, #=dfn=#nothing, Duplicated, tt′, Val(API.DEM_ForwardMode), width)

    res = forward(args...)
    if annotation <: Duplicated
       return Return2(res[1], res[2])
    else
        return Return2(nothing, nothing)
    end
end

function runtime_apply_latest_augfwd(fn::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32, width)
    # Note: We shall not unsafe_wrap any of the Ptr{Any}, since these are stack allocations
    #       As an example, if the Array created by unsafe_wrap get's moved to the remset it
    #       will constitute a leak of the stack allocation, and GC will find delicous garbage.
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)

    args = Any[]
    for i in 1:arg_size
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            if typeof(p) <: AbstractFloat || typeof(p) <: Complex{<:AbstractFloat}
                push!(args, Active(p))
            else
                s = Base.unsafe_load(shadow_ptr, i)
                push!(args, Duplicated(p, s))
            end
        else
            push!(args, Const(p))
        end
    end

    tt = Tuple{map(x->eltype(Core.Typeof(x)), args)...}
    rt = Core.Compiler.return_type(fn, tt)
    annotation = guess_activity(rt)

    tt′ = Tuple{map(Core.Typeof, args)...}
    forward, adjoint = thunk(fn, #=dfn=#nothing, annotation, tt′, Val(API.DEM_ReverseModePrimal), width)

    res = forward(args...)
    origRet = res[2]
    resT = typeof(origRet)

    if annotation <: Active
        shadow_return = Ref(zero(resT))
        ret2 = shadow_return
    elseif annotation <: Const
        shadow_return = nothing
        ret2 = origRet
    elseif annotation <: Duplicated ||  annotation <: DuplicatedNoNeed
        shadow_return = nothing
        ret2 = res[3]
    else
        error("Unknown annotation")
    end
    internal_tape = res[1]

    tape = Tape(adjoint, internal_tape, shadow_return, resT)
    return Return3(origRet, ret2, tape)
end

function runtime_apply_latest_rev(fn::Any, arg_ptr::Ptr{Any}, shadow_ptr::Ptr{Any}, activity_ptr::Ptr{UInt8}, arg_size::UInt32, tape::Any, width)
    __activity = Base.unsafe_wrap(Array, activity_ptr, arg_size)
    args = []
    actives = []
    for i in 1:arg_size
        p = Base.unsafe_load(arg_ptr, i)
        if __activity[i] != 0
            if typeof(p) <: AbstractFloat || typeof(p) <: Complex{<:AbstractFloat}
                push!(args, Active(p))
                push!(actives, (shadow_ptr, i))
            else
                s = Base.unsafe_load(shadow_ptr, i)
                push!(args, Duplicated(p, s))
            end
        else
            push!(args, Const(p))
        end
    end

    tape = tape::Tape

    if tape.shadow_return !== nothing
        val = tape.shadow_return
        if !(val isa tape.resT)
            val = tape.shadow_return[]
        end
        push!(args, val)
    end
    if tape.internal_tape !== nothing
        push!(args, tape.internal_tape)
    end

    tup = tape.thunk(args...)

    for (d, (s, i)) in zip(tup, actives)
        a = unsafe_load(s, i)
        # While `RefValue{T}` and boxed T for immutable are bitwise compatible
        # they are not idempotent on the Julia level. We could "force" `a` to be
        # a boxed T, but would lose the mutable memory semantics that Enzyme requires
        if a isa Base.RefValue
            @assert eltype(a) == typeof(d)
            a[] += d
        else
            ref = unsafe_load(reinterpret(Ptr{Ptr{typeof(a)}}, s), i)
            unsafe_store!(ref, d+a)
        end
    end
    
    return nothing
end

function emit_gc_preserve_begin(B::LLVM.Builder, args=LLVM.Value[])
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    func = get_function!(mod, "llvm.julia.gc_preserve_begin") do mod, ctx, name
        funcT = LLVM.FunctionType(LLVM.TokenType(ctx), vararg=true)
        LLVM.Function(mod, name, funcT)
    end

    token = call!(B, func, args)
    return token
end

function emit_gc_preserve_end(B::LLVM.Builder, token)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    func = get_function!(mod, "llvm.julia.gc_preserve_end") do mod, ctx, name
        funcT = LLVM.FunctionType(LLVM.VoidType(ctx), [LLVM.TokenType(ctx)])
        LLVM.Function(mod, name, funcT)
    end

    call!(B, func, [token])
    return
end

function generic_setup(orig, gutils, start, ctx::LLVM.Context, B::LLVM.Builder, fun, lookup; sret=nothing, tape=nothing)
    T_int8 = LLVM.Int8Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)

    ops = collect(operands(orig))[(start+1):end-1]

    num = convert(UInt32, length(ops))
    llnum = LLVM.ConstantInt(num; ctx)

    EB = LLVM.Builder(ctx)
    position!(EB, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    # TODO: Optimization by emitting liverange
    primal = LLVM.alloca!(EB, LLVM.ArrayType(T_prjlvalue, num))
    shadow = LLVM.alloca!(EB, LLVM.ArrayType(T_prjlvalue, num))
    activity = LLVM.alloca!(EB, LLVM.ArrayType(T_int8, num))

    jl_fn = API.EnzymeGradientUtilsNewFromOriginal(gutils, operands(orig)[start])
    if lookup
        jl_fn = API.EnzymeGradientUtilsLookup(gutils, jl_fn, B)
    end
    vals = LLVM.Value[LLVM.Value(jl_fn), primal, shadow, activity, llnum]

    # to_preserve = LLVM.Value[primal, shadow]
    to_preserve = LLVM.Value[]

    for (i, op) in enumerate(ops)
        idx = LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i-1; ctx)]
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        if lookup
            val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))
        end
        push!(to_preserve, val)
        LLVM.store!(B, val, LLVM.inbounds_gep!(B, primal, idx))

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0
        activeC = LLVM.ConstantInt(T_int8, active)
        LLVM.store!(B, activeC, LLVM.inbounds_gep!(B, activity, idx))

        if active
            inverted = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))
            if lookup
                inverted = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, inverted, B))
            end
            push!(to_preserve, inverted)
            LLVM.store!(B, inverted,
                        LLVM.inbounds_gep!(B, shadow, idx))
        else
            LLVM.store!(B, LLVM.null(llvmtype(op)),
                        LLVM.inbounds_gep!(B, shadow, idx))
        end
    end

    if tape !== nothing
        push!(vals, LLVM.Value(tape))
        push!(to_preserve, LLVM.Value(tape))
    end

    if sret !== nothing
        pushfirst!(vals, sret)
    end

    token = emit_gc_preserve_begin(B, to_preserve)

    params = parameters(eltype(LLVM.llvmtype(fun)))
    for (i, val) in enumerate(vals)
        sourceT = LLVM.llvmtype(val)
        targetT = params[i]
        if sourceT !== targetT
            vals[i] = if sourceT isa LLVM.PointerType && targetT isa LLVM.IntegerType
                LLVM.ptrtoint!(B, val, targetT)
            elseif sourceT isa LLVM.IntegerType && targetT isa LLVM.PointerType
                LLVM.inttoptr!(B, val, targetT)
            else
                LLVM.bitcast!(B, val, targetT)
            end
        end
    end

    cal = LLVM.call!(B, fun, vals)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
   
    return cal, token
end

function allocate_sret!(B::LLVM.Builder, N, ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    al = LLVM.alloca!(B, LLVM.ArrayType(T_prjlvalue, N))
    for i in 1:N
        LLVM.store!(B, LLVM.null(T_prjlvalue),
                    LLVM.inbounds_gep!(B, al, LLVM.Value[LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(i-1; ctx)]))
    end
    return al
end

function allocate_sret!(gutils::API.EnzymeGradientUtilsRef, N, ctx)
    sret = LLVM.Builder(ctx) do B
        position!(B, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
        allocate_sret!(B, N, ctx)
    end
end

function generic_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    B = LLVM.Builder(B)
    sret = allocate_sret!(gutils, 2, ctx)

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    llvmf = nested_codegen!(mod, runtime_generic_fwd, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Val{width}})
    _, token = generic_setup(orig, gutils, #=start=#1, ctx, B, llvmf, false; sret)

    if shadowR != C_NULL
        shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    emit_gc_preserve_end(B, token)

    return nothing
end
function generic_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    B = LLVM.Builder(B)
    sret = allocate_sret!(gutils, 3, ctx)

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    llvmf = nested_codegen!(mod, runtime_generic_augfwd, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Val{width}})
    _, token = generic_setup(orig, gutils, #=start=#1, ctx, B, llvmf, false; sret)

    if shadowR != C_NULL
        shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(2; ctx)]))
    unsafe_store!(tapeR, tape.ref)

    emit_gc_preserve_end(B, token)

    return nothing
end

function generic_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    B = LLVM.Builder(B)

    @assert tape !== C_NULL
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    llvmf = nested_codegen!(mod, runtime_generic_rev, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Any, Val{width}})
    _, token = generic_setup(orig, gutils, #=start=#1, ctx, B, llvmf, true; tape)

    emit_gc_preserve_end(B, token)

    return nothing
end


function invoke_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        ctx = LLVM.context(orig)

        conv = LLVM.API.LLVMGetInstructionCallConv(orig)
        # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
        if conv != 38
            GPUCompiler.@safe_error "Illegal invoke convention ", orig, conv, API.EnzymeGradientUtilsIsConstantValue(gutils, orig),  API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig)
        end
        @assert conv == 38

        B = LLVM.Builder(B)
        sret = allocate_sret!(gutils, 2, ctx)

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        llvmf = nested_codegen!(mod, runtime_invoke_fwd, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Val{width}})
        _, token = generic_setup(orig, gutils, #=start=#1, ctx, B, llvmf, false; sret)

        if shadowR != C_NULL
            shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
            unsafe_store!(shadowR, shadow.ref)
        end

        if normalR != C_NULL
            normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
            unsafe_store!(normalR, normal.ref)
        end

        emit_gc_preserve_end(B, token)
    end

    return nothing
end
function invoke_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        ctx = LLVM.context(orig)

        conv = LLVM.API.LLVMGetInstructionCallConv(orig)
        # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
        @assert conv == 38

        B = LLVM.Builder(B)
        sret = allocate_sret!(gutils, 2, ctx)

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        llvmf = nested_codegen!(mod, runtime_invoke_augfwd, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Val{width}})
        _, token = generic_setup(orig, gutils, #=start=#1, ctx, B, llvmf, false; sret)

        if shadowR != C_NULL
            shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
            unsafe_store!(shadowR, shadow.ref)
        end

        if normalR != C_NULL
            normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
            unsafe_store!(normalR, normal.ref)
        end

        tape = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(2; ctx)]))
        unsafe_store!(tapeR, tape.ref)

        emit_gc_preserve_end(B, token)
    end

    return nothing
end

function invoke_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0 || API.EnzymeGradientUtilsIsConstantInstruction(gutils, orig) == 0 
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        ctx = LLVM.context(orig)
        
        conv = LLVM.API.LLVMGetInstructionCallConv(orig)
        # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
        @assert conv == 38

        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        llvmf = nested_codegen!(mod, runtime_invoke_rev, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Any, Val{width}})
        _, token = generic_setup(orig, gutils, #=start=#1, ctx, B, llvmf, true; tape)

        emit_gc_preserve_end(B, token)
    end

    return nothing
end

function apply_latest_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    B = LLVM.Builder(B)
    sret = allocate_sret!(gutils, 2, ctx)

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    llvmf = nested_codegen!(mod, runtime_apply_latest_fwd, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Val{width}})
    _, token = generic_setup(orig, gutils, #=start=#2, ctx, B, llvmf, false; sret)

    if shadowR != C_NULL
        shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    emit_gc_preserve_end(B, token)

    return nothing
end

function apply_latest_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    B = LLVM.Builder(B)

    sret = allocate_sret!(gutils, 3, ctx)

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    llvmf = nested_codegen!(mod, runtime_apply_latest_augfwd, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Val{width}})
    _, token = generic_setup(orig, gutils, #=start=#2, ctx, B, llvmf, false; sret)

    if shadowR != C_NULL
        shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(2; ctx)]))
    unsafe_store!(tapeR, tape.ref)

    emit_gc_preserve_end(B, token)

    return nothing
end

function apply_latest_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    ctx = LLVM.context(orig)

    B = LLVM.Builder(B)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    llvmf = nested_codegen!(mod, runtime_apply_latest_rev, Tuple{Any, Ptr{Any}, Ptr{Any}, Ptr{UInt8}, UInt32, Any, Val{width}})
    _, token = generic_setup(orig, gutils, #=start=#2, ctx, B, llvmf, true; tape)

    emit_gc_preserve_end(B, token)

    return nothing
end

function emit_error(B::LLVM.Builder, string)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    ctx = context(mod)

    # 1. get the error function
    func = get_function!(mod, "jl_error") do mod, ctx, name
        funcT = LLVM.FunctionType(LLVM.VoidType(ctx), LLVMType[LLVM.PointerType(LLVM.Int8Type(ctx))])
        func = LLVM.Function(mod, name, funcT)
        push!(function_attributes(func), EnumAttribute("noreturn"; ctx))
        return func
    end

    # 2. Call error function and insert unreachable
    call!(B, func, LLVM.Value[globalstring_ptr!(B, string)])

    # FIXME(@wsmoses): Allow for emission of new BB in this code path
    # unreachable!(B)

    # 3. Change insertion point so that we don't stumble later
    # after_error = BasicBlock(fn, "after_error"; ctx)
    # position!(B, after_error)
end

function noop_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    return nothing
end

function noop_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    return nothing
end

function duplicate_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    newg = API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)
    
    B = LLVM.Builder(B)

    real_ops = collect(operands(orig))
    ops = [LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, o), B)) for o in real_ops]
    call!(B, ops[end], ops[1:end-1])
    return nothing
end

let counter = Ref{Int64}(0)
    global getid() = counter[] += 1
end
const leaked_objs = Base.Dict{Int64, Any}()

if VERSION < v"1.8-"

function runtime_pfor_fwd(func, dfunc, width)::Cvoid
    # pfor threads tape[i] = aug(func, dfunc)

    tt = Tuple{}
    # TODO with the MI itself, we should be able to hoist the thunk generation into
    # the original compilation (rather than runtime JIT compiling)
    forward = thunk(func, dfunc, Const, tt, Val(API.DEM_ForwardMode), width)

    function fwd()
        forward()
    end

    Base.Threads.threading_run(fwd)
    return
end

function runtime_pfor_augfwd(func, dfunc, width)::Int64
    # tape = vec{Any}(numthreads())
    # pfor threads tape[i] = aug(func, dfunc)

    tt = Tuple{}
    # TODO with the MI itself, we should be able to hoist the thunk generation into
    # the original compilation (rather than runtime JIT compiling)
    forward, adjoint = thunk(func, dfunc, Const, tt, Val(API.DEM_ReverseModePrimal), width)

    tapes = Vector{Ptr{Cvoid}}(undef, Base.Threads.nthreads())
    # tapes = unsafe_convert(Ptr{Ptr{Cvoid}}, Libc.malloc(sizeof(Ptr{Cvoid})*Base.Threads.nthreads))
    function fwd()
        tapes[Base.Threads.threadid()] = forward()[1]
    end
    adj = () -> adjoint(tapes[Base.Threads.threadid()])
    id = getid()
    leaked_objs[id] = adj
    Base.Threads.threading_run(fwd)
    # Note: `adj` is an immutable object, and thus has pass-by-value semantics
    #       directly returning here allocates a heap object whose pointer is then leaked
    #       onto the tape and GC, will kill it. Until we have proper tape+GC support,
    #       we stash the object in a global dict, and return the id of the object
    return id
end

function runtime_pfor_rev(id::Int64)
    tape = leaked_objs[id]
    delete!(leaked_objs, id)
    Base.Threads.threading_run(tape)
    return nothing
end
else

function runtime_pfor_fwd(func, dfunc, width, dynamic)::Cvoid
    # pfor threads tape[i] = aug(func, dfunc)

    tt = Tuple{Const{Int}}
    # TODO with the MI itself, we should be able to hoist the thunk generation into
    # the original compilation (rather than runtime JIT compiling)
    forward = thunk(func, dfunc, Const, tt, Val(API.DEM_ForwardMode), width)

    function fwd(tid)
        forward(Const(tid))
    end

    Base.Threads.threading_run(fwd, dynamic)
    return
end

function runtime_pfor_augfwd(func, dfunc, width, dynamic)::Int64
    # tape = vec{Any}(numthreads())
    # pfor threads tape[i] = aug(func, dfunc)

    tt = Tuple{Const{Int}}
    # TODO with the MI itself, we should be able to hoist the thunk generation into
    # the original compilation (rather than runtime JIT compiling)
    forward, adjoint = thunk(func, dfunc, Const, tt, Val(API.DEM_ReverseModePrimal), width)

    tapes = Vector{Ptr{Cvoid}}(undef, Base.Threads.nthreads())
    # tapes = unsafe_convert(Ptr{Ptr{Cvoid}}, Libc.malloc(sizeof(Ptr{Cvoid})*Base.Threads.nthreads))
    function fwd(tid)
        tapes[tid] = forward(Const(tid))[1]
    end
    adj = (tid) -> adjoint(Const(tid), tapes[tid])
    id = getid()
    leaked_objs[id] = adj
    Base.Threads.threading_run(fwd, dynamic)
    # Note: `adj` is an immutable object, and thus has pass-by-value semantics
    #       directly returning here allocates a heap object whose pointer is then leaked
    #       onto the tape and GC, will kill it. Until we have proper tape+GC support,
    #       we stash the object in a global dict, and return the id of the object
    return id
end

function runtime_pfor_rev(id::Int64, dynamic)
    tape = leaked_objs[id]
    delete!(leaked_objs, id)
    Base.Threads.threading_run(tape, dynamic)
    return nothing
end
end

function threadsfor_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid

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

    funcT = mi.specTypes.parameters[2]

    width = API.EnzymeGradientUtilsGetWidth(gutils)

@static if VERSION < v"1.8-"
    tt = Tuple{funcT, funcT, Val{width}}
    extraArgs = 0
else
    tt = Tuple{funcT, funcT, Val{width}, Bool}
    extraArgs = 1
end
    entry = nested_codegen!(mod, runtime_pfor_fwd, tt)

    B = LLVM.Builder(B)

    T_int64 = LLVM.Int64Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)

    ops = collect(operands(orig))[1:end-1]

    if !GPUCompiler.isghosttype(funcT) && !Core.Compiler.isconstType(funcT)
		for real in [ LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1])), LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B))]
			push!(vals, real)
		end
        to_preserve = LLVM.Value[vals[1], vals[2]]
    else
        to_preserve = LLVM.Value[]
	end
@static if VERSION < v"1.8-"
else
    if length(vals) ÷ 2 != length(ops)
        push!(vals, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[end])))
    end
end
    token = emit_gc_preserve_begin(B, to_preserve)

    LLVM.call!(B, entry, vals)

    emit_gc_preserve_end(B, token)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end

end

function threadsfor_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
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

	funcT = mi.specTypes.parameters[2]

    width = API.EnzymeGradientUtilsGetWidth(gutils)

@static if VERSION < v"1.8-"
    tt = Tuple{funcT, funcT, Val{width}} #annotate_tuple_type(mi.specTypes, activity)
    extraArgs = 0
else
    tt = Tuple{funcT, funcT, Val{width}, Bool}
    extraArgs = 1
end
    GPUCompiler.@safe_warn "active variables passed by value to jl_threadsfor are not yet supported"
    entry = nested_codegen!(mod, runtime_pfor_augfwd, tt)

    B = LLVM.Builder(B)
    
    T_int64 = LLVM.Int64Type(ctx)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_pprjlvalue = LLVM.PointerType(T_prjlvalue)

    ops = collect(operands(orig))[1:end-1]

	vals = LLVM.Value[]
    if !GPUCompiler.isghosttype(funcT) && !Core.Compiler.isconstType(funcT)
		for real in [ LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1])), LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B))]
			push!(vals, real)
		end
        to_preserve = LLVM.Value[vals[1], vals[2]]
    else
        to_preserve = LLVM.Value[]
	end
@static if VERSION < v"1.8-"
else
    if length(vals) ÷ 2 != length(ops)
        push!(vals, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[end])))
    end
end
    token = emit_gc_preserve_begin(B, to_preserve)

    tape = LLVM.call!(B, entry, vals)
    
    emit_gc_preserve_end(B, token)

    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end

    unsafe_store!(tapeR, tape.ref)

    return nothing
end

function threadsfor_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    tape = LLVM.Value(tape)
    ctx = LLVM.context(orig)
@static if VERSION < v"1.8-"
    fun = @cfunction(runtime_pfor_rev, Cvoid, (Int64,))
else
    fun = @cfunction(runtime_pfor_rev, Cvoid, (Int64,Bool))
end
    
	B = LLVM.Builder(B)
    ops = collect(operands(orig))[1:end-1]
    
	vals = LLVM.Value[tape]
    to_preserve = LLVM.Value[vals[1]]
    token = emit_gc_preserve_begin(B, to_preserve)
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, #= AddressSpace::Tracked =# 10)
    T_int64 = LLVM.Int64Type(ctx)

@static if VERSION < v"1.8-"
    T_args = LLVM.LLVMType[T_int64]
else
    T_args = LLVM.LLVMType[T_int64, LLVM.Int8Type(ctx)]
end
    fnT = LLVM.FunctionType(LLVM.VoidType(ctx), T_args)
    rtfn = LLVM.inttoptr!(B, LLVM.ConstantInt(convert(UInt64, fun); ctx), LLVM.PointerType(fnT))
@static if VERSION < v"1.8-"
else
        push!(vals, LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[end]), B)))
end

    LLVM.call!(B, rtfn, vals)
    
    emit_gc_preserve_end(B, token)
    return nothing
end

include("pmap.jl")

function newtask_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    fun = nested_codegen!(mod, runtime_newtask_fwd, Tuple{Any, Any, Any, Int, Val{width}})

    B = LLVM.Builder(B)
    
    ops = collect(operands(orig))

    vals = LLVM.Value[ LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1])),
                       LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B)),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[2])),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[3]))]

    to_preserve = LLVM.Value[vals[1], vals[2], vals[3]]
    token = emit_gc_preserve_begin(B, to_preserve)

    ntask = LLVM.call!(B, fun, vals)

    # TODO: GC, ret
    if shadowR != C_NULL
        unsafe_store!(shadowR, ntask.ref)
    end

    if normalR != C_NULL
        unsafe_store!(normalR, ntask.ref)
    end

    emit_gc_preserve_end(B, token)

    return nothing
end

function newtask_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    # fn, dfn = augmentAndGradient(fn)
    # t = jl_new_task(fn)
    # # shadow t
    # dt = jl_new_task(dfn)
    orig = LLVM.Instruction(OrigCI)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    ctx = LLVM.context(orig)

    GPUCompiler.@safe_warn "active variables passeed by value to jl_new_task are not yet supported"
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    fun = nested_codegen!(mod, runtime_newtask_augfwd, Tuple{Any, Any, Any, Int, Val{width}})

    B = LLVM.Builder(B)
    sret = allocate_sret!(gutils, 2, ctx)
    
    ops = collect(operands(orig))[1:end-1]

    vals = LLVM.Value[sret,
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[1])),
                       LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, ops[1], B)),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[2])),
                       LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, ops[3]))]

    to_preserve = LLVM.Value[vals[2], vals[3], vals[4]] # All Any should be preserved
    token = emit_gc_preserve_begin(B, to_preserve)

    LLVM.call!(B, fun, vals)

    if shadowR != C_NULL
        shadow = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(1; ctx)]))
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, LLVM.inbounds_gep!(B, sret, [LLVM.ConstantInt(0; ctx), LLVM.ConstantInt(0; ctx)]))
        unsafe_store!(normalR, normal.ref)
    end

    emit_gc_preserve_end(B, token)

    return nothing
end

function newtask_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    return nothing
end

function enq_work_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function enq_work_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
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

function enq_work_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # jl_wait(shadow(t)) 
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    waitfn = find_match(mod, "jl_wait")
    @assert waitfn !== nothing
    shadowtask = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B), B))
    cal = LLVM.call!(LLVM.Builder(B), waitfn, [shadowtask])
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    LLVM.API.LLVMSetInstructionCallConv(cal, conv)
    return nothing
end

function wait_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function wait_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function wait_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # jl_enq_work(shadow(t)) 
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    enq_work_fn = find_match(mod, "jl_enq_work")
    @assert enq_work_fn !== nothing
    shadowtask = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B), B))
    cal = LLVM.call!(LLVM.Builder(B), enq_work_fn, [shadowtask])
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, cal, orig)
    conv = LLVM.API.LLVMGetInstructionCallConv(orig)
    LLVM.API.LLVMSetInstructionCallConv(cal, conv)
    return nothing
end

function arraycopy_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)

    width = API.EnzymeGradientUtilsGetWidth(gutils)

    shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))

    B = LLVM.Builder(B)

    if width == 1
        shadowres = LLVM.call!(B, LLVM.called_value(orig), [shadowin])
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
        for idx in 1:width
            shadowres = insert_value!(B, shadowres, LLVM.call!(B, LLVM.called_value(orig), [
                            extract_value!(B, shadowin, idx-1)
                            ]), idx-1)
        end
    end
 
    unsafe_store!(shadowR, shadowres.ref)
	
	return nothing
end

function arraycopy_common(fwd, B, orig, origArg, gutils)
    # size_t len = jl_array_len(ary);
    # size_t elsz = ary->elsize;
    # memcpy(new_ary->data, ary->data, len * elsz);
	# JL_EXTENSION typedef struct {
	# 	JL_DATA_TYPE
	# 	void *data;
	# #ifdef STORE_ARRAY_LEN
	# 	size_t length;
	# #endif
	# 	jl_array_flags_t flags;
	# 	uint16_t elsize;  // element size including alignment (dim 1 memory stride)

	tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, orig))
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
	dl = string(LLVM.datalayout(mod))
	API.EnzymeTypeTreeLookupEq(tt, 1, dl)
	data0!(tt)
    ct = API.EnzymeTypeTreeInner0(tt)

    if ct == API.DT_Unknown
        GPUCompiler.@safe_warn "Unknown concrete type" tt=string(tt)
        emit_error(B, "Enzyme: Unknown concrete type in arraycopy_common")
        return nothing
    end

    @assert ct != API.DT_Unknown
    ctx = LLVM.context(orig)
    secretty = API.EnzymeConcreteTypeIsFloat(ct, ctx)

    off = sizeof(Cstring)
    if true # STORE_ARRAY_LEN
        off += sizeof(Csize_t)
    end
    #jl_array_flags_t
    off += 2

    actualOp = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origArg))
    if fwd
        B0 = B
    elseif typeof(actualOp) <: LLVM.Argument
        B0 = LLVM.Builder(ctx)
        position!(B0, first(instructions(LLVM.BasicBlock(API.EnzymeGradientUtilsNewFromOriginal(gutils, LLVM.entry(LLVM.parent(LLVM.parent(orig))))))))
    else
        B0 = LLVM.Builder(ctx)
        position!(B0, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(actualOp)))
    end
    
    actualOp = pointercast!(B0, actualOp, LLVM.PointerType(LLVM.IntType(8; ctx), LLVM.addrspace(LLVM.llvmtype(actualOp))))

    elSize = gep!(B0, actualOp, [LLVM.ConstantInt(LLVM.IntType(64; ctx), off)])
    elSize = pointercast!(B0, elSize, LLVM.PointerType(LLVM.IntType(16; ctx), LLVM.addrspace(LLVM.llvmtype(actualOp))))
    elSize = LLVM.load!(B0, elSize)
    elSize = LLVM.zext!(B0, elSize, LLVM.IntType(8*sizeof(Csize_t); ctx))

    len = gep!(B0, actualOp, [LLVM.ConstantInt(LLVM.IntType(64; ctx), sizeof(Cstring))])
    len = pointercast!(B0, len, LLVM.PointerType(LLVM.IntType(8*sizeof(Csize_t); ctx), LLVM.addrspace(LLVM.llvmtype(actualOp))))
    len = LLVM.load!(B0, len)

    length = LLVM.mul!(B0, len, elSize)
    isVolatile = LLVM.ConstantInt(LLVM.IntType(1; ctx), 0)

    # forward pass copy already done by underlying call
    allowForward = false
    intrinsic = LLVM.Intrinsic("llvm.memcpy").id

    shadowdst = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, orig, B))
    if !fwd
        shadowdst = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowdst, B))
    end
    shadowsrc = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origArg, B))
    if !fwd
        shadowsrc = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowsrc, B))
    end
    
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if width == 1
    
    shadowsrc = load!(B, bitcast!(B, shadowsrc, LLVM.PointerType(LLVM.PointerType(LLVM.IntType(8; ctx), 13), LLVM.addrspace(LLVM.llvmtype(shadowsrc)))))
    shadowdst = load!(B, bitcast!(B, shadowdst, LLVM.PointerType(LLVM.PointerType(LLVM.IntType(8; ctx), 13), LLVM.addrspace(LLVM.llvmtype(shadowdst)))))

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst, false, shadowsrc, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    
    else
    for i in 1:width

    evsrc = extract_value!(B, shadowsrc, i-1)
    evdst = extract_value!(B, shadowdst, i-1)

    shadowsrc0 = load!(B, bitcast!(B, evsrc, LLVM.PointerType(LLVM.PointerType(LLVM.IntType(8; ctx), 13), LLVM.addrspace(LLVM.llvmtype(evsrc)))))
    shadowdst0 = load!(B, bitcast!(B, evdst, LLVM.PointerType(LLVM.PointerType(LLVM.IntType(8; ctx), 13), LLVM.addrspace(LLVM.llvmtype(evdst)))))

    API.sub_transfer(gutils, fwd ? API.DEM_ReverseModePrimal : API.DEM_ReverseModeGradient, secretty, intrinsic, #=dstAlign=#1, #=srcAlign=#1, #=offset=#0, false, shadowdst0, false, shadowsrc0, length, isVolatile, orig, allowForward, #=shadowsLookedUp=#!fwd)
    end

    end

    return nothing
end

function arraycopy_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)

    B = LLVM.Builder(B)

    shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
 
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if width == 1
        shadowres = LLVM.call!(B, LLVM.called_value(orig), [shadowin])
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
        for idx in 1:width
            shadowres = insert_value!(B, shadowres, LLVM.call!(B, LLVM.called_value(orig), [
                            extract_value!(B, shadowin, idx-1)
                            ]), idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    
    arraycopy_common(#=fwd=#true, B, orig, origops[1], gutils)
	
	return nothing
end

function arraycopy_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)
    arraycopy_common(#=fwd=#false, LLVM.Builder(B), orig, origops[1], gutils)
    return nothing
end

function arrayreshape_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    
    orig = LLVM.Instruction(OrigCI)
    origops = LLVM.operands(orig)

    B = LLVM.Builder(B)    
    width = API.EnzymeGradientUtilsGetWidth(gutils)

    shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[2], B))
    if width == 1
        args = LLVM.Value[
                          LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                          shadowin
                          LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]))
                          ]
        shadowres = LLVM.call!(B, LLVM.called_value(orig), args)
    else
        shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
        for idx in 1:width
            args = LLVM.Value[
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                              extract_value!(B, shadowin, idx-1)
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[3]))
                              ]
            tmp = LLVM.call!(B, LLVM.called_value(orig), args)
            shadowres = insert_value!(B, shadowres, tmp, idx-1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
	
	return nothing
end

function arrayreshape_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    arrayreshape_fwd(B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function arrayreshape_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented reverse for jl_array_reshape")
    return nothing
end

function f_tuple_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, orig) == 0
        B = LLVM.Builder(B)

        shadowsin = LLVM.Value[
                        LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, o, B)) for o in origops[1:end-1] ]
        if width == 1
            shadowres = LLVM.call!(B, LLVM.called_value(orig), shadowsin)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, s, idx-1) for s in shadowsin
                                  ]
                tmp = LLVM.call!(B, LLVM.called_value(orig), args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(tmp, conv)
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function f_tuple_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    f_tuple_fwd(B, OrigCI, gutils, normalR, shadowR)
    return nothing
end

function f_tuple_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # This function allocates a new return which returns a pointer, thus this instruction itself cannot transfer
    # derivative info, only create a shadow pointer, which is handled by the forward pass.
    return nothing
end

function apply_iterate_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function apply_iterate_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented reverse for jl_f__apply_iterate")
    return nothing
end

function new_structv_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented augmented forward for jl_new_struct")

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    return nothing
end

function new_structv_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: Not yet implemented reverse for jl_new_struct")
    return nothing
end

function gcpreserve_begin_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    ops = collect(operands(orig))[1:end-1]

    to_preserve = LLVM.Value[]

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    B = LLVM.Builder(B)
    for op in ops
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        push!(to_preserve, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0

        if active
            shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))

            if width == 1
                push!(to_preserve, shadowin)
            else
                for idx in 1:width
                    push!(to_preserve, extract_value!(B, shadowin, idx-1))
                end
            end
        end
    end

    token = emit_gc_preserve_begin(B, to_preserve)
    unsafe_store!(normalR, token.ref)

    return nothing
end

function gcpreserve_begin_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)

    ops = collect(operands(orig))[1:end-1]

    to_preserve = LLVM.Value[]

    width = API.EnzymeGradientUtilsGetWidth(gutils)
    B = LLVM.Builder(B)
    for op in ops
        val = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, op))
        push!(to_preserve, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0

        if active
            shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, op, B))

            if width == 1
                push!(to_preserve, shadowin)
            else
                for idx in 1:width
                    push!(to_preserve, extract_value!(B, shadowin, idx-1))
                end
            end
        end
    end

    token = emit_gc_preserve_begin(B, to_preserve)
    unsafe_store!(normalR, token.ref)

    return nothing
end

const GCToks = Dict{LLVM.Instruction, LLVM.Instruction}()

function gcpreserve_begin_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    builder = LLVM.Builder(B)
    orig = LLVM.Instruction(OrigCI)
    if haskey(GCToks, orig)
        token = GCToks[orig]
        delete!(GCToks, orig)
    else
        f   = LLVM.parent(orig)
        mod = LLVM.parent(f)
        ctx = LLVM.context(mod)

        
        token = emit_gc_preserve_begin(LLVM.Builder(B))
        # token = LLVM.phi!(builder, LLVM.TokenType(ctx), "placeholder")
        GCToks[orig] = token
    end
    emit_gc_preserve_end(builder, token)
    return nothing
end

function gcpreserve_end_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    return nothing
end

function gcpreserve_end_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origPres = operands(orig)[1]

    ops = collect(operands(origPres))[1:end-1]

    to_preserve = LLVM.Value[]

    for op in ops
        val = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsNewFromOriginal(gutils, op), B))
        push!(to_preserve, val)

        active = API.EnzymeGradientUtilsIsConstantValue(gutils, op) == 0

        if active
            push!(to_preserve, LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, API.EnzymeGradientUtilsInvertPointer(gutils, op, B), B)))
        end
    end

    token = emit_gc_preserve_begin(LLVM.Builder(B), to_preserve)

    if haskey(GCToks, origPres)
        placeHolder = GCToks[origPres]
        LLVM.replace_uses!(placeHolder, token)
        delete!(GCToks, origPres)
        LLVM.API.LLVMInstructionEraseFromParent(placeHolder)
    else
        GCToks[origPres] = token
    end

    return nothing
end

function jl_array_grow_end_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        if width == 1
            args = LLVM.Value[
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                              ]
            LLVM.call!(B, LLVM.called_value(orig), args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                  ]
                LLVM.call!(B, LLVM.called_value(orig), args)
            end
        end
    end
    return nothing
end

function jl_array_grow_end_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        if width == 1
            args = LLVM.Value[
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                              ]
            LLVM.call!(B, LLVM.called_value(orig), args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                  ]
                LLVM.call!(B, LLVM.called_value(orig), args)
            end
        end
    end

    return nothing
end

function jl_array_grow_end_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)
        
        called_value = origops[end]
        funcT = eltype(llvmtype(called_value)::LLVM.PointerType)::LLVM.FunctionType
        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
        delF = LLVM.Function(mod, "jl_array_del_end", funcT)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        shadowin = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, shadowin, B))

        offset = LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
        offset = LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, offset, B))

        if width == 1
            args = LLVM.Value[
                              shadowin
                              offset
                              ]
            LLVM.call!(B, delF, args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  offset
                                  ]
                LLVM.call!(B, delF, args)
            end
        end
    end
    return nothing
end

function jl_array_del_end_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)

        width = API.EnzymeGradientUtilsGetWidth(gutils)

        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        if width == 1
            args = LLVM.Value[
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                              ]
            LLVM.call!(B, LLVM.called_value(orig), args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                  ]
                LLVM.call!(B, LLVM.called_value(orig), args)
            end
        end
    end
    return nothing
end

function jl_getfield_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)


        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[2], B))
        if width == 1
            args = LLVM.Value[
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                              shadowin
                              ]
            for a in origops[3:end-1]
                push!(args, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, a)))
            end

            shadowres = LLVM.call!(B, LLVM.called_value(orig), args)
            conv = LLVM.API.LLVMGetInstructionCallConv(orig)
            LLVM.API.LLVMSetInstructionCallConv(shadowres, conv)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[1]))
                                  extract_value!(B, shadowin, idx-1)
                                  ]
                for a in origops[3:end-1]
                    push!(args, LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, a)))
                end
                tmp = LLVM.call!(B, LLVM.called_value(orig), args)
                conv = LLVM.API.LLVMGetInstructionCallConv(orig)
                LLVM.API.LLVMSetInstructionCallConv(tmp, conv)
                shadowres = insert_value!(B, shadowres, tmp, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function jl_array_sizehint_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    origops = collect(operands(orig))
    width = API.EnzymeGradientUtilsGetWidth(gutils)
    if API.EnzymeGradientUtilsIsConstantValue(gutils, origops[1]) == 0
        B = LLVM.Builder(B)


        shadowin = LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, origops[1], B))
        if width == 1
            args = LLVM.Value[
                              shadowin
                              LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                              ]
            LLVM.call!(B, LLVM.called_value(orig), args)
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(orig))))
            for idx in 1:width
                args = LLVM.Value[
                                  extract_value!(B, shadowin, idx-1)
                                  LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, origops[2]))
                                  ]
                LLVM.call!(B, LLVM.called_value(orig), args)
            end
        end
    end
    return nothing
end

function jl_array_sizehint_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    jl_array_sizehint_fwd(B, OrigCI, gutils, normalR, shadowR, tapeR)
     return nothing
end

function jl_array_sizehint_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    return nothing
end

function jl_f__apply_iterate_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_f__apply_iterate")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function setfield_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_f_setfield")
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function setfield_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled reverse for jl_f_setfield")
    return nothing
end

function get_binding_or_error_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    err = emit_error(LLVM.Builder(B), "Enzyme: unhandled forward for jl_get_binding_or_error")
    API.moveBefore(CI, err)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing

    if shadowR != C_NULL && normal !== nothing
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        B = LLVM.Builder(B)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function get_binding_or_error_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    err = emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_get_binding_or_error")
    API.moveBefore(CI, err)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        width = API.EnzymeGradientUtilsGetWidth(gutils)
        B = LLVM.Builder(B)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llvmtype(normal))))
            for idx in 1:width
                shadowres = insert_value!(B, shadowres, normal, idx-1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return nothing
end

function get_binding_or_error_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    emit_error(LLVM.Builder(B), "Enzyme: unhandled reverse for jl_get_binding_or_error")
    return nothing
end

function finalizer_fwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    err = emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
    API.moveBefore(CI, err)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return nothing
end

function finalizer_augfwd(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})::Cvoid
    orig = LLVM.Instruction(OrigCI)
    CI = API.EnzymeGradientUtilsNewFromOriginal(gutils, OrigCI)
    # err = emit_error(LLVM.Builder(B), "Enzyme: unhandled augmented forward for jl_gc_add_finalizer_th")
    # API.moveBefore(CI, err)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    # Delete the primal code
    if normal !== nothing
        unsafe_store!(normalR, C_NULL)
    else
        LLVM.API.LLVMInstructionEraseFromParent(LLVM.Instruction(API.EnzymeGradientUtilsNewFromOriginal(gutils, orig)))
    end
    return nothing
end

function finalizer_rev(B::LLVM.API.LLVMBuilderRef, OrigCI::LLVM.API.LLVMValueRef, gutils::API.EnzymeGradientUtilsRef, tape::LLVM.API.LLVMValueRef)::Cvoid
    # emit_error(LLVM.Builder(B), "Enzyme: unhandled reverse for jl_gc_add_finalizer_th")
    return nothing
end


function register_handler!(variants, augfwd_handler, rev_handler, fwd_handler=nothing)
    for variant in variants
        if augfwd_handler !== nothing && rev_handler !== nothing
            API.EnzymeRegisterCallHandler(variant, augfwd_handler, rev_handler)
        end
        if fwd_handler !== nothing
            API.EnzymeRegisterFwdCallHandler(variant, fwd_handler)
        end
    end
end

function register_alloc_handler!(variants, alloc_handler, free_handler)
    for variant in variants
        API.EnzymeRegisterAllocationHandler(variant, alloc_handler, free_handler)
    end
end

abstract type CompilationException <: Base.Exception end
struct NoDerivativeException <: CompilationException
    msg::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::NoDerivativeException)
    print(io, "Enzyme compilation failed.\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct NoShadowException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::NoShadowException)
    print(io, "Enzyme compilation failed due missing shadow.\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, "\n Inverted pointers: \n")
    write(io, ece.sval)
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        print(io,"\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct IllegalTypeAnalysisException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing, String}
    bt::Union{Nothing, Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalTypeAnalysisException)
    print(io, "Enzyme compilation failed due to illegal type analysis.\n")
    if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
    end
    print(io, "\n Type analysis state: \n")
    write(io, ece.sval)
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        print(io,"\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

parent_scope(val::LLVM.Function, depth=0) = depth==0 ? LLVM.parent(val) : val
parent_scope(val::LLVM.Module, depth=0) = val
parent_scope(val::LLVM.Value, depth=0) = parent_scope(LLVM.parent(val), depth+1)

function julia_error(cstr::Cstring, val::LLVM.API.LLVMValueRef, errtype::API.ErrorType, data::Ptr{Cvoid})
    msg = Base.unsafe_string(cstr)
    val = LLVM.Value(val)
    if isa(val, LLVM.Instruction)
        bt = GPUCompiler.backtrace(val)
    else
        bt = nothing
    end
    if isa(val, LLVM.ConstantExpr)
        ir = nothing
    else
        # Need to convert function to string, since when the error is going to be printed
        # the module might have been destroyed
        ir = sprint(io->show(io, parent_scope(val)))
    end
    if errtype == API.ET_NoDerivative
        data = API.EnzymeGradientUtilsRef(data)
        throw(NoDerivativeException(msg, ir, bt))
    elseif errtype == API.ET_NoShadow
        data = API.EnzymeGradientUtilsRef(data)
        ip = API.EnzymeGradientUtilsInvertedPointersToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)
        throw(NoShadowException(msg, sval, ir, bt))
    elseif errtype == API.ET_IllegalTypeAnalysis
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)
        throw(IllegalTypeAnalysisException(msg, sval, ir, bt))
    end
    throw(AssertionError("Unknown errtype"))
end

function __init__()
    API.EnzymeSetHandler(@cfunction(julia_error, Cvoid, (Cstring, LLVM.API.LLVMValueRef, API.ErrorType, Ptr{Cvoid})))
    register_alloc_handler!(
        ("jl_alloc_array_1d", "ijl_alloc_array_1d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_alloc_handler!(
        ("jl_alloc_array_2d", "ijl_alloc_array_2d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_alloc_handler!(
        ("jl_alloc_array_3d", "ijl_alloc_array_3d"),
        @cfunction(array_shadow_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, Csize_t, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(null_free_handler, LLVM.API.LLVMValueRef, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, LLVM.API.LLVMValueRef))
    )
    register_handler!(
        ("jl_apply_generic", "ijl_apply_generic"),
        @cfunction(generic_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(generic_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(generic_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_invoke", "ijl_invoke", "jl_f_invoke"),
        @cfunction(invoke_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(invoke_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(invoke_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_latest", "jl_f__call_latest"),
        @cfunction(apply_latest_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(apply_latest_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(apply_latest_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_threadsfor",),
        @cfunction(threadsfor_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(threadsfor_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(threadsfor_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_pmap",),
        @cfunction(pmap_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(pmap_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(pmap_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_new_task", "ijl_new_task"),
        @cfunction(newtask_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(newtask_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(newtask_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_enq_work",),
        @cfunction(enq_work_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(enq_work_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(enq_work_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}))
    )
    register_handler!(
        ("jl_wait",),
        @cfunction(wait_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(wait_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(wait_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_","jl_breakpoint"),
        @cfunction(noop_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(duplicate_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(noop_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_copy","ijl_array_copy"),
        @cfunction(arraycopy_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(arraycopy_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(arraycopy_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_reshape_array","ijl_reshape_array"),
        @cfunction(arrayreshape_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(arrayreshape_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(arrayreshape_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("llvm.julia.gc_preserve_begin",),
        @cfunction(gcpreserve_begin_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(gcpreserve_begin_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(gcpreserve_begin_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("llvm.julia.gc_preserve_end",),
        @cfunction(gcpreserve_end_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(gcpreserve_end_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_f_setfield",),
        @cfunction(setfield_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(setfield_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_f_tuple","ijl_f_tuple"),
        @cfunction(f_tuple_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(f_tuple_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(f_tuple_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_iterate",),
        @cfunction(apply_iterate_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(apply_iterate_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_new_structv",),
        @cfunction(new_structv_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(new_structv_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
    )
    register_handler!(
        ("jl_get_binding_or_error", "ijl_get_binding_or_error"),
        @cfunction(get_binding_or_error_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(get_binding_or_error_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(get_binding_or_error_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_gc_add_finalizer_th",),
        @cfunction(finalizer_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(finalizer_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(finalizer_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_grow_end","ijl_array_grow_end"),
        @cfunction(jl_array_grow_end_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_grow_end_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_grow_end_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_del_end","ijl_array_del_end"),
        nothing,
        nothing,
        @cfunction(jl_array_del_end_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f_getfield",),
        nothing,
        nothing,
        @cfunction(jl_getfield_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_f__apply_iterate",),
        nothing,
        nothing,
        @cfunction(jl_f__apply_iterate_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
    register_handler!(
        ("jl_array_sizehint","ijl_array_sizehint"),
        @cfunction(jl_array_sizehint_augfwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
        @cfunction(jl_array_sizehint_rev, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, LLVM.API.LLVMValueRef)),
        @cfunction(jl_array_sizehint_fwd, Cvoid, (LLVM.API.LLVMBuilderRef, LLVM.API.LLVMValueRef, API.EnzymeGradientUtilsRef, Ptr{LLVM.API.LLVMValueRef}, Ptr{LLVM.API.LLVMValueRef})),
    )
end

end # module
