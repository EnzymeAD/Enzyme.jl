# Checkpointed loops: `EnzymeCore.checkpoint_for` and `checkpoint_while` become
# Enzyme's `__enzyme_checkpoint_for` and `__enzyme_checkpoint_while` (see
# enzyme/checkpoint.h), whose reverse mode is driven by an external
# checkpointing scheme.
#
# `_checkpoint_for(scheme, data, start, n, box)` calls `checkpoint_step(box, i)`
# for each `i`, and `_checkpoint_while(scheme, data, box)` calls
# `checkpoint_while_step(box)` until it returns false; all are kept out of line
# (`handle_compiled` marks them). Each call to them is replaced by
#
#     __enzyme_checkpoint_for(step, start, n, enzyme_scheme, scheme, data,
#                             box, [pgcstack])
#     __enzyme_checkpoint_while(step, enzyme_scheme, scheme, data,
#                               box, [pgcstack])
#
# where `step(i, box, [pgcstack])` calls the compiled step, which takes the
# task's GC stack (swiftself) first, and the index last if it has one. The box
# is the first argument after the index, so a scheme that copies the state
# itself finds it as the first word of the step's environment. Enzyme then
# lowers the marker to its loop.

const CHECKPOINT_FOR_ATTR = "enzymejl_checkpoint_for"
const CHECKPOINT_WHILE_ATTR = "enzymejl_checkpoint_while"
const CHECKPOINT_STEP_ATTR = "enzymejl_checkpoint_step"

function checkpoint_step_callee(F::LLVM.Function)
    for bb in blocks(F), inst in instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        if callee isa LLVM.Function &&
                has_fn_attr(callee, StringAttribute(CHECKPOINT_STEP_ATTR))
            return callee
        end
    end
    return nothing
end

function is_swiftself(f::LLVM.Function, i::Integer)
    return any(
        a -> a isa EnumAttribute && kind(a) == "swiftself",
        collect(parameter_attributes(f, i)),
    )
end

# `step(i, box, [pgcstack])`, calling `G([pgcstack,] box, i)` for a for loop,
# and returning `G([pgcstack,] box)` for a while loop.
function checkpoint_step_wrapper!(mod::LLVM.Module, G::LLVM.Function, indexed::Bool)
    name = "enzymejl_ckpt_step." * LLVM.name(G)
    haskey(functions(mod), name) && return functions(mod)[name]
    Gparams = parameters(G)
    T_i64 = LLVM.Int64Type()
    nargs = indexed ? length(Gparams) - 1 : length(Gparams)
    indexed && @assert value_type(Gparams[end]) == T_i64
    # G's parameters but the index: the GC stack (if any) goes last.
    order = [i for i in 1:nargs if !is_swiftself(G, i)]
    append!(order, [i for i in 1:nargs if is_swiftself(G, i)])
    rettype = indexed ? LLVM.VoidType() : LLVM.return_type(function_type(G))
    FT = LLVM.FunctionType(
        rettype,
        LLVMType[T_i64, (value_type(Gparams[i]) for i in order)...],
    )
    S = LLVM.Function(mod, name, FT)
    linkage!(S, LLVM.API.LLVMInternalLinkage)
    # The body's parameters keep their Julia type annotations.
    for (j, i) in enumerate(order)
        for a in collect(parameter_attributes(G, i))
            a isa EnumAttribute && kind(a) == "swiftself" && continue
            push!(parameter_attributes(S, j + 1), a)
        end
    end
    builder = IRBuilder()
    position!(builder, BasicBlock(S, "entry"))
    Sparams = parameters(S)
    args = Vector{LLVM.Value}(undef, length(Gparams))
    for (j, i) in enumerate(order)
        args[i] = Sparams[j + 1]
    end
    indexed && (args[end] = Sparams[1])
    call = call!(builder, function_type(G), G, args)
    callconv!(call, callconv(G))
    for i in 1:nargs
        if is_swiftself(G, i)
            LLVM.API.LLVMAddCallSiteAttribute(call, i, EnumAttribute("swiftself", 0))
        end
    end
    indexed ? ret!(builder) : ret!(builder, call)
    dispose(builder)
    return S
end

# Runs before the Julia pipeline, which would drop constant arguments of
# `_checkpoint_for`: the marker is an opaque call it leaves alone.
function rewrite_checkpoint_calls!(mod::LLVM.Module)
    changed = false
    for (attr, indexed) in ((CHECKPOINT_FOR_ATTR, true), (CHECKPOINT_WHILE_ATTR, false))
        changed |= rewrite_checkpoint_calls!(mod, attr, indexed)
    end
    return changed
end

function rewrite_checkpoint_calls!(mod::LLVM.Module, attr::String, indexed::Bool)
    markers = LLVM.Function[
        f for f in functions(mod) if has_fn_attr(f, StringAttribute(attr))
    ]
    isempty(markers) && return false

    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()
    T_ptr = LLVM.PointerType(LLVM.Int8Type())
    scheme_marker = if haskey(globals(mod), "enzyme_scheme")
        globals(mod)["enzyme_scheme"]
    else
        LLVM.GlobalVariable(mod, T_i32, "enzyme_scheme")
    end
    name = indexed ? "__enzyme_checkpoint_for" : "__enzyme_checkpoint_while"
    declFT = LLVM.FunctionType(
        LLVM.VoidType(),
        indexed ? LLVMType[T_ptr, T_i64, T_i64] : LLVMType[T_ptr];
        vararg = true,
    )
    decl = if haskey(functions(mod), name)
        functions(mod)[name]
    else
        LLVM.Function(mod, name, declFT)
    end

    for F in markers
        G = checkpoint_step_callee(F)
        G === nothing && error("Enzyme: $(LLVM.name(F)) does not call its step")
        S = checkpoint_step_wrapper!(mod, G, indexed)
        # F's parameters: [pgcstack,] scheme, data, [start, n,] box
        off = !isempty(parameters(F)) && is_swiftself(F, 1) ? 1 : 0
        nfixed = indexed ? 4 : 2
        for u in collect(LLVM.uses(F))
            ci = LLVM.user(u)
            ci isa LLVM.CallInst && LLVM.called_operand(ci) == F || continue
            args = collect(LLVM.arguments(ci))
            pgcstack = args[1:off]
            scheme, data = args[(off + 1):(off + 2)]
            bounds = args[(off + 3):(off + nfixed)]
            box = args[(off + nfixed + 1):end]
            builder = IRBuilder()
            position!(builder, ci)
            aspointer(v) = value_type(v) isa LLVM.IntegerType ? inttoptr!(builder, v, T_ptr) : v
            marker = load!(builder, T_i32, scheme_marker)
            call!(
                builder,
                declFT,
                decl,
                LLVM.Value[
                    S, bounds..., marker, aspointer(scheme), aspointer(data), box..., pgcstack...,
                ],
            )
            dispose(builder)
            LLVM.API.LLVMInstructionEraseFromParent(ci)
        end
    end
    return true
end

# Loops annotated for checkpointing with their loop metadata, which a Julia
# loop gets from `Expr(:loopinfo, (Symbol("enzyme.checkpoint"), :revolve, k))`
# at the end of its body. Enzyme snapshots what such a loop writes; for memory
# it reaches through Julia objects (the data of an array, the arrays a struct
# holds) it cannot see from the IR how much there is. Here each Julia value
# the loop uses gets, in front of the loop,
#
#     __enzyme_julia_state(value, ptr1, bytes1, ptr2, bytes2, ...)
#
# with all the memory that can be written through it, from its type: the data
# of the arrays it holds, and the mutable objects among it. A value whose type
# does not say (an abstract field, an array of objects) gets none, and Enzyme
# refuses the loop if it writes through it. Enzyme reads and removes these.
# A loop that stores an object reference into its state is refused here.

const CHECKPOINT_LOOP_MD = "enzyme.checkpoint"

function is_checkpoint_loop_id(md)
    md isa LLVM.MDNode || return false
    for op in operands(md)
        op isa LLVM.MDNode || continue
        ops = operands(op)
        if !isempty(ops) && ops[1] isa LLVM.MDString &&
                convert(String, ops[1]) == CHECKPOINT_LOOP_MD
            return true
        end
    end
    return false
end

# The blocks of each loop annotated for checkpointing in `F`, and the block in
# front of it, if it has one.
function checkpoint_loops(F::LLVM.Function)
    loops = Tuple{Set{LLVM.BasicBlock}, Union{Nothing, LLVM.BasicBlock}}[]
    domtree = nothing
    for latch in blocks(F)
        term = terminator(latch)
        md = metadata(term)
        haskey(md, LLVM.MD_loop) && is_checkpoint_loop_id(md[LLVM.MD_loop]) ||
            continue
        domtree === nothing && (domtree = DomTree(F))
        header = nothing
        for succ in successors(term)
            if dominates(domtree, first(instructions(succ)), term)
                header = succ
            end
        end
        header === nothing && continue
        body = Set{LLVM.BasicBlock}([header, latch])
        todo = [latch]
        while !isempty(todo)
            bb = pop!(todo)
            bb == header && continue
            for pred in predecessors(bb)
                pred in body && continue
                push!(body, pred)
                push!(todo, pred)
            end
        end
        outside = [pred for pred in predecessors(header) if !(pred in body)]
        push!(loops, (body, length(outside) == 1 ? only(outside) : nothing))
    end
    domtree === nothing || dispose(domtree)
    return loops
end

# The Julia values a value used in a loop is made from: the array whose data
# pointer it is, the struct it is a field of, and so on.
function julia_objects!(objs, v::LLVM.Value, depth = 0)
    depth > 8 && return objs
    ty = value_type(v)
    if ty isa LLVM.PointerType && addrspace(ty) in (Tracked, Derived)
        push!(objs, v)
    end
    if v isa LLVM.AddrSpaceCastInst || v isa LLVM.BitCastInst ||
            v isa LLVM.GetElementPtrInst || v isa LLVM.LoadInst
        julia_objects!(objs, operands(v)[1], depth + 1)
    elseif v isa LLVM.CallInst
        callee = LLVM.called_operand(v)
        if callee isa LLVM.Function && LLVM.name(callee) == "julia.gc_loaded"
            for op in operands(v)[1:2]
                julia_objects!(objs, op, depth + 1)
            end
        end
    end
    return objs
end

# Objects of the runtime that a loop does not change.
is_runtime_constant(@nospecialize(T)) =
    T <: Union{String, Symbol, Module, Type, Core.SimpleVector, Core.MethodInstance}

function declare!(mod::LLVM.Module, name, ft)
    return haskey(functions(mod), name) ? functions(mod)[name] : LLVM.Function(mod, name, ft)
end

# Push the memory that can be written through a value of type `T` stored at
# `ptr` (in the derived address space), whose object is `obj` if it is boxed,
# to `regions`. False if `T` does not say what it is.
function julia_state_regions!(regions, B::LLVM.IRBuilder, mod::LLVM.Module,
                              ptr::LLVM.Value, obj, @nospecialize(T), depth = 0)
    depth > 8 && return false
    isbitstype(T) && return true
    is_runtime_constant(T) && return true
    isconcretetype(T) || return false
    i64 = LLVM.IntType(64)
    i8 = LLVM.IntType(8)
    if T <: Array
        E = eltype(T)
        isbitstype(E) || return false
        # ref.ptr_or_offset, the first word of the array
        data = load!(B, LLVM.PointerType(), ptr)
        n = LLVM.ConstantInt(i64, sizeof(E))
        off = fieldoffset(T, 2)
        for i in 1:ndims(T)
            p = inbounds_gep!(B, i8, ptr, LLVM.Value[LLVM.ConstantInt(i64, off + 8 * (i - 1))])
            n = mul!(B, n, load!(B, i64, p))
        end
        push!(regions, data, n)
        return true
    end
    T <: GenericMemory && return false
    if ismutabletype(T)
        obj === nothing && return false
        pfo = declare!(mod, "julia.pointer_from_objref", LLVM.FunctionType(
            LLVM.PointerType(), LLVMType[LLVM.PointerType(Derived)]))
        push!(regions, call!(B, LLVM.function_type(pfo), pfo, LLVM.Value[ptr]),
              LLVM.ConstantInt(i64, sizeof(T)))
    end
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        p = inbounds_gep!(B, i8, ptr, LLVM.Value[LLVM.ConstantInt(i64, fieldoffset(T, i))])
        if Base.allocatedinline(F)
            julia_state_regions!(regions, B, mod, p, nothing, F, depth + 1) ||
                return false
        else
            isbitstype(F) && continue
            is_runtime_constant(F) && continue
            # A field that is #undef is not handled.
            fv = load!(B, LLVM.PointerType(Tracked), p)
            julia_state_regions!(regions, B, mod,
                addrspacecast!(B, fv, LLVM.PointerType(Derived)), fv, F, depth + 1) ||
                return false
        end
    end
    return true
end

# Whether `v` is memory made where it is used: a stack slot, a new object.
function is_fresh(v::LLVM.Value)
    while v isa LLVM.AddrSpaceCastInst || v isa LLVM.BitCastInst ||
            v isa LLVM.GetElementPtrInst
        v = operands(v)[1]
    end
    v isa LLVM.AllocaInst && return true
    if v isa LLVM.CallInst
        callee = LLVM.called_operand(v)
        callee isa LLVM.Function || return false
        n = LLVM.name(callee)
        return n == "julia.gc_alloc_obj" || occursin("jl_alloc_genericmemory", n) ||
            occursin("jl_gc_alloc", n)
    end
    return false
end

# A store of an object reference into memory the loop did not make, in the
# loop or what it calls. Not `m.u = copy(m.u)`, whose new array no snapshot
# holds, nor `m.u, m.tmp = m.tmp, m.u`: a snapshot restores the references in
# `m` but not those in its shadow.
function new_object_store(body, mod)
    seen = Set{LLVM.Function}()
    todo = LLVM.Function[]
    check(inst) =
        inst isa LLVM.StoreInst &&
        (ty = value_type(operands(inst)[1]); ty isa LLVM.PointerType && addrspace(ty) == Tracked) &&
        !is_fresh(operands(inst)[2])
    note(inst) = if inst isa LLVM.CallInst
        callee = LLVM.called_operand(inst)
        if callee isa LLVM.Function && !isempty(blocks(callee)) && !(callee in seen)
            push!(seen, callee)
            push!(todo, callee)
        end
    end
    for bb in body, inst in instructions(bb)
        check(inst) && return inst
        note(inst)
    end
    while !isempty(todo)
        F = pop!(todo)
        for bb in blocks(F), inst in instructions(bb)
            check(inst) && return inst
            note(inst)
        end
    end
    return nothing
end

# The type of a value, which abs_typeof finds for most; a struct passed by
# value is stored into a stack slot from the argument, which it does not.
function checkpoint_typeof(V::LLVM.Value)
    legal, T, byref = abs_typeof(V)
    legal && return legal, T, byref
    A = V
    while A isa LLVM.AddrSpaceCastInst || A isa LLVM.BitCastInst
        A = operands(A)[1]
    end
    A isa LLVM.AllocaInst || return false, nothing, nothing
    for u in uses(A)
        st = user(u)
        st isa LLVM.StoreInst && operands(st)[2] == A || continue
        ev = operands(st)[1]
        ev isa LLVM.ExtractValueInst || continue
        arg = operands(ev)[1]
        arg isa LLVM.Argument || continue
        F = LLVM.parent(LLVM.parent(st))
        idx = findfirst(==(arg), collect(parameters(F)))
        T, _ = enzyme_extract_parm_type(F, idx, false)
        T === nothing && continue
        return true, T, GPUCompiler.BITS_REF
    end
    return false, nothing, nothing
end

# The regions of a stack slot of object references, a closure's environment
# say, which the loop may fill each time: what the references stored into it
# hold. Only references made outside the loop, so as to be there in front of
# it, and null.
function stack_slot_regions!(regions, B, mod, V, body)
    A = V
    while A isa LLVM.AddrSpaceCastInst || A isa LLVM.BitCastInst
        A = operands(A)[1]
    end
    A isa LLVM.AllocaInst || return false
    todo = LLVM.Value[A]
    stored = Set{LLVM.Value}()
    while !isempty(todo)
        P = pop!(todo)
        for u in uses(P)
            I = user(u)
            if I isa LLVM.GetElementPtrInst || I isa LLVM.AddrSpaceCastInst ||
                    I isa LLVM.BitCastInst
                push!(todo, I)
            elseif I isa LLVM.StoreInst && operands(I)[2] == P
                push!(stored, operands(I)[1])
            end
        end
    end
    isempty(stored) && return false
    for v in stored
        v isa LLVM.PointerNull && continue
        ty = value_type(v)
        ty isa LLVM.PointerType && addrspace(ty) == Tracked || return false
        v isa LLVM.Instruction && LLVM.parent(v) in body && return false
        legal, T, byref = checkpoint_typeof(v)
        legal && byref == GPUCompiler.MUT_REF || return false
        julia_state_regions!(regions, B, mod,
            addrspacecast!(B, v, LLVM.PointerType(Derived)), v, T) || return false
    end
    return true
end

# Whether julia_state_regions! can describe a value of type `T`, without
# emitting anything.
function static_describable(@nospecialize(T), depth = 0)
    depth > 8 && return false
    (isbitstype(T) || is_runtime_constant(T)) && return true
    isconcretetype(T) || return false
    T <: Array && return isbitstype(eltype(T))
    T <: GenericMemory && return false
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        (isbitstype(F) || is_runtime_constant(F)) && continue
        static_describable(F, depth + 1) || return false
    end
    return true
end

# Whether a value of type `T` can hold a reference the loop could change: a
# mutable object with a field that is not plain bits, or an array of them.
function has_mutable_refs(@nospecialize(T), depth = 0)
    depth > 8 && return true
    (isbitstype(T) || is_runtime_constant(T)) && return false
    isconcretetype(T) || return true
    T <: Array && return !isbitstype(eltype(T))
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        (isbitstype(F) || is_runtime_constant(F)) && continue
        ismutabletype(T) && return true
        has_mutable_refs(F, depth + 1) && return true
    end
    return false
end

#
# Dynamic state: a loop whose state its type does not describe (an array of
# arrays, an abstract field), or which changes the references in it (swaps
# two arrays), is snapshotted at run time by walking the objects reachable
# from it. The value is a callback region (see EnzymeCkptCallbacks in
# enzyme/checkpoint.h) given by
#
#     __enzyme_julia_dynamic_state(value, root1, callbacks1, ...)
#
# whose roots are the objects (or, for a struct kept in a stack slot, the
# slot) the walk starts from. A snapshot holds, for each mutable object
# reachable, its fields or its elements; a restore puts them back, so that
# references are restored too. The shadow's references must follow the
# primal's: the forward sweep runs steps without their derivatives, and a
# restore puts back only the primal. So `enter` pairs each primal object with
# its shadow, and `sync` sets each shadow reference to the shadow of what the
# primal references. A loop that stores a new object into its state is an
# error at run time: its snapshots would hold an object made by a rerun of a
# step, of which the derivative's shadows know nothing.
#

mutable struct CheckpointDynamicState
    shadows::IdDict{Any, Any}
    slots::Dict{Int64, Vector{Any}}
end

const CHECKPOINT_DYNAMIC = Dict{Ptr{Cvoid}, CheckpointDynamicState}()
const CHECKPOINT_DYNAMIC_LOCK = ReentrantLock()

# Same layout as EnzymeCkptCallbacks.
struct CheckpointCallbacks
    enter::Ptr{Cvoid}
    save::Ptr{Cvoid}
    restore::Ptr{Cvoid}
    sync::Ptr{Cvoid}
    leave::Ptr{Cvoid}
    data::Ptr{Cvoid}
end

# Per root type for a root in a stack slot (nothing for an object), kept
# alive here.
const CHECKPOINT_CALLBACKS = IdDict{Any, Base.RefValue{CheckpointCallbacks}}()

struct CheckpointUndef end

# The root type of a stack slot holding one object reference, which the
# outlined loop carries or leaves: a snapshot holds the reference, and sync
# sets the shadow slot to the shadow of the object the primal slot holds.
struct CheckpointRefSlot end

is_ref_slot(cb::Ptr{CheckpointCallbacks}) =
    unsafe_load(cb).data == pointer_from_objref(CheckpointRefSlot)

# The root type of a stack slot holding a pointer into the data of an array
# of the state: sync points the shadow slot at the same place in the
# array's shadow.
struct CheckpointPtrSlot end

is_ptr_slot(cb::Ptr{CheckpointCallbacks}) =
    unsafe_load(cb).data == pointer_from_objref(CheckpointPtrSlot)

# The same place in the shadow of the array whose data `p` points into.
function find_shadow_pointer(p::UInt)
    @lock CHECKPOINT_DYNAMIC_LOCK for st in values(CHECKPOINT_DYNAMIC)
        for (x, s) in st.shadows
            x isa Union{Array, GenericMemory} && isbitstype(eltype(x)) || continue
            s isa Union{Array, GenericMemory} || continue
            base = UInt(pointer(x))
            if base <= p <= base + sizeof(x)
                return UInt(pointer(s)) + (p - base)
            end
        end
    end
    return nothing
end

function load_ref_slot(ptr::Ptr{Cvoid})
    r = unsafe_load(Ptr{Ptr{Cvoid}}(ptr))
    return r == C_NULL ? CheckpointUndef() : unsafe_pointer_to_objref(r)
end

# The shadow of `x` in any state being checkpointed.
function find_shadow(@nospecialize(x))
    @lock CHECKPOINT_DYNAMIC_LOCK for st in values(CHECKPOINT_DYNAMIC)
        haskey(st.shadows, x) && return st.shadows[x]
    end
    return CheckpointUndef()
end


# The objects a root reaches directly: an object root is its object; a
# struct of type `T` kept in a stack slot is read field by field.
function checkpoint_root_values(cb::Ptr{CheckpointCallbacks}, ptr::Ptr{Cvoid})
    ptr == C_NULL && return Any[]
    data = unsafe_load(cb).data
    data == C_NULL && return Any[unsafe_pointer_to_objref(ptr)]
    T = unsafe_pointer_to_objref(data)
    vals = Any[]
    inline_values!(vals, ptr, T)
    return vals
end

function inline_values!(vals, ptr::Ptr{Cvoid}, @nospecialize(T))
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        p = ptr + fieldoffset(T, i)
        if Base.allocatedinline(F)
            isbitstype(F) || inline_values!(vals, p, F)
        else
            r = unsafe_load(Ptr{Ptr{Cvoid}}(p))
            r == C_NULL || push!(vals, unsafe_pointer_to_objref(r))
        end
    end
    return vals
end

is_checkpoint_constant(@nospecialize(x)) =
    x isa Union{String, Symbol, Module, Type, Core.SimpleVector, Core.MethodInstance,
                Core.TypeName, Method, Core.CodeInstance}

# What a value references: its fields, or its elements.
function checkpoint_children(@nospecialize(x))
    if x isa Union{Array, GenericMemory}
        isbitstype(eltype(x)) && return ()
        return (isassigned(x, i) ? x[i] : CheckpointUndef() for i in eachindex(x))
    end
    return (isdefined(x, i) ? getfield(x, i) : CheckpointUndef() for i in 1:Base.nfields(x))
end

# The mutable objects reachable from `vals`.
function checkpoint_reachable(vals)
    seen = Base.IdSet{Any}()
    objs = Any[]
    todo = Any[vals...]
    while !isempty(todo)
        x = pop!(todo)
        (x isa CheckpointUndef || isbits(x) || is_checkpoint_constant(x)) && continue
        if ismutable(x)
            x in seen && continue
            push!(seen, x)
            push!(objs, x)
        end
        for c in checkpoint_children(x)
            push!(todo, c)
        end
    end
    return objs
end

function checkpoint_pair!(shadows, @nospecialize(x), @nospecialize(s))
    (x isa CheckpointUndef || s isa CheckpointUndef || isbits(x) ||
        is_checkpoint_constant(x)) && return
    if ismutable(x)
        haskey(shadows, x) && return
        shadows[x] = s
    end
    typeof(s) === typeof(x) || return
    # An array's buffer, which the loop may hold on its own.
    if x isa Array
        shadows[getfield(x, :ref).mem] = getfield(s, :ref).mem
    end
    for (c, d) in zip(checkpoint_children(x), checkpoint_children(s))
        checkpoint_pair!(shadows, c, d)
    end
end

function checkpoint_enter(cb::Ptr{CheckpointCallbacks}, primal::Ptr{Cvoid}, shadow::Ptr{Cvoid})
    shadows = IdDict{Any, Any}()
    if shadow != C_NULL && !is_ref_slot(cb) && !is_ptr_slot(cb)
        for (x, s) in zip(checkpoint_root_values(cb, primal), checkpoint_root_values(cb, shadow))
            checkpoint_pair!(shadows, x, s)
        end
    end
    @lock CHECKPOINT_DYNAMIC_LOCK CHECKPOINT_DYNAMIC[primal] =
        CheckpointDynamicState(shadows, Dict{Int64, Vector{Any}}())
    return nothing
end

checkpoint_state(primal) = @lock CHECKPOINT_DYNAMIC_LOCK CHECKPOINT_DYNAMIC[primal]

function checkpoint_save(cb::Ptr{CheckpointCallbacks}, primal::Ptr{Cvoid}, shadow::Ptr{Cvoid}, slot::Int64)
    st = checkpoint_state(primal)
    if is_ref_slot(cb)
        st.slots[slot] = Any[load_ref_slot(primal)]
        return nothing
    elseif is_ptr_slot(cb)
        st.slots[slot] = Any[unsafe_load(Ptr{UInt}(primal))]
        return nothing
    end
    snap = Any[]
    for x in checkpoint_reachable(checkpoint_root_values(cb, primal))
        saved = if x isa Union{Array, GenericMemory}
            copy(x)
        else
            Any[isdefined(x, i) ? getfield(x, i) : CheckpointUndef() for i in 1:Base.nfields(x)]
        end
        push!(snap, x, saved)
    end
    st.slots[slot] = snap
    return nothing
end

function checkpoint_restore(cb::Ptr{CheckpointCallbacks}, primal::Ptr{Cvoid}, shadow::Ptr{Cvoid}, slot::Int64)
    st = checkpoint_state(primal)
    snap = st.slots[slot]
    if is_ref_slot(cb)
        x = snap[1]
        unsafe_store!(Ptr{Ptr{Cvoid}}(primal),
                      x isa CheckpointUndef ? C_NULL : pointer_from_objref(x))
        return nothing
    elseif is_ptr_slot(cb)
        unsafe_store!(Ptr{UInt}(primal), snap[1]::UInt)
        return nothing
    end
    for k in 1:2:length(snap)
        x, saved = snap[k], snap[k + 1]
        if x isa Array
            if size(x) != size(saved)
                x isa Vector || error("Enzyme: a checkpointed loop changed the size of a $(typeof(x))")
                resize!(x, length(saved))
            end
            copyto!(x, saved)
        elseif x isa GenericMemory
            copyto!(x, saved)
        else
            T = typeof(x)
            for i in 1:Base.nfields(x)
                v = saved[i]
                (v isa CheckpointUndef || isconst(T, i)) && continue
                setfield!(x, i, v)
            end
        end
    end
    return nothing
end

# The shadow of `x`, which the shadow held as `s` (whose plain bits, the
# derivative's, are kept).
function checkpoint_shadow(shadows, @nospecialize(x), @nospecialize(s))
    (x isa CheckpointUndef || isbits(x) || is_checkpoint_constant(x)) && return s
    if ismutable(x)
        haskey(shadows, x) && return shadows[x]
        error("Enzyme: a checkpointed loop stored a new $(typeof(x)) into its state; " *
              "only objects it held when it started can be in its snapshots")
    end
    # An immutable value holding references: made again with the shadows of
    # what it references.
    T = typeof(x)
    same = typeof(s) === T
    fields = Any[]
    for i in 1:Base.nfields(x)
        isdefined(x, i) || break
        xi = getfield(x, i)
        si = same && isdefined(s, i) ? getfield(s, i) : (isbits(xi) ? Enzyme.make_zero(xi) : CheckpointUndef())
        push!(fields, checkpoint_shadow(shadows, xi, si))
    end
    return ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields, length(fields))
end

function checkpoint_sync(cb::Ptr{CheckpointCallbacks}, primal::Ptr{Cvoid}, shadow::Ptr{Cvoid})
    shadow == C_NULL && return nothing
    if is_ptr_slot(cb)
        p = unsafe_load(Ptr{UInt}(primal))
        p == 0 && return nothing
        sp = find_shadow_pointer(p)
        sp === nothing || unsafe_store!(Ptr{UInt}(shadow), sp)
        return nothing
    end
    if is_ref_slot(cb)
        x = load_ref_slot(primal)
        (x isa CheckpointUndef || isbits(x) || is_checkpoint_constant(x)) && return nothing
        s = find_shadow(x)
        s isa CheckpointUndef &&
            error("Enzyme: a checkpointed loop carries a $(typeof(x)) that is not part of its state")
        unsafe_store!(Ptr{Ptr{Cvoid}}(shadow), pointer_from_objref(s))
        return nothing
    end
    st = checkpoint_state(primal)
    shadows = st.shadows
    for x in checkpoint_reachable(checkpoint_root_values(cb, primal))
        s = checkpoint_shadow(shadows, x, nothing)
        s === x && continue
        if x isa Union{Array, GenericMemory}
            isbitstype(eltype(x)) && continue
            if x isa Vector && length(s) != length(x)
                resize!(s, length(x))
            end
            for i in eachindex(x)
                isassigned(x, i) || continue
                cur = isassigned(s, i) ? s[i] : CheckpointUndef()
                new = checkpoint_shadow(shadows, x[i], cur)
                new === cur || (s[i] = new)
            end
        else
            T = typeof(x)
            for i in 1:Base.nfields(x)
                (isdefined(x, i) && !isconst(T, i)) || continue
                xi = getfield(x, i)
                isbits(xi) && continue
                cur = isdefined(s, i) ? getfield(s, i) : CheckpointUndef()
                new = checkpoint_shadow(shadows, xi, cur)
                new === cur || setfield!(s, i, new)
            end
        end
    end
    return nothing
end

function checkpoint_leave(cb::Ptr{CheckpointCallbacks}, primal::Ptr{Cvoid}, shadow::Ptr{Cvoid})
    @lock CHECKPOINT_DYNAMIC_LOCK delete!(CHECKPOINT_DYNAMIC, primal)
    return nothing
end

function checkpoint_callbacks(@nospecialize(T))
    ref = get!(CHECKPOINT_CALLBACKS, T) do
        Ref(CheckpointCallbacks(
            @cfunction(checkpoint_enter, Cvoid, (Ptr{CheckpointCallbacks}, Ptr{Cvoid}, Ptr{Cvoid})),
            @cfunction(checkpoint_save, Cvoid, (Ptr{CheckpointCallbacks}, Ptr{Cvoid}, Ptr{Cvoid}, Int64)),
            @cfunction(checkpoint_restore, Cvoid, (Ptr{CheckpointCallbacks}, Ptr{Cvoid}, Ptr{Cvoid}, Int64)),
            @cfunction(checkpoint_sync, Cvoid, (Ptr{CheckpointCallbacks}, Ptr{Cvoid}, Ptr{Cvoid})),
            @cfunction(checkpoint_leave, Cvoid, (Ptr{CheckpointCallbacks}, Ptr{Cvoid}, Ptr{Cvoid})),
            T === nothing ? C_NULL :
                pointer_from_objref(T),
        ))
    end
    return Base.unsafe_convert(Ptr{CheckpointCallbacks}, ref)
end

function const_ptr(B, p::Ptr)
    return inttoptr!(B, LLVM.ConstantInt(LLVM.IntType(64), UInt64(p)), LLVM.PointerType())
end

# The dynamic roots of `V`: an object, a struct in a stack slot of type `T`,
# or the objects stored into an untyped stack slot.
function dynamic_roots!(roots, B, mod, V, legal, @nospecialize(T), byref, body)
    if legal && addrspace(value_type(V)) == Tracked
        pfo = declare!(mod, "julia.pointer_from_objref", LLVM.FunctionType(
            LLVM.PointerType(), LLVMType[LLVM.PointerType(Derived)]))
        push!(roots, call!(B, LLVM.function_type(pfo), pfo,
                           LLVM.Value[addrspacecast!(B, V, LLVM.PointerType(Derived))]),
              const_ptr(B, checkpoint_callbacks(nothing)))
        return true
    end
    A = V
    while A isa LLVM.AddrSpaceCastInst || A isa LLVM.BitCastInst
        A = operands(A)[1]
    end
    A isa LLVM.AllocaInst || return false
    if legal
        push!(roots, A, const_ptr(B, checkpoint_callbacks(T)))
        return true
    end
    # Untyped: what is stored into it, made outside the loop.
    todo = LLVM.Value[A]
    stored = Set{LLVM.Value}()
    while !isempty(todo)
        P = pop!(todo)
        for u in uses(P)
            I = user(u)
            if I isa LLVM.GetElementPtrInst || I isa LLVM.AddrSpaceCastInst ||
                    I isa LLVM.BitCastInst
                push!(todo, I)
            elseif I isa LLVM.StoreInst && operands(I)[2] == P
                push!(stored, operands(I)[1])
            end
        end
    end
    isempty(stored) && return false
    for v in stored
        v isa LLVM.PointerNull && continue
        ty = value_type(v)
        ty isa LLVM.PointerType && addrspace(ty) == Tracked || return false
        v isa LLVM.Instruction && LLVM.parent(v) in body && return false
        l, Tv, _ = checkpoint_typeof(v)
        dynamic_roots!(roots, B, mod, v, l, Tv, nothing, body) || return false
    end
    return true
end

function annotate_checkpoint_loop_state!(mod::LLVM.Module)
    changed = false
    force_dynamic = get(ENV, "ENZYME_CHECKPOINT_DYNAMIC", "") == "1"
    for F in functions(mod)
        isempty(blocks(F)) && continue
        for (body, pre) in checkpoint_loops(F)
            pre === nothing && continue
            # A loop that stores references into its state changes them.
            stores_refs = force_dynamic || new_object_store(body, mod) !== nothing
            objs = Set{LLVM.Value}()
            for bb in body, inst in instructions(bb), op in operands(inst)
                if op isa LLVM.Argument ||
                        (op isa LLVM.Instruction && !(LLVM.parent(op) in body))
                    julia_objects!(objs, op)
                end
            end
            isempty(objs) && continue
            static_marker = declare!(mod, "__enzyme_julia_state",
                LLVM.FunctionType(LLVM.VoidType(), LLVMType[]; vararg = true))
            dynamic_marker = declare!(mod, "__enzyme_julia_dynamic_state",
                LLVM.FunctionType(LLVM.VoidType(), LLVMType[]; vararg = true))
            @dispose B = IRBuilder() begin
                position!(B, terminator(pre))
                ref_slots = declare!(mod, "__enzyme_julia_ref_slots",
                    LLVM.FunctionType(LLVM.VoidType(),
                                      LLVMType[LLVM.PointerType(), LLVM.PointerType()]))
                call!(B, LLVM.function_type(ref_slots), ref_slots,
                      LLVM.Value[const_ptr(B, checkpoint_callbacks(CheckpointRefSlot)),
                                 const_ptr(B, checkpoint_callbacks(CheckpointPtrSlot))])
                for V in objs
                    # An object made in the loop is no part of its state.
                    V isa LLVM.Instruction && LLVM.parent(V) in body && continue
                    legal, T, byref = checkpoint_typeof(V)
                    boxed = addrspace(value_type(V)) == Tracked
                    if legal && (boxed ? byref != GPUCompiler.MUT_REF : byref != GPUCompiler.BITS_REF)
                        continue
                    end
                    static = legal && static_describable(T) &&
                        !(stores_refs && has_mutable_refs(T))
                    if !legal && !stores_refs
                        # A stack slot of references, described by what is
                        # stored into it.
                        regions = LLVM.Value[]
                        if stack_slot_regions!(regions, B, mod, V, body)
                            call!(B, LLVM.function_type(static_marker), static_marker,
                                  LLVM.Value[V; regions])
                            changed = true
                            continue
                        end
                    end
                    if static
                        ptr = boxed ? addrspacecast!(B, V, LLVM.PointerType(Derived)) : V
                        regions = LLVM.Value[]
                        julia_state_regions!(regions, B, mod, ptr, boxed ? V : nothing, T)
                        call!(B, LLVM.function_type(static_marker), static_marker,
                              LLVM.Value[V; regions])
                        changed = true
                        continue
                    end
                    roots = LLVM.Value[]
                    dynamic_roots!(roots, B, mod, V, legal, T, byref, body) || continue
                    call!(B, LLVM.function_type(dynamic_marker), dynamic_marker,
                          LLVM.Value[V; roots])
                    changed = true
                end
            end
        end
    end
    return changed
end

# Runs before the pipeline that prepares the module for differentiation:
# an annotated loop is checkpointed one iteration a step, so there it must
# not be unrolled (a step would be several iterations, the rest of them not
# checkpointed, or the loop gone if fully unrolled) or vectorized (its
# remainder would be a second loop). This module is only differentiated;
# the function's own code, from Julia's pipeline, keeps those optimizations,
# as its annotation is metadata no pass reads.
keep_checkpoint_loops!(mod::LLVM.Module) = API.EnzymeKeepCheckpointLoops(mod) != 0

# Runs just before differentiation: Enzyme's loop function has a fixed
# signature the pipeline must not change.
#
# A loop Enzyme cannot checkpoint is reported as a diagnostic; it must be an
# error, or the loop would quietly be differentiated without checkpointing.
function lower_checkpoint_calls!(mod::LLVM.Module)
    ctx = LLVM.context(mod)
    changed = annotate_checkpoint_loop_state!(mod)
    LLVM.prepare_diagnostic(ctx)
    changed |= API.EnzymeLowerCheckpointMarkers(mod) != 0
    LLVM.check_diagnostic(ctx)
    return changed
end
