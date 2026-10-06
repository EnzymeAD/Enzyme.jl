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

function annotate_checkpoint_loop_state!(mod::LLVM.Module)
    changed = false
    for F in functions(mod)
        isempty(blocks(F)) && continue
        for (body, pre) in checkpoint_loops(F)
            pre === nothing && continue
            inst = new_object_store(body, mod)
            if inst !== nothing
                error("Enzyme: a checkpointed loop stores an object reference into its state, " *
                      "which its snapshots cannot restore: $(string(inst)) in " *
                      "$(LLVM.name(LLVM.parent(LLVM.parent(inst))))")
            end
            objs = Set{LLVM.Value}()
            for bb in body, inst in instructions(bb), op in operands(inst)
                if op isa LLVM.Argument ||
                        (op isa LLVM.Instruction && !(LLVM.parent(op) in body))
                    julia_objects!(objs, op)
                end
            end
            isempty(objs) && continue
            marker = declare!(mod, "__enzyme_julia_state",
                LLVM.FunctionType(LLVM.VoidType(), LLVMType[]; vararg = true))
            @dispose B = IRBuilder() begin
                position!(B, terminator(pre))
                for V in objs
                    # An object made in the loop is no part of its state.
                    V isa LLVM.Instruction && LLVM.parent(V) in body && continue
                    legal, T, byref = checkpoint_typeof(V)
                    if !legal
                        regions = LLVM.Value[]
                        if stack_slot_regions!(regions, B, mod, V, body)
                            call!(B, LLVM.function_type(marker), marker, LLVM.Value[V; regions])
                            changed = true
                        end
                        continue
                    end
                    obj = if addrspace(value_type(V)) == Tracked
                        byref == GPUCompiler.MUT_REF || continue
                        V
                    else
                        byref == GPUCompiler.BITS_REF || continue
                        nothing
                    end
                    ptr = obj === nothing ? V :
                        addrspacecast!(B, V, LLVM.PointerType(Derived))
                    regions = LLVM.Value[]
                    julia_state_regions!(regions, B, mod, ptr, obj, T) || continue
                    call!(B, LLVM.function_type(marker), marker, LLVM.Value[V; regions])
                    changed = true
                end
            end
        end
    end
    return changed
end

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
