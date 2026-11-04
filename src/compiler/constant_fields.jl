# Pointer fields of stack objects that only ever hold constant data.
#
# An immutable struct that holds a `Const` array beside an active one, such as
# `Overlay(base, active)` or a `view` or `reshape` of it, is passed by reference to
# a function that is not inlined. The struct's stack slot is active memory, so
# Enzyme needs a shadow for each pointer stored into it, and the constant array
# has none. The mixed-activity handler then gives it a zero-filled copy of the
# whole array (`constant_argument_data`), on every call and at every such store,
# however few of its elements are read.
#
# The copy is needed only because the callee's loads of the field are active: they
# read from an active argument, and Enzyme accumulates the derivative of each
# element read through the loaded pointer into the field's shadow. This pass
# proves, before differentiation, that a field only ever holds constant data
# (`Const` arguments of the differentiated function, memory reached from them,
# and values loaded from other such fields), following the slot through the
# internal functions it is passed to by reference. It then tags every load of the
# field as inactive and every store into it with `enzymejl_constant_field`. No
# derivative flows through the loaded pointer, so the field's shadow is never read
# or written, and the mixed-activity handler stores the primal pointer into it,
# as it already does for data whose type cannot hold derivatives. A value written
# through a tagged load into constant memory is not differentiated, which is the
# documented behaviour for values stored into `Const` arguments.
#
# A read such as `iszero(s) ? base[j] : active[s]` meets the same problem without
# any struct: LLVM folds its two loads into one load through a select or phi of
# the two pointers, which merges a pointer into constant data with an active one.
# That pointer has no shadow either, so the handler raised
# `EnzymeRuntimeActivityError` for it, or copied the whole array on every
# execution. `split_constant_merges!` puts the load back on each side, so the
# constant side loads a value that has no derivative.
#
# Nested modes meet the struct again in the outer compilation. Reverse over
# Forward inlines the inner derivative into the outer function, which then builds
# the struct and passes it to the inner derivative of the non-inlined function, so
# the outer Reverse needs the same proof. The externally visible wrapper of the
# inner derivative stays in the module after it was inlined and builds the same
# struct from its own parameters, which are not `Const`. It never runs in the
# differentiated call, so the pass follows only code the differentiated function
# can reach (`reachable_functions`).
#
# One function can receive several places of a slot, as a vararg method that holds
# a `Const` struct in its argument tuple passes two inline views of it to the same
# `==`. A field is therefore a class of places `(root, offset)`: an access through
# a parameter reaches every place its call sites pass, so those places are proved
# constant together, and the struct's other fields keep their own proof.
#
# The tags hold only for the activities of the current compilation;
# `strip_activity_inactive_md!` removes them after `enzyme!`.
const CONSTANT_FIELD_STORE = "enzymejl_constant_field"

const ConstantField = Tuple{LLVM.Value, Int}

mutable struct ConstantFields
    primalf::LLVM.Function
    const_args::Set{LLVM.Value}
    # Slot bases: allocas and pointer parameters of internal functions.
    # `param => [(base, delta), …]`: a call passes a pointer `delta` bytes into
    # `base` to `param`, so the field `q` bytes into `param` is the one `q + delta`
    # bytes into `base` at that call. A parameter can receive a struct nested inside
    # a slot, and several places of one slot or of several slots.
    edges::Dict{LLVM.Value, Vector{ConstantField}}
    escaped::Set{LLVM.Value}
    loads::Dict{LLVM.Value, Vector{Tuple{LLVM.LoadInst, Int}}}
    stores::Dict{LLVM.Value, Vector{Tuple{LLVM.StoreInst, Int}}}
    # (offset, length) of each zero `memset`, a store of null to the fields it covers.
    zeroed::Dict{LLVM.Value, Vector{Tuple{Int, Int}}}
    reachable::Set{LLVM.Function}
    unreachable::Dict{LLVM.Function, Set{LLVM.BasicBlock}}
    callee::Dict{LLVM.Function, Bool}
    # `base => [(root, delta), …]`: every place in a root (an alloca, or a base no
    # call passes a slot to) that the memory of `base` can be (`constant_field_places`).
    places::Dict{LLVM.Value, Vector{ConstantField}}
    # Union-find over fields `(root, byte offset)`. An access through a base reaches
    # every place of that base, so those places form one class: a store writes any of
    # them and a load reads any of them.
    field_parent::Dict{ConstantField, ConstantField}
    # Classes with a place in a root whose memory escapes.
    escaped_fields::Set{ConstantField}
    # class => whether the field is still assumed constant.
    constant::Dict{ConstantField, Bool}
end

function ConstantFields(primalf::LLVM.Function, const_args::Set{LLVM.Value})
    return ConstantFields(
        primalf, const_args, Dict{LLVM.Value, Vector{ConstantField}}(), Set{LLVM.Value}(),
        Dict{LLVM.Value, Vector{Tuple{LLVM.LoadInst, Int}}}(),
        Dict{LLVM.Value, Vector{Tuple{LLVM.StoreInst, Int}}}(),
        Dict{LLVM.Value, Vector{Tuple{Int, Int}}}(), reachable_functions(primalf),
        Dict{LLVM.Function, Set{LLVM.BasicBlock}}(), Dict{LLVM.Function, Bool}(),
        Dict{LLVM.Value, Vector{ConstantField}}(), Dict{ConstantField, ConstantField}(),
        Set{ConstantField}(), Dict{ConstantField, Bool}(),
    )
end

# The functions that can run in a call of `primalf`: `primalf`, every function
# another constant or global refers to (it may be called through a pointer), and
# every function an instruction of one of these names, transitively.
function reachable_functions(primalf::LLVM.Function)::Set{LLVM.Function}
    reachable = Set{LLVM.Function}([primalf])
    for f in functions(LLVM.parent(primalf))
        for u in LLVM.uses(f)
            isa(LLVM.user(u), LLVM.Instruction) && continue
            push!(reachable, f)
            break
        end
    end
    todo = collect(reachable)
    while !isempty(todo)
        f = pop!(todo)
        for bb in blocks(f), inst in instructions(bb), op in operands(inst)
            if isa(op, LLVM.Function) && !(op in reachable)
                push!(reachable, op)
                push!(todo, op)
            end
        end
    end
    return reachable
end

is_tracked_pointer(@nospecialize(ty::LLVM.LLVMType)) =
    isa(ty, LLVM.PointerType) && LLVM.addrspace(ty) == Tracked

function is_constant_field_load(@nospecialize(inst::LLVM.Value))::Bool
    isa(inst, LLVM.LoadInst) || return false
    md = metadata(inst)
    haskey(md, "enzyme_inactive") || return false
    return inactive_reason(md["enzyme_inactive"]) == INACTIVE_CONSTANT_FIELD
end

# Blocks from which every path ends in `unreachable`. Enzyme does not
# differentiate them (`getGuaranteedUnreachable`), so the throw paths that box a
# struct for an error message never see its shadow.
function guaranteed_unreachable(f::LLVM.Function)::Set{LLVM.BasicBlock}
    dead = Set{LLVM.BasicBlock}()
    changed = true
    while changed
        changed = false
        for bb in blocks(f)
            bb in dead && continue
            term = terminator(bb)
            term === nothing && continue
            isa(term, LLVM.RetInst) && continue
            if isa(term, LLVM.UnreachableInst)
                push!(dead, bb)
                changed = true
                continue
            end
            succs = successors(term)
            isempty(succs) && continue
            if all(in(dead), succs)
                push!(dead, bb)
                changed = true
            end
        end
    end
    return dead
end

# Whether `inst` never runs differentiated: it is in a function the differentiated
# function cannot reach, or in a block from which every path ends in `unreachable`.
function in_unreachable(cf::ConstantFields, inst::LLVM.Instruction)::Bool
    bb = LLVM.parent(inst)
    f = LLVM.parent(bb)
    f in cf.reachable || return true
    dead = get(cf.unreachable, f, nothing)
    if dead === nothing
        dead = guaranteed_unreachable(f)
        cf.unreachable[f] = dead
    end
    return bb in dead
end

# Whether all call sites of `f` are visible: a defined internal function, used only
# as the callee of direct calls, without a special Enzyme handler (custom rules and
# known functions carry `enzyme_math` and external linkage, see `handleCustom`).
function constant_field_callee(cf::ConstantFields, f::LLVM.Function)::Bool
    known = get(cf.callee, f, nothing)
    known === nothing || return known
    ok = !isempty(blocks(f)) && f != cf.primalf &&
        linkage(f) in (LLVM.API.LLVMInternalLinkage, LLVM.API.LLVMPrivateLinkage) &&
        !LLVM.isvararg(LLVM.function_type(f))
    if ok
        for attr in collect(function_attributes(f))
            if isa(attr, LLVM.StringAttribute) && kind(attr) == "enzyme_math"
                ok = false
                break
            end
        end
    end
    if ok
        for u in LLVM.uses(f)
            call = LLVM.user(u)
            if !isa(call, LLVM.CallInst) || called_operand(call) != f || f in arguments(call)
                ok = false
                break
            end
        end
    end
    cf.callee[f] = ok
    return ok
end

is_constant_field_base(cf::ConstantFields, @nospecialize(base::LLVM.Value)) =
    haskey(cf.edges, base)

# The places the memory of `base` can be: `(root, delta)` for every root it can
# point `delta` bytes into, following the edges of each parameter to the slots its
# call sites pass. A base that no call passes a slot to is its own root. A recursive
# call that passes a slot back to the function it is in gives that base up.
function constant_field_places(cf::ConstantFields, @nospecialize(base::LLVM.Value), active::Set{LLVM.Value} = Set{LLVM.Value}())::Vector{ConstantField}
    known = get(cf.places, base, nothing)
    known === nothing || return known
    edges = get(cf.edges, base, ConstantField[])
    if isempty(edges)
        out = ConstantField[(base, 0)]
        cf.places[base] = out
        return out
    end
    if base in active
        push!(cf.escaped, base)
        return ConstantField[(base, 0)]
    end
    push!(active, base)
    out = ConstantField[]
    for (a, delta) in edges
        for (root, d) in constant_field_places(cf, a, active)
            place = (root, d + delta)
            place in out || push!(out, place)
        end
    end
    delete!(active, base)
    cf.places[base] = out
    return out
end

function constant_field_find(cf::ConstantFields, field::ConstantField)::ConstantField
    parent = get(cf.field_parent, field, field)
    parent == field && return field
    root = constant_field_find(cf, parent)
    cf.field_parent[field] = root
    return root
end

function constant_field_union!(cf::ConstantFields, a::ConstantField, b::ConstantField)
    ra = constant_field_find(cf, a)
    rb = constant_field_find(cf, b)
    ra == rb || (cf.field_parent[rb] = ra)
    return
end

# The class of the field `offset` bytes into `base`, or `nothing` when no access
# this pass recorded reaches all of its places.
function constant_field_key(cf::ConstantFields, @nospecialize(base::LLVM.Value), offset::Int)::Union{Nothing, ConstantField}
    is_constant_field_base(cf, base) || return nothing
    key = nothing
    for (root, delta) in constant_field_places(cf, base)
        field = (root, delta + offset)
        haskey(cf.field_parent, field) || return nothing
        class = constant_field_find(cf, field)
        if key === nothing
            key = class
        elseif key != class
            return nothing
        end
    end
    return key
end

# Whether the field `offset` bytes into `base` is (still assumed) constant.
function constant_field_is_constant(cf::ConstantFields, @nospecialize(base::LLVM.Value), offset::Int)::Bool
    key = constant_field_key(cf, base, offset)
    return key !== nothing && get(cf.constant, key, false)
end

# Whether some place of `base` is in a root whose memory escapes.
function constant_field_base_escaped(cf::ConstantFields, @nospecialize(base::LLVM.Value))::Bool
    for (root, _) in constant_field_places(cf, base)
        root in cf.escaped && return true
    end
    return false
end

function add_constant_field_base!(cf::ConstantFields, @nospecialize(base::LLVM.Value))
    is_constant_field_base(cf, base) && return false
    cf.edges[base] = ConstantField[]
    cf.loads[base] = Tuple{LLVM.LoadInst, Int}[]
    cf.stores[base] = Tuple{LLVM.StoreInst, Int}[]
    cf.zeroed[base] = Tuple{Int, Int}[]
    return true
end

# Register `base` (an alloca or a pointer parameter) and record its accesses. The
# memory behind a parameter whose call sites this pass cannot all see escapes.
function constant_field_base!(cf::ConstantFields, @nospecialize(base::LLVM.Value))
    add_constant_field_base!(cf, base) || return
    if isa(base, LLVM.Argument)
        f = LLVM.Function(LLVM.API.LLVMGetParamParent(base))
        if f == cf.primalf || !constant_field_callee(cf, f)
            push!(cf.escaped, base)
            return
        end
    end
    walk_constant_field_base!(cf, base, base, 0)
    return
end

# Record that `b` can point `delta` bytes into `a`: a call passes a pointer that far
# into slot `a` to parameter `b`. One parameter can receive several places, of one
# slot (two inline fields of a struct passed to one function) or of several.
function add_constant_field_edge!(cf::ConstantFields, @nospecialize(a::LLVM.Value), @nospecialize(b::LLVM.Value), delta::Int)
    edges = cf.edges[b]
    (a, delta) in edges || push!(edges, (a, delta))
    return
end

# The byte offset of a GEP with only constant indices, or `nothing`.
function constant_gep_offset(gep::LLVM.GetElementPtrInst)::Union{Nothing, Int}
    for idx in operands(gep)[2:end]
        isa(idx, LLVM.ConstantInt) || return nothing
    end
    b = LLVM.IRBuilder()
    position!(b, gep)
    off = API.EnzymeComputeByteOffsetOfGEP(b, gep, LLVM.IntType(8 * sizeof(Int)))
    dispose(b)
    return isa(off, LLVM.ConstantInt) ? convert(Int, off) : nothing
end

# Strip casts and constant GEPs from a pointer: its base and byte offset.
function constant_field_pointer(@nospecialize(ptr::LLVM.Value))::Tuple{LLVM.Value, Int}
    offset = 0
    while true
        if isa(ptr, LLVM.BitCastInst) || isa(ptr, LLVM.AddrSpaceCastInst)
            ptr = operands(ptr)[1]
        elseif isa(ptr, LLVM.GetElementPtrInst)
            off = constant_gep_offset(ptr)
            off === nothing && break
            offset += off
            ptr = operands(ptr)[1]
        else
            break
        end
    end
    return ptr, offset
end

# Record every access to the memory of `base` reached through `ptr`, which points
# `offset` bytes into it. Any use that could let other code read or write the
# memory without going through such an access makes the whole slot escape.
function walk_constant_field_base!(cf::ConstantFields, @nospecialize(base::LLVM.Value), @nospecialize(ptr::LLVM.Value), offset::Int)
    for u in LLVM.uses(ptr)
        user = LLVM.user(u)
        if !isa(user, LLVM.Instruction)
            push!(cf.escaped, base)
            continue
        end
        in_unreachable(cf, user) && continue
        if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst)
            walk_constant_field_base!(cf, base, user, offset)
        elseif isa(user, LLVM.GetElementPtrInst) && operands(user)[1] == ptr
            off = constant_gep_offset(user)
            if off === nothing
                push!(cf.escaped, base)
            else
                walk_constant_field_base!(cf, base, user, offset + off)
            end
        elseif isa(user, LLVM.LoadInst)
            push!(cf.loads[base], (user, offset))
        elseif isa(user, LLVM.StoreInst) && operands(user)[2] == ptr && operands(user)[1] != ptr
            push!(cf.stores[base], (user, offset))
        elseif isa(user, LLVM.CallInst)
            walk_constant_field_call!(cf, base, user, ptr, offset)
        else
            push!(cf.escaped, base)
        end
    end
    return
end

function walk_constant_field_call!(cf::ConstantFields, @nospecialize(base::LLVM.Value), call::LLVM.CallInst, @nospecialize(ptr::LLVM.Value), offset::Int)
    callee = called_operand(call)
    if isa(callee, LLVM.Function)
        intr = LLVM.API.LLVMGetIntrinsicID(callee)
        if intr == LLVM.Intrinsic("llvm.lifetime.start").id ||
                intr == LLVM.Intrinsic("llvm.lifetime.end").id
            return
        end
        # Julia 1.12 zeroes the roots array of a by-reference argument first.
        if intr == LLVM.Intrinsic("llvm.memset").id
            ops = arguments(call)
            if ops[1] == ptr && ops[2] != ptr && isa(ops[2], LLVM.ConstantInt) &&
                    convert(Int, ops[2]) == 0 && isa(ops[3], LLVM.ConstantInt)
                push!(cf.zeroed[base], (offset, convert(Int, ops[3])))
                return
            end
        end
    end
    if !isa(callee, LLVM.Function) || callee == ptr || !constant_field_callee(cf, callee)
        push!(cf.escaped, base)
        return
    end
    # `offset` is nonzero when the slot holds the struct the callee takes inline,
    # as a tuple of `Const` arguments holds a view.
    params = parameters(callee)
    for (i, arg) in enumerate(arguments(call))
        arg == ptr || continue
        constant_field_base!(cf, params[i])
        add_constant_field_edge!(cf, base, params[i], offset)
    end
    return
end

# Whether `val` is constant data: a `Const` argument of the differentiated
# function, a value read from memory reached from one, a value loaded from a
# field still assumed constant, or a merge of such values. `seen` breaks cycles
# optimistically, as befits the greatest fixpoint `mark_constant_fields!` computes.
function constant_field_data(cf::ConstantFields, @nospecialize(val::LLVM.Value), seen::Set{LLVM.Value})::Bool
    while isa(val, LLVM.BitCastInst) || isa(val, LLVM.AddrSpaceCastInst)
        val = operands(val)[1]
    end
    # A pointer field of a `Const` argument passed by value.
    if isa(val, LLVM.ExtractValueInst)
        return operands(val)[1] in cf.const_args
    end
    if isa(val, LLVM.PointerNull) || isa(val, LLVM.UndefValue) || isa(val, LLVM.PoisonValue)
        return true
    end
    val in seen && return true
    if isa(val, LLVM.Argument)
        val in cf.const_args && return true
        f = LLVM.Function(LLVM.API.LLVMGetParamParent(val))
        (f != cf.primalf && constant_field_callee(cf, f)) || return false
        push!(seen, val)
        idx = findfirst(==(val), collect(parameters(f)))::Int
        for u in LLVM.uses(f)
            call = LLVM.user(u)::LLVM.CallInst
            in_unreachable(cf, call) && continue
            constant_field_data(cf, arguments(call)[idx], seen) || return false
        end
        return true
    end
    if isa(val, LLVM.PHIInst)
        push!(seen, val)
        for (v, _) in LLVM.incoming(val)
            constant_field_data(cf, v, seen) || return false
        end
        return true
    end
    if isa(val, LLVM.SelectInst)
        push!(seen, val)
        return constant_field_data(cf, operands(val)[2], seen) &&
            constant_field_data(cf, operands(val)[3], seen)
    end
    isa(val, LLVM.LoadInst) || return false
    push!(seen, val)
    base, offset = constant_field_pointer(operands(val)[1])
    if isa(base, LLVM.AllocaInst) && haskey(metadata(base), "enzyme_inactive")
        return true
    end
    if is_constant_field_base(cf, base)
        key = constant_field_key(cf, base, offset)
        if key === nothing
            constant_field_base_escaped(cf, base) || return false
        elseif !(key in cf.escaped_fields)
            return get(cf.constant, key, false)
        end
    end
    # Memory reached from constant data (a field of a `Const` struct argument).
    return constant_field_data(cf, base, seen)
end

# The instructions of the block of `phi` that compute loads through it: casts,
# GEPs, loads of pointers and `julia.gc_loaded`, ending in loads of values that are
# not pointers, in block order. Load sinking turns `c ? base[j] : active[s]` into a
# load through a phi of the two data pointers, which `nodecayed_phis!` makes a phi
# of the two arrays: one active and one constant pointer, which has no shadow.
# Repeating the loads in each predecessor (no write may precede them) leaves a
# phi of loaded values instead, whose constant operand simply has no derivative.
# A predecessor that also branches elsewhere first gets a block of its own on the
# edge to this block (`split_constant_merges!`), so that the loads still run only
# on that edge. Under `--check-bounds=yes` the read of `base` can need no block of
# its own, and the block that picks it also branches to the bounds check of the
# read of `active`.
function constant_field_unsink_tree(phi::LLVM.PHIInst)
    bb = LLVM.parent(phi)
    for (_, pred) in LLVM.incoming(phi)
        term = terminator(pred)
        (isa(term, LLVM.BrInst) || isa(term, LLVM.SwitchInst)) || return nothing
        count(Base.Fix2(==, bb), successors(term)) == 1 || return nothing
    end
    tree = Set{LLVM.Instruction}()
    leaves = LLVM.LoadInst[]
    work = LLVM.Value[phi]
    while !isempty(work)
        v = pop!(work)
        for u in LLVM.uses(v)
            user = LLVM.user(u)
            (isa(user, LLVM.Instruction) && LLVM.parent(user) == bb) || return nothing
            user in tree && continue
            if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst) ||
                    isa(user, LLVM.GetElementPtrInst) ||
                    (isa(user, LLVM.CallInst) && called_function_name(user) == "julia.gc_loaded")
                push!(tree, user)
                push!(work, user)
            elseif isa(user, LLVM.LoadInst) && !Bool(LLVM.API.LLVMGetVolatile(user))
                push!(tree, user)
                if isa(value_type(user), LLVM.PointerType)
                    push!(work, user)
                else
                    push!(leaves, user)
                end
            else
                return nothing
            end
        end
    end
    isempty(leaves) && return nothing
    # Every other operand must be available in the predecessors.
    for inst in tree, op in operands(inst)
        op in tree && continue
        if isa(op, LLVM.Instruction) && LLVM.parent(op) == bb && !isa(op, LLVM.PHIInst)
            return nothing
        end
    end
    ordered = LLVM.Instruction[]
    written = false
    for inst in instructions(bb)
        isa(inst, LLVM.PHIInst) && continue
        if inst in tree
            written && return nothing
            push!(ordered, inst)
            length(ordered) == length(tree) && break
        elseif root_path_may_write(inst)
            written = true
        end
    end
    return ordered, leaves
end

function incoming_value(phi::LLVM.PHIInst, pred::LLVM.BasicBlock)
    for (v, b) in LLVM.incoming(phi)
        b == pred && return v
    end
    return nothing
end

function unsink_constant_field_phi!(phi::LLVM.PHIInst)::Bool
    found = constant_field_unsink_tree(phi)
    found === nothing && return false
    ordered, leaves = found
    bb = LLVM.parent(phi)
    builder = LLVM.IRBuilder()
    position!(builder, phi)
    merged = Dict{LLVM.LoadInst, LLVM.PHIInst}()
    for leaf in leaves
        merged[leaf] = phi!(builder, value_type(leaf), LLVM.name(leaf))
    end
    for (_, pred) in LLVM.incoming(phi)
        position!(builder, terminator(pred))
        copies = Dict{LLVM.Value, LLVM.Value}()
        for inst in ordered
            clone = LLVM.Instruction(LLVM.API.LLVMInstructionClone(inst))
            LLVM.API.LLVMInsertIntoBuilder(builder, clone)
            for (i, op) in enumerate(operands(clone))
                new = get(copies, op, nothing)
                if new === nothing && isa(op, LLVM.PHIInst) && LLVM.parent(op) == bb
                    new = incoming_value(op, pred)
                end
                new === nothing || LLVM.API.LLVMSetOperand(clone, i - 1, new)
            end
            copies[inst] = clone
        end
        for leaf in leaves
            push!(LLVM.incoming(merged[leaf]), (copies[leaf], pred))
        end
    end
    for leaf in leaves
        # Memory metadata such as `!tbaa` is not allowed on a phi.
        haskey(metadata(leaf), "enzyme_type") &&
            (metadata(merged[leaf])["enzyme_type"] = metadata(leaf)["enzyme_type"])
        replace_uses!(leaf, merged[leaf])
    end
    for inst in reverse(ordered)
        isempty(LLVM.uses(inst)) && LLVM.erase!(inst)
    end
    isempty(LLVM.uses(phi)) && LLVM.erase!(phi)
    dispose(builder)
    return true
end

# Whether every use of `val`, a pointer read from a constant field (or derived
# from one), stays away from active data: a load or store through it, a store
# into a constant field, a comparison, or an argument of an internal function
# whose parameter is used likewise. A phi or select that also merges active data
# must be one `split_constant_merges!` splits. Anything else would merge the
# pointer with active data where it has no shadow, so the mixed-activity handler
# would copy the whole array there, possibly in a loop, while without the tag one
# copy is made where the struct is built.
function constant_field_uses_safe(cf::ConstantFields, @nospecialize(val::LLVM.Value), visited::Set{LLVM.Value})::Bool
    val in visited && return true
    push!(visited, val)
    for u in LLVM.uses(val)
        user = LLVM.user(u)
        isa(user, LLVM.Instruction) || return false
        in_unreachable(cf, user) && continue
        ok = if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst) ||
                isa(user, LLVM.GetElementPtrInst)
            constant_field_uses_safe(cf, user, visited)
        elseif isa(user, LLVM.LoadInst)
            # A pointer read from constant memory is constant data too.
            !isa(value_type(user), LLVM.PointerType) || constant_field_uses_safe(cf, user, visited)
        elseif isa(user, LLVM.StoreInst)
            if operands(user)[1] != val
                true
            else
                base, offset = constant_field_pointer(operands(user)[2])
                constant_field_is_constant(cf, base, offset)
            end
        elseif isa(user, LLVM.PHIInst) || isa(user, LLVM.SelectInst)
            kind = constant_field_merge(cf, user)
            if kind === :constant
                constant_field_uses_safe(cf, user, visited)
            else
                # `split_constant_merges!` splits it.
                kind === :mixed && constant_merge_splittable(user)
            end
        elseif isa(user, LLVM.ICmpInst)
            true
        elseif isa(user, LLVM.PtrToIntInst)
            # Julia 1.12 compares data addresses (`mightalias`); an integer has no
            # shadow unless it is turned back into a pointer.
            never_pointer_again(user, Set{LLVM.Value}())
        elseif isa(user, LLVM.CallInst)
            constant_field_call_safe(cf, user, val, visited)
        else
            false
        end
        ok || return false
    end
    return true
end

# Whether an integer, and everything computed from it, stays an integer.
function never_pointer_again(@nospecialize(val::LLVM.Value), seen::Set{LLVM.Value})::Bool
    val in seen && return true
    push!(seen, val)
    for u in LLVM.uses(val)
        user = LLVM.user(u)
        isa(user, LLVM.Instruction) || return false
        if isa(value_type(user), LLVM.IntegerType)
            never_pointer_again(user, seen) || return false
        elseif isa(user, LLVM.GetElementPtrInst)
            # An index: the GEP's activity is its pointer operand's (the byte offsets
            # `nodecayed_phis!` computes on Julia 1.12).
            operands(user)[1] == val && return false
        elseif !(isa(user, LLVM.BrInst) || isa(user, LLVM.SwitchInst))
            return false
        end
    end
    return true
end

function constant_field_call_safe(cf::ConstantFields, call::LLVM.CallInst, @nospecialize(val::LLVM.Value), visited::Set{LLVM.Value})::Bool
    callee = called_operand(call)
    (isa(callee, LLVM.Function) && callee != val) || return false
    intr = LLVM.API.LLVMGetIntrinsicID(callee)
    if intr == LLVM.Intrinsic("llvm.lifetime.start").id ||
            intr == LLVM.Intrinsic("llvm.lifetime.end").id
        return true
    end
    nm = LLVM.name(callee)
    nm == "julia.gc_loaded" && return constant_field_uses_safe(cf, call, visited)
    nm == "julia.write_barrier" && return true
    constant_field_callee(cf, callee) || return false
    params = parameters(callee)
    for (i, arg) in enumerate(arguments(call))
        arg == val || continue
        constant_field_uses_safe(cf, params[i], visited) || return false
    end
    return true
end

# Whether `val`, a pointer, points into constant data: casts, GEPs and
# `julia.gc_loaded` of a value `constant_field_data` accepts.
function constant_field_derived(cf::ConstantFields, @nospecialize(val::LLVM.Value))::Bool
    while true
        if isa(val, LLVM.BitCastInst) || isa(val, LLVM.AddrSpaceCastInst) ||
                isa(val, LLVM.GetElementPtrInst)
            val = operands(val)[1]
        elseif isa(val, LLVM.CallInst) && called_function_name(val) == "julia.gc_loaded"
            val = operands(val)[2]
        else
            break
        end
    end
    return constant_field_data(cf, val, Set{LLVM.Value}())
end

# `:constant` if every pointer a phi or select merges is constant data, `:mixed` if
# only some are, `:active` if none is.
function constant_field_merge(cf::ConstantFields, merge::LLVM.Instruction)::Symbol
    vals = isa(merge, LLVM.PHIInst) ? LLVM.Value[v for (v, _) in LLVM.incoming(merge)] :
        LLVM.Value[operands(merge)[2], operands(merge)[3]]
    nconst = 0
    for v in vals
        constant_field_derived(cf, v) && (nconst += 1)
    end
    nconst == length(vals) && return :constant
    return nconst == 0 ? :active : :mixed
end

# The one load through `sel`, a select of pointers, right in its block: not
# volatile or atomic, and of a value that is not a pointer.
function constant_select_load(sel::LLVM.SelectInst)::Union{Nothing, LLVM.LoadInst}
    isa(value_type(sel), LLVM.PointerType) || return nothing
    uses = collect(LLVM.uses(sel))
    length(uses) == 1 || return nothing
    ld = LLVM.user(only(uses))
    (isa(ld, LLVM.LoadInst) && LLVM.parent(ld) == LLVM.parent(sel)) || return nothing
    (Bool(LLVM.API.LLVMGetVolatile(ld)) || LLVM.is_atomic(ld)) && return nothing
    isa(value_type(ld), LLVM.PointerType) && return nothing
    return ld
end

constant_merge_splittable(phi::LLVM.PHIInst) = constant_field_unsink_tree(phi) !== nothing
constant_merge_splittable(sel::LLVM.SelectInst) = constant_select_load(sel) !== nothing

# Rebuild the phis of `succ` with `to` as the predecessor that was `from` (the C
# API cannot set the block of an incoming value).
function retarget_phis!(builder::LLVM.IRBuilder, succ::LLVM.BasicBlock, from::LLVM.BasicBlock, to::LLVM.BasicBlock)
    for phi in collect(instructions(succ))
        isa(phi, LLVM.PHIInst) || break
        incs = collect(LLVM.incoming(phi))
        any(Base.Fix2(==, from) ∘ last, incs) || continue
        position!(builder, phi)
        new = phi!(builder, value_type(phi), LLVM.name(phi))
        for (v, b) in incs
            push!(LLVM.incoming(new), (v, b == from ? to : b))
        end
        for k in ("enzyme_type", "enzyme_inactive", "enzyme_active")
            haskey(metadata(phi), k) && (metadata(new)[k] = metadata(phi)[k])
        end
        replace_uses!(phi, new)
        LLVM.erase!(phi)
    end
    return
end

# `load(select(c, p, q))` becomes `c ? load(p) : load(q)`: a branch with one load
# on each edge and a phi of the two values, as before LLVM folded it into a select
# of pointers. Loading both would read `q` where the program does not, out of
# bounds for `iszero(s) ? base[j] : active[s]` when `s` is 0.
function unselect_constant_load!(sel::LLVM.SelectInst)::Bool
    ld = constant_select_load(sel)
    ld === nothing && return false
    bb = LLVM.parent(ld)
    f = LLVM.parent(bb)
    c, p, q = operands(sel)
    insts = collect(instructions(bb))
    tail = insts[(findfirst(Base.Fix2(==, ld), insts)::Int + 1):end]
    onp = LLVM.BasicBlock(f, "unselect.p")
    onq = LLVM.BasicBlock(f, "unselect.q")
    join = LLVM.BasicBlock(f, "unselect.join")
    builder = LLVM.IRBuilder()
    position!(builder, join)
    for inst in tail
        LLVM.API.LLVMInstructionRemoveFromParent(inst)
        LLVM.API.LLVMInsertIntoBuilder(builder, inst)
    end
    for succ in unique(collect(successors(terminator(join))))
        retarget_phis!(builder, succ, bb, join)
    end
    position!(builder, bb)
    br!(builder, c, onp, onq)
    loaded = LLVM.Instruction[]
    for (blk, ptr) in ((onp, p), (onq, q))
        position!(builder, blk)
        v = load!(builder, value_type(ld), ptr)
        alignment!(v, alignment(ld))
        copy_metadata!(v, ld)
        push!(loaded, v)
        br!(builder, join)
    end
    position!(builder, first(instructions(join)))
    merged = phi!(builder, value_type(ld), LLVM.name(ld))
    push!(LLVM.incoming(merged), (loaded[1], onp))
    push!(LLVM.incoming(merged), (loaded[2], onq))
    haskey(metadata(ld), "enzyme_type") && (metadata(merged)["enzyme_type"] = metadata(ld)["enzyme_type"])
    replace_uses!(ld, merged)
    LLVM.erase!(ld)
    LLVM.erase!(sel)
    dispose(builder)
    return true
end

split_constant_merge!(phi::LLVM.PHIInst) = unsink_constant_field_phi!(phi)
split_constant_merge!(sel::LLVM.SelectInst) = unselect_constant_load!(sel)

# Every phi or select that merges constant data with other pointers and only feeds
# loads, which `split_constant_merge!` can split.
function constant_merges(mod::LLVM.Module, cf::ConstantFields)::Vector{LLVM.Instruction}
    todo = LLVM.Instruction[]
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        (isa(inst, LLVM.PHIInst) || isa(inst, LLVM.SelectInst)) || continue
        isa(value_type(inst), LLVM.PointerType) || continue
        in_unreachable(cf, inst) && continue
        constant_field_merge(cf, inst) === :mixed || continue
        constant_merge_splittable(inst) && push!(todo, inst)
    end
    return todo
end

# Put a block of its own on the edge from `pred`, which also branches elsewhere, to
# `succ`.
function split_edge!(builder::LLVM.IRBuilder, pred::LLVM.BasicBlock, succ::LLVM.BasicBlock)
    edge = LLVM.BasicBlock(LLVM.parent(pred), "unsink.edge")
    term = terminator(pred)
    for i in 0:(Int(LLVM.API.LLVMGetNumSuccessors(term)) - 1)
        if LLVM.BasicBlock(LLVM.API.LLVMGetSuccessor(term, i)) == succ
            LLVM.API.LLVMSetSuccessor(term, i, edge)
        end
    end
    position!(builder, edge)
    br!(builder, succ)
    retarget_phis!(builder, succ, pred, edge)
    return
end

# Split every phi or select that merges constant data with other pointers and only
# feeds loads, so that its constant operand needs no shadow: the mixed-activity
# handler would otherwise copy the whole constant array there on every execution,
# or raise `EnzymeRuntimeActivityError` for a pointer into it.
function split_constant_merges!(mod::LLVM.Module, cf::ConstantFields)
    todo = constant_merges(mod, cf)
    # Splitting an edge rebuilds the phis of the block it leads to, so the merges
    # are collected again afterwards.
    edges = Tuple{LLVM.BasicBlock, LLVM.BasicBlock}[]
    for merge in todo
        isa(merge, LLVM.PHIInst) || continue
        bb = LLVM.parent(merge)
        for (_, pred) in LLVM.incoming(merge)
            length(collect(successors(terminator(pred)))) == 1 && continue
            (pred, bb) in edges || push!(edges, (pred, bb))
        end
    end
    if !isempty(edges)
        builder = LLVM.IRBuilder()
        for (pred, succ) in edges
            split_edge!(builder, pred, succ)
        end
        dispose(builder)
        todo = constant_merges(mod, cf)
    end
    for merge in todo
        split_constant_merge!(merge)
    end
    return length(todo)
end

# Drop a field once one of its stores writes data that is not constant, or one of
# its loads is used unsafely, under the current assumptions.
function settle_constant_fields!(cf::ConstantFields)
    changed = true
    while changed
        changed = false
        for (base, ss) in cf.stores
            for (st, off) in ss
                key = constant_field_key(cf, base, off)
                (key !== nothing && get(cf.constant, key, false)) || continue
                if !constant_field_data(cf, operands(st)[1], Set{LLVM.Value}())
                    cf.constant[key] = false
                    changed = true
                end
            end
        end
        changed && continue
        for (base, ls) in cf.loads
            for (ld, off) in ls
                key = constant_field_key(cf, base, off)
                (key !== nothing && get(cf.constant, key, false)) || continue
                if !constant_field_uses_safe(cf, ld, Set{LLVM.Value}())
                    cf.constant[key] = false
                    changed = true
                end
            end
        end
    end
    return
end

function mark_constant_fields!(mod::LLVM.Module, primalf::LLVM.Function, @nospecialize(args_activity::Vector{API.CDIFFE_TYPE}))
    # A recursive call could pass active data where the outer call passes `Const`.
    isempty(LLVM.uses(primalf)) || return 0
    const_args = Set{LLVM.Value}()
    for (p, act) in zip(parameters(primalf), args_activity)
        act == API.DFT_CONSTANT && push!(const_args, p)
    end
    isempty(const_args) && return 0
    cf = ConstantFields(primalf, const_args)

    for f in functions(mod)
        isempty(blocks(f)) && continue
        if f != primalf && constant_field_callee(cf, f)
            for (i, p) in enumerate(parameters(f))
                isa(value_type(p), LLVM.PointerType) || continue
                constant_field_base!(cf, p)
                # Every call site must pass a pointer into a slot this pass follows.
                for u in LLVM.uses(f)
                    call = LLVM.user(u)::LLVM.CallInst
                    in_unreachable(cf, call) && continue
                    actual, off = constant_field_pointer(arguments(call)[i])
                    if isa(actual, LLVM.AllocaInst) || isa(actual, LLVM.Argument)
                        constant_field_base!(cf, actual)
                        add_constant_field_edge!(cf, actual, p, off)
                    else
                        push!(cf.escaped, p)
                    end
                end
            end
        end
        for bb in blocks(f), inst in instructions(bb)
            isa(inst, LLVM.AllocaInst) && constant_field_base!(cf, inst)
        end
    end
    for base in collect(keys(cf.edges))
        constant_field_places(cf, base)
    end
    # Memory a base escapes to is reachable from every place of the base.
    for base in collect(cf.escaped)
        for (root, _) in constant_field_places(cf, base)
            push!(cf.escaped, root)
        end
    end

    # Map every access into the roots: one place per place of its base, all of them
    # in one class. Offsets are bytes into the root.
    accesses = Dict{LLVM.Value, Vector{Tuple{LLVM.Instruction, Int}}}()
    zeroed = Dict{LLVM.Value, Vector{Tuple{Int, Int}}}()
    for (base, ls) in cf.loads
        places = constant_field_places(cf, base)
        for (inst, off) in Iterators.flatten((ls, cf.stores[base]))
            first = nothing
            for (root, delta) in places
                field = (root, off + delta)
                get!(cf.field_parent, field, field)
                if first === nothing
                    first = field
                else
                    constant_field_union!(cf, first, field)
                end
                push!(get!(accesses, root, Tuple{LLVM.Instruction, Int}[]), (inst, off + delta))
            end
        end
        for (off, len) in cf.zeroed[base], (root, delta) in places
            push!(get!(zeroed, root, Tuple{Int, Int}[]), (off + delta, len))
        end
    end
    for field in collect(keys(cf.field_parent))
        field[1] in cf.escaped && push!(cf.escaped_fields, constant_field_find(cf, field))
    end

    # Candidate places: pointer-sized tracked-pointer accesses at one offset of a root
    # that does not escape, with no other access overlapping it.
    candidates = Set{ConstantField}()
    dl = datalayout(mod)
    for (root, acc) in accesses
        root in cf.escaped && continue
        fields = Dict{Int, Bool}()
        for (inst, off) in acc
            ty = isa(inst, LLVM.LoadInst) ? value_type(inst) : value_type(operands(inst)[1])
            if is_tracked_pointer(ty)
                get!(fields, off, true)
            end
        end
        for (inst, off) in acc
            ty = isa(inst, LLVM.LoadInst) ? value_type(inst) : value_type(operands(inst)[1])
            sz = Int(LLVM.storage_size(dl, ty))
            for (foff, ok) in fields
                ok || continue
                overlaps = off < foff + sizeof(Int) && foff < off + sz
                if overlaps && !(off == foff && is_tracked_pointer(ty))
                    fields[foff] = false
                end
            end
        end
        for (off, len) in get(zeroed, root, Tuple{Int, Int}[])
            for (foff, ok) in fields
                ok || continue
                overlaps = off < foff + sizeof(Int) && foff < off + len
                covers = off <= foff && foff + sizeof(Int) <= off + len
                if overlaps && !covers
                    fields[foff] = false
                end
            end
        end
        for (foff, ok) in fields
            ok && push!(candidates, (root, foff))
        end
    end
    # A class is a candidate field when every place in it is.
    for field in collect(keys(cf.field_parent))
        class = constant_field_find(cf, field)
        if field in candidates && !(class in cf.escaped_fields)
            get!(cf.constant, class, true)
        else
            cf.constant[class] = false
        end
    end
    # A greatest fixpoint.
    settle_constant_fields!(cf)
    split_constant_merges!(mod, cf)

    marked = 0
    for (base, ls) in cf.loads
        for (ld, off) in ls
            constant_field_is_constant(cf, base, off) || continue
            md = metadata(ld)
            haskey(md, "enzyme_inactive") && continue
            md["enzyme_inactive"] = inactive_md(INACTIVE_CONSTANT_FIELD)
            marked += 1
        end
    end
    for (base, ss) in cf.stores
        for (st, off) in ss
            constant_field_is_constant(cf, base, off) || continue
            metadata(st)[CONSTANT_FIELD_STORE] = MDNode(LLVM.Metadata[])
        end
    end
    return marked
end
