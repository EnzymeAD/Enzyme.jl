# The Julia type of a pointer reloaded from a stack slot.
#
# A `Const` argument passed by value and stored into a stack slot, such as the
# tuple a vararg method builds of its arguments, is read back from the slot by
# the callee inlined into the calling-convention wrapper. The callee's safepoint
# and its fences sit between the stores and the loads, so GVN leaves the loads in
# place, and `abs_typeof` cannot type a value loaded from a slot that carries no
# Julia type. The mixed-activity handler then cannot see that such a value, for
# instance a `Vector{Int}` of row indices, holds no derivative. It raises
# `EnzymeRuntimeActivityError` where the value meets active data: stored into the
# by-reference copy of a `view` of an active array, or merged with Base's
# `unalias` copy of it.
#
# When every write into a pointer field of a slot stores values that `abs_typeof`
# gives one Julia type, or clears it (a zero `memset`, or a stored null pointer),
# and no code this pass does not see can write into the slot, each load of the
# field returns a value of that type or null. The loads are tagged
# with it (`SLOT_FIELD_TYPE`, read by `abs_typeof`). A call may read the slot
# through argument positions that are `readonly` and `nocapture`; any other use
# of the slot leaves its loads untyped.

# Record the loads and writes of the slot memory that `ptr` points `offset` bytes
# into. `writes` holds (offset, length, stored value), with `nothing` for a write
# of unknown values and `false` for a zero `memset` or a stored zero constant.
# Returns false if code this walk does not see could write into the slot.
function slot_field_accesses!(
        loads::Vector{Tuple{LLVM.LoadInst, Int}},
        writes::Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}},
        @nospecialize(ptr::LLVM.Value), offset::Int, dl::LLVM.DataLayout,
    )::Bool
    for u in LLVM.uses(ptr)
        user = LLVM.user(u)
        if isa(user, LLVM.BitCastInst) || isa(user, LLVM.AddrSpaceCastInst)
            slot_field_accesses!(loads, writes, user, offset, dl) || return false
        elseif isa(user, LLVM.GetElementPtrInst) && operands(user)[1] == ptr
            off = constant_gep_offset(user)
            off === nothing && return false
            slot_field_accesses!(loads, writes, user, offset + off, dl) || return false
        elseif isa(user, LLVM.LoadInst)
            push!(loads, (user, offset))
        elseif isa(user, LLVM.StoreInst) && operands(user)[2] == ptr && operands(user)[1] != ptr
            val = operands(user)[1]
            len = Int(LLVM.storage_size(dl, value_type(val)))
            # A stored null pointer (or any all-zero constant) clears the field,
            # as a zero `memset` does.
            push!(writes, (offset, len, isa(val, LLVM.Constant) && LLVM.isnull(val) ? false : val))
        elseif isa(user, LLVM.CallInst)
            slot_field_call!(writes, user, ptr, offset) || return false
        else
            return false
        end
    end
    return true
end

function slot_field_call!(
        writes::Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}},
        call::LLVM.CallInst, @nospecialize(ptr::LLVM.Value), offset::Int,
    )::Bool
    callee = called_operand(call)
    callee == ptr && return false
    args = arguments(call)
    if isa(callee, LLVM.Function)
        intr = LLVM.API.LLVMGetIntrinsicID(callee)
        if intr == LLVM.Intrinsic("llvm.lifetime.start").id ||
                intr == LLVM.Intrinsic("llvm.lifetime.end").id
            return true
        end
        if intr == LLVM.Intrinsic("llvm.memset").id
            (args[1] == ptr && args[2] != ptr && isa(args[3], LLVM.ConstantInt)) || return false
            zero = isa(args[2], LLVM.ConstantInt) && convert(Int, args[2]) == 0
            push!(writes, (offset, convert(Int, args[3]), zero ? false : nothing))
            return true
        end
        if intr == LLVM.Intrinsic("llvm.memcpy").id || intr == LLVM.Intrinsic("llvm.memmove").id
            # Copying out of the slot only reads it.
            if args[1] == ptr
                isa(args[3], LLVM.ConstantInt) || return false
                push!(writes, (offset, convert(Int, args[3]), nothing))
            end
            return true
        end
    end
    positions = Int[i for (i, a) in enumerate(args) if a == ptr]
    isempty(positions) && return false
    decay_args_readonly(call, callee, positions) || return false
    for i in positions
        attrs = collect(argument_attributes(call, i))
        if isa(callee, LLVM.Function)
            append!(attrs, collect(parameter_attributes(callee, i)))
        end
        nocapture = false
        for attr in attrs
            kind(attr) == NOCAPTURE_ATTR_KIND && (nocapture = true)
        end
        nocapture || return false
    end
    return true
end

# The Julia type and representation of the values a pointer field `off` bytes
# into a slot holds, or `nothing`.
function slot_field_type(
        writes::Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}}, off::Int,
        enzyme_ctx::Union{EnzymeContext, Nothing},
    )
    found = nothing
    for (woff, len, val) in writes
        (woff < off + sizeof(Int) && off < woff + len) || continue
        if val === false
            # A null pointer, from a zero `memset` or a zero store that covers
            # the field.
            (woff <= off && off + sizeof(Int) <= woff + len) || return nothing
            continue
        end
        (val isa LLVM.Value && woff == off && len == sizeof(Int)) || return nothing
        legal, T, byref = abs_typeof(val, enzyme_ctx)
        legal || return nothing
        (found === nothing || found == (T, byref)) || return nothing
        found = (T, byref)
    end
    return found
end

function type_slot_field_loads!(mod::LLVM.Module, enzyme_ctx::Union{EnzymeContext, Nothing})
    dl = datalayout(mod)
    slots = Tuple{Vector{Tuple{LLVM.LoadInst, Int}}, Vector{Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}}}[]
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        isa(inst, LLVM.AllocaInst) || continue
        loads = Tuple{LLVM.LoadInst, Int}[]
        writes = Tuple{Int, Int, Union{Nothing, Bool, LLVM.Value}}[]
        slot_field_accesses!(loads, writes, inst, 0, dl) || continue
        pointers = Tuple{LLVM.LoadInst, Int}[]
        for (ld, off) in loads
            is_tracked_pointer(value_type(ld)) && push!(pointers, (ld, off))
        end
        isempty(pointers) || push!(slots, (pointers, writes))
    end
    # A value stored into one slot may itself be loaded from another.
    typed = 0
    changed = true
    while changed
        changed = false
        for (loads, writes) in slots, (ld, off) in loads
            haskey(metadata(ld), SLOT_FIELD_TYPE) && continue
            found = slot_field_type(writes, off, enzyme_ctx)
            found === nothing && continue
            T, byref = found
            metadata(ld)[SLOT_FIELD_TYPE] = MDNode(
                LLVM.Metadata[
                    MDString(string(convert(UInt, unsafe_to_pointer(T)))),
                    MDString(string(Int(byref))),
                ]
            )
            typed += 1
            changed = true
        end
    end
    return typed
end
