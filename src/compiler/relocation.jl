# Symbolic references to Julia values from generated code.
#
# Generated code refers to a Julia object through a named global `ejl_v_<id>` whose address
# is the object: the form Enzyme already uses for the values of `JuliaGlobalNameMap` and
# `JuliaEnzymeNameMap`. `RELOC_TARGETS` maps the name to the object, and the JIT binds the
# name to the object's address only when the module is linked (`JIT.prepare!`), so the module
# Enzyme emits carries no address of this session. Targets are rooted for the lifetime of the
# process by `jl_as_global_root`, which also canonicalizes egal immutables, so equal values
# share one name. A `Core.Binding` target stands for the value of the binding; a
# `BindingObject` target for the binding itself.

# A `Core.Binding` as an object: what Julia's codegen refers to when the code reads a
# global through the runtime (`jl_get_binding_value_seqcst` takes the binding, not its
# value). Serializable, since bindings are.
struct BindingObject
    binding::Core.Binding
end

const RELOC_LOCK = ReentrantLock()
const RELOC_TARGETS = Dict{String, Any}()
const RELOC_PREFIX = "ejl_v_"

# Root `val` for the lifetime of the process and return its canonical rooted instance, as
# Julia's own codegen does for values referenced from native code.
function root_value(@nospecialize(val))
    @static if VERSION >= v"1.11-"
        return ccall(:jl_as_global_root, Any, (Any, Cint), val, 1)
    else
        return ccall(:jl_as_global_root, Any, (Any,), val)
    end
end

value_pointer(@nospecialize(val)) = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), val)

# The global name standing for `val`, registering it on first use.
function relocation_name(@nospecialize(val))::String
    val = root_value(val)
    base = RELOC_PREFIX * string(objectid(val); base = 16)
    lock(RELOC_LOCK)
    return try
        name = base
        i = 0
        while true
            prev = get(RELOC_TARGETS, name, nothing)
            if prev === nothing && !haskey(RELOC_TARGETS, name)
                RELOC_TARGETS[name] = val
                return name
            elseif prev === val
                return name
            end
            i += 1
            name = base * "_" * string(i)
        end
    finally
        unlock(RELOC_LOCK)
    end
end

# Register `target` under a name computed elsewhere (e.g. from an artifact's manifest, in a
# session that did not emit the module). Idempotent for an equal target.
function register_relocation!(name::String, @nospecialize(target))
    target = (target isa Core.Binding || target isa BindingObject) ? target : root_value(unbind(target))
    lock(RELOC_LOCK)
    try
        prev = get(RELOC_TARGETS, name, nothing)
        if prev === nothing && !haskey(RELOC_TARGETS, name)
            RELOC_TARGETS[name] = target
        end
    finally
        unlock(RELOC_LOCK)
    end
    return name
end

is_relocation_name(gname::AbstractString) = startswith(gname, RELOC_PREFIX)

# `(true, target)` for a registered global name, `(false, nothing)` otherwise.
function relocation_target(gname::AbstractString)::Tuple{Bool, Any}
    is_relocation_name(gname) || return (false, nothing)
    lock(RELOC_LOCK)
    try
        if haskey(RELOC_TARGETS, gname)
            return (true, RELOC_TARGETS[gname])
        end
    finally
        unlock(RELOC_LOCK)
    end
    return (false, nothing)
end

# The value a registered global denotes: a binding's target is the binding's value.
function relocation_value(gname::AbstractString)::Tuple{Bool, Any}
    found, target = relocation_target(gname)
    found || return (false, nothing)
    target isa BindingObject && return (true, target.binding)
    return (true, unbind(target))
end

# The address the global `gname` must be bound to in this process.
function relocation_pointer(gname::AbstractString)::Ptr{Cvoid}
    found, target = relocation_target(gname)
    found || error("$gname is not a registered Julia value reference")
    target isa BindingObject && return value_pointer(target.binding)
    return value_pointer(root_value(unbind(target)))
end

"""
    manifest(mod::LLVM.Module) -> Vector{Pair{String, Any}}

The Julia values `mod` refers to symbolically, by global name.
"""
function manifest(mod::LLVM.Module)
    m = Pair{String, Any}[]
    for g in globals(mod)
        gname = LLVM.name(g)
        found, target = relocation_target(gname)
        found && push!(m, gname => target)
    end
    return m
end

# An address baked into the code as an integer constant. Anything at or above this is taken
# to be a pointer of this process rather than a genuine small integer.
const MIN_BAKED_ADDRESS = UInt(1) << 16

# Whether `v` is (or contains) an `inttoptr` of a process address.
function has_baked_address(@nospecialize(v::LLVM.Value), seen::Base.IdSet{LLVM.Value})::Bool
    isa(v, LLVM.ConstantExpr) || return false
    v in seen && return false
    push!(seen, v)
    if opcode(v) == LLVM.API.LLVMIntToPtr
        arg = operands(v)[1]
        if isa(arg, LLVM.ConstantInt) && convert(UInt, arg) >= MIN_BAKED_ADDRESS
            return true
        end
    end
    for op in operands(v)
        has_baked_address(op, seen) && return true
    end
    return false
end

"""
    bakes_addresses(mod::LLVM.Module) -> Bool

Whether `mod` refers to anything by an address of this process. Such a module cannot be
reused by another session: the addresses are of Julia objects Julia's own codegen embedded
(the `:bake` relocation strategy), of C functions no name could be found for, or of values
folded from a constant. They are frequently on paths a happy-path test never runs, so this
is checked rather than inferred.
"""
function bakes_addresses(mod::LLVM.Module)::Bool
    seen = Base.IdSet{LLVM.Value}()
    for g in globals(mod)
        init = LLVM.initializer(g)
        init === nothing || has_baked_address(init, seen) && return true
    end
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        for op in operands(inst)
            has_baked_address(op, seen) && return true
        end
    end
    return false
end

# Whether a single value can be reconstructed in another process from its serialized form.
function persistable_value(@nospecialize(v))::Bool
    v isa BindingObject && return true
    v = unbind(v)
    return v isa Type || v isa Symbol || v isa String || v isa Module ||
        v isa Core.MethodInstance || v isa Core.CodeInstance || v isa Method ||
        (isbits(v) && !(v isa Ptr))
end

# Whether every target can be reconstructed in another process from its serialized form.
function persistable(m::AbstractVector{<:Pair{String}})::Bool
    for (_, target) in m
        persistable_value(target) || return false
    end
    return true
end

# Adopting GPUCompiler's relocation metadata for the primal module.
#
# Julia's codegen refers to a Julia object through a word-sized slot global. GPUCompiler's
# default `:bake` strategy fills that slot with the object's address, which puts an address
# of this process into everything Enzyme emits. Enzyme asks for `:patch` instead
# (`relocation_lowering`), which leaves the slots empty and hands back a manifest, and then
# gives each slot an initializer that names the object (the `ejl_v_*` form of
# `unsafe_to_llvm`) rather than an address. The slot and the loads through it are untouched,
# so the shape Enzyme's activity and type analyses see is exactly the one they saw before;
# only the address is gone. `JIT.prepare!` binds the name when the module is linked, and
# `absint`/`try_replace_constant_load!` read the object back out of the registry so that
# constant folding still works (see `relocation_slot_value`).
@static if isdefined(GPUCompiler, :Relocations)

    # The global whose *address* is the Julia object `val`, as `unsafe_to_llvm` emits it.
    function relocation_global!(mod::LLVM.Module, @nospecialize(val))::LLVM.GlobalVariable
        name = relocation_name(val)
        globs = globals(mod)
        haskey(globs, name) && return globs[name]
        gv = LLVM.GlobalVariable(mod, LLVM.StructType(LLVM.LLVMType[]), name, Tracked)
        API.SetMD(gv, "enzyme_ta_norecur", LLVM.MDNode(LLVM.Metadata[]))
        # Julia emits these slots only for compile-time-constant objects, which Enzyme has
        # always treated as constants: the folded form in `try_replace_constant_load!` marks
        # them inactive too. Without it a reference by name would ask for a shadow global
        # that the primal has none of, where the address form asked for nothing.
        API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
        return gv
    end

    # `gv` (the named global whose address is a Julia object) as a constant of type `T`. An
    # integer word goes through the generic address space: the tracked one is non-integral,
    # and the word was a plain address before GPUCompiler recorded the site.
    function relocation_constant(gv::LLVM.GlobalVariable, T::LLVM.LLVMType)
        if !(T isa LLVM.PointerType)
            generic = LLVM.const_addrspacecast(gv, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[])))
            return LLVM.const_ptrtoint(generic, T)
        end
        LLVM.addrspace(T) == LLVM.addrspace(value_type(gv)) && return LLVM.const_pointercast(gv, T)
        return LLVM.const_addrspacecast(gv, T)
    end

    # Replace every load through `v` (the slot, or a constant cast of it) by what
    # `produce(builder, T)` gives for the type `T` the loaded word is used as, positioned
    # at the load. GPUCompiler records a site by loading the slot as an integer word and
    # converting it back to the pointer the original code loaded (`inttoptr`); each such
    # conversion is replaced as the pointer it produces, so what the module sees is the
    # form Julia's codegen emitted, never an integer image of an object's address.
    function replace_slot_loads!(produce, @nospecialize(v::LLVM.Value), name::String)::Nothing
        for use in collect(LLVM.uses(v))
            u = LLVM.user(use)
            if u isa LLVM.LoadInst
                b = IRBuilder()
                position!(b, u)
                for luse in collect(LLVM.uses(u))
                    conv = LLVM.user(luse)
                    conv isa LLVM.IntToPtrInst || continue
                    replace_uses!(conv, produce(b, value_type(conv)))
                    LLVM.API.LLVMInstructionEraseFromParent(conv)
                end
                isempty(LLVM.uses(u)) || replace_uses!(u, produce(b, value_type(u)))
                dispose(b)
                LLVM.API.LLVMInstructionEraseFromParent(u)
            elseif u isa LLVM.ConstantExpr && opcode(u) in (LLVM.API.LLVMBitCast, LLVM.API.LLVMAddrSpaceCast)
                replace_slot_loads!(produce, u, name)
            elseif u isa LLVM.StoreInst
                error("Enzyme: relocation slot $name is written to")
            end
        end
        return nothing
    end

    # Produces, for a slot holding a Julia object, that object by name.
    struct SlotObject
        gv::LLVM.GlobalVariable
    end
    (p::SlotObject)(b::IRBuilder, T::LLVM.LLVMType) = relocation_constant(p.gv, T)

    # Produces, for a slot holding the word at `offset` from a C data global, a load of that
    # word from the global itself, which the JIT resolves by name in any session (the form
    # Julia's codegen emitted before GPUCompiler recorded the site).
    struct SlotCGlobal
        g::LLVM.Value
        offset::Int
    end
    function (p::SlotCGlobal)(b::IRBuilder, T::LLVM.LLVMType)
        T_i8 = LLVM.Int8Type()
        addr = LLVM.const_pointercast(p.g, LLVM.PointerType(T_i8, LLVM.addrspace(value_type(p.g))))
        if p.offset != 0
            addr = inbounds_gep!(b, T_i8, addr, [LLVM.ConstantInt(Int64(p.offset))])
        end
        addr = pointercast!(b, addr, LLVM.PointerType(T, LLVM.addrspace(value_type(addr))))
        return load!(b, T, addr)
    end

    # The C data global `sym` as declared in `mod`, declaring it if the module lost it.
    function cglobal_declaration!(mod::LLVM.Module, sym::String)::LLVM.Value
        globs = globals(mod)
        haskey(globs, sym) && return globs[sym]
        fns = functions(mod)
        haskey(fns, sym) && return fns[sym]
        return LLVM.GlobalVariable(mod, LLVM.Int8Type(), sym)
    end

    # Drop `slot` from the `llvm.compiler.used` and `llvm.used` lists of `mod`, which keep
    # the slots alive for a loader that never comes.
    function forget_slot!(mod::LLVM.Module, slot::LLVM.GlobalVariable)
        globs = globals(mod)
        for lname in ("llvm.compiler.used", "llvm.used")
            haskey(globs, lname) || continue
            used = globs[lname]
            init = LLVM.initializer(used)
            init isa LLVM.ConstantArray || continue
            kept = LLVM.Constant[]
            for i in 1:length(operands(init))
                el = operands(init)[i]
                base, _ = get_base_and_offset(el; offsetAllowed = false, inttoptr = false)
                base == slot || push!(kept, el)
            end
            length(kept) == length(operands(init)) && continue
            LLVM.API.LLVMDeleteGlobal(used)
            isempty(kept) && continue
            T_el = value_type(kept[1])
            newused = LLVM.GlobalVariable(mod, LLVM.ArrayType(T_el, length(kept)), lname)
            LLVM.initializer!(newused, LLVM.ConstantArray(T_el, kept))
            linkage!(newused, LLVM.API.LLVMAppendingLinkage)
            LLVM.section!(newused, "llvm.metadata")
        end
        return nothing
    end

    # Retire a slot whose loads are gone. A use that is not a load takes the slot's address
    # (Julia selects between addresses of type-tag words, say, and loads through the
    # choice); for a C data global that address is the global's own, for a Julia object a
    # word holding the object is kept: this module's own copy, named for the object, so that
    # linking modules together never leaves a slot to resolve.
    function retire_slot!(mod::LLVM.Module, slot::LLVM.GlobalVariable, gv::LLVM.GlobalVariable)
        forget_slot!(mod, slot)
        if isempty(LLVM.uses(slot))
            LLVM.API.LLVMDeleteGlobal(slot)
            return nothing
        end
        name = "ejl_slot\$" * LLVM.name(gv)
        globs = globals(mod)
        if haskey(globs, name)
            replace_uses!(slot, LLVM.const_pointercast(globs[name], value_type(slot)))
            LLVM.API.LLVMDeleteGlobal(slot)
            return nothing
        end
        LLVM.initializer!(slot, relocation_constant(gv, LLVM.global_value_type(slot)))
        linkage!(slot, LLVM.API.LLVMInternalLinkage)
        LLVM.name!(slot, name)
        return nothing
    end

    function retire_slot!(mod::LLVM.Module, slot::LLVM.GlobalVariable, addr::LLVM.Value, ::Nothing)
        forget_slot!(mod, slot)
        isempty(LLVM.uses(slot)) || replace_uses!(slot, LLVM.const_pointercast(addr, value_type(slot)))
        LLVM.API.LLVMDeleteGlobal(slot)
        return nothing
    end

    """
        adopt_relocations!(mod, relocations)

    Replace every load through a relocation slot in `mod` by what the slot stands for, and
    drop the slot: a Julia object becomes a reference to it by name (`ejl_v_*`), the word of
    a C data global (`jl_nothing`, an entry of `jl_small_typeof`, ...) becomes a load from
    that global by its own name. GPUCompiler names a slot for the job it was emitted by, so
    the same object gets differently named slots in the modules Enzyme links together (a
    differentiated function and the rule bodies or runtime functions emitted for it); the
    names used here are the same everywhere, so nothing is left for a later link to resolve.
    """
    function adopt_relocations!(mod::LLVM.Module, relocations)::Nothing
        relocations === nothing && return nothing
        for rec in relocations.records
            rec.kind === GPUCompiler.SlotSite || continue
            target = rec.target
            globs = globals(mod)
            haskey(globs, rec.name) || continue
            slot = globs[rec.name]
            cur = LLVM.initializer(slot)
            (cur === nothing || LLVM.isnull(cur)) || continue
            if target isa GPUCompiler.JuliaValueRef
                val = target.value
                # The code reads the global through the runtime: it needs the binding itself.
                val isa Core.Binding && (val = BindingObject(val))
                gv = relocation_global!(mod, val)
                replace_slot_loads!(SlotObject(gv), slot, rec.name)
                retire_slot!(mod, slot, gv)
            elseif target isa GPUCompiler.CGlobalRef
                target.library === nothing || continue
                g = cglobal_declaration!(mod, String(target.symbol))
                replace_slot_loads!(SlotCGlobal(g, target.offset), slot, rec.name)
                T_i8 = LLVM.Int8Type()
                addr = LLVM.const_pointercast(g, LLVM.PointerType(T_i8, LLVM.addrspace(value_type(g))))
                if target.offset != 0
                    addr = LLVM.const_inbounds_gep(T_i8, addr, [LLVM.ConstantInt(Int64(target.offset))])
                end
                retire_slot!(mod, slot, addr, nothing)
            end
        end
        return nothing
    end

    # The Julia object a slot initialized by `adopt_relocations!` points at, or
    # `(false, nothing)`: the initializer names the object instead of giving its address, so
    # constant folding reads it out of the registry.
    function relocation_slot_value(@nospecialize(init::LLVM.Value))::Tuple{Bool, Any}
        gv, _ = get_base_and_offset(init; offsetAllowed = false, inttoptr = true)
        isa(gv, LLVM.GlobalVariable) || return (false, nothing)
        return relocation_value(LLVM.name(gv))
    end
else
    adopt_relocations!(mod::LLVM.Module, relocations)::Nothing = nothing
    relocation_slot_value(@nospecialize(init::LLVM.Value))::Tuple{Bool, Any} = (false, nothing)
end
