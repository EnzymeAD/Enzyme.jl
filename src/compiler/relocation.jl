# Symbolic references to Julia values from generated code.
#
# Generated code refers to a Julia object through a named global `ejl_v_<id>` whose address
# is the object: the form Enzyme already uses for the values of `JuliaGlobalNameMap` and
# `JuliaEnzymeNameMap`. `RELOC_TARGETS` maps the name to the object, and the JIT binds the
# name to the object's address only when the module is linked (`JIT.prepare!`), so the module
# Enzyme emits carries no address of this session. Targets are rooted for the lifetime of the
# process by `jl_as_global_root`, which also canonicalizes egal immutables, so equal values
# share one name. A `Core.Binding` target stands for the value of the binding.

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
    target = target isa Core.Binding ? target : root_value(unbind(target))
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
    return (true, unbind(target))
end

# The address the global `gname` must be bound to in this process.
function relocation_pointer(gname::AbstractString)::Ptr{Cvoid}
    found, target = relocation_target(gname)
    found || error("$gname is not a registered Julia value reference")
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

    """
        adopt_relocations!(mod, relocations)

    Point every relocation slot in `mod` at its Julia object by name instead of by address.
    """
    function adopt_relocations!(mod::LLVM.Module, relocations)::Nothing
        relocations === nothing && return nothing
        for rec in relocations.records
            rec.kind === GPUCompiler.SlotSite || continue
            target = rec.target
            target isa GPUCompiler.JuliaValueRef || continue
            globs = globals(mod)
            haskey(globs, rec.name) || continue
            slot = globs[rec.name]
            cur = LLVM.initializer(slot)
            (cur === nothing || LLVM.isnull(cur)) || continue
            gv = relocation_global!(mod, target.value)
            init = LLVM.const_addrspacecast(gv, LLVM.PointerType(LLVM.StructType(LLVM.LLVMType[])))
            # The type of what the slot holds, not the slot's own pointer type: with typed
            # pointers those differ, and an initializer of the wrong one is rejected.
            ty = LLVM.global_value_type(slot)
            if LLVM.value_type(init) != ty
                init = LLVM.const_pointercast(init, ty)
            end
            LLVM.initializer!(slot, init)
            # linkage is left as codegen set it: the slot may be referenced from another
            # module that Enzyme links in.
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
