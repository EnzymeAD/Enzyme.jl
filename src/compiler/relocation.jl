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

# Whether every target can be reconstructed in another process from its serialized form.
function persistable(m::AbstractVector{<:Pair{String}})::Bool
    for (_, target) in m
        v = unbind(target)
        (
            v isa Type || v isa Symbol || v isa String || v isa Module || v isa Core.MethodInstance ||
                v isa Method || (isbits(v) && !(v isa Ptr))
        ) || return false
    end
    return true
end
