"""
    relocate_julia_globals!(mod::LLVM.Module, job::GPUCompiler.CompilerJob, meta)

Restate GPUCompiler's relocation of `julia.constgv` slots in the form Enzyme's analyses
consume: a slot whose initializer is the address of the real Julia object.

GPUCompiler v2's `compile_method_instance` strips the initializers of `julia.constgv`
globals to keep the IR session-portable, and `relocate_gvs!` fills them back in at the
toplevel link step -- either with a device-resident replica of the object
(`materialize_box!`) or with the host address. Enzyme is served by neither:

* a replica carries no Julia type and no activity metadata, so
  [`try_replace_constant_load!`](@ref) cannot fold the load into an `ejl_` global. Activity
  analysis then reports a mismatch (`EnzymeRuntimeActivityError`) or fails to find a shadow
  (`EnzymeNoShadowError`);
* worse, a replica is *read-only module data*, not a heap object. Storing such a pointer
  into GC-tracked memory makes the collector fault when it sets the mark bits, and objects
  reached only through it are invisible to the collector;
* for a non-toplevel job `relocate_gvs!` never runs at all, leaving the slot an
  initializer-less external declaration.

Enzyme JITs this module into the running session, so the object it must denote is the real,
GC-rooted one. Point each slot back at it; `try_replace_constant_load!` then turns the loads
into `ejl_` named globals as it always has, which is Enzyme's own (relocatable) reference
form, and the now-unused replica is dropped as dead code.

Modules produced by GPUCompiler v1 carry no relocation table and are left alone, as is device
code: there the replica is the right answer (no Julia GC traverses it, and a host address
would be meaningless), and [`absint`](@ref) decodes it instead.
"""
function relocate_julia_globals!(mod::LLVM.Module, @nospecialize(job::GPUCompiler.CompilerJob), @nospecialize(meta))
    if !hasproperty(meta, :gv_to_value)
        # GPUCompiler v1: Julia's own initializers are still in place.
        return nothing
    end
    if !GPUCompiler.uses_julia_runtime(job)
        return nothing
    end
    globs = LLVM.globals(mod)
    for (name, ptr) in meta.gv_to_value
        ptr == C_NULL && continue
        point_at_object!(globs, name, ptr)
    end
    # The Bool singletons are resolved by name rather than through the table, so they need
    # the same treatment -- and they are the ones that reach GC-tracked memory most often,
    # every time a `Bool` is stored into an `Any` container.
    for (name, obj) in ("jl_true" => true, "jl_false" => false)
        point_at_object!(globs, name, ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj))
    end
    return nothing
end

function point_at_object!(globs::LLVM.ModuleGlobalSet, name::String, ptr::Ptr{Cvoid})
    Base.haskey(globs, name) || return nothing
    gv = globs[name]
    gvty = LLVM.global_value_type(gv)
    isa(gvty, LLVM.PointerType) || return nothing

    LLVM.initializer!(gv, LLVM.const_inttoptr(LLVM.ConstantInt(convert(UInt, ptr)), gvty))
    # `compile_method_instance` demoted the slot to an external declaration so that nothing
    # folds it before relocation; with a value in place it can be internal again.
    if LLVM.linkage(gv) == LLVM.API.LLVMExternalLinkage
        LLVM.linkage!(gv, LLVM.API.LLVMPrivateLinkage)
    end
    return nothing
end
