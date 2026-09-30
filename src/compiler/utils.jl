
"""
    EmitTypeNames[] = true

Also write the printed Julia type next to each type Enzyme records in the IR (the
`enzymejl_parmtype_str` attribute, and the `enzymejl_source_type_<T>` and
`enzymejl_allocart_name` metadata). They help reading IR dumps.
"""
const EmitTypeNames = Ref(false)

@enum(AllocFnKindEnum,
      AFKE_Unknown = 0,
      AFKE_Alloc = 1,
      AFKE_Realloc = 2,
      AFKE_Free = 4,
      AFKE_Uninitialized = 8,
      AFKE_Zeroed = 16,
      AFKE_Aligned = 32,
)

struct AllocFnKind
    data::UInt32
    AllocFnKind() = new(0)
    AllocFnKind(x::UInt32) = new(x)
    AllocFnKind(x::AllocFnKindEnum) = new(UInt32(x))
end

function Base.:|(lhs::AllocFnKind, rhs::AllocFnKind)
    AllocFnKind(UInt32(lhs.data) | UInt32(rhs.data))
end

# Memory effects, as encoded in the `memory` attribute on LLVM 16+, are handled with
# LLVM.jl's `MemoryEffects`, which knows the memory locations of each LLVM version.
set_readonly(effects::LLVM.MemoryEffects) = effects & LLVM.MemoryEffects(:read)

is_readonly(effects::LLVM.MemoryEffects) = effects.access in (:none, :read)
is_readnone(effects::LLVM.MemoryEffects) = effects.access == :none
is_writeonly(effects::LLVM.MemoryEffects) = effects.access in (:none, :write)

Base.@assume_effects :removable :foldable :nothrow is_noreturn(f::LLVM.Function)::Bool =
    haskey(f.function_attributes, :noreturn)

Base.@assume_effects :removable :foldable :nothrow is_nounwind(f::LLVM.Function)::Bool =
    haskey(f.function_attributes, :nounwind)

"""
    is_readonly(attr::LLVM.Attribute)::Bool

Whether `attr` on its own establishes that the function or argument position it
is attached to is only read from. That is `readonly` / `readnone`, and on LLVM
16+ a `memory` effect whose modref is read-only.
"""
Base.@assume_effects :removable :foldable :nothrow function is_readonly(attr::LLVM.Attribute)::Bool
    if attr.kind == :readonly
        return true
    end
    if attr.kind == :readnone
        return true
    end
    if LLVM.version().major > 15 && attr.kind == :memory
        if is_readonly(LLVM.MemoryEffects(attr))
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_readonly(f::LLVM.Function)::Bool
    intr = f.intrinsic
    if intr == LLVM.Intrinsic("llvm.lifetime.start")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume")
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
            f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(f.function_attributes)
        if is_readonly(attr)
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_readnone(f::LLVM.Function)::Bool
    intr = f.intrinsic
    if intr == LLVM.Intrinsic("llvm.lifetime.start")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume")
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
            f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    if haskey(f.function_attributes, :readnone)
        return true
    end
    if LLVM.version().major > 15 && haskey(f.function_attributes, :memory)
        if is_readnone(LLVM.MemoryEffects(f.function_attributes[:memory]))
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_writeonly(f::LLVM.Function)::Bool
    intr = f.intrinsic
    if intr == LLVM.Intrinsic("llvm.lifetime.start")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end")
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume")
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
            f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    if haskey(f.function_attributes, :readnone) || haskey(f.function_attributes, :writeonly)
        return true
    end
    if LLVM.version().major > 15 && haskey(f.function_attributes, :memory)
        if is_writeonly(LLVM.MemoryEffects(f.function_attributes[:memory]))
            return true
        end
    end
    return false
end

function set_readonly!(fn::LLVM.Function)
    attrs = collect(fn.function_attributes)
    if LLVM.version().major <= 15
        if !any(attr.kind == :readonly for attr in attrs) &&
                !any(attr.kind == :readnone for attr in attrs)
            if any(attr.kind == :writeonly for attr in attrs)
                delete!(fn.function_attributes, :writeonly)
                push!(fn.function_attributes, EnumAttribute(:readnone))
            else
                push!(fn.function_attributes, EnumAttribute(:readonly))
            end
            return true
        end
        return false
    else
        # without a `memory` attribute, a function may access any memory
        old = LLVM.MemoryEffects(fn.memory_effects)
        eff = set_readonly(old)
        fn.memory_effects = eff
        return old != eff
    end
end

function get_function!(
    mod::LLVM.Module,
    name::String,
    FT::LLVM.FunctionType,
    attrs::Vector{LLVM.Attribute} = LLVM.Attribute[],
)
    F = get(mod.functions, name, nothing)
    if F === nothing
        F = LLVM.Function(mod, name, FT)
        append!(F.function_attributes, attrs)
    else
        PT = LLVM.PointerType(FT)
        if F.value_type != PT
            F = LLVM.const_pointercast(F, PT)
        end
    end
    return F, FT
end

function get_function!(@nospecialize(builderF), mod::LLVM.Module, name::String, attrs::Vector{LLVM.Attribute} = LLVM.Attribute[])
    get_function!(mod, name, builderF(), attrs)
end

T_ppjlvalue() = LLVM.PointerType(LLVM.PointerType(LLVM.StructType(LLVMType[])))

function declare_pgcstack!(mod::LLVM.Module)
    get_function!(
        mod,
        "julia.get_pgcstack",
        LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue())),
        LLVM.Attribute[StringAttribute("enzyme_inactive"), StringAttribute("enzyme_no_escaping_allocation")]
    )
end

function emit_pgcstack(B::LLVM.IRBuilder, name::String="")
    curent_bb = B.insert_block
    fn = curent_bb.parent
    mod = fn.parent
    func, fty = declare_pgcstack!(mod)
    return call!(B, fty, func, LLVM.Value[], name)
end

function get_pgcstack(func::LLVM.Function)
    entry_bb = first(func.blocks)
    mod = func.parent
    pgcstack_func, _ = declare_pgcstack!(mod)

    # A @cfunction wrapper fetches its pgcstack with julia.get_pgcstack_or_new,
    # which adopts the calling thread when it is not a Julia thread yet. It has
    # to stay the first getter of the entry block: FinalLowerGC roots the GC
    # frame in whichever getter comes first, and a plain getter placed ahead of
    # the adopting one dereferences a NULL pgcstack on a foreign thread.
    if haskey(mod.functions, "julia.get_pgcstack_or_new")
        or_new = mod.functions["julia.get_pgcstack_or_new"]
        for I in entry_bb.instructions
            if I isa LLVM.CallInst && I.called_operand == or_new
                return I
            end
        end
    end

    for I in entry_bb.instructions
        if I isa LLVM.CallInst && I.called_operand == pgcstack_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func::LLVM.Function, @nospecialize(PB::Union{Nothing, LLVM.IRBuilder}) = nothing)
    for i in 1:length(func.parameters)
        for attr in collect(func.parameter_attributes[i])
            if attr isa LLVM.EnumAttribute
                if attr.kind == swiftself_kind
                    return func.parameters[i]
                end
            end
        end
    end

    pgs = get_pgcstack(func)
    if pgs isa Nothing
        entry_bb = first(func.blocks)
        if PB !== nothing &&
                (PB.insert_block.name == "allocsForInversion" || PB.insert_block == entry_bb)
            # emit it with the caller's builder, which we don't own
            emit_pgcstack(PB, "newly_emitted_pgc_stack")
        else
            @dispose B = IRBuilder() begin
                if isempty(entry_bb.instructions)
                    position!(B, LLVM.at_end(entry_bb))
                else
                    position!(B, LLVM.at_begin(entry_bb))
                end
                emit_pgcstack(B, "newly_emitted_pgc_stack")
            end
        end
    else
        entry_bb = first(func.blocks)
        fst = first(entry_bb.instructions)
        if fst != pgs
            API.moveBefore(pgs, fst, PB isa Nothing ? C_NULL : PB.ref)
        end
        pgs
    end
end

# `offsetof(jl_task_t, gcstack)`, which is fixed for a build of Julia. Declaring it foldable
# lets inference turn the load into a constant.
Base.@assume_effects :foldable :nothrow task_gcstack_offset() = Int(unsafe_load(cglobal(:jl_task_gcstack_offset, Cint)))

"""
    current_pgcstack() -> Ptr{Cvoid}

Give the pgcstack of the running task, for an llvmcall that takes it as an argument (see
[`use_gcstack_arg!`](@ref)).

Julia computes `current_task()` from the pgcstack of the function that this is inlined into
(on 1.13 that is the `"gcstack"` argument of the function). So the function gets no
`julia.get_pgcstack` call from this. An llvmcall of `julia.get_pgcstack` would not do: Julia
inlines it where it is called, which is the problem that [`use_gcstack_arg!`](@ref) avoids.
"""
@inline current_pgcstack() = pointer_from_objref(current_task()) + task_gcstack_offset()

"""
    use_gcstack_arg!(f::LLVM.Function, arg::LLVM.Argument)

Make the llvmcall `f` use its argument `arg` as the pgcstack of the code it inlines. The
caller gives this argument with [`current_pgcstack`](@ref), as a `Ptr{Cvoid}`.

Julia inlines an llvmcall into its caller, together with the `alwaysinline` functions that
the llvmcall calls. A `julia.get_pgcstack` call in that code thus goes into the middle of
the caller. On 1.13 the caller takes its pgcstack as its own `"gcstack"` argument. But the
GC lowering prefers a `julia.get_pgcstack` call in the entry block to that argument, and
pushes the GC frame of the caller only after the call. Then no safepoint before the call
has the roots of the caller.

Thus inline the `alwaysinline` functions into `f` here, and replace every
`julia.get_pgcstack` call in `f` with `arg`, as `LowerPTLS` does for a function that takes
its pgcstack as an argument.
"""
function use_gcstack_arg!(f::LLVM.Function, arg::LLVM.Argument)
    mod = f.parent
    run!(AlwaysInlinerPass(), mod)
    if !haskey(mod.functions, "julia.get_pgcstack")
        return
    end
    getter = mod.functions["julia.get_pgcstack"]
    calls = LLVM.CallInst[]
    for use in getter.uses
        call = use.user
        if call isa LLVM.CallInst && call.parent.parent == f
            push!(calls, call)
        end
    end
    if isempty(calls)
        return
    end
    T_pgcstack = getter.function_type.return_type
    pgcstack = @dispose B = IRBuilder() begin
        position!(B, LLVM.at_begin(f.entry))
        # Before 1.12, a `Ptr{Cvoid}` is an integer in Julia's IR.
        if arg.value_type isa LLVM.IntegerType
            inttoptr!(B, arg, T_pgcstack)
        else
            bitcast!(B, arg, T_pgcstack)
        end
    end
    for call in calls
        replace_uses!(call, pgcstack)
        erase!(call)
    end
    return
end

const swiftself_kind = :swiftself

Base.@assume_effects :removable :foldable :nothrow function has_swiftself(fn::LLVM.Function)::Bool
    for i in 1:length(fn.parameters)
        for attr in collect(fn.parameter_attributes[i])
            if attr isa LLVM.EnumAttribute
                if attr.kind == swiftself_kind
                    return true
                end
            end
        end
    end
    return false
end

"""
    gcstack_arg_index(fn::LLVM.Function) -> Int

Give the index of the `pgcstack` parameter of `fn`, or `0` when `fn` has none.

Julia's codegen marks that parameter `swiftself` where the target supports the
swift calling convention, and turns the convention off where it does not
(`jl_codegen_output_t::use_swiftcc`, false on RISC-V). Julia 1.13 also gives
the parameter a `gcstack` string attribute, which `specsig` writes on every
version. Hence recognize either mark.
"""
Base.@assume_effects :removable :foldable :nothrow function gcstack_arg_index(fn::LLVM.Function)::Int
    for i in 1:length(fn.parameters)
        for attr in collect(fn.parameter_attributes[i])
            if attr isa LLVM.EnumAttribute
                if attr.kind == swiftself_kind
                    return i
                end
            elseif attr isa LLVM.StringAttribute
                if attr.kind == "gcstack"
                    return i
                end
            end
        end
    end
    return 0
end

"""
    has_gcstack_arg(fn::LLVM.Function) -> Bool

Say if `fn` takes `pgcstack` as a parameter (see [`gcstack_arg_index`](@ref)).
"""
Base.@assume_effects :removable :foldable :nothrow has_gcstack_arg(fn::LLVM.Function)::Bool = gcstack_arg_index(fn) != 0

"""
    copy_metadata!(dst, src)

Attach every metadata node of the instruction `src`, except its debug location,
to `dst`.
"""
function copy_metadata!(dst::LLVM.Instruction, src::LLVM.Instruction)
    for (kind, md) in src.metadata
        kind == LLVM.MD_dbg && continue
        dst.metadata[kind] = md
    end
    return nothing
end

eraseInst(bb::LLVM.BasicBlock, @nospecialize(inst::LLVM.Instruction)) = erase!(inst)
eraseInst(bb::LLVM.Module, inst::LLVM.Function) = erase!(inst)
eraseInst(bb::LLVM.Module, inst::LLVM.GlobalVariable) = erase!(inst)

function unique_gcmarker!(func::LLVM.Function)
    entry_bb = first(func.blocks)
    pgcstack_func, _ = declare_pgcstack!(func.parent)

    found = LLVM.CallInst[]
    for I in entry_bb.instructions
        if I isa LLVM.CallInst && I.called_operand == pgcstack_func
            push!(found, I)
        end
    end
    if length(found) > 1
        for i = 2:length(found)
            LLVM.replace_uses!(found[i], found[1])
            eraseInst(entry_bb, found[i])
        end
    end
    return nothing
end

@inline AnonymousStruct(::Type{U}) where {U<:Tuple} =
    NamedTuple{ntuple(Symbol, Val(length(U.parameters))),U}

# recursively compute the eltype type indexed by idx[0], idx[1], ...
Base.@assume_effects :removable :foldable :nothrow function recursive_eltype(@nospecialize(val::LLVM.Value), idxs::Vector{Cuint})::LLVM.LLVMType
    ty = val.value_type::LLVM.LLVMType
    for i in idxs
        if isa(ty, LLVM.ArrayType)
            ty = ty.element_type::LLVM.LLVMType
        else
            @assert isa(ty, LLVM.StructType)
            ty = ty.elements[i + 1]::LLVM.LLVMType
        end
    end
    return ty
end

# Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
# and that Bool -> i8, not i1
function calling_conv_fixup(
    builder::LLVM.IRBuilder,
    @nospecialize(val::LLVM.Value),
    @nospecialize(tape::LLVM.LLVMType),
    @nospecialize(prev::LLVM.Value) = LLVM.UndefValue(tape),
    lidxs::Vector{Cuint} = Cuint[],
    ridxs::Vector{Cuint} = Cuint[],
    emesg = nothing,
)::LLVM.Value
    ctype = recursive_eltype(val, lidxs)
    if ctype == tape
        if length(lidxs) != 0
            val = extract_value!(builder, val, lidxs)
        end
        if length(ridxs) == 0
            return val
        else
            return insert_value!(builder, prev, val, ridxs)
        end
    end

    if isa(tape, LLVM.StructType)
        if isa(ctype, LLVM.ArrayType)
            @assert ctype.length == length(tape.elements)
            for (i, ty) in enumerate(tape.elements)
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(ctype.elements) == length(tape.elements)
            for (i, ty) in enumerate(tape.elements)
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
    elseif isa(tape, LLVM.ArrayType)
        if isa(ctype, LLVM.ArrayType)
            @assert ctype.length == tape.length
            for i in 1:tape.length
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, tape.element_type, prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(ctype.elements) == tape.length
            for i in 1:tape.length
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, tape.element_type, prev, ln, rn, emesg)
            end
            return prev
        end
    end

    if isa(tape, LLVM.IntegerType) &&
            tape.width == 1 &&
            ctype.width != tape.width
        if length(lidxs) != 0
            val = extract_value!(builder, val, lidxs)
        end
        val = trunc!(builder, val, tape)
        return if length(ridxs) != 0
            insert_value!(builder, prev, val, ridxs)
        else
            val
        end
    end
    if isa(tape, LLVM.PointerType) &&
       isa(ctype, LLVM.PointerType) &&
            tape.addrspace == ctype.addrspace
        if length(lidxs) != 0
            val = extract_value!(builder, val, lidxs)
        end
        val = pointercast!(builder, val, tape)
        return if length(ridxs) != 0
            insert_value!(builder, prev, val, ridxs)
        else
            val
        end
    end
    if isa(ctype, LLVM.ArrayType) && ctype.length == 1 && ctype.element_type == tape
        lhs_n = copy(lidxs)
        push!(lhs_n, 0)
        return calling_conv_fixup(builder, val, tape, prev, lhs_n, ridxs, emesg)
    end


    msg2 = sprint() do io
        println(io, "Enzyme Internal Error: Illegal calling convention fixup")
        if emesg !== nothing
            emesg(io)
        end
        println(io, "ctype = ", ctype)
        println(io, "tape = ", tape)
        println(io, "val = ", string(val))
        println(io, "prev = ", string(prev))
        println(io, "lidxs = ", lidxs)
        println(io, "ridxs = ", ridxs)
        println(io, "tape_type(tape) = ", tape_type(tape))
        println(
            io,
            "convert(LLVMType, tape_type(tape)) = ",
            convert(LLVM.LLVMType, tape_type(tape); allow_boxed = true),
        )
    end
    throw(AssertionError(msg2))
end
