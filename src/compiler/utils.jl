
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

struct MemoryEffect
    data::UInt32
end


@enum(ModRefInfo, MRI_NoModRef = 0, MRI_Ref = 1, MRI_Mod = 2, MRI_ModRef = 3)

@enum(IRMemLocation, ArgMem = 0, InaccessibleMem = 1, Other = 2)

const BitsPerLoc = UInt32(2)
const LocMask = UInt32((1 << BitsPerLoc) - 1)
function getLocationPos(Loc::IRMemLocation)
    return UInt32(Loc) * BitsPerLoc
end
function Base.:<<(mr::ModRefInfo, rhs::UInt32)
    UInt32(mr) << rhs
end
function Base.:|(lhs::ModRefInfo, rhs::ModRefInfo)
    ModRefInfo(UInt32(lhs) | UInt32(rhs))
end
function Base.:&(lhs::ModRefInfo, rhs::ModRefInfo)
    ModRefInfo(UInt32(lhs) & UInt32(rhs))
end
const AllEffects = MemoryEffect(
    (MRI_ModRef << getLocationPos(ArgMem)) |
    (MRI_ModRef << getLocationPos(InaccessibleMem)) |
    (MRI_ModRef << getLocationPos(Other)),
)
const ReadOnlyEffects = MemoryEffect(
    (MRI_Ref << getLocationPos(ArgMem)) |
    (MRI_Ref << getLocationPos(InaccessibleMem)) |
    (MRI_Ref << getLocationPos(Other)),
)
const ReadOnlyArgMemEffects = MemoryEffect(
    (MRI_Ref << getLocationPos(ArgMem)) |
    (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
    (MRI_NoModRef << getLocationPos(Other)),
)
const WriteOnlyArgMemEffects = MemoryEffect(
    (MRI_Mod << getLocationPos(ArgMem)) |
    (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
    (MRI_NoModRef << getLocationPos(Other)),
)
const NoEffects = MemoryEffect(
    (MRI_NoModRef << getLocationPos(ArgMem)) |
    (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
    (MRI_NoModRef << getLocationPos(Other)),
)
const ReadArgMemReadWriteInaccessibleEffects = MemoryEffect(
    (MRI_Ref << getLocationPos(ArgMem)) |
        (MRI_ModRef << getLocationPos(InaccessibleMem)) |
        (MRI_NoModRef << getLocationPos(Other)),
)

const ReadArgMemWriteInaccessibleEffects = MemoryEffect(
    (MRI_Ref << getLocationPos(ArgMem)) |
    (MRI_Mod << getLocationPos(InaccessibleMem)) |
    (MRI_NoModRef << getLocationPos(Other)),
)

# Get ModRefInfo for any location.
function getModRef(effect::MemoryEffect, loc::IRMemLocation)::ModRefInfo
    ModRefInfo((effect.data >> getLocationPos(loc)) & LocMask)
end

function getModRef(effect::MemoryEffect)::ModRefInfo
    cur = MRI_NoModRef
    for loc in (ArgMem, InaccessibleMem, Other)
        cur |= getModRef(effect, loc)
    end
    return cur
end

function setModRef(effect::MemoryEffect, Loc::IRMemLocation, MR::ModRefInfo)::MemoryEffect
    data = effect.data
    Data &= ~(LocMask << getLocationPos(Loc))
    Data |= MR << getLocationPos(Loc)
    return MemoryEffect(data)
end

function setModRef(effect::MemoryEffect)::MemoryEffect
    for loc in (ArgMem, InaccessibleMem, Other)
        effect = setModRef(effect, mri) = getModRef(effect, loc)
    end
    return effect
end

function set_readonly(mri::ModRefInfo)
    return mri & MRI_Ref
end
function set_writeonly(mri::ModRefInfo)
    return mri & MRI_Mod
end
function set_reading(mri::ModRefInfo)
    return mri | MRI_Ref
end
function set_writing(mri::ModRefInfo)
    return mri | MRI_Mod
end

function set_readonly(effect::MemoryEffect)::MemoryEffect
    data = UInt32(0)
    for loc in (ArgMem, InaccessibleMem, Other)
        data |= UInt32(set_readonly(getModRef(effect, loc))) << getLocationPos(loc)
    end
    return MemoryEffect(data)
end

function is_readonly(mri::ModRefInfo)::Bool
    return mri == MRI_NoModRef || mri == MRI_Ref
end

function is_readnone(mri::ModRefInfo)::Bool
    return mri == MRI_NoModRef
end

function is_writeonly(mri::ModRefInfo)::Bool
    return mri == MRI_NoModRef || mri == MRI_Mod
end

for n in (:is_readonly, :is_readnone, :is_writeonly)
    @eval begin
        function $n(memeffect::MemoryEffect)
            return $n(getModRef(memeffect))
        end
    end
end

Base.@assume_effects :removable :foldable :nothrow function is_noreturn(f::LLVM.Function)::Bool
    for attr in collect(f.function_attributes)
        if attr.kind == :noreturn
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_nounwind(f::LLVM.Function)::Bool
    for attr in collect(f.function_attributes)
        if attr.kind == :nounwind
            return true
        end
    end
    return false
end

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
    if LLVM.version().major > 15 && isa(attr, LLVM.EnumAttribute)
        if attr.kind == :memory
            if is_readonly(MemoryEffect(attr.value))
                return true
            end
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_readonly(f::LLVM.Function)::Bool
    intr = LLVM.API.LLVMGetIntrinsicID(f)
    if intr == LLVM.Intrinsic("llvm.lifetime.start").id
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end").id
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume").id
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
    intr = LLVM.API.LLVMGetIntrinsicID(f)
    if intr == LLVM.Intrinsic("llvm.lifetime.start").id
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end").id
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume").id
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
       f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(cur.function_attributes)
        if attr.kind == :readnone
            return true
        end
        if LLVM.version().major > 15
            if attr.kind == :memory
                if is_readnone(MemoryEffect(attr.value))
                    return true
                end
            end
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_writeonly(f::LLVM.Function)::Bool
    intr = LLVM.API.LLVMGetIntrinsicID(f)
    if intr == LLVM.Intrinsic("llvm.lifetime.start").id
        return true
    end
    if intr == LLVM.Intrinsic("llvm.lifetime.end").id
        return true
    end
    if intr == LLVM.Intrinsic("llvm.assume").id
        return true
    end
    if f.name == "llvm.julia.gc_preserve_begin" ||
       f.name == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(cur.function_attributes)
        if attr.kind == :readnone
            return true
        end
        if attr.kind == :writeonly
            return true
        end
        if LLVM.version().major > 15
            if attr.kind == :memory
                if is_writeonly(MemoryEffect(attr.value))
                    return true
                end
            end
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
                delete!(fn.function_attributes, EnumAttribute("writeonly"))
                push!(fn.function_attributes, EnumAttribute("readnone"))
            else
                push!(fn.function_attributes, EnumAttribute("readonly"))
            end
            return true
        end
        return false
    else
        for attr in attrs
            if attr.kind == :memory
                old = MemoryEffect(attr.value)
                eff = set_readonly(old)
                push!(fn.function_attributes, EnumAttribute("memory", eff.data))
                return old != eff
            end
        end
        push!(
            fn.function_attributes,
            EnumAttribute("memory", set_readonly(AllEffects).data),
        )
        return true
    end
end

function get_function!(
    mod::LLVM.Module,
    name::String,
    FT::LLVM.FunctionType,
    attrs::Vector{LLVM.Attribute} = LLVM.Attribute[],
)
    if haskey(mod.functions, name)
        F = mod.functions[name]
        PT = LLVM.PointerType(FT)
        if F.value_type != PT
            F = LLVM.const_pointercast(F, PT)
        end
    else
        F = LLVM.Function(mod, name, FT)
        for attr in attrs
            push!(F.function_attributes, attr)
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
        func.parent.context
        B = IRBuilder()
        entry_bb = first(func.blocks)
	if PB !== nothing && PB.insert_block.name == "allocsForInversion"
	    B = PB
	elseif !isempty(entry_bb.instructions)
	    if PB === nothing || PB.insert_block != entry_bb 
		    position!(B, LLVM.at_begin(entry_bb))
	    else
		    B = PB
	    end
        else
	    if PB === nothing || PB.insert_block != entry_bb 
               position!(B, LLVM.at_end(entry_bb))
	    else
	       B = PB
	    end
        end
        emit_pgcstack(B, "newly_emitted_pgc_stack")
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
    B = IRBuilder()
    position!(B, LLVM.at_begin(f.entry))
    T_pgcstack = getter.function_type.return_type
    # Before 1.12, a `Ptr{Cvoid}` is an integer in Julia's IR.
    pgcstack = if arg.value_type isa LLVM.IntegerType
        inttoptr!(B, arg, T_pgcstack)
    else
        bitcast!(B, arg, T_pgcstack)
    end
    dispose(B)
    for call in calls
        replace_uses!(call, pgcstack)
        LLVM.API.LLVMInstructionEraseFromParent(call)
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

Base.@assume_effects :removable :foldable :nothrow function has_fn_attr(fn::LLVM.Function, attr::LLVM.EnumAttribute)::Bool
    ekind = attr.kind
    for attr in collect(fn.function_attributes)
        if attr isa LLVM.EnumAttribute
            if attr.kind == ekind
                return true
            end
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function has_fn_attr(fn::LLVM.Function, attr::LLVM.StringAttribute)::Bool
    ekind = attr.kind
    for attr in collect(fn.function_attributes)
        if attr isa LLVM.StringAttribute
            if attr.kind == ekind
                return true
            end
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function has_arg_attr(fn::LLVM.Function, i::Int, attr::LLVM.StringAttribute)::Bool
    ekind = attr.kind
    for attr in collect(fn.parameter_attributes[i])
        if attr isa LLVM.StringAttribute
            if attr.kind == ekind
                return true
            end
        end
    end
    return false
end

"""
    precedes(a, b)

Whether instruction `a` comes before `b` in their common basic block.
"""
function precedes(a::LLVM.Instruction, b::LLVM.Instruction)::Bool
    @assert a.parent == b.parent
    for inst in a.parent.instructions
        inst == a && return true
        inst == b && return false
    end
    return false
end

"""
    copy_metadata!(dst, src)

Attach every metadata node of the instruction `src`, except its debug location,
to `dst`.
"""
function copy_metadata!(dst::LLVM.Instruction, src::LLVM.Instruction)
    num = Ref{Csize_t}()
    entries = LLVM.API.LLVMInstructionGetAllMetadataOtherThanDebugLoc(src, num)
    ctx = src.context
    for i in 1:num[]
        kind = LLVM.API.LLVMValueMetadataEntriesGetKind(entries, i - 1)
        md = LLVM.API.LLVMValueMetadataEntriesGetMetadata(entries, i - 1)
        LLVM.API.LLVMSetMetadata(dst, kind, LLVM.API.LLVMMetadataAsValue(ctx, md))
    end
    num[] > 0 && LLVM.API.LLVMDisposeValueMetadataEntries(entries)
    return nothing
end

function eraseInst(bb::LLVM.BasicBlock, @nospecialize(inst::LLVM.Instruction))
    @static if isdefined(LLVM, Symbol("erase!"))
        LLVM.erase!(inst)
    else
        erase!(inst)
    end
end
function eraseInst(bb::LLVM.Module, inst::LLVM.Function)
    @static if isdefined(LLVM, Symbol("erase!"))
        LLVM.erase!(inst)
    else
        erase!(inst)
    end
end
function eraseInst(bb::LLVM.Module, inst::LLVM.GlobalVariable)
    @static if isdefined(LLVM, Symbol("erase!"))
        LLVM.erase!(inst)
    else
        erase!(inst)
    end
end

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
            ty = ty.elements[i+1]::LLVM.LLVMType
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
            val = API.e_extract_value!(builder, val, lidxs)
        end
        if length(ridxs) == 0
            return val
        else
            return API.e_insert_value!(builder, prev, val, ridxs)
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
            for i = 1:tape.length
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
            for i = 1:tape.length
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
            val = API.e_extract_value!(builder, val, lidxs)
        end
        val = trunc!(builder, val, tape)
        return if length(ridxs) != 0
            API.e_insert_value!(builder, prev, val, ridxs)
        else
            val
        end
    end
    if isa(tape, LLVM.PointerType) &&
       isa(ctype, LLVM.PointerType) &&
       tape.addrspace == ctype.addrspace
        if length(lidxs) != 0
            val = API.e_extract_value!(builder, val, lidxs)
        end
        val = pointercast!(builder, val, tape)
        return if length(ridxs) != 0
            API.e_insert_value!(builder, prev, val, ridxs)
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
