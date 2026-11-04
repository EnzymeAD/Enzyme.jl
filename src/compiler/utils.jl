
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
    for attr in collect(function_attributes(f))
        if kind(attr) == kind(EnumAttribute("noreturn"))
            return true
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function is_nounwind(f::LLVM.Function)::Bool
    for attr in collect(function_attributes(f))
        if kind(attr) == kind(EnumAttribute("nounwind"))
            return true
        end
    end
    return false
end

"""
    clear_nounwind!(f::LLVM.Function)

Remove `nounwind` from `f`, from every call of `f`, and from each calling
function in turn, so that a throw inserted into `f` may unwind through its
callers.

Enzyme's preprocessing marks every call it differentiates `willreturn`, and
inlining a `nounwind` function marks the calls it inlines `nounwind`. A
`noreturn` throw that is also `nounwind` and `willreturn` can neither return,
unwind nor run forever, so LLVM treats it as unreachable and deletes it,
together with every path that leads to it.
"""
function clear_nounwind!(f::LLVM.Function)
    idx = reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex)
    nounwind = kind(EnumAttribute("nounwind"))
    todo = LLVM.Function[f]
    done = Set{LLVM.Function}()
    while !isempty(todo)
        cur = pop!(todo)
        if cur in done
            continue
        end
        push!(done, cur)
        LLVM.API.LLVMRemoveEnumAttributeAtIndex(cur, idx, nounwind)
        for u in LLVM.uses(cur)
            call = LLVM.user(u)
            if !isa(call, LLVM.CallBase) || LLVM.called_operand(call) != cur
                continue
            end
            LLVM.API.LLVMRemoveCallSiteEnumAttribute(call, idx, nounwind)
            push!(todo, LLVM.parent(LLVM.parent(call)))
        end
    end
    return nothing
end

"""
    is_float_data(ty::LLVM.LLVMType)::Bool

Whether a value of type `ty` is floating-point data (a float or a vector of
floats), which cannot refer to memory.
"""
is_float_data(::LLVM.LLVMType) = false
is_float_data(::LLVM.FloatingPointType) = true
is_float_data(ty::LLVM.VectorType) = is_float_data(eltype(ty))

"""
    is_inactive_operand(call, fn, val, world)::Bool

Whether `call` passes `val` only by reference (`BITS_REF` or `MUT_REF`) to
parameters of the Julia function `fn` whose declared type (`enzymejl_parmtype`)
cannot hold derivative data that a write could change (`active_reg` is
`ActiveState` or `AnyState`). Nothing `fn` does through `val` then writes
derivative data, whether or not its body can be inspected.
"""
function is_inactive_operand(call::LLVM.CallInst, fn::LLVM.Function, @nospecialize(val::LLVM.Value), world::UInt)::Bool
    found = false
    nparms = length(LLVM.parameters(fn))
    for (i, op) in enumerate(arg_operands_view(call))
        op == val || continue
        i <= nparms || return false
        typ, byref = enzyme_extract_parm_type(fn, i, false)
        (typ === nothing || byref === nothing) && return false
        (byref == GPUCompiler.BITS_REF || byref == GPUCompiler.MUT_REF) || return false
        reg = active_reg(typ, world)
        (reg == ActiveState || reg == AnyState) || return false
        found = true
    end
    return found
end

"""
    readonly_checked_params(primalf, TT, err_if_written, world)::Vector{LLVM.Value}

The parameters of `primalf` whose data the readonly check follows. `err_if_written`
is `true` for the function argument alone (`err_if_func_written`), or a tuple with
one entry per element of `TT.parameters`, the function first, as the rules of a
dynamic call request it (`src/rules/jitrules.jl`). Only arguments whose type can
hold derivative data (`DupState` / `MixedState`) are followed.
"""
function readonly_checked_params(primalf::LLVM.Function, @nospecialize(TT::Type), err_if_written::Union{Bool, Tuple}, world::UInt)::Vector{LLVM.Value}
    checked = LLVM.Value[]
    parms = collect(LLVM.parameters(primalf))
    # Lowered parameters: `swiftself`, then one per argument that is not a ghost.
    idx = Int(has_swiftself(primalf))
    for (i, T) in enumerate(TT.parameters)
        source_typ = eltype(T)
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end
        idx += 1
        check = err_if_written isa Bool ? (i == 1 && err_if_written) : err_if_written[i]
        check || continue
        reg = active_reg(source_typ, world)
        if reg == DupState || reg == MixedState
            push!(checked, parms[idx])
        end
    end
    return checked
end

"""
    is_fresh_object(val::LLVM.Value)::Bool

Whether `val` is an object this function allocates (`julia.gc_alloc_obj`), such as
the box Julia makes of a value it passes to a dynamic call.
"""
function is_fresh_object(@nospecialize(val::LLVM.Value))::Bool
    isa(val, LLVM.CallInst) || return false
    fn = LLVM.called_operand(val)
    return isa(fn, LLVM.Function) && LLVM.name(fn) == "julia.gc_alloc_obj"
end

"""
    is_nonwriting_runtime_call(call::LLVM.CallInst)::Bool

Whether `call` reads its arguments, or builds a new object from them, but writes
no memory they refer to: a `julia.call` of a Julia builtin such as `tuple`,
`getfield` or `isa` (builtins take and return boxed values, as the code around a
dynamic call does), or `jl_array_copy` (`copy` of an `Array` before Julia 1.11).
"""
function is_nonwriting_runtime_call(call::LLVM.CallInst)::Bool
    fn = LLVM.called_operand(call)
    isa(fn, LLVM.Function) || return false
    builtin = LLVM.name(fn) == "julia.call"
    if builtin
        fn = operands(call)[1]
        isa(fn, LLVM.Function) || return false
    end
    nm = LLVM.name(fn)
    if startswith(nm, "ijl_")
        nm = nm[2:end]
    end
    builtin || return nm == "jl_array_copy"
    return nm in (
        "jl_f_tuple",
        "jl_f_getfield",
        "jl_f__svec_ref",
        "jl_f_typeassert",
        "jl_f_isa",
        "jl_f_is",
        "jl_f_typeof",
        "jl_f_sizeof",
        "jl_f_nfields",
        "jl_f_isdefined",
        "jl_f_ifelse",
    )
end

"""
    local_memory_base(ptr::LLVM.Value)

The memory `ptr` points into, through GEPs and casts, when it belongs to the
function: a stack slot (`alloca`) or an object it allocates (`is_fresh_object`).
`nothing` otherwise.
"""
function local_memory_base(@nospecialize(ptr::LLVM.Value))
    base = ptr
    while isa(base, LLVM.BitCastInst) ||
            isa(base, LLVM.AddrSpaceCastInst) ||
            isa(base, LLVM.GetElementPtrInst)
        base = operands(base)[1]
    end
    return (isa(base, LLVM.AllocaInst) || is_fresh_object(base)) ? base : nothing
end

"""
    is_pointerfree_copy(call::LLVM.CallInst, enzyme_ctx)::Bool

Whether a `memcpy` / `memmove` copies bytes of a Julia type that holds no
references, known from the type either pointer points to. Such bytes cannot
refer to memory.
"""
function is_pointerfree_copy(call::LLVM.CallInst, enzyme_ctx::EnzymeContext)::Bool
    for ptr in (operands(call)[1], operands(call)[2])
        base, _ = get_base_and_offset(ptr)
        legal, jTy, byref = abs_typeof(base, enzyme_ctx)
        legal || continue
        if byref == GPUCompiler.BITS_VALUE && jTy <: Ptr
            jTy = eltype(jTy)
        end
        if isa(jTy, DataType) && Base.isconcretetype(jTy) && Base.datatype_pointerfree(jTy)
            return true
        end
    end
    return false
end

"""
    is_pointerfree_runtime_copy(call::LLVM.CallInst, val, enzyme_ctx)::Bool

Whether `call` is a Julia runtime copy of elements that hold no references
(`isbitstype`, or a union of such types), which reads `val` and fills fresh memory
with bytes: `jl_array_copy` (`copy` of an `Array` before Julia 1.11), or
`jl_genericmemory_copyto` (`unsafe_copyto!` of `Memory` on Julia 1.11+, for an
element type Julia does not copy with a `memmove`, such as
`Union{Missing, Float64}`) with `val` among its source operands only. The element
type comes from the Julia type of the array or memory. Such bytes cannot refer to
memory, so nothing that writes the copy, or passes it on, reaches memory of the
argument; `is_pointerfree_copy` says the same of a `memcpy` / `memmove`.
"""
function is_pointerfree_runtime_copy(call::LLVM.CallInst, @nospecialize(val::LLVM.Value), enzyme_ctx::EnzymeContext)::Bool
    fn = LLVM.called_operand(call)
    isa(fn, LLVM.Function) || return false
    nm = LLVM.name(fn)
    if startswith(nm, "ijl_")
        nm = nm[2:end]
    end
    if nm == "jl_array_copy"
        legal, jTy, _ = abs_typeof(call, enzyme_ctx)
        (legal && isa(jTy, DataType) && jTy <: Array) || return false
    elseif nm == "jl_genericmemory_copyto"
        # (dest memory, dest data, src memory, src data, length)
        ops = arg_operands_view(call)
        length(ops) == 5 || return false
        (val == ops[1] || val == ops[2]) && return false
        (val == ops[3] || val == ops[4]) || return false
        legal, jTy, _ = abs_typeof(ops[3], enzyme_ctx)
        @static if VERSION >= v"1.11-"
            (legal && isa(jTy, DataType) && jTy <: GenericMemory) || return false
        else
            return false
        end
    else
        return false
    end
    T = eltype(jTy)
    return isbitstype(T) || Base.isbitsunion(T)
end

"""
    is_throw_runtime_call(call::LLVM.CallInst)::Bool

Whether `call` is a Julia runtime function that throws what it receives (`jl_throw`,
`jl_rethrow_other`) or a `BoundsError` holding it (`jl_bounds_error_int`, …). It
writes nothing it receives, but a `catch` can read what it throws.
"""
function is_throw_runtime_call(call::LLVM.CallInst)::Bool
    fn = LLVM.called_operand(call)
    isa(fn, LLVM.Function) || return false
    nm = LLVM.name(fn)
    if startswith(nm, "ijl_")
        nm = nm[2:end]
    end
    return nm in (
        "jl_throw",
        "jl_rethrow_other",
        "jl_bounds_error",
        "jl_bounds_error_v",
        "jl_bounds_error_int",
        "jl_bounds_error_ints",
        "jl_bounds_error_tuple_int",
        "jl_bounds_error_unboxed_int",
    )
end

"""
    caught_exception_reads(mod::LLVM.Module)::Vector{LLVM.Value}

The calls in `mod` that read a caught exception: `jl_current_exception` (`catch e`)
and `jl_get_excstack` (`current_exceptions()`).
"""
function caught_exception_reads(mod::LLVM.Module)::Vector{LLVM.Value}
    reads = LLVM.Value[]
    fns = functions(mod)
    for nm in ("jl_current_exception", "ijl_current_exception", "jl_get_excstack", "ijl_get_excstack")
        haskey(fns, nm) || continue
        for u in LLVM.uses(fns[nm])
            call = LLVM.user(u)
            if isa(call, LLVM.CallInst)
                push!(reads, call)
            end
        end
    end
    return reads
end

"""
    ReadonlyWalkState

What the walks of one readonly check share (`readonly_walk!`).

- `shared`: per callee called from several sites and parameter positions that
  receive the data, whether a walk of its body found a possible write and whether
  the data reached its `ret`.
- `caught`: the calls that read a caught exception, not yet followed.
- `thrown`: whether data may have been thrown, by a runtime call or by the method a
  dynamic call reaches. A `catch` can then read it, so `caught` is followed too.
"""
struct ReadonlyWalkState
    shared::Dict{Tuple{LLVM.Function, Vector{Int}}, Tuple{Bool, Bool}}
    caught::Vector{LLVM.Value}
    thrown::Base.RefValue{Bool}
end
ReadonlyWalkState(mod::LLVM.Module) = ReadonlyWalkState(
    Dict{Tuple{LLVM.Function, Vector{Int}}, Tuple{Bool, Bool}}(), caught_exception_reads(mod), Ref(false),
)

const READONLY_OPERANDS_MD = "enzymejl_readonly_operands"

"""
    dynamic_call_operands(call::LLVM.CallInst)

For a `julia.call` / `julia.call2` that dispatches at run time (`jl_apply_generic`,
`jl_invoke`, `invokelatest`, `invoke_in_world`, `_apply_iterate`), the index in
`arg_operands_view(call)` of the function it calls, the first of the operands the
dynamic-call rules (`generic_setup`, `src/rules/jitrules.jl`) pass on, and whether
the arguments after it are splatted (`_apply_iterate`). `nothing` for any other
call. The indices follow the handlers in `src/rules/llvmrules.jl`.
"""
function dynamic_call_operands(call::LLVM.CallInst)
    jlcall = LLVM.called_operand(call)
    isa(jlcall, LLVM.Function) || return nothing
    jlname = LLVM.name(jlcall)
    (jlname == "julia.call" || jlname == "julia.call2") || return nothing
    fn = operands(call)[1]
    isa(fn, LLVM.Function) || return nothing
    nm = LLVM.name(fn)
    if jlname == "julia.call2"
        return nm in ("ijl_invoke", "jl_invoke") ? (3, false) : nothing
    end
    if nm in ("ijl_apply_generic", "jl_apply_generic")
        return (2, false)
    elseif nm in (
            "ijl_f__apply_latest",
            "ijl_f__call_latest",
            "ijl_f_invokelatest",
            "jl_f__apply_latest",
            "jl_f__call_latest",
            "jl_f_invokelatest",
        )
        return (3, false)
    elseif nm in ("ijl_f_invoke_in_world", "jl_f_invoke_in_world", "ijl_f__call_in_world", "jl_f__call_in_world")
        return (4, false)
    elseif nm in ("ijl_f__apply_iterate", "jl_f__apply_iterate")
        return (4, true)
    end
    return nothing
end

"""
    readonly_operands(call::LLVM.Instruction)::Vector{Int}

The operands of a dynamic call whose data the readonly check left to the method
the call reaches (`defer_readonly_operand!`), numbered as the dynamic-call rules
see them: 1 is the called function, 2 its first argument.
"""
function readonly_operands(call::LLVM.Instruction)::Vector{Int}
    haskey(metadata(call), READONLY_OPERANDS_MD) || return Int[]
    mds = operands(metadata(call)[READONLY_OPERANDS_MD])[1]::MDString
    return Int[parse(Int, s) for s in split(Base.convert(String, mds), ",")]
end

"""
    defer_readonly_operand!(call::LLVM.CallInst, val::LLVM.Value)::Bool

Whether the readonly check can leave `val`, an operand of `call`, to the method a
dynamic call reaches (`dynamic_call_operands`). It can when `val` is the called
function or, unless the arguments are splatted, one of its arguments. The
positions are then recorded on `call` (`readonly_operands`), and the rules of
`src/rules/jitrules.jl` compile the derivative of that method with the check on
them. What the call returns is followed like the other values the walk meets.
"""
function defer_readonly_operand!(call::LLVM.CallInst, @nospecialize(val::LLVM.Value))::Bool
    dyn = dynamic_call_operands(call)
    dyn === nothing && return false
    first, splat = dyn
    positions = readonly_operands(call)
    for (i, op) in enumerate(arg_operands_view(call))
        op == val || continue
        (i < first || (splat && i != first)) && return false
        push!(positions, i - first + 1)
    end
    metadata(call)[READONLY_OPERANDS_MD] =
        MDNode(LLVM.Metadata[MDString(join(sort!(unique!(positions)), ","))])
    return true
end

"""
    is_readonly(attr::LLVM.Attribute)::Bool

Whether `attr` on its own establishes that the function or argument position it
is attached to is only read from. That is `readonly` / `readnone`, and on LLVM
16+ a `memory` effect whose modref is read-only.
"""
Base.@assume_effects :removable :foldable :nothrow function is_readonly(attr::LLVM.Attribute)::Bool
    if kind(attr) == kind(EnumAttribute("readonly"))
        return true
    end
    if kind(attr) == kind(EnumAttribute("readnone"))
        return true
    end
    if LLVM.version().major > 15 && isa(attr, LLVM.EnumAttribute)
        if kind(attr) == kind(EnumAttribute("memory"))
            if is_readonly(MemoryEffect(value(attr)))
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
    if LLVM.name(f) == "llvm.julia.gc_preserve_begin" ||
       LLVM.name(f) == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(function_attributes(f))
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
    if LLVM.name(f) == "llvm.julia.gc_preserve_begin" ||
       LLVM.name(f) == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(function_attributes(cur))
        if kind(attr) == kind(EnumAttribute("readnone"))
            return true
        end
        if LLVM.version().major > 15
            if kind(attr) == kind(EnumAttribute("memory"))
                if is_readnone(MemoryEffect(value(attr)))
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
    if LLVM.name(f) == "llvm.julia.gc_preserve_begin" ||
       LLVM.name(f) == "llvm.julia.gc_preserve_end"
        return true
    end
    for attr in collect(function_attributes(cur))
        if kind(attr) == kind(EnumAttribute("readnone"))
            return true
        end
        if kind(attr) == kind(EnumAttribute("writeonly"))
            return true
        end
        if LLVM.version().major > 15
            if kind(attr) == kind(EnumAttribute("memory"))
                if is_writeonly(MemoryEffect(value(attr)))
                    return true
                end
            end
        end
    end
    return false
end

function set_readonly!(fn::LLVM.Function)
    attrs = collect(function_attributes(fn))
    if LLVM.version().major <= 15
        if !any(kind(attr) == kind(EnumAttribute("readonly")) for attr in attrs) &&
           !any(kind(attr) == kind(EnumAttribute("readnone")) for attr in attrs)
            if any(kind(attr) == kind(EnumAttribute("writeonly")) for attr in attrs)
                delete!(function_attributes(fn), EnumAttribute("writeonly"))
                push!(function_attributes(fn), EnumAttribute("readnone"))
            else
                push!(function_attributes(fn), EnumAttribute("readonly"))
            end
            return true
        end
        return false
    else
        for attr in attrs
            if kind(attr) == kind(EnumAttribute("memory"))
                old = MemoryEffect(value(attr))
                eff = set_readonly(old)
                push!(function_attributes(fn), EnumAttribute("memory", eff.data))
                return old != eff
            end
        end
        push!(
            function_attributes(fn),
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
    if haskey(functions(mod), name)
        F = functions(mod)[name]
        PT = LLVM.PointerType(FT)
        if value_type(F) != PT
            F = LLVM.const_pointercast(F, PT)
        end
    else
        F = LLVM.Function(mod, name, FT)
        if LLVM.name(F) != name
            # another kind of global value uses the name, so LLVM gave the function a
            # different one that nothing would define
            LLVM.erase!(F)
            throw(ArgumentError("Module already contains a global value called \"$name\" that is not a function"))
        end
        for attr in attrs
            push!(function_attributes(F), attr)
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
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, fty = declare_pgcstack!(mod)
    return call!(B, fty, func, LLVM.Value[], name)
end

function get_pgcstack(func::LLVM.Function)
    entry_bb = first(blocks(func))
    mod = LLVM.parent(func)
    pgcstack_func, _ = declare_pgcstack!(mod)

    # A @cfunction wrapper fetches its pgcstack with julia.get_pgcstack_or_new,
    # which adopts the calling thread when it is not a Julia thread yet. It has
    # to stay the first getter of the entry block: FinalLowerGC roots the GC
    # frame in whichever getter comes first, and a plain getter placed ahead of
    # the adopting one dereferences a NULL pgcstack on a foreign thread.
    if haskey(functions(mod), "julia.get_pgcstack_or_new")
        or_new = functions(mod)["julia.get_pgcstack_or_new"]
        for I in instructions(entry_bb)
            if I isa LLVM.CallInst && called_operand(I) == or_new
                return I
            end
        end
    end

    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_operand(I) == pgcstack_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func::LLVM.Function, @nospecialize(PB::Union{Nothing, LLVM.IRBuilder}) = nothing)
    for i in 1:length(LLVM.parameters(func))
        for attr in collect(LLVM.parameter_attributes(func, i))
            if attr isa LLVM.EnumAttribute
                if kind(attr) == swiftself_kind
                    return parameters(func)[i]
                end
            end
        end
    end

    pgs = get_pgcstack(func)
    return if pgs isa Nothing
        entry_bb = first(blocks(func))
        if PB !== nothing &&
                (LLVM.name(Base.position(PB)) == "allocsForInversion" || Base.position(PB) == entry_bb)
            # emit it with the caller's builder, which we don't own
            emit_pgcstack(PB, "newly_emitted_pgc_stack")
        else
            @dispose B = IRBuilder() begin
                if isempty(instructions(entry_bb))
                    position!(B, entry_bb)
                else
                    position!(B, first(instructions(entry_bb)))
                end
                emit_pgcstack(B, "newly_emitted_pgc_stack")
            end
        end
    else
        entry_bb = first(blocks(func))
        fst = first(instructions(entry_bb))
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
    mod = LLVM.parent(f)
    run!(AlwaysInlinerPass(), mod)
    if !haskey(functions(mod), "julia.get_pgcstack")
        return
    end
    getter = functions(mod)["julia.get_pgcstack"]
    calls = LLVM.CallInst[]
    for use in uses(getter)
        call = user(use)
        if call isa LLVM.CallInst && LLVM.parent(LLVM.parent(call)) == f
            push!(calls, call)
        end
    end
    if isempty(calls)
        return
    end
    T_pgcstack = LLVM.return_type(LLVM.function_type(getter))
    pgcstack = @dispose B = IRBuilder() begin
        position!(B, first(instructions(first(blocks(f)))))
        # Before 1.12, a `Ptr{Cvoid}` is an integer in Julia's IR.
        if value_type(arg) isa LLVM.IntegerType
            inttoptr!(B, arg, T_pgcstack)
        else
            bitcast!(B, arg, T_pgcstack)
        end
    end
    for call in calls
        replace_uses!(call, pgcstack)
        LLVM.API.LLVMInstructionEraseFromParent(call)
    end
    return
end

@inline enum_attr_kind(kind::String) = LLVM.API.LLVMGetEnumAttributeKindForName(kind, Csize_t(length(kind)))

const swiftself_kind = enum_attr_kind("swiftself")

Base.@assume_effects :removable :foldable :nothrow function has_swiftself(fn::LLVM.Function)::Bool
    for i in 1:length(LLVM.parameters(fn))
        for attr in collect(LLVM.parameter_attributes(fn, i))
            if attr isa LLVM.EnumAttribute
                if kind(attr) == swiftself_kind
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
    for i in 1:length(LLVM.parameters(fn))
        for attr in collect(LLVM.parameter_attributes(fn, i))
            if attr isa LLVM.EnumAttribute
                if kind(attr) == swiftself_kind
                    return i
                end
            elseif attr isa LLVM.StringAttribute
                if kind(attr) == "gcstack"
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
    ekind = LLVM.kind(attr)
    for attr in collect(function_attributes(fn))
        if attr isa LLVM.EnumAttribute
            if kind(attr) == ekind
                return true
            end
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function has_fn_attr(fn::LLVM.Function, attr::LLVM.StringAttribute)::Bool
    ekind = LLVM.kind(attr)
    for attr in collect(function_attributes(fn))
        if attr isa LLVM.StringAttribute
            if kind(attr) == ekind
                return true
            end
        end
    end
    return false
end

Base.@assume_effects :removable :foldable :nothrow function has_arg_attr(fn::LLVM.Function, i::Int, attr::LLVM.StringAttribute)::Bool
    ekind = LLVM.kind(attr)
    for attr in collect(parameter_attributes(fn, i))
        if attr isa LLVM.StringAttribute
            if kind(attr) == ekind
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
    @assert LLVM.parent(a) == LLVM.parent(b)
    for inst in instructions(LLVM.parent(a))
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
    ctx = LLVM.context(src)
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
        unsafe_delete!(bb, inst)
    end
end
function eraseInst(bb::LLVM.Module, inst::LLVM.Function)
    @static if isdefined(LLVM, Symbol("erase!"))
        LLVM.erase!(inst)
    else
        unsafe_delete!(bb, inst)
    end
end
function eraseInst(bb::LLVM.Module, inst::LLVM.GlobalVariable)
    @static if isdefined(LLVM, Symbol("erase!"))
        LLVM.erase!(inst)
    else
        unsafe_delete!(bb, inst)
    end
end

function unique_gcmarker!(func::LLVM.Function)
    entry_bb = first(blocks(func))
    pgcstack_func, _ = declare_pgcstack!(LLVM.parent(func))

    found = LLVM.CallInst[]
    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_operand(I) == pgcstack_func
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
    ty = LLVM.value_type(val)::LLVM.LLVMType
    for i in idxs
        if isa(ty, LLVM.ArrayType)
            ty = eltype(ty)::LLVM.LLVMType
        else
            @assert isa(ty, LLVM.StructType)
            ty = elements(ty)[i+1]::LLVM.LLVMType
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
            @assert length(ctype) == length(elements(tape))
            for (i, ty) in enumerate(elements(tape))
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(elements(ctype)) == length(elements(tape))
            for (i, ty) in enumerate(elements(tape))
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
            @assert length(ctype) == length(tape)
            for i = 1:length(tape)
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, eltype(tape), prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(elements(ctype)) == length(tape)
            for i = 1:length(tape)
                ln = copy(lidxs)
                push!(ln, i - 1)
                rn = copy(ridxs)
                push!(rn, i - 1)
                prev = calling_conv_fixup(builder, val, eltype(tape), prev, ln, rn, emesg)
            end
            return prev
        end
    end

    if isa(tape, LLVM.IntegerType) &&
       LLVM.width(tape) == 1 &&
       LLVM.width(ctype) != LLVM.width(tape)
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
       LLVM.addrspace(tape) == LLVM.addrspace(ctype)
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
    if isa(ctype, LLVM.ArrayType) && length(ctype) == 1 && eltype(ctype) == tape
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
