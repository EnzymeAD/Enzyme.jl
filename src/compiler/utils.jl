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

function set_readonly(effect::MemoryEffect)
    data = UInt32(0)
    for loc in (ArgMem, InaccessibleMem, Other)
        data = UInt32(set_readonly(getModRef(effect, loc))) << getLocationPos(loc)
    end
    return MemoryEffect(data)
end

function is_readonly(mri::ModRefInfo)
    return mri == MRI_NoModRef || mri == MRI_Ref
end

function is_readnone(mri::ModRefInfo)
    return mri == MRI_NoModRef
end

function is_writeonly(mri::ModRefInfo)
    return mri == MRI_NoModRef || mri == MRI_Mod
end

for n in (:is_readonly, :is_readnone, :is_writeonly)
    @eval begin
        function $n(memeffect::MemoryEffect)
            return $n(getModRef(memeffect))
        end
    end
end

function is_noreturn(f::LLVM.Function)
    for attr in collect(function_attributes(f))
        if kind(attr) == kind(EnumAttribute("noreturn"))
            return true
        end
    end
    return false
end

function is_readonly(f::LLVM.Function)
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
        if kind(attr) == kind(EnumAttribute("readonly"))
            return true
        end
        if kind(attr) == kind(EnumAttribute("readnone"))
            return true
        end
        if LLVM.version().major > 15
            if kind(attr) == kind(EnumAttribute("memory"))
                if is_readonly(MemoryEffect(value(attr)))
                    return true
                end
            end
        end
    end
    return false
end

function is_readnone(f::LLVM.Function)
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

function is_writeonly(f::LLVM.Function)
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
    name::AbstractString,
    FT::LLVM.FunctionType,
    attrs = [],
)
    if haskey(functions(mod), name)
        F = functions(mod)[name]
        PT = LLVM.PointerType(FT)
        if value_type(F) != PT
            F = LLVM.const_pointercast(F, PT)
        end
    else
        F = LLVM.Function(mod, name, FT)
        for attr in attrs
            push!(function_attributes(F), attr)
        end
    end
    return F, FT
end

function get_function!(builderF, mod::LLVM.Module, name)
    get_function!(mod, name, builderF())
end

T_ppjlvalue() = LLVM.PointerType(LLVM.PointerType(LLVM.StructType(LLVMType[])))

@inline function get_base_object(v)
    if isa(v, LLVM.AddrSpaceCastInst) || isa(v, LLVM.BitCastInst)
        return get_base_object(operands(v)[1])
    end
    if isa(v, LLVM.GetElementPtrInst)
        return get_base_object(operands(v)[1])
    end
    return v
end

function declare_pgcstack!(mod)
    get_function!(
        mod,
        "julia.get_pgcstack",
        LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue())),
    )
end

function emit_pgcstack(B)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, fty = declare_pgcstack!(mod)
    return call!(B, fty, func)
end

function get_pgcstack(func)
    entry_bb = first(blocks(func))
    pgcstack_func = declare_pgcstack!(LLVM.parent(func))

    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_operand(I) == pgcstack_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func, PB = nothing)
    for (i, v) in enumerate(parameters(func))
        if any(
            map(
                k -> kind(k) == kind(EnumAttribute("swiftself")),
                collect(parameter_attributes(func, i)),
            ),
        )
            return v
        end
    end

    pgs = get_pgcstack(func)
    if pgs === nothing
        context(LLVM.parent(func))
        B = IRBuilder()
        entry_bb = first(blocks(func))
        if !isempty(instructions(entry_bb))
            position!(B, first(instructions(entry_bb)))
        else
            position!(B, entry_bb)
        end
        emit_pgcstack(B)
    else
        entry_bb = first(blocks(func))
        fst = first(instructions(entry_bb))
        if fst != pgs
            API.moveBefore(pgs, fst, PB === nothing ? C_NULL : PB.ref)
        end
        pgs
    end
end

function eraseInst(bb, inst)
    @static if isdefined(LLVM, Symbol("erase!"))
        LLVM.erase!(inst)
    else
        unsafe_delete!(bb, inst)
    end
end

function unique_gcmarker!(func)
    entry_bb = first(blocks(func))
    pgcstack_func = declare_pgcstack!(LLVM.parent(func))

    found = LLVM.CallInst[]
    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_operand(I) == pgcstack_func
            push!(found, I)
        end
    end
    if length(found) > 1
        for i = 2:length(found)
            LLVM.replace_uses!(found[i], found[1])
            ops = LLVM.collect(operands(found[i]))
            eraseInst(entry_bb, found[i])
        end
    end
    return nothing
end

@inline AnonymousStruct(::Type{U}) where {U<:Tuple} =
    NamedTuple{ntuple(i -> Symbol(i), Val(length(U.parameters))),U}

# recursively compute the eltype type indexed by idx[0], idx[1], ...
function recursive_eltype(val::LLVM.Value, idxs::Vector{Cuint})
    ty = LLVM.value_type(val)
    for i in idxs
        if isa(ty, LLVM.ArrayType)
            ty = eltype(ty)
        else
            @assert isa(ty, LLVM.StructType)
            ty = elements(ty)[i+1]
        end
    end
    return ty
end

# Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
# and that Bool -> i8, not i1
function calling_conv_fixup(
    builder,
    val::LLVM.Value,
    tape::LLVM.LLVMType,
    prev::LLVM.Value = LLVM.UndefValue(tape),
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
