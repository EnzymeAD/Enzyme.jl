
function get_function!(mod::LLVM.Module, name::AbstractString, FT::LLVM.FunctionType, attrs=[])
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


if VERSION < v"1.7.0-DEV.1205"

declare_ptls!(mod) = get_function!(mod, "julia.ptls_states", LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue())))

function emit_ptls!(B)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func, fty = declare_ptls!(mod)
    return call!(B, fty, func)
end

function get_ptls(func)
    entry_bb = first(blocks(func))
    ptls_func = declare_ptls!(LLVM.parent(func))

    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_operand(I) == ptls_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func, PB=nothing)
	ptls = get_ptls(func) 
    if isnothing(ptls)
        B = IRBuilder()
        entry_bb = first(blocks(func))
        if !isempty(instructions(entry_bb))
            position!(B, first(instructions(entry_bb)))
        else
            position!(B, entry_bb)
        end
        emit_ptls!(B)
	else
        entry_bb = first(blocks(func))
        fst = first(instructions(entry_bb))
        if fst != ptls
            API.moveBefore(ptls, fst, PB === nothing ? C_NULL : PB.ref)
        end
		ptls
    end
end

function unique_gcmarker!(func)
    entry_bb = first(blocks(func))
    ptls_func = declare_ptls!(LLVM.parent(func))

    found = LLVM.CallInst[]
    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_operand(I) == ptls_func
            push!(found, I)
        end
    end
    if length(found) > 1
        for i in 2:length(found)
            LLVM.replace_uses!(found[i], found[1])
            Base.unsafe_delete!(entry_bb, found[i])
        end
    end
    return nothing
end

else

function declare_pgcstack!(mod) 
        get_function!(mod, "julia.get_pgcstack", LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue())))
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

function reinsert_gcmarker!(func, PB=nothing)
    for (i, v) in enumerate(parameters(func))
        if any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(func, i))))
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
        for i in 2:length(found)
            LLVM.replace_uses!(found[i], found[1])
            ops = LLVM.collect(operands(found[i]))
            Base.unsafe_delete!(entry_bb, found[i])
        end
    end
    return nothing
end
end

@inline AnonymousStruct(::Type{U}) where U<:Tuple = NamedTuple{ntuple(i->Symbol(i), Val(length(U.parameters))), U}

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
function calling_conv_fixup(builder, val::LLVM.Value, tape::LLVM.LLVMType, prev::LLVM.Value=LLVM.UndefValue(tape), lidxs::Vector{Cuint}=Cuint[], ridxs::Vector{Cuint}=Cuint[], emesg=nothing)::LLVM.Value
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
                push!(ln, i-1)
                rn = copy(ridxs)
                push!(rn, i-1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(elements(ctype)) == length(elements(tape))
            for (i, ty) in enumerate(elements(tape))
                ln = copy(lidxs)
                push!(ln, i-1)
                rn = copy(ridxs)
                push!(rn, i-1)
                prev = calling_conv_fixup(builder, val, ty, prev, ln, rn, emesg)
            end
            return prev
        end
    elseif isa(tape, LLVM.ArrayType)
        if isa(ctype, LLVM.ArrayType)
            @assert length(ctype) == length(tape)
            for i in 1:length(tape)
                ln = copy(lidxs)
                push!(ln, i-1)
                rn = copy(ridxs)
                push!(rn, i-1)
                prev = calling_conv_fixup(builder, val, eltype(tape), prev, ln, rn, emesg)
            end
            return prev
        end
        if isa(ctype, LLVM.StructType)
            @assert length(elements(ctype)) == length(tape)
            for i in 1:length(tape)
                ln = copy(lidxs)
                push!(ln, i-1)
                rn = copy(ridxs)
                push!(rn, i-1)
                prev = calling_conv_fixup(builder, val, eltype(tape), prev, ln, rn, emesg)
            end
            return prev
        end
    end

    if isa(tape, LLVM.IntegerType) && LLVM.width(tape) == 1 && LLVM.width(ctype) != LLVM.width(tape)
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
    if isa(tape, LLVM.PointerType) && isa(ctype, LLVM.PointerType) && LLVM.addrspace(tape) == LLVM.addrspace(ctype)
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
        if  emesg !== nothing
            emesg(io)
        end
        println(io, "ctype = ", ctype)
        println(io, "tape = ", tape)
        println(io, "val = ", val)
        println(io, "prev = ", prev)
        println(io, "lidxs = ", lidxs)
        println(io, "ridxs = ", ridxs)
        println(io, "tape_type(tape) = ", tape_type(tape))
        println(io, "convert(LLVMType, tape_type(tape)) = ", convert(LLVM.LLVMType, tape_type(tape); allow_boxed=true))
    end
    throw(AssertionError(msg2))
end
