
function get_function!(mod::LLVM.Module, name, FT::LLVM.FunctionType, attrs=[])
    if haskey(functions(mod), name)
        F = functions(mod)[name]
        PT = LLVM.PointerType(FT)
        if llvmtype(F) != PT
            F = LLVM.const_pointercast(F, PT)
        end
    else
        F = LLVM.Function(mod, name, FT)
        for attr in attrs
            push!(function_attributes(F), attr)
        end
    end
    return F
end

function get_function!(builderF, mod::LLVM.Module, name)
    get_function!(mod, name, builderF(context(mod)))
end


T_ppjlvalue(ctx) = LLVM.PointerType(LLVM.PointerType(LLVM.StructType(LLVMType[]; ctx)))

if VERSION < v"1.7.0-DEV.1205"

declare_ptls!(mod) = get_function!(mod, "julia.ptls_states", LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue(context(mod)))))

function emit_ptls!(B)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func = declare_ptls!(mod)
    return call!(B, func)
end

function get_ptls(func)
    entry_bb = first(blocks(func))
    ptls_func = declare_ptls!(LLVM.parent(func))

    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_value(I) == ptls_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func, PB=nothing)
	ptls = get_ptls(func) 
    if ptls === nothing
        B = Builder(context(LLVM.parent(func)))
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

else

declare_pgcstack!(mod) = get_function!(mod, "julia.get_pgcstack", LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue(context(mod)))))

function emit_pgcstack(B)
    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    mod = LLVM.parent(fn)
    func = declare_pgcstack!(mod)
    return call!(B, func)
end

function get_pgcstack(func)
    entry_bb = first(blocks(func))
    pgcstack_func = declare_pgcstack!(LLVM.parent(func))

    for I in instructions(entry_bb)
        if I isa LLVM.CallInst && called_value(I) == pgcstack_func
            return I
        end
    end
    return nothing
end

function reinsert_gcmarker!(func, PB=nothing)
	pgs = get_pgcstack(func)
    if pgs === nothing
        B = Builder(context(LLVM.parent(func)))
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

end

AnonymousStruct(::Type{U}) where U<:Tuple = NamedTuple{ntuple(i->Symbol(i), Val(length(U.parameters))), U}
