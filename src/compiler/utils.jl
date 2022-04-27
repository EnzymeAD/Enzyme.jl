
function get_function!(builderF, mod, name)
    if haskey(functions(mod), name)
        return functions(mod)[name]
    else
        return builderF(mod, context(mod), name)
    end
end

if VERSION < v"1.7.0-DEV.1205"

declare_ptls!(mod) = get_function!(mod, "julia.ptls_states") do mod, ctx, name
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)

    funcT = LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue))
    LLVM.Function(mod, name, funcT)
end

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

function reinsert_gcmarker!(func)
	ptls = get_ptls(func) 
    if ptls === nothing
        B = Builder(context(LLVM.parent(func)))
        entry_bb = first(blocks(func))
        position!(B, first(instructions(entry_bb)))
        emit_ptls!(B)
	else
		ptls
    end
end

else

declare_pgcstack!(mod) = get_function!(mod, "julia.get_pgcstack") do mod, ctx, name
    T_jlvalue = LLVM.StructType(LLVMType[]; ctx)
    T_pjlvalue = LLVM.PointerType(T_jlvalue)
    T_ppjlvalue = LLVM.PointerType(T_pjlvalue)

    funcT = LLVM.FunctionType(LLVM.PointerType(T_ppjlvalue))
    LLVM.Function(mod, name, funcT)
end

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

function reinsert_gcmarker!(func)
	pgs = get_pgcstack(func)
    if pgs === nothing
        B = Builder(context(LLVM.parent(func)))
        entry_bb = first(blocks(func))
        position!(B, first(instructions(entry_bb)))
        emit_pgcstack(B)
	else
		pgs
    end
end

end