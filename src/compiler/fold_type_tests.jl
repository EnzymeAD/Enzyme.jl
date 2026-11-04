# Strip only casts that preserve the complete type tag. In particular, a
# truncated ptrtoint is not an exact type comparison.
function type_test_operand(@nospecialize(v::LLVM.Value))
    for _ in 1:32
        if v isa LLVM.BitCastInst || v isa LLVM.AddrSpaceCastInst
            v = operands(v)[1]
        elseif v isa LLVM.PtrToIntInst && value_type(v) == LLVM.IntType(Sys.WORD_SIZE)
            v = operands(v)[1]
        elseif v isa LLVM.CallInst && called_operand(v) isa LLVM.Function &&
                name(called_operand(v)) == "julia.pointer_from_objref"
            v = first(arg_operands_view(v))
        else
            return v
        end
    end
    return nothing
end

# Read an address without interpreting it as a Julia object: an arbitrary
# pointer compared with typeof need not point to a valid object at all.
function type_test_constant(@nospecialize(v::LLVM.Value), enzyme_ctx::Union{EnzymeContext, Nothing})
    for _ in 1:32
        if v isa LLVM.ConstantInt
            value_type(v) == LLVM.IntType(Sys.WORD_SIZE) || return nothing
            return convert(UInt, v)
        elseif v isa LLVM.PointerNull
            return UInt(0)
        elseif v isa LLVM.ConstantExpr && opcode(v) in
                (LLVM.API.LLVMBitCast, LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMIntToPtr)
            v = operands(v)[1]
        elseif v isa LLVM.GlobalVariable
            found = named_global(v, inserted_values(enzyme_ctx))
            found === nothing && return nothing
            T = something(found)
            return T isa Type ? UInt(unsafe_to_pointer(T)) : nothing
        elseif v isa LLVM.LoadInst && operands(v)[1] isa LLVM.GlobalVariable
            found = julia_value_of_slot(operands(v)[1], enzyme_ctx)
            found === nothing && return nothing
            T = something(found)
            return T isa Type ? UInt(unsafe_to_pointer(T)) : nothing
        else
            return nothing
        end
    end
    return nothing
end

# julia.typeof produces the GC header tag, which is a small integer for some
# built-in types and the datatype address otherwise. Do not confuse that tag
# with the datatype pointer returned by the Julia-level typeof function.
function type_test_tag(@nospecialize(T::Type))
    address = UInt(unsafe_to_pointer(T))
    small_types = Ptr{UInt}(cglobal(:jl_small_typeof))
    for tag in 0:(JL_MAX_TAGS - 1)
        if unsafe_load(small_types, (tag << 4) ÷ sizeof(UInt) + 1) == address
            return UInt(tag << 4)
        end
    end
    return address
end

# A loop may carry several concrete types through a cycle of PHIs and selects.
# An exact type test is constant only when every leaf gives the same answer.
# Unknown leaves, conflicting answers and exhausted budgets preserve the test.
function constant_type_test(@nospecialize(root::LLVM.Value), target::UInt, enzyme_ctx::Union{EnzymeContext, Nothing})
    todo = LLVM.Value[root]
    seen = Set{LLVM.Value}()
    result = nothing
    budget = 256
    while !isempty(todo)
        v = pop!(todo)
        v in seen && continue
        push!(seen, v)
        if v isa LLVM.PHIInst
            budget -= length(LLVM.incoming(v))
            budget >= 0 || return nothing
            for (value, _) in LLVM.incoming(v)
                push!(todo, value)
            end
        elseif v isa LLVM.SelectInst
            budget -= 2
            budget >= 0 || return nothing
            push!(todo, operands(v)[2], operands(v)[3])
        elseif v isa LLVM.BitCastInst || v isa LLVM.AddrSpaceCastInst
            budget -= 1
            budget >= 0 || return nothing
            push!(todo, operands(v)[1])
        else
            legal, T, _ = abs_typeof(v, enzyme_ctx)
            legal && isconcretetype(T) || return nothing
            answer = type_test_tag(T) == target
            result === nothing || result == answer || return nothing
            result = answer
        end
    end
    return result
end

# Remove impossible type arms before Enzyme's path-insensitive type analysis
# combines their incompatible object layouts with the reachable ones.
function fold_type_tests!(mod::LLVM.Module, enzyme_ctx::Union{EnzymeContext, Nothing})
    changed = 0
    for f in functions(mod), bb in blocks(f), inst in instructions(bb)
        inst isa LLVM.ICmpInst || continue
        pred = LLVM.API.LLVMGetICmpPredicate(inst)
        pred in (LLVM.API.LLVMIntEQ, LLVM.API.LLVMIntNE) || continue
        lhs, rhs = operands(inst)
        for (value, constant) in ((lhs, rhs), (rhs, lhs))
            value = type_test_operand(value)
            value isa LLVM.CallInst || continue
            callee = called_operand(value)
            callee isa LLVM.Function && name(callee) == "julia.typeof" || continue
            target = type_test_constant(constant, enzyme_ctx)
            target === nothing && continue
            answer = constant_type_test(first(arg_operands_view(value)), target, enzyme_ctx)
            answer === nothing && continue
            replace_uses!(inst, LLVM.ConstantInt(pred == LLVM.API.LLVMIntEQ ? answer : !answer))
            changed += 1
            break
        end
    end
    if changed > 0
        run!(SimplifyCFGPass(), mod)
    end
    return changed
end
