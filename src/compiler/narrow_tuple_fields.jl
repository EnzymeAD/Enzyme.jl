# Identify a runtime type test of a checked field from an immutable tuple.
function tuple_field_type_test(term::LLVM.Instruction)
    if term isa LLVM.SwitchInst
        cond = operands(term)[1]
    elseif term isa LLVM.BrInst && LLVM.API.LLVMGetNumSuccessors(term) == 2
        # Julia can lower a two-way type test to an equality branch instead
        # of a switch (notably with Julia 1.12).
        cmp = LLVM.Value(LLVM.API.LLVMGetCondition(term))
        cmp isa LLVM.ICmpInst || return nothing
        LLVM.API.LLVMGetICmpPredicate(cmp) in
            (LLVM.API.LLVMIntEQ, LLVM.API.LLVMIntNE) || return nothing
        lhs, rhs = operands(cmp)
        if rhs isa LLVM.Constant
            cond = lhs
        elseif lhs isa LLVM.Constant
            cond = rhs
        else
            return nothing
        end
    else
        return nothing
    end
    while true
        if cond isa LLVM.PtrToIntInst || cond isa LLVM.AddrSpaceCastInst || cond isa LLVM.BitCastInst
            cond = operands(cond)[1]
        elseif cond isa LLVM.CallInst && called_operand(cond) isa LLVM.Function &&
                name(called_operand(cond)) == "julia.pointer_from_objref"
            cond = first(arg_operands_view(cond))
        else
            break
        end
    end
    cond isa LLVM.CallInst || return nothing
    cf = called_operand(cond)
    cf isa LLVM.Function && name(cf) == "julia.typeof" || return nothing
    source = first(arg_operands_view(cond))
    source isa LLVM.CallInst || return nothing
    sf = called_operand(source)
    sf isa LLVM.Function || return nothing
    name(sf) in ("jl_get_nth_field_checked", "ijl_get_nth_field_checked") || return nothing
    legal, tuplety, _ = abs_typeof(first(arg_operands_view(source)))
    legal && tuplety <: Tuple && isconcretetype(tuplety) || return nothing
    return source
end

# Give each runtime type arm its own read from an immutable tuple. A boxed
# heterogeneous field otherwise shares one SSA value between distinct layouts,
# causing Enzyme's path-insensitive type analysis to combine their pointer depths.
# Preserve the original checked read and type test; only rematerialize the
# already-checked field where the original read strictly dominates the type arm,
# and only redirect uses that the new read dominates.
function narrow_tuple_fields!(mod::LLVM.Module)
    changed = 0
    for f in functions(mod)
        bbs = collect(blocks(f))
        isempty(bbs) && continue
        tests = Pair{LLVM.Instruction, LLVM.CallInst}[]
        for bb in bbs
            term = terminator(bb)
            term === nothing && continue
            source = tuple_field_type_test(term)
            source === nothing && continue
            push!(tests, term => source)
        end
        isempty(tests) && continue

        tree = LLVM.DomTree(f)
        for (term, source) in tests
            users = Set{LLVM.Value}()
            for use in LLVM.uses(source)
                push!(users, LLVM.user(use))
            end
            successors = Set{LLVM.BasicBlock}()
            for i in 0:(Int(LLVM.API.LLVMGetNumSuccessors(term)) - 1)
                push!(successors, LLVM.BasicBlock(LLVM.API.LLVMGetSuccessor(term, i)))
            end
            for succ in successors
                LLVM.parent(source) != succ || continue
                anchor = nothing
                for inst in instructions(succ)
                    inst isa LLVM.PHIInst && continue
                    anchor = inst
                    break
                end
                LLVM.dominates(tree, source, anchor) || continue
                builder = IRBuilder()
                position!(builder, anchor)
                clone = LLVM.Value(LLVM.API.LLVMInstructionClone(source))
                LLVM.API.LLVMInsertIntoBuilderWithName(builder, clone, "tuplefield.narrow")
                dispose(builder)
                replaced = false
                for user in users
                    # PHI operands are used on incoming edges, not in their block.
                    user isa LLVM.Instruction && !(user isa LLVM.PHIInst) || continue
                    LLVM.dominates(tree, clone, user) || continue
                    for i in 1:length(operands(user))
                        if operands(user)[i] == source
                            LLVM.API.LLVMSetOperand(user, i - 1, clone)
                            replaced = true
                        end
                    end
                end
                if replaced
                    changed += 1
                else
                    LLVM.API.LLVMInstructionEraseFromParent(clone)
                end
            end
        end
        dispose(tree)
    end
    return changed
end
