# Pass large aggregate arguments -- in practice the tape Enzyme hands a `diffe`
# function it calls without inlining (recursion) -- by reference instead of by value.
#
# The caller keeps each such tape in a `*_cache` alloca between the forward and the
# reverse sweep, then loads it whole and passes it by value. Once that alloca is split
# and promoted (InstCombine + SimpleGVN in `fixup_callconv!`, SROA later), every field
# becomes its own SSA value, and because the forward sweep stores the tape on only
# one of many paths, each field needs a phi at the block where the paths rejoin. For
# case 04 of the slow-compile corpus (EnzymeAD/Enzyme.jl#1156, 3+3 operators) that is
# 15 tapes x 276 fields -> ~9k phis x 25 predecessors in a single block, and
# InstCombine's identical-phi scan in `visitPHINode` is quadratic in phis per block.
#
# Passing the tape's address instead makes the cache alloca escape (into a
# `nocapture readonly` argument), so it is never promoted and no phi web forms. The
# callee loads the whole tape on entry, which is exactly what by-value passing did.

# Tapes with at least this many fields are passed by reference; 0 disables the rewrite.
const TapeByRefMinFields = Ref(16)

agg_nfields(@nospecialize(T::LLVM.LLVMType)) =
    T isa LLVM.StructType ? length(LLVM.elements(T)) :
    T isa LLVM.ArrayType ? length(T) : 0

function copy_callsite_attrs!(dst::LLVM.CallInst, src::LLVM.CallInst, idxmap)
    for (from, to) in idxmap
        n = LLVM.API.LLVMGetCallSiteAttributeCount(src, from)
        n == 0 && continue
        attrs = Vector{LLVM.API.LLVMAttributeRef}(undef, n)
        LLVM.API.LLVMGetCallSiteAttributes(src, from, attrs)
        for a in attrs
            LLVM.API.LLVMAddCallSiteAttribute(dst, to, a)
        end
    end
    return
end

function tape_byval_to_byref!(mod::LLVM.Module; minfields::Int = TapeByRefMinFields[])
    minfields <= 0 && return false
    dl = datalayout(mod)
    changed = false
    for f in collect(functions(mod))
        isdeclaration(f) && continue
        linkage(f) in (LLVM.API.LLVMPrivateLinkage, LLVM.API.LLVMInternalLinkage) || continue
        ft = LLVM.function_type(f)
        isvararg(ft) && continue
        ptys = LLVM.parameters(ft)
        # Only Enzyme's own tape argument (named `tapeArg`, no Julia type attributes).
        # User-level aggregates carry `enzyme_type`/`enzymejl_parmtype` attributes that
        # nested differentiation reads back from the cached IR; don't touch those.
        params = collect(LLVM.parameters(f))
        idx = Int[i for (i, T) in enumerate(ptys)
                  if agg_nfields(T) >= minfields && startswith(LLVM.name(params[i]), "tapeArg")]
        isempty(idx) && continue

        # Every use must be a direct call; anything else (address taken) keeps the ABI.
        calls = LLVM.CallInst[]
        ok = true
        for u in LLVM.uses(f)
            c = LLVM.user(u)
            if !(c isa LLVM.CallInst) || LLVM.called_operand(c) != f ||
                    any(==(f), collect(LLVM.arguments(c)))
                ok = false
                break
            end
            push!(calls, c)
        end
        ok || continue

        # `PointerType(T)` is `T*` under typed pointers and plain `ptr` otherwise.
        newptys = LLVM.LLVMType[i in idx ? LLVM.PointerType(T) : T for (i, T) in enumerate(ptys)]
        nft = LLVM.FunctionType(LLVM.return_type(ft), newptys)
        name = LLVM.name(f)
        LLVM.name!(f, name * ".byval")
        nf = LLVM.Function(mod, name, nft)
        linkage!(nf, linkage(f))
        LLVM.callconv!(nf, LLVM.callconv(f))

        # Placeholders for the loaded aggregates, materialized in a scratch block so the
        # clone can map the old by-value arguments to them.
        scratch = BasicBlock(nf, "scratch")
        entry_loads = Dict{Int, LLVM.LoadInst}()
        min_align = Dict{Int, Int}(i => LLVM.abi_alignment(dl, ptys[i]) for i in idx)
        value_map = Dict{LLVM.Value, LLVM.Value}()
        placeholders = Pair{LLVM.Instruction, Int}[]
        @dispose B = IRBuilder() begin
            position!(B, scratch)
            for (i, (op, np)) in enumerate(zip(LLVM.parameters(f), LLVM.parameters(nf)))
                LLVM.name!(np, LLVM.name(op))
                if i in idx
                    ph = load!(B, ptys[i], np)
                    value_map[op] = ph
                    push!(placeholders, ph => i)
                else
                    value_map[op] = np
                end
            end
            unreachable!(B)
        end
        clone_into!(nf, f; value_map, changes = LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly)
        # clone_into! appended the body after `scratch`; the cloned entry is the second block.
        entry = collect(blocks(nf))[2]
        for (i, np) in enumerate(LLVM.parameters(nf))
            i in idx || continue
            push!(parameter_attributes(nf, i), EnumAttribute("nocapture"))
            push!(parameter_attributes(nf, i), EnumAttribute("readonly"))
            push!(parameter_attributes(nf, i), EnumAttribute("nonnull"))
            push!(parameter_attributes(nf, i), EnumAttribute("dereferenceable", LLVM.storage_size(dl, ptys[i])))
        end
        @dispose B = IRBuilder() begin
            position!(B, first(instructions(entry)))
            # keep allocas first
            for inst in instructions(entry)
                inst isa LLVM.AllocaInst || (position!(B, inst); break)
            end
            for (ph, i) in placeholders
                ld = load!(B, ptys[i], LLVM.parameters(nf)[i], "tape.byref")
                entry_loads[i] = ld
                LLVM.replace_uses!(ph, ld)
                LLVM.erase!(ph)
            end
        end
        LLVM.erase!(scratch)

        # Rewrite call sites (re-collected: recursive calls now also live in the clone).
        calls = LLVM.CallInst[LLVM.user(u) for u in LLVM.uses(f)]
        for c in calls
            caller = LLVM.parent(LLVM.parent(c))
            args = collect(LLVM.arguments(c))
            newargs = LLVM.Value[]
            @dispose B = IRBuilder() begin
                for (i, a) in enumerate(args)
                    if !(i in idx)
                        push!(newargs, a)
                        continue
                    end
                    T = ptys[i]
                    al = LLVM.abi_alignment(dl, T)
                    # `load T, p` straight into the call with nothing in between that
                    # could write memory: hand the callee `p` itself. The callee loads the
                    # whole value on entry, so this is the same as passing it by value, and
                    # it keeps `p` (usually Enzyme's `*_cache` alloca) from being promoted
                    # into one SSA value per field -- which is where the phi web comes from.
                    if a isa LLVM.LoadInst && LLVM.parent(a) == LLVM.parent(c) &&
                            LLVM.value_type(LLVM.operands(a)[1]) == newptys[i] &&
                            LLVM.API.LLVMGetVolatile(a) == 0 &&
                            LLVM.API.LLVMGetOrdering(a) == LLVM.API.LLVMAtomicOrderingNotAtomic &&
                            !any(i -> i isa LLVM.StoreInst || i isa LLVM.CallInst, _insts_between(a, c))
                        push!(newargs, LLVM.operands(a)[1])
                        min_align[i] = min(min_align[i], LLVM.alignment(a))
                        continue
                    end
                    position!(B, first(instructions(first(blocks(caller)))))
                    slot = alloca!(B, T, "tape.slot")
                    LLVM.alignment!(slot, al)
                    position!(B, c)
                    st = store!(B, a, slot)
                    LLVM.alignment!(st, al)
                    push!(newargs, slot)
                end
                position!(B, c)
                nc = call!(B, nft, nf, newargs)
                LLVM.callconv!(nc, LLVM.callconv(c))
                # no `tail`: the callee now reads the caller's stack
                LLVM.API.LLVMSetTailCall(nc, false)
                LLVM.API.LLVMInstructionSetDebugLoc(nc, LLVM.API.LLVMInstructionGetDebugLoc(c))
                idxmap = Tuple{UInt32, UInt32}[]
                push!(idxmap, (typemax(UInt32), typemax(UInt32)))  # function index (~0U)
                push!(idxmap, (UInt32(0), UInt32(0)))  # return index
                for i in eachindex(args)
                    i in idx || push!(idxmap, (UInt32(i), UInt32(i)))
                end
                copy_callsite_attrs!(nc, c, idxmap)
                LLVM.replace_uses!(c, nc)
                LLVM.erase!(c)
            end
        end
        for (i, ld) in entry_loads
            LLVM.alignment!(ld, min_align[i])
            push!(parameter_attributes(nf, i), EnumAttribute("align", min_align[i]))
        end
        @assert isempty(LLVM.uses(f))
        LLVM.erase!(f)
        changed = true
    end
    return changed
end

function _insts_between(a::LLVM.Instruction, c::LLVM.Instruction)
    res = LLVM.Instruction[]
    i = LLVM.API.LLVMGetNextInstruction(a)
    while i != C_NULL && i != c.ref
        push!(res, LLVM.Value(i))
        i = LLVM.API.LLVMGetNextInstruction(i)
    end
    return res
end
