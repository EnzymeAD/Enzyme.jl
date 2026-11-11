# Checkpointed loops: `EnzymeCore.checkpoint_for` and `checkpoint_while` become
# Enzyme's `__enzyme_checkpoint_for` and `__enzyme_checkpoint_while` (see
# enzyme/checkpoint.h), whose reverse mode is driven by an external
# checkpointing scheme.
#
# `_checkpoint_for(scheme, data, start, n, box)` calls `checkpoint_step(box, i)`
# for each `i`, and `_checkpoint_while(scheme, data, box)` calls
# `checkpoint_while_step(box)` until it returns false; all are kept out of line
# (`handle_compiled` marks them). Each call to them is replaced by
#
#     __enzyme_checkpoint_for(step, start, n, enzyme_scheme, scheme, data,
#                             box, [pgcstack])
#     __enzyme_checkpoint_while(step, enzyme_scheme, scheme, data,
#                               box, [pgcstack])
#
# where `step(i, box, [pgcstack])` calls the compiled step, which takes the
# task's GC stack (swiftself) first, and the index last if it has one. The box
# is the first argument after the index, so a scheme that copies the state
# itself finds it as the first word of the step's environment. Enzyme then
# lowers the marker to its loop.

const CHECKPOINT_FOR_ATTR = "enzymejl_checkpoint_for"
const CHECKPOINT_WHILE_ATTR = "enzymejl_checkpoint_while"
const CHECKPOINT_STEP_ATTR = "enzymejl_checkpoint_step"

function checkpoint_step_callee(F::LLVM.Function)
    for bb in blocks(F), inst in instructions(bb)
        inst isa LLVM.CallInst || continue
        callee = LLVM.called_operand(inst)
        if callee isa LLVM.Function &&
                has_fn_attr(callee, StringAttribute(CHECKPOINT_STEP_ATTR))
            return callee
        end
    end
    return nothing
end

function is_swiftself(f::LLVM.Function, i::Integer)
    return any(
        a -> a isa EnumAttribute && kind(a) == "swiftself",
        collect(parameter_attributes(f, i)),
    )
end

# `step(i, box, [pgcstack])`, calling `G([pgcstack,] box, i)` for a for loop,
# and returning `G([pgcstack,] box)` for a while loop.
function checkpoint_step_wrapper!(mod::LLVM.Module, G::LLVM.Function, indexed::Bool)
    name = "enzymejl_ckpt_step." * LLVM.name(G)
    haskey(functions(mod), name) && return functions(mod)[name]
    Gparams = parameters(G)
    T_i64 = LLVM.Int64Type()
    nargs = indexed ? length(Gparams) - 1 : length(Gparams)
    indexed && @assert value_type(Gparams[end]) == T_i64
    # G's parameters but the index: the GC stack (if any) goes last.
    order = [i for i in 1:nargs if !is_swiftself(G, i)]
    append!(order, [i for i in 1:nargs if is_swiftself(G, i)])
    rettype = indexed ? LLVM.VoidType() : LLVM.return_type(function_type(G))
    FT = LLVM.FunctionType(
        rettype,
        LLVMType[T_i64, (value_type(Gparams[i]) for i in order)...],
    )
    S = LLVM.Function(mod, name, FT)
    linkage!(S, LLVM.API.LLVMInternalLinkage)
    # The body's parameters keep their Julia type annotations.
    for (j, i) in enumerate(order)
        for a in collect(parameter_attributes(G, i))
            a isa EnumAttribute && kind(a) == "swiftself" && continue
            push!(parameter_attributes(S, j + 1), a)
        end
    end
    builder = IRBuilder()
    position!(builder, BasicBlock(S, "entry"))
    Sparams = parameters(S)
    args = Vector{LLVM.Value}(undef, length(Gparams))
    for (j, i) in enumerate(order)
        args[i] = Sparams[j + 1]
    end
    indexed && (args[end] = Sparams[1])
    call = call!(builder, function_type(G), G, args)
    callconv!(call, callconv(G))
    for i in 1:nargs
        if is_swiftself(G, i)
            LLVM.API.LLVMAddCallSiteAttribute(call, i, EnumAttribute("swiftself", 0))
        end
    end
    indexed ? ret!(builder) : ret!(builder, call)
    dispose(builder)
    return S
end

# Runs before the Julia pipeline, which would drop constant arguments of
# `_checkpoint_for`: the marker is an opaque call it leaves alone.
function rewrite_checkpoint_calls!(mod::LLVM.Module)
    changed = false
    for (attr, indexed) in ((CHECKPOINT_FOR_ATTR, true), (CHECKPOINT_WHILE_ATTR, false))
        changed |= rewrite_checkpoint_calls!(mod, attr, indexed)
    end
    return changed
end

function rewrite_checkpoint_calls!(mod::LLVM.Module, attr::String, indexed::Bool)
    markers = LLVM.Function[
        f for f in functions(mod) if has_fn_attr(f, StringAttribute(attr))
    ]
    isempty(markers) && return false

    T_i32 = LLVM.Int32Type()
    T_i64 = LLVM.Int64Type()
    T_ptr = LLVM.PointerType(LLVM.Int8Type())
    scheme_marker = if haskey(globals(mod), "enzyme_scheme")
        globals(mod)["enzyme_scheme"]
    else
        LLVM.GlobalVariable(mod, T_i32, "enzyme_scheme")
    end
    name = indexed ? "__enzyme_checkpoint_for" : "__enzyme_checkpoint_while"
    declFT = LLVM.FunctionType(
        LLVM.VoidType(),
        indexed ? LLVMType[T_ptr, T_i64, T_i64] : LLVMType[T_ptr];
        vararg = true,
    )
    decl = if haskey(functions(mod), name)
        functions(mod)[name]
    else
        LLVM.Function(mod, name, declFT)
    end

    for F in markers
        G = checkpoint_step_callee(F)
        G === nothing && error("Enzyme: $(LLVM.name(F)) does not call its step")
        S = checkpoint_step_wrapper!(mod, G, indexed)
        # F's parameters: [pgcstack,] scheme, data, [start, n,] box
        off = !isempty(parameters(F)) && is_swiftself(F, 1) ? 1 : 0
        nfixed = indexed ? 4 : 2
        for u in collect(LLVM.uses(F))
            ci = LLVM.user(u)
            ci isa LLVM.CallInst && LLVM.called_operand(ci) == F || continue
            args = collect(LLVM.arguments(ci))
            pgcstack = args[1:off]
            scheme, data = args[(off + 1):(off + 2)]
            bounds = args[(off + 3):(off + nfixed)]
            box = args[(off + nfixed + 1):end]
            builder = IRBuilder()
            position!(builder, ci)
            aspointer(v) = value_type(v) isa LLVM.IntegerType ? inttoptr!(builder, v, T_ptr) : v
            marker = load!(builder, T_i32, scheme_marker)
            call!(
                builder,
                declFT,
                decl,
                LLVM.Value[
                    S, bounds..., marker, aspointer(scheme), aspointer(data), box..., pgcstack...,
                ],
            )
            dispose(builder)
            LLVM.API.LLVMInstructionEraseFromParent(ci)
        end
    end
    return true
end

# Runs just before differentiation: Enzyme's loop function has a fixed
# signature the pipeline must not change.
lower_checkpoint_calls!(mod::LLVM.Module) = API.EnzymeLowerCheckpointMarkers(mod) != 0
