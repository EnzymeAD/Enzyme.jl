# Calling derivative-free callees natively instead of emitting them.
#
# Most of the code a differentiated function calls is never differentiated: callees
# that are inactive by an `EnzymeRules.inactive` rule, or whose argument types and
# return type are all guaranteed const. Emitting them into every thunk means Enzyme
# optimizes, analyzes and code generates the same code again for each thunk, although
# Julia compiles it natively for the primal anyway.
#
# The same holds for the primal of a function with a custom rule, on Julia 1.12 and
# 1.13: Enzyme calls the rule, not the primal, unless no rule applies to a constant call
# (its function, arguments and return are all constant, see `has_rule`), and then it
# calls the primal as is. So such a callee is called natively too, through a declaration
# the rule handlers recognize as they recognize an emitted one (see
# `bind_native_callees!`).
#
# During inference, `typeinf_edge` resolves a call of a derivative-free method to the
# code Julia's native interpreter and JIT produce for it, without inferring the method
# with `EnzymeInterpreter`, and `compileable_specialization` makes the optimizer emit an
# `:invoke` of that native CodeInstance.
#
# On Julia 1.12 and 1.13, GPUCompiler emits exactly the CodeInstances `ci_cache_populate`
# returns (`jl_emit_native`). A call of a MethodInstance or CodeInstance that is not in
# that list gets a small specsig stub, which boxes the arguments and calls `jl_invoke` on
# the callee; none of the callee's body, nor of its callees, is emitted.
# `ci_cache_populate` below also leaves out derivative-free callees `EnzymeInterpreter`
# did infer, and `bind_native_callees!` replaces each stub by a declaration bound to the
# native specsig entry point of the callee (see `invoke_codegen!`, which does the same
# for custom rules).
#
# On Julia 1.10 and 1.11 the call goes through `jl_invoke` instead (see the end of this
# file).

"""
    host_interp(interp::EnzymeInterpreter) -> Bool

Say if `interp` infers code for the host: it looks methods up in GPUCompiler's global
method table only. Back-ends stack their own overlay tables for device code, which
cannot call code Julia compiled for the host.
"""
host_interp(interp::Interpreter.EnzymeInterpreter) =
    method_tables(interp.method_table) == (GPUCompiler.GLOBAL_METHOD_TABLE,)

"""
    use_native_callees(interp) -> Bool

Say if `interp` resolves derivative-free callees to native code: it infers code for the
host (see [`host_interp`](@ref)), and Julia is not generating a package image, which
must not refer to code compiled in this session.
"""
use_native_callees(interp::Interpreter.EnzymeInterpreter) =
    ccall(:jl_generating_output, Cint, ()) == 0 && host_interp(interp)

"""
    is_native_edge(interp, x) -> Bool

Say if `x` is a derivative-free MethodInstance `interp` resolves to native code (see
`typeinf_edge` below). Codegen calls one it has no CodeInstance for through `jl_invoke`,
which `check_ir!` marks inactive. A callee with a custom rule is not derivative-free:
its call must reach the rule handlers.
"""
function is_native_edge(@nospecialize(interp), @nospecialize(x))::Bool
    interp isa Interpreter.EnzymeInterpreter || return false
    x isa Core.MethodInstance || return false
    use_native_callees(interp) || return false
    native = native_entry(interp, x)
    return native !== nothing && native[3] === :inactive
end

"""
    all_guaranteed_const(params, world) -> Bool

Say if every type in `params` is guaranteed const.
"""
function all_guaranteed_const(params::Core.SimpleVector, world::UInt)::Bool
    for T in params
        (T isa Type && guaranteed_const_nongen(T, world)) || return false
    end
    return true
end

# Julia 1.12 and 1.13: GPUCompiler emits exactly the CodeInstances it gathers for
# `jl_emit_native`, with `ci_cache_populate` on GPUCompiler 1.x and with CompilerCaching's
# `get_codeinfos` on 2.x. The overrides below follow those for these versions.
@static if v"1.12-beta3" <= VERSION < v"1.14-"

    """
        native_uses_specsig(specTypes, RT) -> Bool

    Say if Julia's native codegen compiles a function of signature `specTypes` and return
    type `RT` to a specialized entry point, as `uses_specsig` in codegen.cpp does with
    `prefer_specsig` off: when an argument or the return is unboxed, the return is a union,
    or all arguments are singletons. Otherwise it only emits the boxed `jl_fptr_args` ABI,
    although the CodeInstance may still report a specialized entry.
    """
    function native_uses_specsig(@nospecialize(specTypes::DataType), @nospecialize(RT))::Bool
        if !GPUCompiler.deserves_retbox(RT) && !Base.issingletontype(RT) && RT !== Bool
            return true
        end
        RT isa Union && return true
        all_singletons = true
        for T in specTypes.parameters
            if isconcretetype(T) && !Base.issingletontype(T) && !GPUCompiler.deserves_argbox(T)
                return true
            end
            Base.issingletontype(T) || (all_singletons = false)
        end
        return all_singletons
    end

    """
        native_entry(interp::EnzymeInterpreter, mi::MethodInstance)

    Return the native `CodeInstance` of `mi`, its specialized entry point and how Enzyme
    handles a call of `mi` when Enzyme never differentiates the code of `mi` and can call
    its natively compiled code instead, and `nothing` otherwise.

    `mi` must either be derivative-free (`:inactive`): have an `EnzymeRules.inactive` rule,
    or take and return only guaranteed-const types, and no other rule or special handling.
    Or it must have a custom rule that `interp` applies (`:frule` or `:rrule`, see
    `Interpreter.enzyme_call_kind`): Enzyme calls the rule instead of `mi`, and calls `mi`
    itself only when no rule applies to a constant call (its function, arguments and return
    are all constant, see `has_rule`), which needs no derivative of `mi` either. Julia's
    native interpreter infers `mi` and its JIT compiles it, exactly as for an ordinary call
    (see [`codeinst`](@ref)), and it must get a specialized entry point for the same
    MethodInstance (see [`native_uses_specsig`](@ref)). A small derivative-free method
    Julia would inline (not the `:call` convention of [`call_convention`](@ref)) gets
    `nothing`, so that it is inlined as before. A method with a rule is never inlined
    (`NoInlineCallInfo`).
    """
    function native_entry(interp::Interpreter.EnzymeInterpreter, mi::Core.MethodInstance)
        CC = Core.Compiler
        specTypes = mi.specTypes
        specTypes isa DataType || return nothing
        params = specTypes.parameters
        isempty(params) && return nothing
        Base.isvarargtype(params[end]) && return nothing
        # Julia compiles a varargs method with the boxed `jl_fptr_args` ABI, whatever the
        # specialization, although its CodeInstance may report a specsig entry.
        method = mi.def
        method isa Method || return nothing
        method.isva && return nothing
        kind = native_call_kind(interp, specTypes)
        kind === nothing && return nothing
        ruled = kind === :frule || kind === :rrule
        world = interp.world
        # Ask the cheap question before inferring natively.
        if kind === :const
            all_guaranteed_const(params, world) || return nothing
        end
        # Infer natively first, and compile only what is called natively: most calls with
        # const arguments go to small methods that are inlined.
        inferred = CC.typeinf_ext(CC.NativeInterpreter(world), mi, CC.SOURCE_MODE_NOT_REQUIRED)
        inferred isa Core.CodeInstance || return nothing
        CC.get_ci_mi(inferred) === mi || return nothing
        RT = inferred.rettype
        if kind === :const
            guaranteed_const_nongen(RT, world) || return nothing
        end
        # `NoInlineCallInfo` keeps a call of a method with a rule from being inlined.
        (ruled || call_convention(mi, inferred) === :call) || return nothing
        native_uses_specsig(specTypes, RT) || return nothing
        ci = codeinst(mi, world)
        ci === nothing && return nothing
        (CC.get_ci_mi(ci) === mi && ci.rettype == RT) || return nothing
        specptr, _ = Interpreter.codeinst_entry(ci)
        specptr == C_NULL && return nothing
        return (ci, specptr, ruled ? kind : :inactive)
    end

    """
        native_call_kind(interp::EnzymeInterpreter, specTypes) -> Union{Nothing, Symbol}

    Say which callees of signature `specTypes` [`native_entry`](@ref) may resolve to native
    code: `:inactive` for one with an inactive rule, `:const` for one without special
    handling, which is derivative-free only when its argument and return types are
    guaranteed const, `:frule` or `:rrule` for one with a custom rule `interp` applies, and
    `nothing` for any other. A keyword call has the rule of the function it calls, as
    `Interpreter.FutureCallinfoByType` finds it.
    """
    function native_call_kind(interp::Interpreter.EnzymeInterpreter, @nospecialize(specTypes))::Union{Nothing, Symbol}
        kind = Interpreter.enzyme_call_kind(interp, specTypes)
        if kind === nothing && Interpreter.isKWCallSignature(specTypes)
            kwkind = Interpreter.enzyme_call_kind(interp, Interpreter.simplify_kw(specTypes))
            (kwkind === :frule || kwkind === :rrule) && return kwkind
        end
        kind === nothing && return :const
        (kind === :inactive || kind === :frule || kind === :rrule) && return kind
        return nothing
    end

    """
        native_callee(interp::EnzymeInterpreter, callee::CodeInstance)

    Like [`native_entry`](@ref) for a callee `interp` inferred: the native code must have
    the return type `callee` has.
    """
    function native_callee(interp::Interpreter.EnzymeInterpreter, callee::Core.CodeInstance)
        native = native_entry(interp, Core.Compiler.get_ci_mi(callee))
        native === nothing && return nothing
        native[1].rettype == callee.rettype || return nothing
        return native
    end

    # Resolve a call of a derivative-free method to its native code, without inferring the
    # method with `EnzymeInterpreter`: the caller sees the native return type and effects,
    # and the optimizer emits an `:invoke` of the native CodeInstance (see
    # `compileable_specialization` below). Codegen turns that into a stub that
    # `bind_native_callees!` points at the native entry.
    function Core.Compiler.typeinf_edge(interp::Interpreter.EnzymeInterpreter, method::Method, @nospecialize(atype), sparams::Core.SimpleVector, caller::Core.Compiler.AbsIntState, edgecycle::Bool, edgelimited::Bool)
        CC = Core.Compiler
        if use_native_callees(interp)
            mi = CC.specialize_method(method, atype, sparams)
            native = mi isa Core.MethodInstance ? native_entry(interp, mi) : nothing
            if native !== nothing
                return CC.return_cached_result(interp, method, native[1], caller, edgecycle, edgelimited)
            end
        end
        return @invoke CC.typeinf_edge(interp::CC.AbstractInterpreter, method::Method, atype::Any, sparams::Core.SimpleVector, caller::CC.AbsIntState, edgecycle::Bool, edgelimited::Bool)
    end

    # The optimizer emits an `:invoke` of the CodeInstance it finds in the cache of `interp`
    # for the callee, or else of its MethodInstance, which codegen calls through `jl_invoke`
    # with boxed arguments and result: Enzyme then sees neither the callee nor the type of
    # the result. Give it the native CodeInstance of a call `typeinf_edge` resolved to native
    # code instead, so that codegen emits a specsig call of a stub for it.
    function Core.Compiler.compileable_specialization(code::Union{Core.MethodInstance, Core.CodeInstance}, effects::Core.Compiler.Effects, et::Core.Compiler.InliningEdgeTracker, @nospecialize(info::Core.Compiler.CallInfo), state::Core.Compiler.InliningState{<:Interpreter.EnzymeInterpreter})
        CC = Core.Compiler
        case = @invoke CC.compileable_specialization(code::Union{Core.MethodInstance, Core.CodeInstance}, effects::CC.Effects, et::CC.InliningEdgeTracker, info::CC.CallInfo, state::CC.InliningState)
        if case isa CC.InvokeCase && case.invoke isa Core.MethodInstance && use_native_callees(state.interp)
            native = native_entry(state.interp, case.invoke)
            if native !== nothing && native[1].min_world <= state.world <= native[1].max_world
                return CC.InvokeCase(native[1], case.effects, case.info)
            end
        end
        return case
    end

    @static if HAS_GPUCOMPILER_2

        const CompilerCaching = GPUCompiler.CompilerCaching

        # As CompilerCaching's, except that derivative-free callees are left out, and recorded in
        # the `EnzymeContext` of the compilation for `bind_native_callees!`.
        function CompilerCaching.get_codeinfos(interp::Interpreter.EnzymeInterpreter, root::Core.CodeInstance)
            if !isassigned(ENZYME_CONTEXT) || !use_native_callees(interp)
                return @invoke CompilerCaching.get_codeinfos(interp::Core.Compiler.AbstractInterpreter, root::Core.CodeInstance)
            end
            CC = Core.Compiler
            codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]
            skipped = enzyme_context().native_callees
            visited = Base.IdSet{Core.CodeInstance}()
            workqueue = Core.CodeInstance[root]
            while !isempty(workqueue)
                callee = pop!(workqueue)
                callee in visited && continue
                push!(visited, callee)
                if callee !== root
                    native = native_callee(interp, callee)
                    if native !== nothing
                        skipped[callee] = native
                        skipped[CC.get_ci_mi(callee)] = native
                        continue
                    end
                end
                src = CompilerCaching.get_source(callee)
                if src === nothing
                    CompilerCaching.typeinf!(interp, CC.get_ci_mi(callee))
                    src = CompilerCaching.get_source(callee)
                    src === nothing && continue
                end
                @static if isdefined(GPUCompiler.CompilerCaching, :resolve_invoke_targets)
                    src = CompilerCaching.resolve_invoke_targets(interp, src)
                end
                push!(codeinfos, callee => src)
                for stmt in src.code
                    if stmt isa Expr && stmt.head === :(=)
                        stmt = stmt.args[2]
                    end
                    if stmt isa Expr && (stmt.head === :invoke || stmt.head === :invoke_modify)
                        target = stmt.args[1]
                        target isa Core.CodeInstance && push!(workqueue, target)
                    end
                end
            end
            return codeinfos
        end

    else

        # As GPUCompiler's, except that derivative-free callees are left out, and recorded in the
        # `EnzymeContext` of the compilation for `bind_native_callees!`.
        function GPUCompiler.ci_cache_populate(interp::Interpreter.EnzymeInterpreter, cache, mi, min_world, max_world)
            if !isassigned(ENZYME_CONTEXT) || !use_native_callees(interp)
                return @invoke GPUCompiler.ci_cache_populate(interp::Any, cache::Any, mi::Any, min_world::Any, max_world::Any)
            end
            CC = Core.Compiler
            codeinfos = Pair{Core.CodeInstance, Core.CodeInfo}[]
            skipped = enzyme_context().native_callees
            root = CC.typeinf_ext(interp, mi, CC.SOURCE_MODE_NOT_REQUIRED)
            workqueue = CC.CompilationQueue(; interp)
            push!(workqueue, root)
            while !isempty(workqueue)
                callee = pop!(workqueue)
                CC.isinspected(workqueue, callee) && continue
                CC.markinspected!(workqueue, callee)
                cmi = CC.get_ci_mi(callee)
                if CC.use_const_api(callee)
                    src = @static if VERSION >= v"1.13.0-DEV.1121"
                        CC.codeinfo_for_const(interp, cmi, CC.WorldRange(callee.min_world, callee.max_world), callee.edges, callee.rettype_const)
                    else
                        CC.codeinfo_for_const(interp, cmi, callee.rettype_const)
                    end
                else
                    if callee !== root
                        native = native_callee(interp, callee)
                        if native !== nothing
                            skipped[callee] = native
                            skipped[cmi] = native
                            continue
                        end
                    end
                    src = CC.typeinf_code(interp, cmi, true)
                end
                if src isa Core.CodeInfo
                    sptypes = CC.sptypes_from_meth_instance(cmi)
                    CC.collectinvokes!(workqueue, src, sptypes)
                    push!(codeinfos, callee => src)
                end
            end
            return codeinfos
        end

    end

    """
        stub_callee(tojlinvoke::LLVM.Function, ctx::EnzymeContext) -> Union{Nothing, CodeInstance, MethodInstance}

    Return the CodeInstance, or MethodInstance, a `tojlinvoke` function of Julia's codegen
    invokes. Codegen emits one for each CodeInstance called but not emitted: it loads the
    CodeInstance or its MethodInstance from a `julia.constgv` global and passes it to
    `ijl_invoke`. The value of a global left a declaration (see [`emit_unresolved_llvm`](@ref))
    is the one `ctx` records for it (see [`slot_value`](@ref)).
    """
    function stub_callee(tojlinvoke::LLVM.Function, ctx::EnzymeContext)
        for bb in LLVM.blocks(tojlinvoke), inst in LLVM.instructions(bb)
            inst isa LLVM.CallInst || continue
            fn = LLVM.called_operand(inst)
            (fn isa LLVM.Function && LLVM.name(fn) in ("ijl_invoke", "jl_invoke")) || continue
            v = LLVM.arguments(inst)[4]
            while v isa LLVM.ConstantExpr || v isa LLVM.AddrSpaceCastInst || v isa LLVM.BitCastInst
                v = LLVM.operands(v)[1]
            end
            v isa LLVM.LoadInst || return nothing
            gv = LLVM.operands(v)[1]
            gv isa LLVM.GlobalVariable || return nothing
            obj = if LLVM.isdeclaration(gv)
                found = slot_value(ctx, LLVM.name(gv))
                found === nothing && return nothing
                something(found)
            else
                init = LLVM.initializer(gv)
                while init isa LLVM.ConstantExpr
                    init = LLVM.operands(init)[1]
                end
                init isa LLVM.ConstantInt || return nothing
                Base.unsafe_pointer_to_objref(reinterpret(Ptr{Cvoid}, convert(UInt, init)))
            end
            return obj isa Union{Core.CodeInstance, Core.MethodInstance} ? obj : nothing
        end
        return nothing
    end

    """
        bind_native_callees!(mod::LLVM.Module, world::UInt)

    Replace the stub Julia's codegen emitted for each callee `ci_cache_populate` left out
    by a declaration bound to the native entry point of the callee. The stub is a specsig
    function that calls `julia.call(@tojlinvokeN, ...)`.

    The declaration of a derivative-free callee is marked `enzyme_inactive`, and
    `enzymejl_native_inactive` so that [`materialize_native_invokes!`](@ref) leaves it
    without a body even for nested differentiation, which does not differentiate it either.
    It is `nofree`: Julia code frees nothing its caller can observe.

    A callee with a custom rule gets none of these: its calls must reach the rule handlers.
    Its stub stays, and calls the declaration instead (see [`bind_ruled_stub!`](@ref)).
    Binding such a callee is not allowed to fail: the stub would call it through
    `jl_invoke`, and Enzyme would not apply its rule.

    A return with GC roots gets an sret type with the tracked pointers stripped, as the
    buffer holds (the roots array keeps the objects alive), because Enzyme may move that
    buffer to the heap to keep the result for the reverse pass, and LateLowerGC requires a
    stack buffer for an sret with tracked pointers.
    """
    function bind_native_callees!(mod::LLVM.Module, world::UInt)
        isassigned(ENZYME_CONTEXT) || return nothing
        haskey(LLVM.functions(mod), "julia.call") || return nothing
        julia_call = LLVM.functions(mod)["julia.call"]
        enzyme_ctx = enzyme_context()
        skipped = enzyme_ctx.native_callees
        for tojl in collect(LLVM.functions(mod))
            startswith(LLVM.name(tojl), "tojlinvoke") || continue
            callee = stub_callee(tojl, enzyme_ctx)
            callee === nothing && continue
            native = get(skipped, callee, nothing)
            native === nothing && continue
            ci, specptr, kind = native
            ruled = kind !== :inactive
            mi = Core.Compiler.get_ci_mi(ci)
            RT = ci.rettype
            stubs = LLVM.Function[]
            for u in LLVM.uses(tojl)
                c = LLVM.user(u)
                (c isa LLVM.CallInst && LLVM.called_operand(c) == julia_call) || continue
                push!(stubs, LLVM.parent(LLVM.parent(c)))
            end
            for stub in stubs
                try
                    check_specsig(stub, mi, RT)
                catch
                    ruled && rethrow()
                    continue
                end
                if ruled
                    bind_ruled_stub!(mod, stub, mi, RT, specptr, world)
                    push!(enzyme_ctx.edges, mi)
                    push!(enzyme_ctx.edges, ci)
                    continue
                end
                name = LLVM.name(stub)
                LLVM.name!(stub, name * ".stub")
                fn = declare_native!(mod, mi, RT, specptr, name, world)
                fattrs = function_attributes(fn)
                push!(fattrs, StringAttribute("enzyme_inactive"))
                push!(fattrs, StringAttribute("enzymejl_native_inactive"))
                push!(fattrs, EnumAttribute("nofree"))
                sret_attr = strip_native_sret!(fn, RT)
                if Core.Compiler.is_effect_free(Core.Compiler.decode_effects(ci.ipo_purity_bits))
                    _, sret, returnRoots = get_return_info(RT)
                    mark_read_only_or_throw!(fn, sret === nothing ? 0 : returnRoots === nothing ? 1 : 2)
                end
                if !retarget_calls!(stub, fn, sret_attr)
                    LLVM.erase!(fn)
                    LLVM.name!(stub, name)
                    continue
                end
                LLVM.erase!(stub)
                push!(enzyme_ctx.edges, mi)
                push!(enzyme_ctx.edges, ci)
            end
            if isempty(LLVM.uses(tojl))
                LLVM.erase!(tojl)
            elseif ruled
                throw(AssertionError("Enzyme: the function $(mi), which has a custom rule, is called through $(LLVM.name(tojl)) other than from a specsig stub that calls it with `julia.call`. This is not expected to happen, please report it."))
            end
        end
        return nothing
    end

    """
        strip_native_sret!(fn::LLVM.Function, RT) -> Union{Nothing, LLVM.Attribute}

    Give the sret parameter of the native declaration `fn`, of a function returning `RT`
    with GC roots, the sret type with the tracked pointers stripped (see
    [`bind_native_callees!`](@ref)), and return that attribute for the calls of `fn`.
    Return `nothing`, and change nothing, for any other return.
    """
    function strip_native_sret!(fn::LLVM.Function, @nospecialize(RT::Type))
        _, sret, returnRoots = get_return_info(RT)
        (returnRoots !== nothing && sret !== nothing && !is_sret_union(RT)) || return nothing
        full = convert(LLVMType, eltype(sret))
        sret_attr = TypeAttribute("sret", strip_tracked_pointers(full))
        delete!(parameter_attributes(fn, 1), TypeAttribute("sret", full))
        push!(parameter_attributes(fn, 1), sret_attr)
        return sret_attr
    end

    """
        bind_ruled_stub!(mod::LLVM.Module, stub::LLVM.Function, mi::MethodInstance, RT, specptr, world)

    Make `stub`, the stub Julia's codegen emitted for a call of `mi`, which has a custom
    rule, call the native entry `specptr` of `mi` through a declaration, instead of calling
    `mi` through `jl_invoke`.

    The rule handlers handle the calls of `stub` as they would the calls of the function
    Julia's codegen emits for `mi` in `mod`, which `stub` stands for: it keeps its
    signature, without the `pgcstack` parameter of the native entry, which the handlers do
    not expect, and gets the attributes Enzyme reads from an emitted function (see
    `mark_compiled!`), so that `handle_compiled` marks it for the rule handlers. They call
    the rule instead of `stub`, or `stub` as is when no rule applies to a constant call
    (see `has_rule`). Its body only forwards to the declaration, which Enzyme never
    differentiates (see [`materialize_native_invokes!`](@ref) for nested differentiation).
    """
    function bind_ruled_stub!(mod::LLVM.Module, stub::LLVM.Function, mi::Core.MethodInstance, @nospecialize(RT::Type), specptr::Ptr{Cvoid}, world::UInt)
        fn = declare_native!(mod, mi, RT, specptr, LLVM.name(stub) * ".native", world)
        sret_attr = strip_native_sret!(fn, RT)
        if sret_attr === nothing
            _, sret, _ = get_return_info(RT)
            if sret !== nothing && !is_sret_union(RT)
                sret_attr = TypeAttribute("sret", convert(LLVMType, eltype(sret)))
            end
        end

        # Drop the body of the stub, which boxes the arguments and calls `jl_invoke`.
        for bb in LLVM.blocks(stub), inst in LLVM.instructions(bb)
            isempty(LLVM.uses(inst)) || LLVM.replace_uses!(inst, LLVM.UndefValue(LLVM.value_type(inst)))
        end
        for bb in collect(LLVM.blocks(stub))
            for inst in reverse(collect(LLVM.instructions(bb)))
                LLVM.API.LLVMInstructionEraseFromParent(inst)
            end
            LLVM.API.LLVMDeleteBasicBlock(bb)
        end

        B = LLVM.IRBuilder()
        entry = LLVM.BasicBlock(stub, "entry")
        LLVM.position!(B, entry)
        args = collect(LLVM.Value, LLVM.parameters(stub))
        gi = gcstack_arg_index(fn)
        if gi != 0
            pgcstack = reinsert_gcmarker!(stub, B)
            LLVM.position!(B, entry)
            insert!(args, gi, pgcstack)
        end
        # `enzyme-fixup-julia` rejects an sret parameter with GC roots that is passed on
        # to a call. So with return roots, the native entry returns into buffers of `stub`,
        # as in the thunks of `enzyme_call`, and `stub` splits the value into its own sret
        # and roots as Julia's codegen does.
        _, sret, returnRoots = get_return_info(RT)
        rooted = sret !== nothing && returnRoots !== nothing && !is_sret_union(RT)
        if rooted
            jltype = convert(LLVMType, eltype(sret))
            args[1] = LLVM.alloca!(B, strip_tracked_pointers(jltype), "native.sret")
            args[2] = LLVM.alloca!(B, convert(LLVMType, eltype(returnRoots)), "native.return_roots")
        end
        ft = LLVM.function_type(fn)
        @assert length(args) == length(LLVM.parameters(ft))
        call = LLVM.call!(B, ft, fn, args)
        LLVM.callconv!(call, LLVM.callconv(fn))
        copy_abi_attrs!(call, fn)
        if sret_attr !== nothing
            LLVM.API.LLVMAddCallSiteAttribute(call, UInt32(1), sret_attr)
        end
        if rooted
            val = recombine_value_ptr!(B, jltype, args[1], args[2])
            stub_params = LLVM.parameters(stub)
            split_value_into!(B, val, stub_params[1], stub_params[2])
        end
        if RT === Union{}
            LLVM.unreachable!(B)
        elseif LLVM.return_type(LLVM.function_type(stub)) isa LLVM.VoidType
            LLVM.ret!(B)
        else
            LLVM.ret!(B, call)
        end
        LLVM.dispose(B)

        fattrs = function_attributes(stub)
        delete!(fattrs, EnumAttribute("alwaysinline"))
        delete!(fattrs, EnumAttribute("inlinehint"))
        mark_compiled!(mod, stub, mi, RT, world)
        return nothing
    end

    """
        mark_read_only_or_throw!(fn::LLVM.Function, nout::Int)

    Mark the declaration `fn` of a native callee Julia proved effect-free: it writes no
    memory its caller can observe, other than its first `nout` arguments (the sret buffer
    and the GC roots of the return), although it may throw. Besides Enzyme's attribute,
    state it in the LLVM memory attributes, as Enzyme does for the functions it finds
    read-only-or-throw (`addReadOnlyOrThrowAttributes`), which it does not do for
    declarations: reads anything, writes only inaccessible memory (the exception, GC
    bookkeeping), and every other pointer argument is `readonly`. The GC stack the callee
    pushes its frame on through `pgcstack` is restored before it returns. This lets LICM
    hoist loads past a call of it.
    """
    function mark_read_only_or_throw!(fn::LLVM.Function, nout::Int)
        fattrs = function_attributes(fn)
        push!(fattrs, StringAttribute(nout == 0 ? "enzyme_ReadOnlyOrThrow" : "enzyme_LocalReadOnlyOrThrow"))
        argmem = nout == 0 ? MRI_Ref : MRI_ModRef
        effects = MemoryEffect(
            (argmem << getLocationPos(ArgMem)) |
                (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                (MRI_Ref << getLocationPos(Other)),
        )
        push!(fattrs, EnumAttribute("memory", effects.data))
        for (i, p) in enumerate(LLVM.parameters(fn))
            i <= nout && continue
            LLVM.value_type(p) isa LLVM.PointerType || continue
            push!(parameter_attributes(fn, i), EnumAttribute("readonly"))
        end
        return nothing
    end

    """
        retarget_calls!(f::LLVM.Function, fn::LLVM.Function, sret_attr) -> Bool

    Point every call of `f`, which Julia's codegen emitted without a `pgcstack` parameter,
    at `fn`, which has the one of Julia's JIT ABI (see [`specsig`](@ref)). Replace the sret
    attribute of the calls with `sret_attr` unless it is `nothing`. Return `false`, and
    change nothing, when `f` has a use other than a direct call.
    """
    function retarget_calls!(f::LLVM.Function, fn::LLVM.Function, sret_attr)::Bool
        calls = LLVM.CallInst[]
        for u in LLVM.uses(f)
            c = LLVM.user(u)
            (c isa LLVM.CallInst && LLVM.called_operand(c) == f) || return false
            push!(calls, c)
        end
        gi = gcstack_arg_index(fn)
        fgi = gcstack_arg_index(f)
        ft = LLVM.function_type(fn)
        for c in calls
            caller = LLVM.parent(LLVM.parent(c))
            B = LLVM.IRBuilder()
            LLVM.position!(B, c)
            args = collect(LLVM.arguments(c))
            # argument index of the old call => argument index of the new one
            idxmap = Pair{Int, Int}[]
            if gi != 0 && fgi == 0
                pgcstack = reinsert_gcmarker!(caller, B)
                LLVM.position!(B, c)
                for i in 1:length(args)
                    push!(idxmap, i => (i < gi ? i : i + 1))
                end
                insert!(args, gi, pgcstack)
            else
                for i in 1:length(args)
                    push!(idxmap, i => i)
                end
            end
            @assert length(args) == length(LLVM.parameters(ft))
            call = LLVM.call!(B, ft, fn, args)
            LLVM.callconv!(call, LLVM.callconv(fn))
            for (from, to) in idxmap
                copy_callsite_attrs!(call, c, from, to)
            end
            copy_callsite_attrs!(call, c, UInt32(LLVM.API.LLVMAttributeReturnIndex), UInt32(LLVM.API.LLVMAttributeReturnIndex))
            copy_callsite_attrs!(call, c, typemax(UInt32), typemax(UInt32))
            copy_abi_attrs!(call, fn)
            if sret_attr !== nothing
                LLVM.API.LLVMRemoveCallSiteEnumAttribute(call, UInt32(1), enum_attr_kind("sret"))
                LLVM.API.LLVMAddCallSiteAttribute(call, UInt32(1), sret_attr)
            end
            copy_metadata!(call, c)
            LLVM.API.LLVMInstructionSetDebugLoc(call, LLVM.API.LLVMInstructionGetDebugLoc(c))
            LLVM.replace_uses!(c, call)
            LLVM.erase!(c)
            LLVM.dispose(B)
        end
        return true
    end

    function copy_callsite_attrs!(dst::LLVM.CallInst, src::LLVM.CallInst, from::Integer, to::Integer)
        n = LLVM.API.LLVMGetCallSiteAttributeCount(src, UInt32(from))
        n == 0 && return nothing
        attrs = Vector{LLVM.API.LLVMAttributeRef}(undef, n)
        LLVM.API.LLVMGetCallSiteAttributes(src, UInt32(from), attrs)
        for a in attrs
            LLVM.API.LLVMAddCallSiteAttribute(dst, UInt32(to), a)
        end
        return nothing
    end

elseif VERSION < v"1.12-"

    # Julia 1.10 and 1.11: GPUCompiler emits every callee codegen finds in the cache of
    # `EnzymeInterpreter` (through its lookup callback). `typeinf_edge` keeps a
    # derivative-free callee out of that cache by resolving its call to Julia's native
    # inference result instead, so the optimizer emits an `:invoke` of its MethodInstance,
    # and codegen, which finds no CodeInstance for it, calls it through `jl_invoke` with boxed
    # arguments and result. `check_ir!` marks such a call inactive (see `is_native_edge`).

    """
        native_entry(interp::EnzymeInterpreter, mi::MethodInstance)

    Return the CodeInstance Julia's native interpreter infers for `mi` when Enzyme never
    differentiates `mi` and can call its native code instead, and `nothing` otherwise: as
    on Julia 1.12, `mi` must have an `EnzymeRules.inactive` rule, or take and return only
    guaranteed-const types, and no other rule or special handling, and a small method
    Julia would inline gets `nothing`. The second element is unused (`C_NULL`): the call
    goes through `jl_invoke`, which compiles the callee when it first runs. A callee with a
    custom rule stays emitted: the rule handlers cannot read the boxed arguments of a
    `jl_invoke` call. The third element is always `:inactive`.
    """
    function native_entry(interp::Interpreter.EnzymeInterpreter, mi::Core.MethodInstance)
        CC = Core.Compiler
        specTypes = mi.specTypes
        specTypes isa DataType || return nothing
        params = specTypes.parameters
        isempty(params) && return nothing
        Base.isvarargtype(params[end]) && return nothing
        method = mi.def
        method isa Method || return nothing
        kind = Interpreter.enzyme_call_kind(interp, specTypes)
        (kind === nothing || kind === :inactive) || return nothing
        world = interp.world
        # Ask the cheap question before inferring natively.
        (kind === :inactive || all_guaranteed_const(params, world)) || return nothing
        native = CC.NativeInterpreter(world)
        CC.typeinf_type(native, method, specTypes, mi.sparam_vals)
        inferred = CC.get(CC.code_cache(native), mi, nothing)
        inferred isa Core.CodeInstance || return nothing
        (kind === :inactive || guaranteed_const_nongen(inferred.rettype, world)) || return nothing
        call_convention(mi, inferred) === :call || return nothing
        return (inferred, C_NULL, :inactive)
    end

    # The result of a call edge to the cached `ci`, as `typeinf_edge` returns it.
    @static if isdefined(Core.Compiler, :return_cached_result)
        cached_edge_result(interp, ci::Core.CodeInstance, caller::Core.Compiler.AbsIntState) =
            Core.Compiler.return_cached_result(interp, ci, caller)
    else
        function cached_edge_result(interp, ci::Core.CodeInstance, caller::Core.Compiler.AbsIntState)
            CC = Core.Compiler
            CC.update_valid_age!(caller, CC.WorldRange(CC.min_world(ci), CC.max_world(ci)))
            rettype = ci.rettype
            if isdefined(ci, :rettype_const)
                rettype_const = ci.rettype_const
                # As `typeinf_edge` does for a cached edge.
                if isa(rettype_const, Vector{Any}) && !(Vector{Any} <: rettype)
                    rettype = CC.PartialStruct(rettype, rettype_const)
                elseif isa(rettype_const, CC.PartialOpaque) && rettype <: Core.OpaqueClosure
                    rettype = rettype_const
                elseif isa(rettype_const, CC.InterConditional) && rettype !== CC.InterConditional
                    rettype = rettype_const
                elseif isa(rettype_const, CC.InterMustAlias) && rettype !== CC.InterMustAlias
                    rettype = rettype_const
                else
                    rettype = CC.Const(rettype_const)
                end
            end
            return CC.EdgeCallResult(rettype, ci.def, CC.ipo_effects(ci))
        end
    end

    # Resolve a call of a derivative-free method to Julia's native inference result, without
    # inferring the method with `EnzymeInterpreter`: the caller sees the native return type
    # and effects, and nothing for the method is in the cache of `interp`.
    function Core.Compiler.typeinf_edge(interp::Interpreter.EnzymeInterpreter, method::Method, @nospecialize(atype), sparams::Core.SimpleVector, caller::Core.Compiler.AbsIntState)
        CC = Core.Compiler
        if use_native_callees(interp)
            mi = CC.specialize_method(method, atype, sparams)
            native = mi isa Core.MethodInstance ? native_entry(interp, mi) : nothing
            if native !== nothing
                return cached_edge_result(interp, native[1], caller)
            end
        end
        return @invoke CC.typeinf_edge(interp::CC.AbstractInterpreter, method::Method, atype::Any, sparams::Core.SimpleVector, caller::CC.AbsIntState)
    end

    bind_native_callees!(mod::LLVM.Module, world::UInt) = nothing

else

    native_entry(interp, mi) = nothing
    bind_native_callees!(mod::LLVM.Module, world::UInt) = nothing

end
