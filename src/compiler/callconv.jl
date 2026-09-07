# Calling natively compiled Julia code from a differentiated module.
#
# Julia 1.12 lets a caller infer a MethodInstance with an interpreter of its
# choice, hand the result to Julia's JIT, and read back the entry points of
# the compiled code. The functions here derive the signature Julia's codegen
# gives that entry point (`get_specsig_function`), declare it in an LLVM
# module bound to its address, and decide, per inlining annotation, whether
# a function reaches the module that way or is emitted into it by
# `nested_codegen!`. The custom rule handlers are the first users
# (`invoke_codegen!`), and nested differentiation gets the emitted bodies it
# needs from `materialize_native_invokes!`.

"""
    copy_abi_attrs!(call::LLVM.CallInst, fn::LLVM.Function)

Copy the `zeroext`, `signext` and `swiftself` parameter attributes of `fn` to
the call site `call`. `restore_lookups` replaces the callee with a constant
address, after which only the call site's attributes describe the convention
the callee expects: the extension of a small integer argument, and the
register `pgcstack` goes in. A target without the swift calling convention
takes `pgcstack` as a plain parameter, which carries no attribute to copy.
"""
function copy_abi_attrs!(call::LLVM.CallInst, fn::LLVM.Function)
    zeroext = enum_attr_kind("zeroext")
    signext = enum_attr_kind("signext")
    for i in 1:length(parameters(fn))
        for attr in collect(parameter_attributes(fn, i))
            if attr isa EnumAttribute && (kind(attr) == zeroext || kind(attr) == signext || kind(attr) == swiftself_kind)
                LLVM.API.LLVMAddCallSiteAttribute(call, LLVM.API.LLVMAttributeIndex(i), attr)
            end
        end
    end
    return nothing
end

@static if Interpreter.HAS_INVOKE_RULES

    function read_jit_gcstack_arg()::Bool
        off = fieldoffset(Base.CodegenParams, Base.fieldindex(Base.CodegenParams, :gcstack_arg))
        return unsafe_load(Ptr{Cint}(cglobal(:jl_default_cgparams) + off)) != 0
    end
    const jit_gcstack_arg_once = Base.OncePerProcess{Bool}(read_jit_gcstack_arg)

    """
        jit_gcstack_arg() -> Bool

    Say if Julia's JIT passes `pgcstack` as an argument to compiled code. The
    JIT compiles with `jl_default_cgparams`, so read its `gcstack_arg` field.
    The `Base.CodegenParams` mirror gives the field offset. The value is a
    process constant, so read it once and cache it.
    """
    jit_gcstack_arg()::Bool = jit_gcstack_arg_once()

    """
        jit_uses_swiftcc() -> Bool

    Say if Julia's codegen uses the swift calling convention on this target.
    LLVM does not support the convention on RISC-V, so Julia turns it off there
    (`jl_codegen_output_t::use_swiftcc`). With the convention on, and
    [`jit_gcstack_arg`](@ref) set, the `pgcstack` parameter gets the
    `swiftself` attribute; without it, `pgcstack` keeps its place in the
    signature but carries no attribute and the function keeps the C calling
    convention.
    """
    jit_uses_swiftcc()::Bool = !startswith(string(Sys.ARCH), "riscv")

    """
        module_targets_host(mod) -> Bool

    Say if `mod` targets the machine this process runs on. Compare the
    architecture component of the module's target triple with the host triple
    (`Sys.MACHINE`) and with `Sys.ARCH`. The two spellings can differ: Darwin
    writes `arm64` where `Sys.ARCH` says `aarch64`.
    """
    function module_targets_host(mod::LLVM.Module)::Bool
        arch = first(split(LLVM.triple(mod), '-'))
        return arch == first(split(Sys.MACHINE, '-')) || arch == string(Sys.ARCH)
    end

    """
        specsig(mi, RT; gcstack_arg = jit_gcstack_arg()) -> (retty, params, param_attrs)

    Derive the signature Julia's codegen (`get_specsig_function`) gives the
    specialized entry of the function `mi` with return type `RT`. The parameter
    order is `[sret][return roots][pgcstack] args...`.

    The return:

    - A return that deserves an `sret` becomes a leading pointer parameter with
      the `sret` attribute. On 1.12+ the buffer holds the layout with the
      tracked pointers stripped, and the tracked pointers go through the
      return-roots parameter.
    - A small `Union` return gets its byte buffer as the leading pointer
      parameter, and the function returns `{boxed-or-null, selector}`.
    - A boxed return is a tracked pointer.
    - Every other return keeps the value's LLVM type.

    The arguments:

    - Ghost and `Type{T}` arguments are omitted.
    - Boxed arguments are tracked pointers.
    - Aggregates go by reference in the derived address space. On 1.12+ a
      pointer to their inline roots follows.
    - Scalars go by value.

    With `gcstack_arg` set, `pgcstack` follows the return parameters, with the
    `swiftself` attribute where [`jit_uses_swiftcc`](@ref) holds. It always
    carries the `gcstack` string attribute, as Julia 1.13 writes, so that
    [`gcstack_arg_index`](@ref) finds it on a target without the convention
    too. Pass `false` to model a function compiled without `pgcstack`, as
    `nested_codegen!` emits.

    `classify_arguments`, `get_return_info` and `enzyme_custom_setup_args`
    follow the same rules, so the derived signature matches the arguments
    the rule handlers in `customrules.jl` pass.
    """
    function specsig(mi::Core.MethodInstance, @nospecialize(RT::Type); gcstack_arg::Bool = jit_gcstack_arg())
        T_void = LLVM.VoidType()
        T_int8 = LLVM.Int8Type()
        T_ptr = LLVM.PointerType(T_int8)
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
        T_derived = LLVM.PointerType(T_jlvalue, Derived)

        params = LLVMType[]
        param_attrs = Vector{LLVM.Attribute}[]

        rt, sret, returnRoots = get_return_info(RT)
        if sret !== nothing
            if is_sret_union(RT)
                push!(params, T_ptr)
                push!(param_attrs, LLVM.Attribute[StringAttribute("enzymejl_sret_union_bytes", string(union_alloca_type(RT)))])
                retty = LLVM.StructType(LLVMType[T_prjlvalue, T_int8])
            else
                # As in Julia's codegen, the sret attribute carries the full type,
                # and `recombine_value_ptr!` reads it from there. With return
                # roots, the buffer holds the layout with the tracked pointers
                # stripped.
                sret_lty = convert(LLVMType, eltype(sret))
                push!(params, T_ptr)
                push!(param_attrs, LLVM.Attribute[TypeAttribute("sret", sret_lty), EnumAttribute("noalias"), EnumAttribute("nocapture"), EnumAttribute("noundef")])
                if returnRoots !== nothing
                    push!(params, T_ptr)
                    push!(param_attrs, LLVM.Attribute[EnumAttribute("noalias"), EnumAttribute("nocapture"), EnumAttribute("noundef")])
                end
                retty = T_void
            end
        elseif rt === Nothing
            retty = T_void
        elseif RT isa Union && rt === UInt8
            # A `Union` whose members are all ghosts returns only the selector
            # byte (the `Ghosts` return convention of `get_specsig_function`).
            retty = T_int8
        elseif rt === Any || GPUCompiler.deserves_retbox(RT)
            retty = T_prjlvalue
        else
            retty = convert(LLVMType, rt)
        end

        if gcstack_arg
            push!(params, T_ptr)
            attrs = LLVM.Attribute[StringAttribute("gcstack"), EnumAttribute("nonnull")]
            if jit_uses_swiftcc()
                pushfirst!(attrs, EnumAttribute("swiftself"))
            end
            push!(param_attrs, attrs)
        end

        for T in (mi.specTypes::DataType).parameters
            kind = arg_kind(T)
            if kind === :ghost
                continue
            elseif kind === :boxed
                push!(params, T_prjlvalue)
                attrs = LLVM.Attribute[]
                if T isa DataType && !Base.isabstracttype(T) && !ismutabletype(T)
                    push!(attrs, EnumAttribute("readonly"))
                end
                push!(param_attrs, attrs)
            elseif kind === :byref
                push!(params, T_derived)
                push!(param_attrs, LLVM.Attribute[EnumAttribute("noalias"), EnumAttribute("nocapture"), EnumAttribute("readonly")])
                if inline_roots_type(T) != 0
                    push!(params, T_ptr)
                    push!(param_attrs, LLVM.Attribute[EnumAttribute("noalias"), EnumAttribute("nocapture"), EnumAttribute("readonly")])
                end
            else
                lty = convert(LLVMType, T)
                push!(params, lty)
                attrs = LLVM.Attribute[]
                if Base.isprimitivetype(T) && lty isa LLVM.IntegerType
                    # Julia's codegen extends small integers to the register
                    # width at the call site, and the callee relies on it.
                    push!(attrs, EnumAttribute(T <: Signed ? "signext" : "zeroext"))
                end
                push!(param_attrs, attrs)
            end
        end

        return retty, params, param_attrs
    end

    """
        deserves_argbox(T) -> Bool

    Say if Julia's codegen passes a value of type `T` boxed, as a tracked
    `jl_value_t*`, rather than unboxed on the stack (`deserves_argbox` in
    `codegen.cpp`). Only a concrete immutable type that is a singleton, or
    that Julia can allocate inline (`jl_datatype_isinlinealloc`), goes
    unboxed. A struct with uninitialized fields, or one whose field layout
    the GC cannot describe inline, is boxed even though `jl_type_to_llvm`
    gives it an LLVM struct type.
    """
    function deserves_argbox(@nospecialize(T::Type))::Bool
        T isa DataType || return true
        (Base.isconcretetype(T) && !ismutabletype(T)) || return true
        Base.issingletontype(T) && return false
        return !Base.allocatedinline(T)
    end

    """
        arg_kind(T) -> Symbol

    Classify an argument of type `T` as `get_specsig_function` does:
    `:ghost` (omitted), `:boxed` (a tracked pointer), `:byref` (an aggregate
    passed through a derived pointer, plus an inline-roots pointer when it
    holds tracked pointers), or `:byval` (a scalar in a register).
    """
    function arg_kind(@nospecialize(T::Type))::Symbol
        if Core.Compiler.isconstType(T)
            return :ghost
        elseif deserves_argbox(T)
            return :boxed
        elseif isghostty(T)
            return :ghost
        end
        lty = convert(LLVMType, T)
        if lty isa LLVM.StructType || lty isa LLVM.ArrayType
            return :byref
        else
            return :byval
        end
    end


    # Create the function `customrules.jl` calls a rule through. Shape it like
    # the codegen'd rule function it replaces: the specsig, the calling
    # convention, and the `enzymejl_mi` / `enzymejl_rt` attributes the rule
    # handlers read. Also add the per-parameter attributes `prepare_llvm` gives
    # every compiled function (`enzymejl_parmtype*`, `enzymejl_rooted_typ`,
    # `enzymejl_returnRoots`). `fix_decayaddr!` recognizes root-carrying
    # arguments by those markers.
    function specsig_function!(mod::LLVM.Module, mi::Core.MethodInstance, @nospecialize(RT::Type), name::String, world::UInt)::LLVM.Function
        retty, params, param_attrs = specsig(mi, RT)
        fn = LLVM.Function(mod, name, LLVM.FunctionType(retty, params))
        # Julia's codegen uses the swift calling convention only when the
        # `pgcstack` parameter exists and the target supports it
        # (`get_specsig_function`).
        if jit_gcstack_arg() && jit_uses_swiftcc()
            callconv!(fn, LLVM.API.LLVMSwiftCallConv)
        end
        fattrs = function_attributes(fn)
        push!(fattrs, StringAttribute("enzymejl_mi", string(convert(UInt, pointer_from_objref(mi)))))
        push!(fattrs, StringAttribute("enzymejl_rt", string(convert(UInt, unsafe_to_pointer(RT)))))
        push!(fattrs, StringAttribute("enzymejl_world", string(world)))
        if RT === Union{}
            push!(fattrs, EnumAttribute("noreturn"))
        end
        for (i, attrs) in enumerate(param_attrs)
            for attr in attrs
                push!(parameter_attributes(fn, i), attr)
            end
        end

        # Classify the arguments to place the per-parameter attributes.
        # `classify_arguments` also asserts that the derived signature matches
        # `mi.specTypes` as it walks them. The signature has a `pgcstack`
        # parameter only when the JIT passes one, so tell `classify_arguments`
        # the same.
        _, sret, returnRoots = get_return_info(RT)
        jlargs = classify_arguments(mi.specTypes, LLVM.function_type(fn), sret !== nothing, returnRoots !== nothing, jit_gcstack_arg(), UInt64[], mi, world)
        for arg in jlargs
            if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
                continue
            end
            pattrs = parameter_attributes(fn, arg.codegen.i)
            push!(pattrs, StringAttribute("enzymejl_parmtype", string(convert(UInt, unsafe_to_pointer(arg.typ)))))
            push!(pattrs, StringAttribute("enzymejl_parmtype_str", string(arg.typ)))
            push!(pattrs, StringAttribute("enzymejl_parmtype_ref", string(UInt(arg.cc))))
            if arg.rooted_typ !== nothing
                push!(pattrs, StringAttribute("enzymejl_rooted_typ", string(convert(UInt, unsafe_to_pointer(arg.rooted_typ)))))
            end
        end
        if returnRoots !== nothing
            push!(parameter_attributes(fn, 2), StringAttribute("enzymejl_returnRoots", string(length(eltype(returnRoots).parameters[1]))))
        end
        return fn
    end

    """
        check_specsig(llvmf, mi, RT)

    Throw an `AssertionError` when the signature of the emitted function `llvmf`
    differs from the one [`specsig`](@ref) derives for `mi` and `RT`. The
    comparison covers the return type, the parameter types, and the `sret` and
    `swiftself` attributes. The other derived attributes only aid optimization
    and are skipped.

    `nested_codegen!` compiles without `gcstack_arg`, so the derivation takes
    the presence of `pgcstack` from [`has_gcstack_arg`](@ref).
    """
    function check_specsig(llvmf::LLVM.Function, mi::Core.MethodInstance, @nospecialize(RT::Type))
        retty, params, param_attrs = specsig(mi, RT; gcstack_arg = has_gcstack_arg(llvmf))
        ft = LLVM.function_type(llvmf)
        ok = LLVM.return_type(ft) == retty && parameters(ft) == params
        if ok
            sretkind = enum_attr_kind("sret")
            for (i, attrs) in enumerate(param_attrs)
                actual = collect(parameter_attributes(llvmf, i))
                for attr in attrs
                    is_sret = attr isa TypeAttribute && kind(attr) == sretkind
                    is_swiftself = attr isa EnumAttribute && kind(attr) == swiftself_kind
                    if !is_sret && !is_swiftself
                        continue
                    end
                    found = false
                    for a in actual
                        if is_sret && a isa TypeAttribute && kind(a) == sretkind && LLVM.value(a) == LLVM.value(attr)
                            found = true
                        elseif is_swiftself && a isa EnumAttribute && kind(a) == swiftself_kind
                            found = true
                        end
                    end
                    ok &= found
                end
            end
        end
        if !ok
            msg = sprint() do io
                println(io, "Enzyme: the emitted function does not match the derived signature")
                println(io, "mi = ", mi)
                println(io, "specTypes = ", mi.specTypes)
                println(io, "RT = ", RT)
                println(io, "function = ", LLVM.name(llvmf))
                println(io, "emitted = ", string(ft))
                println(io, "derived retty = ", string(retty))
                println(io, "derived params = ", string.(params))
            end
            throw(AssertionError(msg))
        end
        return nothing
    end

    """
        declare_native!(mod, mi, RT, specptr, name, world)

    Declare the natively compiled function `mi`, with return type `RT`, in `mod`,
    with the signature [`specsig`](@ref) derives. Store the entry point
            `bind_native_invokes!` later gives the declaration a body that calls the
            function through its `CodeInstance`.
    """
    function declare_native!(mod::LLVM.Module, mi::Core.MethodInstance, @nospecialize(RT::Type), specptr::Ptr{Cvoid}, name::String, world::UInt)::LLVM.Function
        fn = specsig_function!(mod, mi, RT, name, world)
        # The declaration stays symbolic until `bind_native_invokes!` gives it a body (after
        # the module string for nested differentiation is taken), and
        # `materialize_native_invokes!` finds it by this marker.
        push!(function_attributes(fn), StringAttribute("enzymejl_native_invoke"))
        return fn
    end

    """
        check_emitted_specsig(mod::LLVM.Module, llvmf::LLVM.Function, mi::MethodInstance, RT::Type)

    Compare the signature of `llvmf`, which Julia's codegen just emitted into
    `mod` for the MethodInstance `mi` with return type `RT`, against the one
    [`specsig`](@ref) derives, and throw an `AssertionError` on a difference
    (see [`check_specsig`](@ref)). Only [`invoke_codegen!`](@ref) relies on
    the derivation, but `prepare_llvm` checks every function Julia emits, so
    the derivation meets every signature shape the differentiated code
    contains and a mismatch is reported where it arises rather than as
    corruption at the first native call.
    """
    function check_emitted_specsig(mod::LLVM.Module, llvmf::LLVM.Function, mi::Core.MethodInstance, @nospecialize(RT::Type))
        (Interpreter.HAS_INVOKE_RULES && module_targets_host(mod)) || return nothing
        mi.specTypes isa DataType || return nothing
        check_specsig(llvmf, mi, RT)
        return nothing
    end

    """
        native_invoke_available(mod::LLVM.Module) -> Bool

    Say if functions called from `mod` may be bound to natively compiled code (see
    [`invoke_codegen!`](@ref)). That binds a process address, so not during
    precompilation, and only for modules that target the host.
    """
    function native_invoke_available(mod::LLVM.Module)::Bool
        return !Base.generating_output() && module_targets_host(mod)
    end

    """
        codeinst(mi::MethodInstance, world::UInt) -> Union{Nothing, CodeInstance}

    Infer and compile the function `mi` at `world` with Julia's native
    interpreter and JIT, exactly as an ordinary call of it would, and return
    its `CodeInstance`. Return `nothing` when inference fails. The result is
    the same `CodeInstance` ordinary callers use, so the function is compiled
    at most once per process. See [`invoke_codegen!`](@ref) for why the native
    interpreter, and not `EnzymeInterpreter`, infers the functions that are
    called natively.
    """
    function codeinst(mi::Core.MethodInstance, world::UInt)::Union{Nothing, Core.CodeInstance}
        CC = Core.Compiler
        ci = CC.typeinf_ext_toplevel(CC.NativeInterpreter(world), mi, CC.SOURCE_MODE_ABI)
        return ci isa Core.CodeInstance ? ci : nothing
    end

    """
        call_convention(mi::MethodInstance, ci::CodeInstance) -> Symbol

    Decide from the inlining annotation how the function `mi` reaches the
    calling module. `:inline` means: emit its IR into the calling module and
    always-inline it. `:call` means: call its natively compiled entry point.
    An `@inline` method gets `:inline`, and a `@noinline` method gets `:call`.
    An unannotated method gets `:inline` exactly when Julia would inline it, as
    recorded on its native `CodeInstance` `ci` (see [`codeinst`](@ref)):
    the native compiler keeps the source of a method only when the method is
    inlineable. A method with a constant result gets `:inline` whatever its
    annotation, because Julia compiles no code for it.
    """
    function call_convention(mi::Core.MethodInstance, ci::Core.CodeInstance)::Symbol
        CC = Core.Compiler
        # A constant result has no code to call: Julia serves it through the
        # constant-return trampoline, whatever the annotation says.
        CC.use_const_api(ci) && return :inline
        method = mi.def
        if method isa Method
            CC.is_declared_noinline(method) && return :call
            CC.is_declared_inline(method) && return :inline
        end
        inferred = @atomic :monotonic ci.inferred
        inferred isa CC.MaybeCompressed && CC.is_inlineable(inferred) && return :inline
        return :call
    end

    """
        native_codeinst(mod, mi, world) -> Union{Nothing, Tuple{CodeInstance, Ptr{Cvoid}}}

    Return the native `CodeInstance` of the function `mi` and its specialized
    entry point when [`invoke_codegen!`](@ref) binds the function, called from
    `mod`, to that code. Return `nothing` when it emits the function with
    `nested_codegen!` instead: where native calls are unavailable (see
    [`native_invoke_available`](@ref)), and for the `:inline` convention.

    Throw a `CallingConventionMismatchError` when Julia compiled the function for
    a MethodInstance other than `mi`, or without a specialized entry point
    (the boxed `jl_fptr_args` ABI, which Julia picks when every argument is
    boxed and so is the return). The declaration is derived from
    `mi.specTypes`, so the first would bind it to code with another
    signature. Every rule takes an annotation, an immutable struct passed
    unboxed, so Julia always gives a rule a specialized entry point. Neither
    case is known to occur; an error reports the broken assumption instead
    of hiding it.
    """
    function native_codeinst(mod::LLVM.Module, mi::Core.MethodInstance, world::UInt)::Union{Nothing, Tuple{Core.CodeInstance, Ptr{Cvoid}}}
        (mi.specTypes isa DataType && native_invoke_available(mod)) || return nothing
        ci = codeinst(mi, world)
        ci === nothing && return nothing
        # Julia may compile a normalized MethodInstance (for example with
        # `@nospecialize` arguments widened) whose signature differs from
        # `mi.specTypes`, which the declaration is derived from.
        if Core.Compiler.get_ci_mi(ci) !== mi
            throw(CallingConventionMismatchError{String}("Enzyme: Julia compiled the custom rule $(mi) for the MethodInstance $(Core.Compiler.get_ci_mi(ci)). This is not expected to happen, please report it.", mi, world))
        end
        call_convention(mi, ci) === :call || return nothing
        specptr, _ = Interpreter.codeinst_entry(ci)
        if specptr == C_NULL
            throw(CallingConventionMismatchError{String}("Enzyme: Julia compiled the custom rule $(mi) without a specialized entry point. This is not expected to happen, please report it.", mi, world))
        end
        return (ci, specptr)
    end

    # The declaration of a natively called function is named after the relocation name of the
    # `CodeInstance` it calls, which registers that instance as a Julia-value reference like any
    # other (`compiler/relocation.jl`); `bind_native_invokes!` reads it back from the name.
    native_symbol_name(ci::Core.CodeInstance)::String = "ejlnative\$" * relocation_name(ci)

    const SPECPTR_OFFSET = fieldoffset(Core.CodeInstance, Base.fieldindex(Core.CodeInstance, :specptr))

    # Compile `ci` (waiting for it) so that its `specptr` is set before code that loads it runs.
    function ensure_native_compiled!(ci::Core.CodeInstance)::Nothing
        specptr, _ = Interpreter.codeinst_entry(ci)
        specptr == C_NULL && error("Enzyme: CodeInstance $(ci) has no specialized entry point")
        return nothing
    end

    """
        bind_native_invokes!(mod)

    Give every natively called declaration (`declare_native!`) a body that calls the
    function through its `CodeInstance`: the instance is referenced as a Julia value
    (an `ejl_v_*` global, so the module carries no address and Julia's serialization of
    the instance is what makes the reference valid in another session), and its specialized
    entry point is loaded from the instance's `specptr` field at run time, the shape Julia's
    own image-mode codegen gives an `invoke`. The body is always inlined, so each call site
    becomes a load and an indirect call with the declaration's calling convention and
    parameter attributes. Runs after the module string for nested differentiation is taken,
    which must still see declarations (`materialize_native_invokes!`).
    """
    function bind_native_invokes!(mod::LLVM.Module)::Nothing
        marker = StringAttribute("enzymejl_native_invoke")
        prefix = "ejlnative\$"
        for fn in collect(functions(mod))
            isdeclaration(fn) || continue
            has_fn_attr(fn, marker) || continue
            fname = LLVM.name(fn)
            startswith(fname, prefix) || continue
            found, ci = relocation_value(fname[(ncodeunits(prefix) + 1):end])
            (found && ci isa Core.CodeInstance) || error("Enzyme: native invoke $fname has no CodeInstance")
            ft = LLVM.function_type(fn)
            entry = BasicBlock(fn, "entry")
            b = IRBuilder()
            position!(b, entry)
            civ = unsafe_to_llvm(b, ci)
            T_i8 = LLVM.Int8Type()
            T_fnptr = LLVM.PointerType(ft)
            p0 = addrspacecast!(b, civ, LLVM.PointerType(T_i8))
            slot = inbounds_gep!(b, T_i8, p0, [ConstantInt(Int64(SPECPTR_OFFSET))])
            slot = bitcast!(b, slot, LLVM.PointerType(T_fnptr))
            fp = load!(b, T_fnptr, slot)
            ordering!(fp, LLVM.API.LLVMAtomicOrderingMonotonic)
            alignment!(fp, 8)
            args = collect(LLVM.Value, parameters(fn))
            call = call!(b, ft, fp, args)
            callconv!(call, callconv(fn))
            for i in 1:length(args)
                for attr in collect(parameter_attributes(fn, i))
                    LLVM.API.LLVMAddCallSiteAttribute(call, LLVM.API.LLVMAttributeIndex(i), attr)
                end
            end
            if has_fn_attr(fn, EnumAttribute("noreturn"))
                unreachable!(b)
            elseif LLVM.return_type(ft) == LLVM.VoidType()
                ret!(b)
            else
                ret!(b, call)
            end
            dispose(b)
            push!(function_attributes(fn), EnumAttribute("alwaysinline"))
            linkage!(fn, LLVM.API.LLVMInternalLinkage)
        end
        return nothing
    end

    """
        native_return_type(mod::LLVM.Module, mi::MethodInstance, world::UInt) -> Union{Nothing, Type}

    Return the return type of the function `mi` as its natively compiled code
    has it, when [`invoke_codegen!`](@ref) binds the function, called from
    `mod`, to that code. Return `nothing` when it emits the function with
    `nested_codegen!` instead. The rule handlers derive the tape type and check the returned
    derivatives against this type, so it must come from the inference that
    produced the code they call.
    """
    function native_return_type(mod::LLVM.Module, mi::Core.MethodInstance, world::UInt)::Union{Nothing, Type}
        native = native_codeinst(mod, mi, world)
        native === nothing && return nothing
        return native[1].rettype
    end

    """
        invoke_codegen!(enzyme_context, mode, mod, funcspec, world, alwaysinline = false)

    Make the rule `funcspec` callable from `mod`.

    Rules with the `:inline` convention (see [`call_convention`](@ref))
    are emitted into `mod` by `nested_codegen!` and marked always-inline. Like
    all code emitted into `mod`, they are inferred by `EnzymeInterpreter`.

    Rules with the `:call` convention are not emitted at all. Julia's native
    interpreter infers the rule, exactly as an ordinary call of it would, and
    Julia's JIT compiles the resulting `CodeInstance` and its callees (see
    [`codeinst`](@ref)). `declare_native!` declares the specialized
    entry point in `mod`, and `restore_lookups` binds the entry's address. Such
    a rule is compiled once for the whole process, shares its `CodeInstance`s
    and native code with ordinary callers, and is called directly through its
    specsig, without boxing or dispatch. Callers build its arguments as for an
    emitted rule. A rule that Julia compiled without a specialized entry
    point, or for another MethodInstance, is an error (see
    [`native_codeinst`](@ref)).

    The two interpreters must not mix. Code inferred by the native interpreter
    has Julia's semantics: no call in it is marked for a rule,
    `within_autodiff()` is `false`, and no Enzyme intrinsic (such as
    `ignore_derivatives`) is left for the Enzyme pipeline to lower. Such code
    is legal to *call*, because a rule body is primal code that Enzyme never
    differentiates. It is not legal to emit into `mod`, where `check_ir`, the
    rule handlers and the Enzyme pipeline expect `EnzymeInterpreter` output
    and its `enzymejl_*` attributes. Hence a natively called rule reaches
    `mod` only as a declaration bound to an address: it has no body that LLVM
    could inline. Conversely, code inferred by `EnzymeInterpreter` must not be
    handed to Julia's JIT: `within_autodiff()` is `true` there, so a rule body
    that calls `ignore_derivatives` emits a call to
    `__enzyme_ignore_derivatives`, which only the Enzyme pipeline resolves.

    Nested differentiation is the exception to "never differentiates". When a
    derivative is differentiated again, its rule calls must be
    differentiated through, so they need a body. [`materialize_native_invokes!`](@ref)
    gives the declarations one before the outer differentiation runs.

    Every rule is emitted by `nested_codegen!` where the call ABI does not
    exist (see [`native_invoke_available`](@ref)) and on Julia without the 1.12
    compiler API (`Interpreter.HAS_INVOKE_RULES`).
    """
    function invoke_codegen!(
            enzyme_context::EnzymeContext,
            mode::API.CDerivativeMode,
            mod::LLVM.Module,
            funcspec::Core.MethodInstance,
            world::UInt,
            alwaysinline::Bool = false,
        )
        if !(funcspec.specTypes isa DataType) || !native_invoke_available(mod)
            return nested_codegen!(enzyme_context, mode, mod, funcspec, world, alwaysinline)
        end

        if haskey(enzyme_context.nested_cache, funcspec)
            fname = enzyme_context.nested_cache[funcspec]
            if haskey(functions(mod), fname)
                return functions(mod)[fname]
            end
        end

        native = native_codeinst(mod, funcspec, world)
        if native === nothing
            llvmf = nested_codegen!(enzyme_context, mode, mod, funcspec, world, true)
            # Emitted rules do not use the derived signature. Check it against
            # them anyway, so `specsig` stays correct for every rule shape.
            # `EnzymeInterpreter` inferred the emitted rule, so take the return
            # type from the emitted function.
            check_specsig(llvmf, funcspec, enzyme_custom_extract_mi(llvmf)[2])
            return llvmf
        end
        ci, specptr = native

        name = native_symbol_name(ci)
        fn = declare_native!(mod, funcspec, ci.rettype, specptr, name, world)

        # The native code stays valid as long as its CodeInstance does.
        push!(enzyme_context.edges, funcspec)
        push!(enzyme_context.edges, ci)
        enzyme_context.nested_cache[funcspec] = name
        return fn
    end

    """
        materialize_native_invokes!(enzyme_context, mode, mod, world)

    Give every natively called function that `mod` declares a body, so that
    the differentiation of `mod` can differentiate through it.

    A derivative that is differentiated again reaches the outer primal module
    either linked in as a deferred job, or embedded through `enzyme_call`.
    Either way it calls its rules through the declarations
    [`declare_native!`](@ref) made, which `restore_lookups` leaves
    symbolic until the module is compiled (see `_thunk`). Such a declaration
    is an opaque call to the outer differentiation. This emits the rule with
    `nested_codegen!`, as an `:inline` rule, and defines the declaration as an
    always-inline wrapper that forwards to it. The wrapper drops the
    `pgcstack` parameter when the emitted rule takes none.
    """
    function materialize_native_invokes!(enzyme_context::EnzymeContext, mode::API.CDerivativeMode, mod::LLVM.Module, world::UInt)
        marker = StringAttribute("enzymejl_native_invoke")
        for fn in collect(functions(mod))
            isdeclaration(fn) || continue
            has_fn_attr(fn, marker) || continue
            mi, RT = enzyme_custom_extract_mi(fn)
            llvmf = nested_codegen!(enzyme_context, mode, mod, mi, world, true)
            check_specsig(llvmf, mi, enzyme_custom_extract_mi(llvmf)[2])
            # `nested_codegen!` defers linking the emitted module until after
            # the differentiation. The body is needed before it.
            fname = LLVM.name(llvmf)
            for otherMod in enzyme_context.modules_to_link
                link_split_existing!(mod, otherMod)
            end
            empty!(enzyme_context.modules_to_link)
            llvmf = functions(mod)[fname]

            # Drop the `pgcstack` parameter when the emitted rule has none.
            fparams = collect(parameters(fn))
            drop = has_gcstack_arg(llvmf) ? 0 : gcstack_arg_index(fn)
            args = LLVM.Value[p for (i, p) in enumerate(fparams) if i != drop]
            lft = LLVM.function_type(llvmf)
            fretty = LLVM.return_type(LLVM.function_type(fn))
            if length(args) != length(parameters(lft)) || any(value_type(a) != t for (a, t) in zip(args, parameters(lft))) || fretty != LLVM.return_type(lft)
                msg = sprint() do io
                    println(io, "Enzyme: the emitted function does not match the natively called function it replaces")
                    println(io, "mi = ", mi)
                    println(io, "native = ", string(LLVM.function_type(fn)))
                    println(io, "emitted = ", string(lft))
                end
                throw(CallingConventionMismatchError{String}(msg, mi, world))
            end

            entry = BasicBlock(fn, "entry")
            B = IRBuilder()
            position!(B, entry)
            res = call!(B, lft, llvmf, args)
            callconv!(res, callconv(llvmf))
            if fretty isa LLVM.VoidType
                ret!(B)
            else
                ret!(B, res)
            end
            dispose(B)

            fattrs = function_attributes(fn)
            delete!(fattrs, StringAttribute("enzymejl_needs_restoration"))
            delete!(fattrs, marker)
            push!(fattrs, EnumAttribute("alwaysinline"))
            linkage!(fn, LLVM.API.LLVMInternalLinkage)
        end
        return nothing
    end

else

    invoke_codegen!(
        enzyme_context::EnzymeContext,
        mode::API.CDerivativeMode,
        mod::LLVM.Module,
        funcspec::Core.MethodInstance,
        world::UInt,
        alwaysinline::Bool = false,
    ) = nested_codegen!(enzyme_context, mode, mod, funcspec, world, alwaysinline)

    native_return_type(mod::LLVM.Module, mi::Core.MethodInstance, world::UInt) = nothing

    check_emitted_specsig(mod::LLVM.Module, llvmf::LLVM.Function, mi::Core.MethodInstance, @nospecialize(RT::Type)) = nothing

    materialize_native_invokes!(enzyme_context::EnzymeContext, mode::API.CDerivativeMode, mod::LLVM.Module, world::UInt) = nothing

    bind_native_invokes!(mod::LLVM.Module)::Nothing = nothing
    ensure_native_compiled!(ci::Core.CodeInstance)::Nothing = nothing

end
