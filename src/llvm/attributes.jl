const nofreefns = Set{String}((
    "ClientNumAddressableDevices",
    "BufferToDevice",
    "BufferToClient",
    "jl_typeof",
    "julia.gc_loaded",
    "jl_egal__unboxed", "ijl_egal__unboxed",
    "jl_restore_excstack",
    "ijl_restore_excstack",
    "ClientGetDevice",
    "BufferOnCPU",
    "pcre2_match_8",
    "julia.gcroot_flush",
    "pcre2_jit_stack_assign_8",
    "pcre2_match_context_create_8",
    "pcre2_jit_stack_create_8",
    "ijl_gc_enable_finalizers_internal",
    "jl_gc_enable_finalizers_internal",
    "pcre2_match_data_create_from_pattern_8",
    "ijl_gc_run_pending_finalizers",
    "jl_gc_run_pending_finalizers",
    "ijl_typeassert",
    "jl_typeassert",
    "ijl_f_isdefined",
    "jl_f_isdefined",
    "ijl_field_index",
    "jl_field_index",
    "ijl_specializations_get_linfo",
    "jl_specializations_get_linfo",
    "ijl_gf_invoke_lookup_worlds",
    "jl_gf_invoke_lookup_worlds",
    "ijl_gc_get_total_bytes",
    "jl_gc_get_total_bytes",
    "ijl_array_grow_at",
    "jl_array_grow_at",
    "ijl_try_substrtod",
    "jl_try_substrtod",
    "jl_f__apply_iterate",
    "ijl_field_index",
    "jl_field_index",
    "julia.call",
    "julia.call2",
    "ijl_tagged_gensym",
    "jl_tagged_gensym",
    "ijl_array_ptr_copy",
    "jl_array_ptr_copy",
    "ijl_array_copy",
    "jl_array_copy",
    "ijl_genericmemory_slice",
    "jl_genericmemory_slice",
    "ijl_genericmemory_copy_slice",
    "jl_genericmemory_copy_slice",
    "ijl_get_nth_field_checked",
    "ijl_get_nth_field_checked",
    "jl_array_del_end",
    "ijl_array_del_end",
    "jl_get_world_counter",
    "ijl_get_world_counter",
    "memhash32_seed",
    "memhash_seed",
    "ijl_module_parent",
    "jl_module_parent",
    "julia.safepoint",
    "ijl_set_task_tid",
    "jl_set_task_tid",
    "ijl_get_task_tid",
    "jl_get_task_tid",
    "julia.get_pgcstack_or_new",
    "ijl_global_event_loop",
    "jl_global_event_loop",
    "ijl_gf_invoke_lookup",
    "jl_gf_invoke_lookup",
    "ijl_f_typeassert",
    "jl_f_typeassert",
    "ijl_type_unionall",
    "jl_type_unionall",
    "jl_gc_queue_root",
    "gpu_report_exception",
    "gpu_signal_exception",
    "julia.ptls_states",
    "julia.write_barrier",
    "julia.typeof",
    "jl_backtrace_from_here",
    "ijl_backtrace_from_here",
    "jl_box_int64",
    "jl_box_int32",
    "ijl_box_int64",
    "ijl_box_int32",
    "jl_box_uint64",
    "jl_box_uint32",
    "ijl_box_uint64",
    "ijl_box_uint32",
    "ijl_box_char",
    "jl_box_char",
    "ijl_subtype",
    "jl_subtype",
    "julia.get_pgcstack",
    "jl_in_threaded_region",
    "jl_object_id_",
    "jl_object_id",
    "ijl_object_id_",
    "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin",
    "llvm.julia.gc_preserve_end",
    "jl_get_ptls_states",
    "ijl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline",
    "ijl_stored_inline",
    "jl_f_apply_type",
    "jl_f_issubtype",
    "jl_isa",
    "ijl_isa",
    "jl_matching_methods",
    "ijl_matching_methods",
    "jl_excstack_state",
    "ijl_excstack_state",
    "jl_current_exception",
    "ijl_current_exception",
    "memhash_seed",
    "jl_f__typevar",
    "ijl_f__typevar",
    "jl_f_isa",
    "ijl_f_isa",
    "jl_set_task_threadpoolid",
    "ijl_set_task_threadpoolid",
    "jl_types_equal",
    "ijl_types_equal",
    "jl_invoke",
    "ijl_invoke",
    "jl_apply_generic",
    "ijl_apply_generic",
    "jl_egal__unboxed",
    "julia.pointer_from_objref",
    "_platform_memcmp",
    "memcmp",
    "julia.except_enter",
    "jl_array_grow_end",
    "ijl_array_grow_end",
    "jl_f_getfield",
    "ijl_f_getfield",
    "jl_pop_handler",
    "ijl_pop_handler",
    "jl_pop_handler_noexcept",
    "ijl_pop_handler_noexcept",
    "jl_string_to_array",
    "ijl_string_to_array",
    "jl_alloc_string",
    "ijl_alloc_string",
    "getenv",
    "jl_cstr_to_string",
    "ijl_cstr_to_string",
    "jl_symbol_n",
    "ijl_symbol_n",
    "uv_os_homedir",
    "jl_array_to_string",
    "ijl_array_to_string",
    "pcre2_jit_compile_8",
    "memmove",
))

const inactivefns = Set{String}((
    "ClientNumAddressableDevices",
    "BufferToDevice",
    "BufferToClient",
    "jl_typeof",
    "jl_egal__unboxed", "ijl_egal__unboxed",
    "ClientGetDevice",
    "BufferOnCPU",
    "pcre2_match_data_create_from_pattern_8",
    "ijl_typeassert",
    "jl_typeassert",
    "ijl_f_isdefined",
    "jl_f_isdefined",
    "ijl_field_index",
    "jl_field_index",
    "ijl_specializations_get_linfo",
    "jl_specializations_get_linfo",
    "ijl_gf_invoke_lookup_worlds",
    "jl_gf_invoke_lookup_worlds",
    "ijl_gc_get_total_bytes",
    "jl_gc_get_total_bytes",
    "ijl_try_substrtod",
    "jl_try_substrtod",
    "ijl_tagged_gensym",
    "jl_tagged_gensym",
    "jl_get_world_counter",
    "ijl_get_world_counter",
    "memhash32_seed",
    "memhash_seed",
    "ijl_module_parent",
    "jl_module_parent",
    "julia.safepoint",
    "ijl_set_task_tid",
    "jl_set_task_tid",
    "ijl_get_task_tid",
    "jl_get_task_tid",
    "julia.get_pgcstack_or_new",
    "ijl_global_event_loop",
    "jl_global_event_loop",
    "ijl_gf_invoke_lookup",
    "jl_gf_invoke_lookup",
    "ijl_f_typeassert",
    "jl_f_typeassert",
    "ijl_type_unionall",
    "jl_type_unionall",
    "jl_gc_queue_root",
    "gpu_report_exception",
    "gpu_signal_exception",
    "julia.ptls_states",
    "julia.write_barrier",
    "julia.typeof",
    "jl_backtrace_from_here",
    "ijl_backtrace_from_here",
    "jl_box_int64",
    "jl_box_int32",
    "ijl_box_int64",
    "ijl_box_int32",
    "jl_box_uint64",
    "jl_box_uint32",
    "ijl_box_uint64",
    "ijl_box_uint32",
    "ijl_box_char",
    "jl_box_char",
    "ijl_subtype",
    "jl_subtype",
    "julia.get_pgcstack",
    "jl_in_threaded_region",
    "jl_object_id_",
    "jl_object_id",
    "ijl_object_id_",
    "ijl_object_id",
    "jl_breakpoint",
    "llvm.julia.gc_preserve_begin",
    "llvm.julia.gc_preserve_end",
    "jl_get_ptls_states",
    "ijl_get_ptls_states",
    "jl_f_fieldtype",
    "jl_symbol_n",
    "jl_stored_inline",
    "ijl_stored_inline",
    "jl_f_apply_type",
    "jl_f_issubtype",
    "jl_isa",
    "ijl_isa",
    "jl_matching_methods",
    "ijl_matching_methods",
    "jl_excstack_state",
    "ijl_excstack_state",
    "jl_current_exception",
    "ijl_current_exception",
    "memhash_seed",
    "jl_f__typevar",
    "ijl_f__typevar",
    "jl_f_isa",
    "ijl_f_isa",
    "jl_set_task_threadpoolid",
    "ijl_set_task_threadpoolid",
    "jl_types_equal",
    "ijl_types_equal",
    "jl_string_to_array",
    "ijl_string_to_array",
    "jl_alloc_string",
    "ijl_alloc_string",
    "getenv",
    "jl_cstr_to_string",
    "ijl_cstr_to_string",
    "jl_symbol_n",
    "ijl_symbol_n",
    "uv_os_homedir",
    "jl_array_to_string",
    "ijl_array_to_string",
    "pcre2_jit_compile_8",
    # "jl_"
))

const activefns = Set{String}(("jl_",))

const inactiveglobs = Set{String}((
    "ijl_boxed_uint8_cache",
    "jl_boxed_uint8_cache",
    "ijl_boxed_int8_cache",
    "jl_boxed_int8_cache",
    "jl_nothing",
))

function annotate!(mod::LLVM.Module)
    inactive = LLVM.StringAttribute("enzyme_inactive", "")
    active = LLVM.StringAttribute("enzyme_active", "")
    no_escaping_alloc = LLVM.StringAttribute("enzyme_no_escaping_allocation")

    funcs = Dict{String, Vector{LLVM.Function}}()
    for f in functions(mod)
        fname = LLVM.name(f)
        for fattr in collect(function_attributes(f))
            if isa(fattr, LLVM.StringAttribute)
                if kind(fattr) == "enzyme_math"
                    fname = LLVM.value(fattr)
                    break
                end
            end
        end
        fname = String(fname)
        if !haskey(funcs, fname)
            funcs[fname] = LLVM.Function[]
        end
        push!(funcs[String(fname)], f)
        API.EnzymeAttributeKnownFunctions(f.ref)
    end

    for gname in inactiveglobs
        globs = LLVM.globals(mod)
        if haskey(globs, gname)
            glob = globs[gname]
            API.SetMD(glob, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
        end
    end

    for fname in inactivefns
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), inactive)
                push!(function_attributes(fn), no_escaping_alloc)
                for u in LLVM.uses(fn)
                    c = LLVM.user(u)
                    if !isa(c, LLVM.CallInst)
                        continue
                    end
                    cf = LLVM.called_operand(c)
                    if !isa(cf, LLVM.Function)
                        continue
                    end
                    if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                        continue
                    end
                    if operands(c)[1] != fn
                        continue
                    end
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        reinterpret(
                            LLVM.API.LLVMAttributeIndex,
                            LLVM.API.LLVMAttributeFunctionIndex,
                        ),
                        inactive,
                    )
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        reinterpret(
                            LLVM.API.LLVMAttributeIndex,
                            LLVM.API.LLVMAttributeFunctionIndex,
                        ),
                        no_escaping_alloc,
                    )
                end
            end
        end
    end

    for fname in nofreefns
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), LLVM.EnumAttribute("nofree", 0))
                for u in LLVM.uses(fn)
                    c = LLVM.user(u)
                    if !isa(c, LLVM.CallInst)
                        continue
                    end
                    cf = LLVM.called_operand(c)
                    if !isa(cf, LLVM.Function)
                        continue
                    end
                    if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                        continue
                    end
                    if operands(c)[1] != fn
                        continue
                    end
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        reinterpret(
                            LLVM.API.LLVMAttributeIndex,
                            LLVM.API.LLVMAttributeFunctionIndex,
                        ),
                        LLVM.EnumAttribute("nofree", 0),
                    )
                end
            end
        end
    end

    for fname in activefns
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), active)
            end
        end
    end

    for fname in
        ("julia.typeof", "jl_object_id_", "jl_object_id", "ijl_object_id_", "ijl_object_id")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readnone"))
                else
                    push!(function_attributes(fn), EnumAttribute("memory", NoEffects.data))
                end
                push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
            end
        end
    end
    for fname in ("julia.typeof",)
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), LLVM.StringAttribute("enzyme_nocache"))
                push!(parameter_attributes(fn, 1), LLVM.EnumAttribute("nocapture"))
            end
        end
    end

    for fname in
        ("jl_excstack_state", "ijl_excstack_state", "ijl_field_index", "jl_field_index")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readonly"))
                    push!(function_attributes(fn), LLVM.StringAttribute("inaccessiblememonly"))
                else
                    push!(
                        function_attributes(fn),
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_Ref << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                    )
                end
            end
        end
    end

    for fname in ("jl_types_equal", "ijl_types_equal")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
            end
        end
    end

    for fname in (
        "UnsafeBufferPointer",
    )
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.StringAttribute("enzyme_math", "__dynamic_cast"))
                end
            end
        end
    end

    for fname in (
        "jl_f_getfield",
        "ijl_f_getfield",
        "jl_get_nth_field_checked",
        "ijl_get_nth_field_checked",
        "jl_f__svec_ref",
        "ijl_f__svec_ref",
        "UnsafeBufferPointer"
    )
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readonly", 0))
                else
                    push!(function_attributes(fn), 
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_Ref << getLocationPos(ArgMem)) |
                                (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        )
                    )
                end
                for u in LLVM.uses(fn)
                    c = LLVM.user(u)
                    if !isa(c, LLVM.CallInst)
                        continue
                    end
                    cf = LLVM.called_operand(c)
                    if !isa(cf, LLVM.Function)
                        continue
                    end
                    if LLVM.name(cf) != "julia.call" && LLVM.name(cf) != "julia.call2"
                        continue
                    end
                    if operands(c)[1] != fn
                        continue
                    end
                    attr = if LLVM.version().major <= 15
                        LLVM.EnumAttribute("readonly")
                    else
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_Ref << getLocationPos(ArgMem)) |
                                (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        )
                    end
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        reinterpret(
                            LLVM.API.LLVMAttributeIndex,
                            LLVM.API.LLVMAttributeFunctionIndex,
                        ),
                        attr,
                    )
                end
            end
        end
    end

    for fname in ("julia.get_pgcstack", "julia.ptls_states", "jl_get_ptls_states")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                # TODO per discussion w keno perhaps this should change to readonly / inaccessiblememonly
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readnone"))
                else
                    push!(function_attributes(fn), EnumAttribute("memory", NoEffects.data))
                end
                push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
            end
        end
    end

    for fname in ("julia.gc_loaded",)
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), LLVM.StringAttribute("enzyme_shouldrecompute"))
                push!(function_attributes(fn), LLVM.StringAttribute("enzyme_nocache"))
            end
        end
    end

    for fname in (
        "julia.get_pgcstack",
        "julia.ptls_states",
        "jl_get_ptls_states",
        "julia.safepoint",
        "ijl_throw",
        "julia.pointer_from_objref",
        "ijl_array_grow_end",
        "jl_array_grow_end",
        "ijl_array_del_end",
        "jl_array_del_end",
        "ijl_array_grow_beg",
        "jl_array_grow_beg",
        "ijl_array_del_beg",
        "jl_array_del_beg",
        "ijl_array_grow_at",
        "jl_array_grow_at",
        "ijl_array_del_at",
        "jl_array_del_at",
        "ijl_pop_handler",
        "jl_pop_handler",
        "ijl_pop_handler_noexcept",
        "jl_pop_handler_noexcept",
        "ijl_push_handler",
        "jl_push_handler",
        "ijl_module_name",
        "jl_module_name",
        "ijl_restore_excstack",
        "jl_restore_excstack",
        "julia.except_enter",
        "ijl_get_nth_field_checked",
        "jl_get_nth_field_checked",
        "jl_egal__unboxed",
        "ijl_reshape_array",
        "jl_reshape_array",
        "ijl_eqtable_get",
        "jl_eqtable_get",
        "jl_gc_run_pending_finalizers",
        "ijl_try_substrtod",
        "jl_try_substrtod",
    )
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(function_attributes(fn), no_escaping_alloc)
            end
        end
    end



    for fname in ("julia.pointer_from_objref",)
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readnone"))
                else
                    push!(function_attributes(fn), EnumAttribute("memory", NoEffects.data))
                end
            end
        end
    end

    for fname in (
        "julia.gc_alloc_obj",
        "jl_gc_alloc_typed",
        "ijl_gc_alloc_typed",
    )
        if haskey(funcs, fname)
            for fn in funcs[fname]
                LLVM.API.LLVMRemoveEnumAttributeAtIndex(
                    fn,
                    reinterpret(LLVM.API.LLVMAttributeIndex, LLVM.API.LLVMAttributeFunctionIndex),
                    kind(EnumAttribute("allockind", AllocFnKind(AFKE_Alloc).data)),
                )
                push!(function_attributes(fn), no_escaping_alloc)
                push!(function_attributes(fn), LLVM.EnumAttribute("allockind", (AllocFnKind(AFKE_Alloc) | AllocFnKind(AFKE_Uninitialized)).data))
            end
        end
    end

    for fname in (
        "julia.gc_alloc_obj",
        "jl_gc_alloc_typed",
        "ijl_gc_alloc_typed",
        "jl_box_float32",
        "jl_box_float64",
        "jl_box_int32",
        "jl_box_int64",
        "ijl_box_float32",
        "ijl_box_float64",
        "ijl_box_int32",
        "ijl_box_int64",
        "jl_alloc_genericmemory",
        "ijl_alloc_genericmemory",
        "jl_alloc_array_1d",
        "jl_alloc_array_2d",
        "jl_alloc_array_3d",
        "ijl_alloc_array_1d",
        "ijl_alloc_array_2d",
        "ijl_alloc_array_3d",
        "jl_array_copy",
        "ijl_array_copy",
        "jl_genericmemory_slice",
        "ijl_genericmemory_slice",
        "jl_genericmemory_copy_slice",
        "ijl_genericmemory_copy_slice",
        "jl_alloc_genericmemory",
        "ijl_alloc_genericmemory",
        "jl_idtable_rehash",
        "ijl_idtable_rehash",
        "jl_f_tuple",
        "ijl_f_tuple",
        "jl_new_structv",
        "ijl_new_structv",
        "ijl_new_array",
        "jl_new_array",
    )
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(return_attributes(fn), LLVM.EnumAttribute("noalias", 0))
                push!(return_attributes(fn), LLVM.EnumAttribute("nonnull", 0))
                push!(function_attributes(fn), no_escaping_alloc)
                push!(function_attributes(fn), LLVM.EnumAttribute("mustprogress"))
                push!(function_attributes(fn), LLVM.EnumAttribute("willreturn"))
                push!(function_attributes(fn), LLVM.EnumAttribute("nounwind"))
                push!(function_attributes(fn), LLVM.EnumAttribute("nofree"))
                accattr = if LLVM.version().major <= 15
                    LLVM.EnumAttribute("inaccessiblememonly")
                else
                    if fname in (
                        "jl_genericmemory_slice",
                        "ijl_genericmemory_slice",
                        "jl_genericmemory_copy_slice",
                        "ijl_genericmemory_copy_slice",
                        )
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_Ref << getLocationPos(ArgMem)) |
                                (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        )
                    else 
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        )
                    end
                end
                if !(
                    fname in (
                        "jl_array_copy",
                        "ijl_array_copy",
                        "jl_idtable_rehash",
                        "ijl_idtable_rehash",
                    )
                )
                    push!(function_attributes(fn), accattr)
                end
                for u in LLVM.uses(fn)
                    c = LLVM.user(u)
                    if !isa(c, LLVM.CallInst)
                        continue
                    end
                    cf = LLVM.called_operand(c)
                    if cf == fn
                        LLVM.API.LLVMAddCallSiteAttribute(
                            c,
                            LLVM.API.LLVMAttributeReturnIndex,
                            LLVM.EnumAttribute("noalias", 0),
                        )
                        if !(
                            fname in (
                                "jl_array_copy",
                                "ijl_array_copy",
                                "jl_idtable_rehash",
                                "ijl_idtable_rehash",
                            )
                        )
                            LLVM.API.LLVMAddCallSiteAttribute(
                                c,
                                reinterpret(
                                    LLVM.API.LLVMAttributeIndex,
                                    LLVM.API.LLVMAttributeFunctionIndex,
                                ),
                                accattr,
                            )
                        end
                    end
                    if !isa(cf, LLVM.Function)
                        continue
                    end
                    if !(cf == fn ||
                         ((LLVM.name(cf) == "julia.call" || LLVM.name(cf) != "julia.call2") && operands(c)[1] == fn))
                        continue
                    end
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        LLVM.API.LLVMAttributeReturnIndex,
                        LLVM.EnumAttribute("noalias", 0),
                    )
                    LLVM.API.LLVMAddCallSiteAttribute(
                        c,
                        reinterpret(
                            LLVM.API.LLVMAttributeIndex,
                            LLVM.API.LLVMAttributeFunctionIndex,
                        ),
                        no_escaping_alloc,
                    )
                    if !(
                        fname in (
                            "jl_array_copy",
                            "ijl_array_copy",
                            "jl_idtable_rehash",
                            "ijl_idtable_rehash",
                        )
                    )
                        LLVM.API.LLVMAddCallSiteAttribute(
                            c,
                            reinterpret(
                                LLVM.API.LLVMAttributeIndex,
                                LLVM.API.LLVMAttributeFunctionIndex,
                            ),
                            accattr,
                        )
                    end
                end
            end
        end
    end

    for fname in ("llvm.julia.gc_preserve_begin", "llvm.julia.gc_preserve_end")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly"))
                else
                    push!(
                        function_attributes(fn),
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_ModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                    )
                end
            end
        end
    end

    # Key of jl_eqtable_get/put is inactive, definitionally
    for fname in ("jl_eqtable_get", "ijl_eqtable_get")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(parameter_attributes(fn, 2), LLVM.StringAttribute("enzyme_inactive"))
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readonly"))
                    push!(function_attributes(fn), LLVM.EnumAttribute("argmemonly"))
                else
                    push!(
                        function_attributes(fn),
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_Ref << getLocationPos(ArgMem)) |
                                (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                    )
                end
            end
        end
    end
    
    for fname in ("jl_reshape_array", "ijl_reshape_array")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(parameter_attributes(fn, 3), LLVM.EnumAttribute("readonly"))
                push!(parameter_attributes(fn, 3), LLVM.EnumAttribute("nocapture"))
            end
        end
    end
    
    # Key of jl_eqtable_get/put is inactive, definitionally
    for fname in ("jl_eqtable_put", "ijl_eqtable_put")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                push!(parameter_attributes(fn, 2), LLVM.StringAttribute("enzyme_inactive"))
                push!(parameter_attributes(fn, 4), LLVM.StringAttribute("enzyme_inactive"))
                if value_type(LLVM.parameters(fn)[4]) isa LLVM.PointerType
                    push!(parameter_attributes(fn, 4), LLVM.EnumAttribute("writeonly"))
                    push!(parameter_attributes(fn, 4), LLVM.EnumAttribute("nocapture"))
                end
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("argmemonly"))
                else
                    push!(
                        function_attributes(fn),
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_ModRef << getLocationPos(ArgMem)) |
                                (MRI_NoModRef << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                    )
                end
            end
        end
    end

    for fname in ("jl_in_threaded_region_", "jl_in_threaded_region")
        if haskey(funcs, fname)
            for fn in funcs[fname]
                if LLVM.version().major <= 15
                    push!(function_attributes(fn), LLVM.EnumAttribute("readonly"))
                    push!(function_attributes(fn), LLVM.EnumAttribute("inaccessiblememonly"))
                else
                    push!(
                        function_attributes(fn),
                        EnumAttribute(
                            "memory",
                            MemoryEffect(
                                (MRI_NoModRef << getLocationPos(ArgMem)) |
                                (MRI_Ref << getLocationPos(InaccessibleMem)) |
                                (MRI_NoModRef << getLocationPos(Other)),
                            ).data,
                        ),
                    )
                end
            end
        end
    end
end

function mark_gpu_intrinsics!(target, mod::LLVM.Module)
    if target isa GPUCompiler.PTXCompilerTarget
    
        arg1 = (
            "sin",
            "cos",
            "tan",
            "log2",
            "exp",
            "exp2",
            "exp10",
            "cosh",
            "sinh",
            "tanh",
            "atan",
            "asin",
            "acos",
            "log",
            "log10",
            "log1p",
            "acosh",
            "asinh",
            "atanh",
            "expm1",
            "cbrt",
            "rcbrt",
            "j0",
            "j1",
            "y0",
            "y1",
            "erf",
            "erfinv",
            "erfc",
            "erfcx",
            "erfcinv",
            "remquo",
            "tgamma",
            "round",
            "fdim",
            "logb",
            "isinf",
            "sqrt",
            "fabs",
            "atan2",
        )
        # isinf, finite "modf",       "fmod",    "remainder", 
        # "rnorm3d",    "norm4d",  "rnorm4d",   "norm",   "rnorm",
        #   "hypot",  "rhypot",
        # "yn", "jn", "norm3d", "ilogb", powi
        # "normcdfinv", "normcdf", "lgamma",    "ldexp",  "scalbn", "frexp",
        # arg1 = ("atan2", "fmax", "pow")
        for n in arg1,
            (T, pf, lpf) in
            ((LLVM.DoubleType(), "", "f64"), (LLVM.FloatType(), "f", "f32"))

            fname = "__nv_" * n * pf
            if !haskey(functions(mod), fname)
                FT = LLVM.FunctionType(T, [T], vararg = false)
                wrapper_f = LLVM.Function(mod, fname, FT)
                llname = "llvm." * n * "." * lpf
                push!(
                    function_attributes(wrapper_f),
                    StringAttribute("implements", llname),
                )
                push!(
                    function_attributes(wrapper_f),
        StringAttribute("implements2", n * pf)
                )
            end
        end
    end
    if target isa GPUCompiler.GCNCompilerTarget
        arg1 = (
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan2",
            "atan",
            "atanh",
            "cbrt",
            "ceil",
            "copysign",
            "cos",
            "native_cos",
            "cosh",
            "cospi",
            "i0",
            "i1",
            "erfc",
            "erfcinv",
            "erfcx",
            "erf",
            "erfinv",
            "exp10",
            "native_exp10",
            "exp2",
            "exp",
            "native_exp",
            "expm1",
            "fabs",
            "fdim",
            "floor",
            "fma",
            "fmax",
            "fmin",
            "fmod",
            "frexp",
            "hypot",
            "ilogb",
            "isfinite",
            "isinf",
            "isnan",
            "j0",
            "j1",
            "ldexp",
            "lgamma",
            "log10",
            "native_log10",
            "log1p",
            "log2",
            "log2",
            "logb",
            "log",
            "native_log",
            "modf",
            "nearbyint",
            "nextafter",
            "len3",
            "len4",
            "ncdf",
            "ncdfinv",
            "pow",
            "pown",
            "rcbrt",
            "remainder",
            "remquo",
            "rhypot",
            "rint",
            "rlen3",
            "rlen4",
            "round",
            "rsqrt",
            "scalb",
            "scalbn",
            "signbit",
            "sincos",
            "sincospi",
            "sin",
            "native_sin",
            "sinh",
            "sinpi",
            "sqrt",
            "native_sqrt",
            "tan",
            "tanh",
            "tgamma",
            "trunc",
            "y0",
            "y1",
        )
        for n in arg1,
            (T, pf, lpf) in
            ((LLVM.DoubleType(), "", "f64"), (LLVM.FloatType(), "f", "f32"))

            fname = "__ocml_" * n * "_" * lpf
            if !haskey(functions(mod), fname)
                FT = LLVM.FunctionType(T, [T], vararg = false)
                wrapper_f = LLVM.Function(mod, fname, FT)
                llname = "llvm." * n * "." * lpf
                push!(
                    function_attributes(wrapper_f),
                    StringAttribute("implements", llname),
                )
                push!(
                    function_attributes(wrapper_f),
        StringAttribute("implements2", n * pf)
                )
            end
        end
    end
end
