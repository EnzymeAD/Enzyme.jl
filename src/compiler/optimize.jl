function enzyme!(pm)
    # ccall((:AddEnzymePass, Enzyme_jll.libEnzyme), Nothing, (LLVM.API.LLVMPassManagerRef,), LLVM.ref(pm))
    # Let's not talk about this...
    ptr = Libdl.dlsym(Enzyme_jll.libEnzyme_handle, :AddEnzymePass)
    ccall(ptr, Nothing, (LLVM.API.LLVMPassManagerRef,), pm)
end

function optimize!(mod::LLVM.Module, entry::LLVM.Function; run_enzyme=true, second_stage=true)
    # everying except unroll, slpvec, loop-vec
    # then finish Julia GC
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm[])

        propagate_julia_addrsp!(pm)
        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cfgsimplification!(pm)
        # TODO: DCE (doesn't exist in llvm-c)
        scalar_repl_aggregates!(pm) # SSA variant?
        mem_cpy_opt!(pm)
        always_inliner!(pm)
        alloc_opt!(pm)
        instruction_combining!(pm)
        cfgsimplification!(pm)
        scalar_repl_aggregates!(pm) # SSA variant?
        instruction_combining!(pm)
        jump_threading!(pm)
        instruction_combining!(pm)
        reassociate!(pm)
        early_cse!(pm)
        alloc_opt!(pm)
        loop_idiom!(pm)
        loop_rotate!(pm)
        lower_simdloop!(pm)
        licm!(pm)
        loop_unswitch!(pm)
        instruction_combining!(pm)
        ind_var_simplify!(pm)
        loop_deletion!(pm)
        # SimpleLoopUnroll -- not for Enzyme
        alloc_opt!(pm)
        scalar_repl_aggregates!(pm) # SSA variant?
        instruction_combining!(pm)
        gvn!(pm)
        mem_cpy_opt!(pm)
        sccp!(pm)
        # TODO: Sinking Pass
        # TODO: LLVM <7 InstructionSimplifier
        instruction_combining!(pm)
        jump_threading!(pm)
        dead_store_elimination!(pm)
        alloc_opt!(pm)
        cfgsimplification!(pm)
        loop_idiom!(pm)
        loop_deletion!(pm)
        jump_threading!(pm)
        # SLP_Vectorizer -- not for Enzyme
        aggressive_dce!(pm)
        instruction_combining!(pm)
        # Loop Vectorize -- not for Enzyme
        # InstCombine

        # GC passes
        barrier_noop!(pm)
        lower_exc_handlers!(pm)
        gc_invariant_verifier!(pm, false)
        late_lower_gc_frame!(pm)
        final_lower_gc!(pm)
        # TODO: DCE doesn't exist in llvm-c
        lower_ptls!(pm, #=dump_native=# false)

        # FIXME: Currently crashes printing
        # remove_julia_addrspaces!(pm)
        cfgsimplification!(pm)
        instruction_combining!(pm) # Extra for Enzyme

        run!(pm, mod)
    end

    inactive = LLVM.StringAttribute("enzyme_inactive", "", context(mod))
    for inactivefn in ["jl_gc_queue_root"]
        if haskey(functions(mod), inactivefn)
            fn = functions(mod)[inactivefn]
            push!(function_attributes(fn), inactive)
        end
    end

    ModulePassManager() do pm
        if run_enzyme
            # Enzyme pass
            # barrier_noop!(pm)
            enzyme!(pm)
        end

        if second_stage
            barrier_noop!(pm)

            # run second set of optimizations post enzyme
            scoped_no_alias_aa!(pm)
            type_based_alias_analysis!(pm)
            basic_alias_analysis!(pm)
            cfgsimplification!(pm)
            # TODO: DCE (doesn't exist in llvm-c)
            scalar_repl_aggregates!(pm) # SSA variant?
            mem_cpy_opt!(pm)
            always_inliner!(pm)
            instruction_combining!(pm)
            cfgsimplification!(pm)
            scalar_repl_aggregates!(pm) # SSA variant?
            instruction_combining!(pm)
            jump_threading!(pm)
            instruction_combining!(pm)
            reassociate!(pm)
            early_cse!(pm)
            loop_idiom!(pm)
            loop_rotate!(pm)
            lower_simdloop!(pm)
            licm!(pm)
            loop_unswitch!(pm)
            instruction_combining!(pm)
            ind_var_simplify!(pm)
            loop_deletion!(pm)
            # SimpleLoopUnroll -- not for Enzyme
            scalar_repl_aggregates!(pm) # SSA variant?
            instruction_combining!(pm)
            gvn!(pm)
            mem_cpy_opt!(pm)
            sccp!(pm)
            # TODO: Sinking Pass
            # TODO: LLVM <7 InstructionSimplifier
            instruction_combining!(pm)
            jump_threading!(pm)
            dead_store_elimination!(pm)
            cfgsimplification!(pm)
            loop_idiom!(pm)
            loop_deletion!(pm)
            jump_threading!(pm)
            # SLP_Vectorizer -- not for Enzyme
            aggressive_dce!(pm)
            instruction_combining!(pm)
            # Loop Vectorize -- not for Enzyme
            # InstCombine

            cfgsimplification!(pm)
            instruction_combining!(pm)
            # CombineMulAddPass will run on second pass
        end

        run!(pm, mod)
    end
    return entry
end
