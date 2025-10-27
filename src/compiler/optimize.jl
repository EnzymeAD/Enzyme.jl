function registerEnzymeAndPassPipeline!(pb::NewPMPassBuilder)
    enzyme_callback = cglobal((:registerEnzymeAndPassPipeline, API.libEnzyme))
    LLVM.API.LLVMPassBuilderExtensionsPushRegistrationCallbacks(pb.exts, enzyme_callback)
end

LLVM.@function_pass "jl-inst-simplify" JLInstSimplifyPass

const RunAttributor = Ref(true)

function enzyme_attributor_pass!(mod::LLVM.Module)
    ccall(
        (:RunAttributorOnModule, API.libEnzyme),
        Cvoid,
        (LLVM.API.LLVMModuleRef,),
        mod,
    )
    return true
end

EnzymeAttributorPass() = NewPMModulePass("enzyme_attributor", enzyme_attributor_pass!)
ReinsertGCMarkerPass() = NewPMFunctionPass("reinsert_gcmarker", reinsert_gcmarker_pass!)
SafeAtomicToRegularStorePass() = NewPMFunctionPass("safe_atomic_to_regular_store", safe_atomic_to_regular_store!)

@static if VERSION < v"1.11-"
    function propagate_julia_addrsp_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        propagate_julia_addrsp!(pm)
    end
else
    function propagate_julia_addrsp_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function prop_julia_addr(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, PropagateJuliaAddrspacesPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("PropagateJuliaAddrSpace", prop_julia_addr))
    end
end

@static if VERSION < v"1.11-"
    function alloc_opt_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        alloc_opt!(pm)
    end
else
    function alloc_opt_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function alloc_opt(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, AllocOptPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("AllocOpt", alloc_opt))
    end
end

@static if VERSION < v"1.11-"
    function lower_simdloop_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        lower_simdloop!(pm)
    end
else
    function lower_simdloop_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function lower_simdloop(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, NewPMLoopPassManager()) do lpm
                            add!(lpm, LowerSIMDLoopPass())
                        end
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        # really looppass
        add!(pm, ModulePass("LowerSIMDLoop", lower_simdloop))
    end
end

function loop_optimizations_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
    lower_simdloop_tm!(pm, tm)
    licm!(pm)
    if LLVM.version() >= v"15"
        simple_loop_unswitch_legacy!(pm)
    else
        loop_unswitch!(pm)
    end
end

function more_loop_optimizations_newPM!(fpm::LLVM.NewPMPassManager)
    add!(fpm, NewPMLoopPassManager()) do lpm
        add!(lpm, LoopRotatePass())
        # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
        # add!(lpm, LoopIdiomPass()) TODO(NewPM): This seems to have gotten removed

        # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
        add!(lpm, LowerSIMDLoopPass()) # Annotate loop marked with "loopinfo" as LLVM parallel loop
        add!(lpm, LICMPass())
        add!(lpm, JuliaLICMPass())
    end
    add!(fpm, InstCombinePass())
    add!(fpm, JLInstSimplifyPass())
    add!(fpm, NewPMLoopPassManager()) do lpm
        add!(lpm, IndVarSimplifyPass())
        add!(lpm, LoopDeletionPass())
    end
    add!(fpm, LoopUnrollPass(opt_level=2))
end

@static if VERSION < v"1.11-"
    function cpu_features_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        @static if isdefined(LLVM.Interop, :cpu_features!)
            LLVM.Interop.cpu_features!(pm)
        else
            @static if isdefined(GPUCompiler, :cpu_features!)
                GPUCompiler.cpu_features!(pm)
            end
        end
    end
else
    function cpu_features_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function cpu_features(mod)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, CPUFeaturesPass())
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("CPUFeatures", cpu_features))
    end
end

function jl_inst_simplify!(PM::LLVM.ModulePassManager)
    ccall(
        (:LLVMAddJLInstSimplifyPass, API.libEnzyme),
        Cvoid,
        (LLVM.API.LLVMPassManagerRef,),
        PM,
    )
end

cse!(pm) = LLVM.API.LLVMAddEarlyCSEPass(pm)

function optimize!(mod::LLVM.Module, tm::LLVM.TargetMachine)
    addr13NoAlias(mod)
    if !LLVM.has_oldpm()
        # TODO(NewPM)
        return
    end
    # everying except unroll, slpvec, loop-vec
    # then finish Julia GC
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        propagate_julia_addrsp_tm!(pm, tm)
        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cfgsimplification!(pm)
        dce!(pm)
        cpu_features_tm!(pm, tm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        mem_cpy_opt!(pm)
        always_inliner!(pm)
        alloc_opt_tm!(pm, tm)
        LLVM.run!(pm, mod)
    end

    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    ModulePassManager() do pm

        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cpu_features_tm!(pm, tm)

        LLVM.API.LLVMAddGlobalOptimizerPass(pm) # Extra
        gvn!(pm) # Extra
        LLVM.run!(pm, mod)
    end

    rewrite_generic_memory!(mod)
    
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cpu_features_tm!(pm, tm)

        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        cfgsimplification!(pm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        jump_threading!(pm)
        correlated_value_propagation!(pm)
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        reassociate!(pm)
        early_cse!(pm)
        alloc_opt_tm!(pm, tm)
        loop_idiom!(pm)
        loop_rotate!(pm)

        loop_optimizations_tm!(pm, tm)

        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        ind_var_simplify!(pm)
        loop_deletion!(pm)
        loop_unroll!(pm)
        alloc_opt_tm!(pm, tm)
        scalar_repl_aggregates_ssa!(pm) # SSA variant?
        gvn!(pm)

        # This InstCombine needs to be after GVN
        # Otherwise it will generate load chains in GPU code...
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        mem_cpy_opt!(pm)
        sccp!(pm)
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        jump_threading!(pm)
        dead_store_elimination!(pm)
        alloc_opt_tm!(pm, tm)
        cfgsimplification!(pm)
        loop_idiom!(pm)
        loop_deletion!(pm)
        jump_threading!(pm)
        correlated_value_propagation!(pm)
        # SLP_Vectorizer -- not for Enzyme

        LLVM.run!(pm, mod)

        aggressive_dce!(pm)
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        # Loop Vectorize -- not for Enzyme
        # InstCombine

        # GC passes
        barrier_noop!(pm)
        gc_invariant_verifier_tm!(pm, tm, false)

        # FIXME: Currently crashes printing
        cfgsimplification!(pm)
        instruction_combining!(pm) # Extra for Enzyme
        jl_inst_simplify!(pm)
        LLVM.run!(pm, mod)
    end
    
    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)

        scoped_no_alias_aa!(pm)
        type_based_alias_analysis!(pm)
        basic_alias_analysis!(pm)
        cpu_features_tm!(pm, tm)

        LLVM.API.LLVMAddGlobalOptimizerPass(pm) # Exxtra
        gvn!(pm) # Exxtra
        LLVM.run!(pm, mod)
    end
    removeDeadArgs!(mod, tm)
    detect_writeonly!(mod)
    nodecayed_phis!(mod)
end

function addOptimizationPasses!(mpm::LLVM.NewPMPassManager)
    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, ReinsertGCMarkerPass())
    end

    add!(mpm, ConstantMergePass())

    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, PropagateJuliaAddrspacesPass())

        add!(fpm, SimplifyCFGPass())
        add!(fpm, DCEPass())
        add!(fpm, SROAPass())
    end

    add!(mpm, AlwaysInlinerPass())

    add!(mpm, NewPMFunctionPassManager()) do fpm
        # Running `memcpyopt` between this and `sroa` seems to give `sroa` a hard time
        # merging the `alloca` for the unboxed data and the `alloca` created by the `alloc_opt`
        # pass.


        add!(fpm, AllocOptPass())
        # consider AggressiveInstCombinePass at optlevel > 2

        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, SimplifyCFGPass())
        add!(fpm, SROAPass())
        add!(fpm, InstSimplifyPass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, JumpThreadingPass())
        add!(fpm, CorrelatedValuePropagationPass())

        add!(fpm, ReassociatePass())
        add!(fpm, EarlyCSEPass())

        # Load forwarding above can expose allocations that aren't actually used
        # remove those before optimizing loops.
        add!(fpm, AllocOptPass())

        more_loop_optimizations_newPM!(fpm)

        # Run our own SROA on heap objects before LLVM's
        add!(fpm, AllocOptPass())
        # Re-run SROA after loop-unrolling (useful for small loops that operate,
        # over the structure of an aggregate)
        add!(fpm, SROAPass())
        add!(fpm, InstSimplifyPass())

        add!(fpm, GVNPass())
        add!(fpm, MemCpyOptPass())
        add!(fpm, SCCPPass())

        # Run instcombine after redundancy elimination to exploit opportunities
        # opened up by them.
        # This needs to be InstCombine instead of InstSimplify to allow
        # loops over Union-typed arrays to vectorize.
        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, JumpThreadingPass())
        add!(fpm, DSEPass())
        add!(fpm, SafeAtomicToRegularStorePass())

        # More dead allocation (store) deletion before loop optimization
        # consider removing this:
        add!(fpm, AllocOptPass())

        # see if all of the constant folding has exposed more loops
        # to simplification and deletion
        # this helps significantly with cleaning up iteration
        add!(fpm, SimplifyCFGPass())
        add!(fpm, LoopDeletionPass())
        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, LoopVectorizePass())
        add!(fpm, SimplifyCFGPass())
        add!(fpm, SLPVectorizerPass())
        add!(fpm, ADCEPass())
    end
end

function addMachinePasses!(mpm::LLVM.NewPMPassManager)
    add!(mpm, NewPMFunctionPassManager()) do fpm
        if VERSION < v"1.12.0-DEV.1390"
            add!(fpm, CombineMulAddPass())
        end
        add!(fpm, DivRemPairsPass())
        add!(fpm, DemoteFloat16Pass())
        add!(fpm, GVNPass())              
    end
end

function addJuliaLegalizationPasses!(mpm::LLVM.NewPMPassManager, lower_intrinsics::Bool = true)
    if lower_intrinsics
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, ReinsertGCMarkerPass())
            if VERSION < v"1.13.0-DEV.36"
                add!(fpm, LowerExcHandlersPass())
            end
            # TODO: strong=false?
            add!(fpm, GCInvariantVerifierPass())
        end
        add!(mpm, VerifierPass())
        add!(mpm, RemoveNIPass())
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, LateLowerGCPass())
            if VERSION >= v"1.11.0-DEV.208"
                add!(fpm, FinalLowerGCPass())
            end
        end
        if VERSION < v"1.11.0-DEV.208"
            add!(mpm, FinalLowerGCPass())
        end
        # We need these two passes and the instcombine below
        # after GC lowering to let LLVM do some constant propagation on the tags.
        # and remove some unnecessary write barrier checks.        
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, GVNPass())
            add!(fpm, SCCPPass())
            # Remove dead use of ptls
            add!(fpm, DCEPass())
        end
        add!(mpm, LowerPTLSPass())
        # Clean up write barrier and ptls lowering
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, InstCombinePass())
            add!(fpm, JLInstSimplifyPass())
            aggressiveSimplifyCFGOptions =
                (forward_switch_cond=true,
                   switch_range_to_icmp=true,
                   switch_to_lookup=true,
                   hoist_common_insts=true)
            add!(fpm, SimplifyCFGPass(; aggressiveSimplifyCFGOptions...))
        end
    else
        add!(mpm, RemoveNIPass())
    end
end

function post_optimize!(mod::LLVM.Module, tm::LLVM.TargetMachine, machine::Bool = true)
    addr13NoAlias(mod)
    removeDeadArgs!(mod, tm)
    for f in collect(functions(mod))
        API.EnzymeFixupJuliaCallingConvention(f)
    end
    for f in collect(functions(mod))
        API.EnzymeFixupBatchedJuliaCallingConvention(f)
    end
    for g in collect(globals(mod))
        if startswith(LLVM.name(g), "ccall")
            hasuse = false
            for u in LLVM.uses(g)
                hasuse = true
                break
            end
            if !hasuse
                eraseInst(mod, g)
            end
        end
    end
    out_error = Ref{Cstring}()
    if LLVM.API.LLVMVerifyModule(mod, LLVM.API.LLVMReturnStatusAction, out_error) != 0
        throw(
            LLVM.LLVMException(
                "broken gc calling conv fix\n" *
                string(unsafe_string(out_error[])) *
                "\n" *
                string(mod),
            ),
        )
    end
    @dispose pb = NewPMPassBuilder() begin
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, ReinsertGCMarkerPass())
        register!(pb, SafeAtomicToRegularStorePass())
        add!(pb, NewPMAAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, NewPMModulePassManager()) do mpm
            addOptimizationPasses!(mpm)
            if machine
                # TODO enable validate_return_roots
                # validate_return_roots!(mod)
                addJuliaLegalizationPasses!(mpm, true)
                addMachinePasses!(mpm)
            end
        end
        run!(pb, mod, tm)
    end
    for f in functions(mod)
	if isempty(blocks(f))
		continue
	end
	if !has_fn_attr(f, StringAttribute("frame-pointer"))
		push!(function_attributes(f), StringAttribute("frame-pointer", "all"))
	end
    end
    # @safe_show "post_mod", mod
    # flush(stdout)
    # flush(stderr)
end
