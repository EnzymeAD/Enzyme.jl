function registerEnzymeAndPassPipeline!(pb::NewPMPassBuilder)
    enzyme_callback = cglobal((:registerEnzymeAndPassPipeline, API.libEnzyme))
    LLVM.API.LLVMPassBuilderExtensionsPushRegistrationCallbacks(pb.exts, enzyme_callback)
end

LLVM.@function_pass "jl-inst-simplify" JLInstSimplifyPass

struct PipelineConfig
    Speedup::Cint
    Size::Cint
    lower_intrinsics::Cint
    dump_native::Cint
    external_use::Cint
    llvm_only::Cint
    always_inline::Cint
    enable_early_simplifications::Cint
    enable_early_optimizations::Cint
    enable_scalar_optimizations::Cint
    enable_loop_optimizations::Cint
    enable_vector_pipeline::Cint
    remove_ni::Cint
    cleanup::Cint
end

const RunAttributor = Ref(true)

function pipeline_options(;
    lower_intrinsics::Bool = true,
    dump_native::Bool = false,
    external_use::Bool = false,
    llvm_only::Bool = false,
    always_inline::Bool = true,
    enable_early_simplifications::Bool = true,
    enable_early_optimizations::Bool = true,
    enable_scalar_optimizations::Bool = true,
    enable_loop_optimizations::Bool = true,
    enable_vector_pipeline::Bool = true,
    remove_ni::Bool = true,
    cleanup::Bool = true,
    Size::Cint = 0,
    Speedup::Cint = 3,
)
    return PipelineConfig(
        Speedup,
        Size,
        lower_intrinsics,
        dump_native,
        external_use,
        llvm_only,
        always_inline,
        enable_early_simplifications,
        enable_early_optimizations,
        enable_scalar_optimizations,
        enable_loop_optimizations,
        enable_vector_pipeline,
        remove_ni,
        cleanup,
    )
end

function run_jl_pipeline(pm::ModulePassManager, tm::LLVM.TargetMachine; kwargs...)
    config = Ref(pipeline_options(; kwargs...))
    function jl_pipeline(m)
        @dispose pb = NewPMPassBuilder() begin
            add!(pb, NewPMModulePassManager()) do mpm
                @ccall jl_build_newpm_pipeline(
                    mpm.ref::Ptr{Cvoid},
                    pb.ref::Ptr{Cvoid},
                    config::Ptr{PipelineConfig},
                )::Cvoid
            end
            LLVM.run!(mpm, m, tm)
        end
        return true
    end
    add!(pm, ModulePass("JLPipeline", jl_pipeline))
end

@static if VERSION < v"1.11.0-DEV.428"
else
    barrier_noop!(pm) = nothing
end

@static if VERSION < v"1.11-"
    function gc_invariant_verifier_tm!(pm::ModulePassManager, tm::LLVM.TargetMachine, cond::Bool)
        gc_invariant_verifier!(pm, cond)
    end
else
    function gc_invariant_verifier_tm!(pm::ModulePassManager, tm::LLVM.TargetMachine, cond::Bool)
        function gc_invariant_verifier(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, GCInvariantVerifierPass(; strong = cond))
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("GCInvariantVerifier", gc_invariant_verifier))
    end
end

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
    function remove_ni_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        remove_ni!(pm)
    end
else
    function remove_ni_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function remove_ni(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, RemoveNIPass())
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("RemoveNI", remove_ni))
    end
end

@static if VERSION < v"1.11-"
    function julia_licm_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        julia_licm!(pm)
    end
else
    function julia_licm_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function julia_licm(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, NewPMLoopPassManager()) do lpm
                            add!(lpm, JuliaLICMPass())
                        end
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        # really looppass
        add!(pm, ModulePass("JuliaLICM", julia_licm))
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
    @static if true || VERSION < v"1.11-"
        lower_simdloop_tm!(pm, tm)
        licm!(pm)
        if LLVM.version() >= v"15"
            simple_loop_unswitch_legacy!(pm)
        else
            loop_unswitch!(pm)
        end
    else
        run_jl_pipeline(
            pm,
            tm;
            lower_intrinsics = false,
            dump_native = false,
            external_use = false,
            llvm_only = false,
            always_inline = false,
            enable_early_simplifications = false,
            enable_early_optimizations = false,
            enable_scalar_optimizations = false,
            enable_loop_optimizations = true,
            enable_vector_pipeline = false,
            remove_ni = false,
            cleanup = false,
        )
    end
end


function more_loop_optimizations_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
    @static if true || VERSION < v"1.11-"
        loop_rotate!(pm)
        # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
        loop_idiom!(pm)

        # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
        lower_simdloop_tm!(pm, tm) # Annotate loop marked with "loopinfo" as LLVM parallel loop
        licm!(pm)
        julia_licm_tm!(pm, tm)
        # Subsequent passes not stripping metadata from terminator
        instruction_combining!(pm) # TODO: createInstSimplifyLegacy
        jl_inst_simplify!(pm)

        ind_var_simplify!(pm)
        loop_deletion!(pm)
        loop_unroll!(pm) # TODO: in Julia createSimpleLoopUnroll
    else
        # LowerSIMDLoopPass
        # LoopRotatePass [opt >= 2]
        # LICMPass
        # JuliaLICMPass
        # SimpleLoopUnswitchPass
        # LICMPass
        # JuliaLICMPass
        # IRCEPass
        # LoopInstSimplifyPass
        #   - in ours this is instcombine with jlinstsimplify
        # LoopIdiomRecognizePass
        # IndVarSimplifyPass
        # LoopDeletionPass
        # LoopFullUnrollPass
        run_jl_pipeline(
            pm,
            tm;
            lower_intrinsics = false,
            dump_native = false,
            external_use = false,
            llvm_only = false,
            always_inline = false,
            enable_early_simplifications = false,
            enable_early_optimizations = false,
            enable_scalar_optimizations = false,
            enable_loop_optimizations = true,
            enable_vector_pipeline = false,
            remove_ni = false,
            cleanup = false,
        )
    end
end

@static if VERSION < v"1.11-"
    function demote_float16_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        demote_float16!(pm)
    end
else
    function demote_float16_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function demote_float16(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, DemoteFloat16Pass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("DemoteFloat16", demote_float16))
    end
end

@static if VERSION < v"1.11-"
    function lower_exc_handlers_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        lower_exc_handlers!(pm)
    end
else
    function lower_exc_handlers_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function lower_exc_handlers(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, LowerExcHandlersPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("LowerExcHandlers", lower_exc_handlers))
    end
end

@static if VERSION < v"1.11-"
    function lower_ptls_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine, dump_native::Bool)
        lower_ptls!(pm, dump_native)
    end
else
    function lower_ptls_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine, dump_native::Bool)
        function lower_ptls(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, LowerPTLSPass())
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("LowerPTLS", lower_ptls))
    end
end

@static if VERSION < v"1.11-"
    function combine_mul_add_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        combine_mul_add!(pm)
    end
else
    function combine_mul_add_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
@static if VERSION < v"1.12.0-DEV.1390"
        function combine_mul_add(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, CombineMulAddPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("CombineMulAdd", combine_mul_add))
end
    end
end

@static if VERSION < v"1.11-"
    function late_lower_gc_frame_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        late_lower_gc_frame!(pm)
    end
else
    function late_lower_gc_frame_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function late_lower_gc_frame(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, LateLowerGCPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("LateLowerGCFrame", late_lower_gc_frame))
    end
end

@static if VERSION < v"1.11-"
    function final_lower_gc_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        final_lower_gc!(pm)
    end
else
    function final_lower_gc_tm!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
        function final_lower_gc(mod::LLVM.Module)
            @dispose pb = NewPMPassBuilder() begin
                add!(pb, NewPMModulePassManager()) do mpm
                    add!(mpm, NewPMFunctionPassManager()) do fpm
                        add!(fpm, FinalLowerGCPass())
                    end
                end
                run!(pb, mod)
            end
            return true
        end
        add!(pm, ModulePass("FinalLowerGCFrame", final_lower_gc))
    end
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

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L603
function addTargetPasses!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine, trip::String)
    add_library_info!(pm, trip)
    add_transform_info!(pm, tm)
end

# https://github.com/JuliaLang/julia/blob/2eb5da0e25756c33d1845348836a0a92984861ac/src/aotcompile.cpp#L620
function addOptimizationPasses!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
    add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))

    constant_merge!(pm)

    propagate_julia_addrsp_tm!(pm, tm)
    scoped_no_alias_aa!(pm)
    type_based_alias_analysis!(pm)
    basic_alias_analysis!(pm)
    cfgsimplification!(pm)
    dce!(pm)
    scalar_repl_aggregates!(pm)

    # mem_cpy_opt!(pm)

    always_inliner!(pm) # Respect always_inline

    # Running `memcpyopt` between this and `sroa` seems to give `sroa` a hard time
    # merging the `alloca` for the unboxed data and the `alloca` created by the `alloc_opt`
    # pass.

    alloc_opt_tm!(pm, tm)
    # consider AggressiveInstCombinePass at optlevel > 2

    instruction_combining!(pm)
    jl_inst_simplify!(pm)
    cfgsimplification!(pm)
    scalar_repl_aggregates!(pm)
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy
    jl_inst_simplify!(pm)
    jump_threading!(pm)
    correlated_value_propagation!(pm)

    reassociate!(pm)

    early_cse!(pm)

    # Load forwarding above can expose allocations that aren't actually used
    # remove those before optimizing loops.
    alloc_opt_tm!(pm, tm)

    more_loop_optimizations_tm!(pm, tm)

    # Run our own SROA on heap objects before LLVM's
    alloc_opt_tm!(pm, tm)
    # Re-run SROA after loop-unrolling (useful for small loops that operate,
    # over the structure of an aggregate)
    scalar_repl_aggregates!(pm)
    instruction_combining!(pm) # TODO: createInstSimplifyLegacy
    jl_inst_simplify!(pm)

    gvn!(pm)
    mem_cpy_opt!(pm)
    sccp!(pm)

    # Run instcombine after redundancy elimination to exploit opportunities
    # opened up by them.
    # This needs to be InstCombine instead of InstSimplify to allow
    # loops over Union-typed arrays to vectorize.
    instruction_combining!(pm)
    jl_inst_simplify!(pm)
    jump_threading!(pm)
    dead_store_elimination!(pm)
    add!(pm, FunctionPass("SafeAtomicToRegularStore", safe_atomic_to_regular_store!))

    # More dead allocation (store) deletion before loop optimization
    # consider removing this:
    alloc_opt_tm!(pm, tm)

    # see if all of the constant folding has exposed more loops
    # to simplification and deletion
    # this helps significantly with cleaning up iteration
    cfgsimplification!(pm)
    loop_deletion!(pm)
    instruction_combining!(pm)
    jl_inst_simplify!(pm)
    loop_vectorize!(pm)
    # TODO: createLoopLoadEliminationPass
    cfgsimplification!(pm)
    slpvectorize!(pm)
    # might need this after LLVM 11:
    # TODO: createVectorCombinePass()

    aggressive_dce!(pm)
end

function addMachinePasses!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine)
    combine_mul_add_tm!(pm, tm)
    # TODO: createDivRemPairs[]

    demote_float16_tm!(pm, tm)
    gvn!(pm)
end

function addMachinePasses_newPM!(mpm::LLVM.NewPMPassManager)
    add!(mpm, NewPMFunctionPassManager()) do fpm
        if VERSION < v"1.12.0-DEV.1390"
            add!(fpm, CombineMulAddPass())
        end
        add!(fpm, DivRemPairsPass())
        add!(fpm, DemoteFloat16Pass())
        add!(fpm, GVNPass())              
    end
end

function addJuliaLegalizationPasses!(pm::LLVM.ModulePassManager, tm::LLVM.TargetMachine, lower_intrinsics::Bool = true)
    if lower_intrinsics
        # LowerPTLS removes an indirect call. As a result, it is likely to trigger
        # LLVM's devirtualization heuristics, which would result in the entire
        # pass pipeline being re-exectuted. Prevent this by inserting a barrier.
        barrier_noop!(pm)
        add!(pm, FunctionPass("ReinsertGCMarker", reinsert_gcmarker_pass!))
        lower_exc_handlers_tm!(pm, tm)
        # BUDE.jl demonstrates a bug here TODO
        gc_invariant_verifier_tm!(pm, tm, false)
        verifier!(pm)

        # Needed **before** LateLowerGCFrame on LLVM < 12
        # due to bug in `CreateAlignmentAssumption`.
        remove_ni_tm!(pm, tm)
        late_lower_gc_frame_tm!(pm, tm)
        final_lower_gc_tm!(pm, tm)
        # We need these two passes and the instcombine below
        # after GC lowering to let LLVM do some constant propagation on the tags.
        # and remove some unnecessary write barrier checks.
        gvn!(pm)
        sccp!(pm)
        # Remove dead use of ptls
        dce!(pm)
        lower_ptls_tm!(pm, tm, false) #=dump_native=#
        instruction_combining!(pm)
        jl_inst_simplify!(pm)
        # Clean up write barrier and ptls lowering
        cfgsimplification!(pm)
    else
        barrier_noop!(pm)
        remove_ni_tm!(pm, tm)
    end
end

ReinsertGCMarkerPass() = NewPMFunctionPass("reinsert_gcmarker", reinsert_gcmarker_pass!)

function addJuliaLegalizationPasses_newPM!(mpm::LLVM.NewPMPassManager, lower_intrinsics::Bool = true)
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
        add!(mpm, NewPMFunctionPassManager()) do fpm
            add!(fpm, GVNPass())
            add!(fpm, SCCPPass())
            add!(fpm, DCEPass())
        end
        add!(mpm, LowerPTLSPass())
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
    if LLVM.has_oldpm()
        LLVM.ModulePassManager() do pm
            addTargetPasses!(pm, tm, LLVM.triple(mod))
            addOptimizationPasses!(pm, tm)
            LLVM.run!(pm, mod)
        end
        if machine
            # TODO enable validate_return_roots
            # validate_return_roots!(mod)
            LLVM.ModulePassManager() do pm
                addJuliaLegalizationPasses!(pm, tm, true)
                addMachinePasses!(pm, tm)
                LLVM.run!(pm, mod)
            end
        end
    else
        @dispose pb = NewPMPassBuilder() begin
            registerEnzymeAndPassPipeline!(pb)
            register!(pb, ReinsertGCMarkerPass())
            add!(pb, NewPMModulePassManager()) do mpm
                # TODO(NewPM)
                # addTargetPasses!(mpm, tm, LLVM.triple(mod))
                # addOptimizationPasses!(mpm, tm)
            end
            if machine
                add!(pb, NewPMModulePassManager()) do mpm
                    addJuliaLegalizationPasses_newPM!(mpm, true)
                    addMachinePasses_newPM!(mpm)
                end
            end
            run!(pb, mod, tm)
        end
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
