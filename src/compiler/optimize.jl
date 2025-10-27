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
Addr13NoAliasPass() = NewPMModulePass("addr13_noalias", addr13NoAlias)
RewriteGenericMemoryPass() = NewPMModulePass("rewrite_generic_memory", rewrite_generic_memory)

function optimize!(mod::LLVM.Module, tm::LLVM.TargetMachine)
    @dispose pb = NewPMPassBuilder() begin
        register!(pb, Addr13NoAliasPass())
        register!(pb, RewriteGenericMemoryPass())
        add!(pb, NewPMAAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, NewPMModulePassManager()) do mpm
            add!(mpm, Addr13NoAliasPass())
            add!(mpm, PropagateJuliaAddrspacesPass())

            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, SimplifyCFGPass())
                add!(fpm, DCEPass())
                add!(fpm, CPUFeaturesPass())
                add!(fpm, SROAPass())
                add!(fpm, MemCpyOptPass())
                add!(fpm, AlwaysInlinerPass())
                add!(fpm, AllocOptPass())
            end            

            add!(mpm, GlobalOptPass())
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, GVNPass())
            end

            add!(mpm, RewriteGenericMemoryPass())

            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, SimplifyCFGPass())
                add!(fpm, SROAPass())
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, JumpThreadingPass())
                add!(fpm, CorrelatedValuePropagationPass())
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, ReassociatePass())
                add!(fpm, EarlyCSEPass())
                add!(fpm, AllocOptPass())
                add!(fpm, NewPMLoopPassManager()) do lpm
                    add!(lpm, LoopIdiomRecognizePass())
                    add!(lpm, LoopRotatePass())
                    add!(lpm, LowerSIMDLoopPass())
                    add!(lpm, LICMPass())
                    add!(lpm, JuliaLICMPass())
                    add!(lpm, SimpleLoopUnswitchPass())
                end

                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, NewPMLoopPassManager()) do lpm
                    add!(lpm, IndVarSimplifyPass())
                    add!(lpm, LoopDeletionPass())
                end
                add!(fpm, LoopUnrollPass(opt_level=2))
                add!(fpm, AllocOptPass())
                add!(fpm, SROAPass())
                add!(fpm, GVNPass())

                # This InstCombine needs to be after GVN
                # Otherwise it will generate load chains in GPU code...
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, MemCpyOptPass())
                add!(fpm, SCCPPass())
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
                add!(fpm, JumpThreadingPass())
                add!(fpm, DSEPass())
                add!(fpm, AllocOptPass())
                add!(fpm, SimplifyCFGPass())


                add!(fpm, NewPMLoopPassManager()) do lpm
                    add!(lpm, LoopIdiomRecognizePass())
                    add!(lpm, LoopDeletionPass())
                end
                add!(fpm, JumpThreadingPass())
                add!(fpm, CorrelatedValuePropagationPass())

                add!(fpm, ADCEPass())
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())

                # GC passes
                add!(fpm, GCInvariantVerifierPass(strong=false))
                add!(fpm, SimplifyCFGPass())
                add!(fpm, InstCombinePass())
                add!(fpm, JLInstSimplifyPass())
            end

            add!(mpm, GlobalOptPass())
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, GVNPass())
            end
        end

        run!(pb, mod, tm)

        # TODO: Turn into passes?
        removeDeadArgs!(mod, tm)
        detect_writeonly!(mod)
        nodecayed_phis!(mod)
    end
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

        add!(fpm, NewPMLoopPassManager()) do lpm
            add!(lpm, LoopRotatePass())
            # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
            add!(lpm, LoopIdiomRecognizePass())

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
        add!(fpm, NewPMLoopPassManager()) do lpm
            add!(lpm, LoopDeletionPass())
        end
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
