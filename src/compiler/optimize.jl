function registerEnzymeAndPassPipeline!(pb::PassBuilder)
    enzyme_callback = cglobal((:registerEnzymeAndPassPipeline, API.libEnzyme))
    return register_callbacks!(pb, enzyme_callback)
end

LLVM.@function_pass "jl-inst-simplify" JLInstSimplifyPass
LLVM.@module_pass "preserve-nvvm" PreserveNVVMPass
LLVM.@module_pass "preserve-nvvm-end" PreserveNVVMEndPass
LLVM.@module_pass "simple-gvn" SimpleGVNPass
LLVM.@module_pass "enzyme-fixup-julia" FixupJuliaCallingConventionPass
LLVM.@module_pass "enzyme-fixup-julia-sret" FixupJuliaCallingConventionSRetPass
LLVM.@module_pass "enzyme-fixup-batched-julia" FixupBatchedJuliaCallingConventionPass

const RunAttributor = Ref(VERSION < v"1.12")

function enzyme_attributor_pass!(mod::LLVM.Module)
    ccall(
        (:RunAttributorOnModule, API.libEnzyme),
        Cvoid,
        (LLVM.API.LLVMModuleRef,),
        mod,
    )
    return true
end

EnzymeAttributorPass() = ModulePass("enzyme_attributor", enzyme_attributor_pass!)
ReinsertGCMarkerPass() = FunctionPass("reinsert_gcmarker", reinsert_gcmarker_pass!; required=true)
RestoreAllocaType() = FunctionPass("restore_alloca_type", restore_alloca_type!)
SafeAtomicToRegularStorePass() = FunctionPass("safe_atomic_to_regular_store", safe_atomic_to_regular_store!)
Addr13NoAliasPass() = ModulePass("addr13_noalias", addr13NoAlias)
RemoveAlwaysInlineRootsPass() = ModulePass("remove_alwaysinline_roots", remove_alwaysinline_roots!)

# `mark_loads_dereferenceable!` for the compilation `enzyme_ctx`.
struct MarkLoadsDereferenceable
    enzyme_ctx::Union{EnzymeContext, Nothing}
end
(pass::MarkLoadsDereferenceable)(fn::LLVM.Function) = mark_loads_dereferenceable!(fn, pass.enzyme_ctx)

MarkLoadsDereferenceablePass(enzyme_ctx::Union{EnzymeContext, Nothing}) =
    FunctionPass("enzyme_mark_loads_dereferenceable", MarkLoadsDereferenceable(enzyme_ctx))

# `enzyme_ctx` is the compilation `mod` belongs to, `nothing` outside of one.
function optimize!(mod::LLVM.Module, tm::Union{LLVM.TargetMachine, Nothing}, enzyme_ctx::Union{EnzymeContext, Nothing}, tti = nothing)
    @dispose pb = PassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, Addr13NoAliasPass())
        register!(pb, RestoreAllocaType())
        register!(pb, RemoveAlwaysInlineRootsPass())
        add!(pb, AAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, ModulePassManager()) do mpm
            add!(mpm, Addr13NoAliasPass())

            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, PropagateJuliaAddrspacesPass())
                add!(fpm, SimplifyCFGPass())
                add!(fpm, DCEPass())
            end
            add!(mpm, CPUFeaturesPass())
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, SROAPass())
                add!(fpm, MemCpyOptPass())
            end
            add!(mpm, RemoveAlwaysInlineRootsPass())
            add!(mpm, AlwaysInlinerPass())
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, AllocOptPass())
                add!(fpm, RestoreAllocaType())
            end
        end
        run!(pb, mod, tm)
    end

    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    @dispose pb = PassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        add!(pb, AAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, ModulePassManager()) do mpm
            add!(mpm, CPUFeaturesPass()) # why is this duplicated?
            add!(mpm, GlobalOptPass())
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, GVNPass())
            end
        end
        run!(pb, mod, tm)
    end

    function middle_optimize!(second_stage = false)
        # Infer which functions only write on paths that throw before the loop
        # passes below run: with EnzymeAD/Enzyme#3264, Enzyme also states this as
        # LLVM `memory` attributes, which lets LICM hoist loads (e.g. of an array's
        # `Memory` pointer) past calls to such functions instead of Enzyme having
        # to cache them per loop iteration.
        API.EnzymeDetectReadonlyOrThrow(mod)
        return @dispose pb = PassBuilder() begin
            if tti !== nothing
                LLVM.target_transform_info!(pb, tti)
            end
            registerEnzymeAndPassPipeline!(pb)
            register!(pb, RestoreAllocaType())
            register!(pb, MarkLoadsDereferenceablePass(enzyme_ctx))
            add!(pb, AAManager()) do aam
                add!(aam, ScopedNoAliasAA())
                add!(aam, TypeBasedAA())
                add!(aam, BasicAA())
            end
            add!(pb, ModulePassManager()) do mpm
                add!(mpm, CPUFeaturesPass()) # why is this duplicated?

                add!(mpm, FunctionPassManager()) do fpm
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
                    add!(fpm, RestoreAllocaType())

                    # Loaded pointers to heap objects of a known type are `dereferenceable`
                    # (see `mark_load_dereferenceable!`), so that LICM can hoist loads such as
                    # an array's `Memory` pointer out of loops.
                    add!(fpm, MarkLoadsDereferenceablePass(enzyme_ctx))
                    add!(fpm, LoopPassManager(use_memory_ssa = true)) do lpm
                        add!(lpm, LoopIdiomRecognizePass())
                        add!(lpm, LoopRotatePass())
                        add!(lpm, LowerSIMDLoopPass())
                        add!(lpm, LICMPass())
                        add!(lpm, JuliaLICMPass())
                        add!(lpm, SimpleLoopUnswitchPass())
                    end

                    add!(fpm, InstCombinePass())
                    add!(fpm, JLInstSimplifyPass())
                    add!(fpm, LoopPassManager()) do lpm
                        add!(lpm, IndVarSimplifyPass())
                        add!(lpm, LoopDeletionPass())
                    end
                    # todo peeling=false?
                    # Only fully unroll before AD. Runtime unrolling a loop with an
                    # unknown trip count gives it a strided body and a remainder loop,
                    # which the reverse pass inherits and the vectorizer after AD turns
                    # into gathers and scatters.
                    add!(fpm, LoopUnrollPass(opt_level = 2, partial = false, runtime = false)) # what opt level?
                    add!(fpm, AllocOptPass())
                    add!(fpm, RestoreAllocaType())
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
                    add!(fpm, RestoreAllocaType())
                    add!(fpm, SimplifyCFGPass())


                    add!(fpm, LoopPassManager()) do lpm
                        add!(lpm, LoopIdiomRecognizePass())
                        add!(lpm, LoopDeletionPass())
                    end
                    add!(fpm, JumpThreadingPass())
                    add!(fpm, CorrelatedValuePropagationPass())
                    if second_stage

                        add!(fpm, ADCEPass())
                        add!(fpm, InstCombinePass())
                        add!(fpm, JLInstSimplifyPass())

                        # GC passes
                        add!(fpm, GCInvariantVerifierPass(strong = false))
                        add!(fpm, SimplifyCFGPass())
                        add!(fpm, InstCombinePass())
                        add!(fpm, JLInstSimplifyPass())
                    end # second_stage
                end
            end
            run!(pb, mod, tm)
        end
    end # middle_optimize!

    run!(GCInvariantVerifierPass(strong = false), mod)

    middle_optimize!()

    run!(GCInvariantVerifierPass(strong = false), mod)

    middle_optimize!(true)

    run!(GCInvariantVerifierPass(strong = false), mod)

    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    @dispose pb = PassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        add!(pb, AAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, ModulePassManager()) do mpm
            add!(mpm, GlobalOptPass())
            add!(mpm, FunctionPassManager()) do fpm
                add!(fpm, GVNPass())
            end
        end
        run!(pb, mod, tm)
    end

    run!(GCInvariantVerifierPass(strong = false), mod)

    removeDeadArgs!(mod, tm, #=post_gc_fixup=# false)

    run!(GCInvariantVerifierPass(strong = false), mod)

    API.EnzymeDetectReadonlyOrThrow(mod)

    run!(GCInvariantVerifierPass(strong = false), mod)

    nodecayed_phis!(mod)

    return run!(GCInvariantVerifierPass(strong = false), mod)
end

# Julia's `aggressiveSimplifyCFGOptions`
const aggressiveSimplifyCFGOptions = (
    forward_switch_cond = true,
    switch_range_to_icmp = true,
    switch_to_lookup = true,
    hoist_common_insts = true,
)

function addOptimizationPasses!(mpm::LLVM.PassManager)
    add!(mpm, FunctionPassManager()) do fpm
        add!(fpm, ReinsertGCMarkerPass())
    end

    add!(mpm, ConstantMergePass())

    add!(mpm, FunctionPassManager()) do fpm
        add!(fpm, PropagateJuliaAddrspacesPass())

        add!(fpm, SimplifyCFGPass())
        add!(fpm, DCEPass())
        add!(fpm, SROAPass())
    end
    add!(mpm, RemoveAlwaysInlineRootsPass())
    add!(mpm, AlwaysInlinerPass())

    return add!(mpm, FunctionPassManager()) do fpm
        # Running `memcpyopt` between this and `sroa` seems to give `sroa` a hard time
        # merging the `alloca` for the unboxed data and the `alloca` created by the `alloc_opt`
        # pass.

        add!(fpm, AllocOptPass())
        add!(fpm, RestoreAllocaType())
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
        add!(fpm, RestoreAllocaType())

        add!(fpm, LoopPassManager(use_memory_ssa = true)) do lpm
            add!(lpm, LoopRotatePass())
            # moving IndVarSimplify here prevented removing the loop in perf_sumcartesian(10:-1:1)
            add!(lpm, LoopIdiomRecognizePass())

            # LoopRotate strips metadata from terminator, so run LowerSIMD afterwards
            add!(lpm, LowerSIMDLoopPass()) # Annotate loop marked with "loopinfo" as LLVM parallel loop
            add!(lpm, LICMPass())
            # Runtime-activity checks compare loop-invariant pointers inside loops;
            # hoist them so the loop bodies can vectorize.
            add!(lpm, SimpleLoopUnswitchPass(; nontrivial = true, trivial = true))
            add!(lpm, JuliaLICMPass())
        end
        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, IRCEPass())
        add!(fpm, LoopPassManager()) do lpm
            add!(lpm, LoopInstSimplifyPass())
            add!(lpm, IndVarSimplifyPass())
            add!(lpm, LoopDeletionPass())
            # As in Julia, only unroll loops whose trip count is known and small
            # here, so that no loop remains. Partial and runtime unrolling happen
            # after vectorization, as unrolling first leaves the vectorizer a
            # strided loop that it can only vectorize with gathers and scatters.
            add!(lpm, LoopFullUnrollPass())
        end

        # Run our own SROA on heap objects before LLVM's
        add!(fpm, AllocOptPass())
        add!(fpm, RestoreAllocaType())
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
        add!(fpm, RestoreAllocaType())

        # see if all of the constant folding has exposed more loops
        # to simplification and deletion
        # this helps significantly with cleaning up iteration
        add!(fpm, SimplifyCFGPass())
        add!(fpm, LoopPassManager()) do lpm
            add!(lpm, LoopDeletionPass())
        end
        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())

        # Vectorization, following Julia's `buildVectorPipeline`.
        add!(fpm, InjectTLIMappings())
        add!(fpm, LoopVectorizePass())
        add!(fpm, LoopLoadEliminationPass())
        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, SimplifyCFGPass(; aggressiveSimplifyCFGOptions...))
        add!(fpm, SLPVectorizerPass())
        add!(fpm, VectorCombinePass())
        add!(fpm, ADCEPass())
        # Unroll vectorized loops, as well as loops that failed to vectorize.
        add!(fpm, LoopUnrollPass(opt_level = 2))
    end
end

if VERSION < v"1.14.0-DEV.61"
    import Libdl
    const RUN_ASAN_PASS = any(contains("libclang_rt.asan"), Libdl.dllist())
end

function addMachinePasses!(mpm::LLVM.PassManager)
    add!(mpm, FunctionPassManager()) do fpm
        if VERSION < v"1.12.0-DEV.1390"
            add!(fpm, CombineMulAddPass())
        end
        add!(fpm, DivRemPairsPass())
        add!(fpm, AnnotationRemarksPass())
    end
    @static if VERSION >= v"1.14.0-DEV.61"
        if Base.JLOptions().target_sanitize_address != 0
            add!(mpm, AddressSanitizerPass())
        end
    else
        if RUN_ASAN_PASS
            add!(mpm, AddressSanitizerPass())
        end
    end
    return add!(mpm, FunctionPassManager()) do fpm
        add!(fpm, DemoteFloat16Pass())
        add!(fpm, GVNPass())
    end
end

function addJuliaLegalizationPasses!(mpm::LLVM.PassManager, lower_intrinsics::Bool = true)
    return if lower_intrinsics
        add!(mpm, FunctionPassManager()) do fpm
            add!(fpm, ReinsertGCMarkerPass())
            if VERSION < v"1.13.0-DEV.36"
                add!(fpm, LowerExcHandlersPass())
            end
            # TODO: strong=false?
            add!(fpm, GCInvariantVerifierPass())
        end
        add!(mpm, VerifierPass())
        add!(mpm, RemoveNIPass())
        add!(mpm, FunctionPassManager()) do fpm
            add!(fpm, LateLowerGCPass())
            if VERSION >= v"1.11.0-DEV.208"
                add!(fpm, FinalLowerGCPass())
            end
            if VERSION >= v"1.13.0-DEV.321"
                # after LateLowerGCPass so that all IPO is valid
                add!(fpm, ExpandAtomicModifyPass())
            end
        end
        if VERSION < v"1.11.0-DEV.208"
            add!(mpm, FinalLowerGCPass())
        end
        # We need these two passes and the instcombine below
        # after GC lowering to let LLVM do some constant propagation on the tags.
        # and remove some unnecessary write barrier checks.
        add!(mpm, FunctionPassManager()) do fpm
            add!(fpm, GVNPass())
            add!(fpm, SCCPPass())
            # Remove dead use of ptls
            add!(fpm, DCEPass())
        end
        add!(mpm, LowerPTLSPass())
        # Clean up write barrier and ptls lowering
        add!(mpm, FunctionPassManager()) do fpm
            add!(fpm, InstCombinePass())
            add!(fpm, JLInstSimplifyPass())
            add!(fpm, SimplifyCFGPass(; aggressiveSimplifyCFGOptions...))
        end
    else
        add!(mpm, RemoveNIPass())
    end
end

const DumpPreCallConv = Ref(false)
const DumpPostCallConv = Ref(false)

function fixup_callconv!(mod::LLVM.Module, tm::Union{LLVM.TargetMachine, Nothing}, tti = nothing)
    addr13NoAlias(mod)

    removeDeadArgs!(mod, tm, #=post_gc_fixup=# false)

    memcpy_sret_split!(mod)
    # if we did the move_sret_tofrom_roots, we will have loaded out of the sret, then stored into the rooted.
    # we should forward the value we actually stored [fixing the sret to therefore be writeonly and also ensuring
    # we can find the root store from the jlvaluet]
    # Instcombine breaks apart struct stores into individual components
    run!(InstCombinePass(), mod)
    # GVN actually forwards
    @dispose pb = PassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        registerEnzymeAndPassPipeline!(pb)
        add!(pb, SimpleGVNPass())
        run!(pb, mod, tm)
    end

    if DumpPreCallConv[]
        API.EnzymeDumpModuleRef(mod.ref)
    end

    @dispose pb = PassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        registerEnzymeAndPassPipeline!(pb)
        add!(pb, "enzyme-fixup-batched-julia")
        if VERSION < v"1.12"
            add!(pb, "enzyme-fixup-julia-sret")
        else
            add!(pb, "enzyme-fixup-julia")
        end
        run!(pb, mod, tm)
    end
    if DumpPostCallConv[]
        API.EnzymeDumpModuleRef(mod.ref)
    end
    for g in collect(mod.globals)
        if startswith(g.name, "ccall")
            hasuse = false
            for u in g.uses
                hasuse = true
                break
            end
            if !hasuse
                eraseInst(mod, g)
            end
        end
    end
    verifier_msg = verification_error(mod)
    if verifier_msg !== nothing
        throw(
            LLVM.LLVMException(
                "broken gc calling conv fix\n" *
                    verifier_msg *
                    "\n" *
                    string(mod),
            ),
        )
    end
    return
end

function post_optimize!(mod::LLVM.Module, tm::Union{LLVM.TargetMachine, Nothing}, machine::Bool = true; callconv::Bool = true, tti = nothing)
    define_ntuple_type!(mod)
    if callconv
        fixup_callconv!(mod, tm, tti)
    end

    for f in mod.functions
        if isempty(f.blocks)
            continue
        end
        # Before additional dead arg removal, get rid of the body of functions
        # that we will retain the original calling convention for.
        if startswith(f.name, "ejlstr\$") || startswith(f.name, "ejlptr\$")
            Base.empty!(f)
        end

        if haskey(f.function_attributes, "enzyme_preserve_primal")
            delete!(f.function_attributes, "enzyme_preserve_primal")
        end
    end

    removeDeadArgs!(mod, tm, #=post_gc_fixup=# true)

    @dispose pb = PassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, ReinsertGCMarkerPass())
        register!(pb, SafeAtomicToRegularStorePass())
        register!(pb, RestoreAllocaType())
        register!(pb, RemoveAlwaysInlineRootsPass())
        add!(pb, AAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, ModulePassManager()) do mpm
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
    for f in mod.functions
        if isempty(f.blocks)
            continue
        end
        if !haskey(f.function_attributes, "frame-pointer")
            push!(f.function_attributes, StringAttribute("frame-pointer", "all"))
        end
    end
    # @safe_show "post_mod", mod
    # flush(stdout)
    # flush(stderr)
    return
end
