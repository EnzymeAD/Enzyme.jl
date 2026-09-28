function registerEnzymeAndPassPipeline!(pb::NewPMPassBuilder)
    enzyme_callback = cglobal((:registerEnzymeAndPassPipeline, API.libEnzyme))
    return LLVM.API.LLVMPassBuilderExtensionsPushRegistrationCallbacks(pb.exts, enzyme_callback)
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

EnzymeAttributorPass() = NewPMModulePass("enzyme_attributor", enzyme_attributor_pass!)
ReinsertGCMarkerPass() = NewPMFunctionPass("reinsert_gcmarker", reinsert_gcmarker_pass!)
RestoreAllocaType() = NewPMFunctionPass("restore_alloca_type", restore_alloca_type!)
SafeAtomicToRegularStorePass() = NewPMFunctionPass("safe_atomic_to_regular_store", safe_atomic_to_regular_store!)
Addr13NoAliasPass() = NewPMModulePass("addr13_noalias", addr13NoAlias)
RemoveAlwaysInlineRootsPass() = NewPMModulePass("remove_alwaysinline_roots", remove_alwaysinline_roots!)

function optimize!(mod::LLVM.Module, tm::Union{LLVM.TargetMachine, Nothing}, tti = nothing)
    @dispose pb = NewPMPassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, Addr13NoAliasPass())
        register!(pb, RestoreAllocaType())
        register!(pb, RemoveAlwaysInlineRootsPass())
        add!(pb, NewPMAAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, NewPMModulePassManager()) do mpm
            add!(mpm, Addr13NoAliasPass())

            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, PropagateJuliaAddrspacesPass())
                add!(fpm, SimplifyCFGPass())
                add!(fpm, DCEPass())
            end
            add!(mpm, CPUFeaturesPass())
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, SROAPass())
                add!(fpm, MemCpyOptPass())
            end
            add!(mpm, RemoveAlwaysInlineRootsPass())
            add!(mpm, AlwaysInlinerPass())
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, AllocOptPass())
                add!(fpm, RestoreAllocaType())
            end
        end
        run!(pb, mod, tm)
    end

    # Globalopt is separated as it can delete functions, which invalidates the Julia hardcoded pointers to
    # known functions
    @dispose pb = NewPMPassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        add!(pb, NewPMAAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, NewPMModulePassManager()) do mpm
            add!(mpm, CPUFeaturesPass()) # why is this duplicated?
            add!(mpm, GlobalOptPass())
            add!(mpm, NewPMFunctionPassManager()) do fpm
                add!(fpm, GVNPass())
            end
        end
        run!(pb, mod, tm)
    end

    function middle_optimize!(second_stage = false)
        return @dispose pb = NewPMPassBuilder() begin
            if tti !== nothing
                LLVM.target_transform_info!(pb, tti)
            end
            registerEnzymeAndPassPipeline!(pb)
            register!(pb, RestoreAllocaType())
            add!(pb, NewPMAAManager()) do aam
                add!(aam, ScopedNoAliasAA())
                add!(aam, TypeBasedAA())
                add!(aam, BasicAA())
            end
            add!(pb, NewPMModulePassManager()) do mpm
                add!(mpm, CPUFeaturesPass()) # why is this duplicated?

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
                    add!(fpm, RestoreAllocaType())

                    add!(fpm, NewPMLoopPassManager(use_memory_ssa = true)) do lpm
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
                    # todo peeling=false?
                    add!(fpm, LoopUnrollPass(opt_level = 2, partial = false)) # what opt level?
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


                    add!(fpm, NewPMLoopPassManager()) do lpm
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
    @dispose pb = NewPMPassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        add!(pb, NewPMAAManager()) do aam
            add!(aam, ScopedNoAliasAA())
            add!(aam, TypeBasedAA())
            add!(aam, BasicAA())
        end
        add!(pb, NewPMModulePassManager()) do mpm
            add!(mpm, GlobalOptPass())
            add!(mpm, NewPMFunctionPassManager()) do fpm
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
    add!(mpm, RemoveAlwaysInlineRootsPass())
    add!(mpm, AlwaysInlinerPass())

    return add!(mpm, NewPMFunctionPassManager()) do fpm
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

        add!(fpm, NewPMLoopPassManager(use_memory_ssa = true)) do lpm
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
        add!(fpm, NewPMLoopPassManager()) do lpm
            add!(lpm, IndVarSimplifyPass())
            add!(lpm, LoopDeletionPass())
        end
        add!(fpm, LoopUnrollPass(opt_level = 2))

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

if VERSION < v"1.14.0-DEV.61"
    import Libdl
    const RUN_ASAN_PASS = any(contains("libclang_rt.asan"), Libdl.dllist())
end

function addMachinePasses!(mpm::LLVM.NewPMPassManager)
    add!(mpm, NewPMFunctionPassManager()) do fpm
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
    return add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, DemoteFloat16Pass())
        add!(fpm, GVNPass())
    end
end

function addJuliaLegalizationPasses!(mpm::LLVM.NewPMPassManager, lower_intrinsics::Bool = true)
    return if lower_intrinsics
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
                (
                forward_switch_cond = true,
                switch_range_to_icmp = true,
                switch_to_lookup = true,
                hoist_common_insts = true,
            )
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
    @dispose pb = NewPMPassBuilder() begin
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

    @dispose pb = NewPMPassBuilder() begin
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
    return
end

function has_todense(mod::LLVM.Module)
    for f in functions(mod)
        if isempty(blocks(f)) && startswith(LLVM.name(f), "__enzyme_todense") && !isempty(LLVM.uses(f))
            return true
        end
    end
    return false
end

# The simplification the Enzyme clang plugin runs before lowering sparsity
# (it lowers at the start of the optimizer pipeline): inline, clean up, and
# canonicalize the loops, but do not unroll or vectorize them.
function addSparsityPrePasses!(mpm::LLVM.NewPMPassManager)
    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, ReinsertGCMarkerPass())
        add!(fpm, PropagateJuliaAddrspacesPass())
        add!(fpm, SimplifyCFGPass())
        add!(fpm, DCEPass())
        add!(fpm, SROAPass())
    end
    add!(mpm, RemoveAlwaysInlineRootsPass())
    add!(mpm, AlwaysInlinerPass())
    add!(mpm, NewPMFunctionPassManager()) do fpm
        add!(fpm, AllocOptPass())
        add!(fpm, RestoreAllocaType())
        add!(fpm, InstCombinePass())
        add!(fpm, JLInstSimplifyPass())
        add!(fpm, SimplifyCFGPass())
        add!(fpm, SROAPass())
        add!(fpm, EarlyCSEPass())
        add!(fpm, GVNPass())
        add!(fpm, NewPMLoopPassManager(use_memory_ssa = true)) do lpm
            add!(lpm, LoopRotatePass())
            add!(lpm, LICMPass())
        end
        add!(fpm, InstCombinePass())
        add!(fpm, NewPMLoopPassManager()) do lpm
            add!(lpm, IndVarSimplifyPass())
            add!(lpm, LoopDeletionPass())
        end
        add!(fpm, SimplifyCFGPass())
    end
end

strip_constexpr(v::LLVM.Value) = v isa LLVM.ConstantExpr ? strip_constexpr(operands(v)[1]) : v

# `Enzyme.sparse_accumulate(f, args...)` calls `__enzyme_sparse_accumulate_call`
# with a pointer to the compiled `f`. Call `f` directly, and mark it as the
# accumulation that the sparsity rewrite of a `todense` loop keeps.
function lower_sparse_accumulate!(mod::LLVM.Module)
    for marker in collect(functions(mod))
        startswith(LLVM.name(marker), "__enzyme_sparse_accumulate_call") || continue
        for u in collect(LLVM.uses(marker))
            ci = LLVM.user(u)::LLVM.CallInst
            ops = collect(LLVM.Value, arg_operands_view(ci))
            fn = strip_constexpr(ops[1])
            if !(fn isa LLVM.Function)
                error("Enzyme.sparse_accumulate: could not resolve the accumulation function in $(string(ci))")
            end
            attrs = function_attributes(fn)
            push!(attrs, StringAttribute("enzyme_sparse_accumulate"))
            delete!(attrs, EnumAttribute("alwaysinline", 0))
            push!(attrs, EnumAttribute("noinline", 0))
            @dispose b = IRBuilder() begin
                position!(b, ci)
                debuglocation!(b, ci)
                call!(b, LLVM.function_type(fn), fn, ops[2:end])
            end
            LLVM.erase!(ci)
        end
        isempty(LLVM.uses(marker)) && LLVM.erase!(marker)
    end
    return
end

# The loaders and storers of a `todense` pointer are inlined where the pointer
# is used, so that automatic sparsity can reason about their index conditions.
function inline_todense_callbacks!(mod::LLVM.Module)
    for f in functions(mod)
        (isempty(blocks(f)) && startswith(LLVM.name(f), "__enzyme_todense")) || continue
        for u in LLVM.uses(f)
            ci = LLVM.user(u)
            ci isa LLVM.CallInst || continue
            for op in operands(ci)[1:2]
                fn = strip_constexpr(op)
                fn isa LLVM.Function || continue
                attrs = function_attributes(fn)
                delete!(attrs, EnumAttribute("noinline", 0))
                push!(attrs, EnumAttribute("alwaysinline", 0))
            end
        end
    end
    return
end

"""
    lower_sparsification!(mod::LLVM.Module)

Replace every `__enzyme_todense(load, store, args...)` pointer by calls to its
`load`/`store` functions (see [`Enzyme.todense`](@ref)). With automatic
sparsity, the loops that contain `todense` calls are then rewritten to visit
only the indices at which an `enzyme_sparse_accumulate` call can be reached.

This must run once the derivatives that use the pointers are inlined into the
function that creates them. A loop that Enzyme cannot sparsify is kept dense,
with a warning.
"""
function lower_sparsification!(mod::LLVM.Module)
    lower_sparse_accumulate!(mod)
    inline_todense_callbacks!(mod)
    # Without accumulations there is nothing to sparsify.
    autosparsity = any(f -> has_fn_attr(f, StringAttribute("enzyme_sparse_accumulate")), functions(mod))
    ctx = LLVM.context(mod)
    prev = API.autosparsity()
    API.autosparsity!(autosparsity)
    try
        for f in collect(functions(mod))
            isempty(blocks(f)) && continue
            # Enzyme reports a loop it cannot sparsify as an error diagnostic,
            # which LLVM.jl records for the context.
            LLVM.prepare_diagnostic(ctx)
            API.EnzymeLowerSparsification(f, true)
            try
                LLVM.check_diagnostic(ctx)
            catch err
                err isa LLVM.LLVMException || rethrow()
                # The message dumps the function; its last line is the reason.
                reason = last(filter(!isempty, split(err.info, '\n')))
                # A function with `todense` pointers but no accumulation.
                occursin("Found no stores for sparsification", reason) && continue
                @warn "Enzyme could not sparsify a loop of $(LLVM.name(f)); it is evaluated densely" reason
            end
        end
    finally
        API.autosparsity!(prev)
    end
    for f in collect(functions(mod))
        if isempty(blocks(f)) && isempty(LLVM.uses(f)) &&
                any(p -> startswith(LLVM.name(f), p), ("__enzyme_todense", "__enzyme_post_sparse_todense", "__enzyme_sum", "__enzyme_product", "enzyme.sparse.inbounds"))
            LLVM.erase!(f)
        end
    end
    return
end

function post_optimize!(mod::LLVM.Module, tm::Union{LLVM.TargetMachine, Nothing}, machine::Bool = true; callconv::Bool = true, tti = nothing)
    define_ntuple_type!(mod)
    if callconv
        fixup_callconv!(mod, tm, tti)
    end

    for f in functions(mod)
        if isempty(blocks(f))
            continue
        end
        # Before additional dead arg removal, get rid of the body of functions
        # that we will retain the original calling convention for.
        if startswith(LLVM.name(f), "ejlstr\$") || startswith(LLVM.name(f), "ejlptr\$")
            Base.empty!(f)
        end

        if has_fn_attr(f, StringAttribute("enzyme_preserve_primal"))
            delete!(LLVM.function_attributes(f), StringAttribute("enzyme_preserve_primal"))
        end
    end

    removeDeadArgs!(mod, tm, #=post_gc_fixup=# true)

    if has_todense(mod)
        # Inline the derivatives into the functions that create the `todense`
        # pointers before lowering them.
        @dispose pb = NewPMPassBuilder() begin
            if tti !== nothing
                LLVM.target_transform_info!(pb, tti)
            end
            registerEnzymeAndPassPipeline!(pb)
            register!(pb, ReinsertGCMarkerPass())
            register!(pb, RestoreAllocaType())
            register!(pb, RemoveAlwaysInlineRootsPass())
            add!(pb, NewPMAAManager()) do aam
                add!(aam, ScopedNoAliasAA())
                add!(aam, TypeBasedAA())
                add!(aam, BasicAA())
            end
            add!(pb, NewPMModulePassManager()) do mpm
                addSparsityPrePasses!(mpm)
            end
            run!(pb, mod, tm)
        end
        lower_sparsification!(mod)
    end

    @dispose pb = NewPMPassBuilder() begin
        if tti !== nothing
            LLVM.target_transform_info!(pb, tti)
        end
        registerEnzymeAndPassPipeline!(pb)
        register!(pb, ReinsertGCMarkerPass())
        register!(pb, SafeAtomicToRegularStorePass())
        register!(pb, RestoreAllocaType())
        register!(pb, RemoveAlwaysInlineRootsPass())
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
    return
end
