import LLVM: refcheck
import GPUCompiler
LLVM.@checked struct GradientUtils
    ref::API.EnzymeGradientUtilsRef
end
Base.unsafe_convert(::Type{API.EnzymeGradientUtilsRef}, gutils::GradientUtils) = gutils.ref
LLVM.dispose(gutils::GradientUtils) = throw("Cannot free gutils")

function call_samefunc_with_inverted_bundles!(
    B::LLVM.IRBuilder,
    gutils::GradientUtils,
    orig::LLVM.CallInst,
    args::Vector{<:LLVM.Value},
    valTys::Vector{API.CValueType},
    lookup::Bool,
)
    @assert length(args) == length(valTys)
    return LLVM.Value(
        API.EnzymeGradientUtilsCallWithInvertedBundles(
            gutils,
            LLVM.called_operand(orig),
            LLVM.called_type(orig),
            args,
            length(args),
            orig,
            valTys,
            length(valTys),
            B,
            false,
        ),
    ) #=lookup=#
end

get_width(gutils::GradientUtils) = API.EnzymeGradientUtilsGetWidth(gutils)
get_mode(gutils::GradientUtils) = API.EnzymeGradientUtilsGetMode(gutils)
get_runtime_activity(gutils::GradientUtils) =
    API.EnzymeGradientUtilsGetRuntimeActivity(gutils)

get_strong_zero(gutils::GradientUtils) =
    API.EnzymeGradientUtilsGetStrongZero(gutils)

function get_shadow_type(gutils::GradientUtils, T::LLVM.LLVMType)
    w = get_width(gutils)
    if w == 1
        return T
    else
        return LLVM.ArrayType(T, Int(w))
    end
end
function get_uncacheable(gutils::GradientUtils, orig::LLVM.CallInst)
    uncacheable = Vector{UInt8}(undef, length(collect(LLVM.operands(orig))) - 1)
    if get_mode(gutils) == API.DEM_ForwardMode
        fill!(uncacheable, 0)
        return uncacheable
    end
    if API.EnzymeGradientUtilsGetUncacheableArgs(
        gutils,
        orig,
        uncacheable,
        length(uncacheable),
    ) != 1
        fill!(uncacheable, 1)
    end
    return uncacheable
end

erase_with_placeholder(
    gutils::GradientUtils,
    inst::LLVM.Instruction,
    orig::LLVM.Instruction,
    erase::Bool = true,
) = API.EnzymeGradientUtilsEraseWithPlaceholder(gutils, inst, orig, erase)
is_constant_value(gutils::GradientUtils, val::LLVM.Value) =
    API.EnzymeGradientUtilsIsConstantValue(gutils, val) != 0

is_constant_inst(gutils::GradientUtils, inst::LLVM.Instruction) =
    API.EnzymeGradientUtilsIsConstantInstruction(gutils, inst) != 0

new_from_original(gutils::GradientUtils, val::LLVM.Value) =
    LLVM.Value(API.EnzymeGradientUtilsNewFromOriginal(gutils, val))

lookup_value(gutils::GradientUtils, val::LLVM.Value, B::LLVM.IRBuilder) =
    LLVM.Value(API.EnzymeGradientUtilsLookup(gutils, val, B))

invert_pointer(gutils::GradientUtils, val::LLVM.Value, B::LLVM.IRBuilder) =
    LLVM.Value(API.EnzymeGradientUtilsInvertPointer(gutils, val, B))

function debug_from_orig!(
    gutils::GradientUtils,
    nval::LLVM.Instruction,
    oval::LLVM.Instruction,
)
    API.EnzymeGradientUtilsSetDebugLocFromOriginal(gutils, nval, oval)
    nothing
end

function add_reverse_block!(gutils::GradientUtils, block::LLVM.BasicBlock, name::String, forkCache::Bool = true, push::Bool = true)
    return LLVM.BasicBlock(API.EnzymeGradientUtilsAddReverseBlock(gutils, block, name, forkCache, push))
end

function set_reverse_block!(gutils::GradientUtils, block::LLVM.BasicBlock)
    return API.EnzymeGradientUtilsSetReverseBlock(gutils, block)
end

function get_or_insert_conditional_execute!(fn::LLVM.Function; force_run=false, need_result=true, preprocess=nothing, postprocess=nothing, postprocess_const=nothing, cmpidx::Int = 1)
    FT0 = LLVM.function_type(fn)
    ptys = LLVM.parameters(FT0)
    insert!(ptys, 1, ptys[cmpidx])

    void_rt = LLVM.return_type(FT0) == LLVM.VoidType() || !need_result
    extra_rt = !void_rt && postprocess_const === nothing
    if extra_rt
        insert!(ptys, 1, LLVM.return_type(FT0))
    end
    FT = LLVM.FunctionType(need_result ? LLVM.return_type(FT0) : LLVM.VoidType(), ptys; vararg=LLVM.isvararg(FT0))
    mod = LLVM.parent(fn)
    newname = "julia.enzyme.conditionally_execute."
    if !need_result
        newname = newname * "noresult."
    end
    if force_run
        newname = newname * "forcerun."
    end
    if preprocess !== nothing
        newname = newname * ".po_$(preprocess)"
    end
    if postprocess !== nothing
        newname = newname * ".po_$(postprocess)"
    end
    if postprocess_const !== nothing
        newname = newname * ".poc_$(postprocess_const)"
    end
    newname = newname * LLVM.name(fn)
    cfn, _ = get_function!(mod, newname, FT)
    if isempty(blocks(cfn))
        linkage!(cfn, LLVM.API.LLVMInternalLinkage)
        let builder = IRBuilder()
            entry = BasicBlock(cfn, "entry")
            good = BasicBlock(cfn, "good")
            bad = BasicBlock(cfn, "bad")
            position!(builder, entry)
            parms = collect(parameters(cfn))

            rparms = parms[(2+extra_rt):end]

            ppr = nothing

            if force_run
                if preprocess !== nothing
                    ppr = preprocess(builder, args)
                end
                res = call!(builder, FT0, fn, rparms)
                callconv!(res, callconv(fn))
            end

            cmp = icmp!(builder, LLVM.API.LLVMIntNE, parms[1 + extra_rt], parms[1 + cmpidx + extra_rt])

            br!(builder, cmp, good, bad)
            position!(builder, good)

            if !force_run
                if preprocess !== nothing
                    ppr = preprocess(builder, rparms)
                end
                res = call!(builder, FT0, fn, rparms)
                callconv!(res, callconv(fn))
            end
            if postprocess !== nothing
                postprocess(builder, res, rparms, ppr)
            end
            if void_rt
                ret!(builder)
            else
                ret!(builder, res)
            end

            position!(builder, bad)
            if postprocess_const !== nothing
                postprocess_const(builder, res, rparms, ppr)
                if void_rt
                    ret!(builder)
                else
                    ret!(builder, res)
                end
            elseif void_rt
                ret!(builder)
            else
                ret!(builder, parms[1])
            end
        end
        push!(function_attributes(fn), EnumAttribute("alwaysinline"))
    end
    return cfn
end

"""

Helper function for llvm-level rule generation. Will call the same function (and optional postprocessing),
if the argument at index `cmpidx` isn't active. This takes into account runtime activity as a reason
the value may not be active. 

If postprocess_const is set, the original function will always be called, but the postprocessing will be
conditionally gated as follows.

If the relevant input is active (and verified by runtime activity), 
    postprocess(B, result, args) will run as normal
Otherwise
    postprocess_const(B, result, args) will run
"""
function call_same_with_inverted_arg_if_active!(
    B::LLVM.IRBuilder,
    gutils::GradientUtils,
    orig::LLVM.CallInst,
    args::Vector{<:LLVM.Value},
    valTys::Vector{API.CValueType},
    lookup::Bool;
    preprocess=nothing,
    postprocess=nothing,
    postprocess_const = nothing,
    force_run = postprocess_const !== nothing,
    cmpidx::Int = 1,
    movebefore = true,
    need_result = true
)::Union{LLVM.Value, Nothing}
    @assert length(args) == length(valTys)

    origops = collect(operands(orig))
    if !force_run && is_constant_value(gutils, origops[cmpidx])
        if !need_result
            return nothing
        else
            return new_from_original(gutils, orig)
        end
    end

    if !get_runtime_activity(gutils) || (!force_run && is_constant_value(gutils, origops[cmpidx]))
        ppr = nothing
        if preprocess !== nothing
            ppr = preprocess(B, args)
        end
        res = call_samefunc_with_inverted_bundles!(
            B,
            gutils,
            orig,
            args,
            valTys,
            lookup
        )
        callconv!(res, callconv(orig))
        debug_from_orig!(gutils, res, orig)

        if postprocess_const === nothing
            if postprocess !== nothing
                postprocess(B, res, args, ppr)
            end
        elseif is_constant_value(gutils, origops[cmpidx])
            postprocess_const(B, res, args, ppr)
        elseif postprocess !== nothing
            postprocess(B, res, args, ppr)
        end

        if !need_result
            return nothing
        else
            return res
        end
    end

    if cmpidx isa Int
        valTys = copy(valTys)
        @assert valTys[cmpidx] == API.VT_Shadow
        valTys[cmpidx] = API.VT_Both
    end
    args = collect(LLVM.Value, args)
    insert!(args, 1, new_from_original(gutils, origops[cmpidx]))
    newval = nothing
    if value_type(orig) != LLVM.VoidType() && postprocess_const === nothing && need_result
        newval = new_from_original(gutils, orig)
        insert!(args, 1, newval)
    end
    prefn = LLVM.called_operand(orig)::LLVM.Function
    condfn = get_or_insert_conditional_execute!(prefn; force_run, preprocess, postprocess, postprocess_const, need_result, cmpidx)

    res = LLVM.Value(
        API.EnzymeGradientUtilsCallWithInvertedBundles(
            gutils,
            condfn,
            LLVM.function_type(condfn),
            args,
            length(args),
            orig,
            valTys,
            length(valTys),
            B,
            false,
        ),
    ) #=lookup=#
    callconv!(res, callconv(orig))

    debug_from_orig!(gutils, res, orig)
    if movebefore && newval !== nothing
        API.moveBefore(newval, res, B)
    end

    if !need_result
        return nothing
    else
        return res
    end
end


"""
Helper function for llvm-level rule generation. Will call call_same_with_inverted_arg_if_active with
corresponding extracted batches if width > 1, otherwise it will call it once.
"""
function batch_call_same_with_inverted_arg_if_active!(
    B::LLVM.IRBuilder,
    gutils::GradientUtils,
    orig::LLVM.CallInst,
    args::Vector{<:LLVM.Value},
    valTys::Vector{API.CValueType},
    lookup::Bool;
    need_result = true,
    kwargs...
)

    width = get_width(gutils)

    void_rt = value_type(orig) ==LLVM.VoidType()
    shadow = if !void_rt && need_result
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        LLVM.UndefValue(ST)::LLVM.Value
    end


    for idx in 1:width
        args2 = args
        if width > 1
            args2 = collect(LLVM.Value, args)
            for i in 1:length(valTys)
                if valTys[i] == API.VT_Shadow
                    args2[i] = extract_value!(B, args2[i], idx - 1)
                end
            end
        end
        res = call_same_with_inverted_arg_if_active!(B, gutils, orig, args2, valTys, lookup; need_result, kwargs..., movebefore=idx == 1)
        if shadow === nothing
            continue
        end
        if width == 1
            shadow = res
        else            
            shadow = insert_value!(B, shadow, res, idx - 1)
            if idx == 1
                norm = new_from_original(gutils, orig)
                if norm == res
                    API.moveBefore(norm, shadow, B)
                end
            end
        end
    end

    return shadow
end
