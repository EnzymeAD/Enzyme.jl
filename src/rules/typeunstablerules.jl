function body_construct_augfwd(
    N,
    Width,
    primtypes,
    active_refs,
    primargs,
    batchshadowargs,
    tuple,
)
    shadow_rets = Vector{Expr}[]
    results = quote
        $(active_refs...)
    end
    @assert length(primtypes) == N
    @assert length(primargs) == N
    @assert length(batchshadowargs) == N
    for i = 1:N
        @assert length(batchshadowargs[i]) == Width
        shadow_rets_i = Expr[]
        aref = Symbol("active_ref_$i")
        for w = 1:Width
            sref = Symbol("sub_shadow_" * string(i) * "_" * string(w))
            push!(
                shadow_rets_i,
                quote
                    $sref = if $aref == AnyState
                        $(primargs[i])
                    else
                        if !ActivityTup[$i]
                            if ($aref == DupState || $aref == MixedState) &&
                               $(batchshadowargs[i][w]) === nothing
                                prim = $(primargs[i])
                                throw(
                                    "Error cannot store inactive but differentiable variable $prim into active tuple",
                                )
                            end
                        end
                        if $aref == DupState
                            $(batchshadowargs[i][w])
                        else
                            $(batchshadowargs[i][w])[]
                        end
                    end
                end,
            )
        end
        push!(shadow_rets, shadow_rets_i)
    end

    refs = Expr[]
    ref_syms = Symbol[]
    res_syms = Symbol[]
    for w = 1:Width
        sres = Symbol("result_$w")
        ref_res = Symbol("ref_result_$w")
        combined = Expr[]
        for i = 1:N
            push!(combined, shadow_rets[i][w])
        end
        if tuple
            results = quote
                $results
                $sres = ($(combined...),)
            end
        else
            results = quote
                $results
                $sres = $(Expr(:new, :NewType, combined...))
            end
        end
        push!(refs, quote
            $ref_res = Ref($sres)
        end)
        push!(ref_syms, ref_res)
        push!(res_syms, sres)
    end

    if Width == 1
        return quote
            $results
            if any_mixed
                $(refs...)
                $(ref_syms[1])
            else
                $(res_syms[1])
            end
        end
    else
        return quote
            $results
            if any_mixed
                $(refs...)
                ReturnType(($(ref_syms...),))
            else
                ReturnType(($(res_syms...),))
            end
        end
    end
end


function body_construct_rev(
    N,
    Width,
    primtypes,
    active_refs,
    primargs,
    batchshadowargs,
    tuple,
)
    outs = []
    for i = 1:N
        for w = 1:Width
            tsym = Symbol("tval_$w")
            expr = if tuple
                :($tsym[$i])
            else
                :(getfield($tsym, $i))
            end
            shad = batchshadowargs[i][w]
            out = :(
                if $(Symbol("active_ref_$i")) == MixedState ||
                   $(Symbol("active_ref_$i")) == ActiveState
                    if $shad isa Base.RefValue
                        $shad[] = recursive_add($shad[], $expr, identity, guaranteed_nonactive)
                    else
                        error(
                            "Enzyme Mutability Error: Cannot add one in place to immutable value " *
                            string($shad),
                        )
                    end
                end
            )
            push!(outs, out)
        end
    end

    tapes = Expr[:(tval_1 = tape[])]
    for w = 2:Width
        sym = Symbol("tval_$w")
        df = Symbol("df_$w")
        push!(tapes, :($sym = $df[]))
    end

    quote
        $(active_refs...)

        if any_mixed
            $(tapes...)
            $(outs...)
        end
        return nothing
    end
end


function body_runtime_tuple_rev(N, Width, primtypes, active_refs, primargs, batchshadowargs)
    body_construct_rev(N, Width, primtypes, active_refs, primargs, batchshadowargs, true)
end

function body_runtime_newstruct_rev(
    N,
    Width,
    primtypes,
    active_refs,
    primargs,
    batchshadowargs,
)
    body_construct_rev(N, Width, primtypes, active_refs, primargs, batchshadowargs, false)
end


function body_runtime_tuple_augfwd(
    N,
    Width,
    primtypes,
    active_refs,
    primargs,
    batchshadowargs,
)
    body_construct_augfwd(N, Width, primtypes, active_refs, primargs, batchshadowargs, true)
end

function func_runtime_tuple_augfwd(N, Width)
    primargs, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width; func = false, mixed_or_active = true)
    body = body_runtime_tuple_augfwd(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )

    quote
        function runtime_tuple_augfwd(
            activity::Type{Val{ActivityTup}},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            RT::Val{ReturnType},
            $(allargs...),
        )::ReturnType where {ActivityTup,MB,ReturnType,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_tuple_augfwd(
    activity::Type{Val{ActivityTup}},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    RT::Val{ReturnType},
    allargs...,
)::ReturnType where {ActivityTup,MB,Width,ReturnType}
    N = div(length(allargs), Width)
    primargs, _, primtypes, _, _, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width, :allargs; func = false, mixed_or_active = true)
    return body_runtime_tuple_augfwd(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )
end


function func_runtime_tuple_rev(N, Width)
    primargs, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width; mixed_or_active = true)
    body =
        body_runtime_tuple_rev(N, Width, primtypes, active_refs, primargs, batchshadowargs)

    quote
        function runtime_tuple_rev(
            activity::Type{Val{ActivityTup}},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            tape::TapeType,
            $(allargs...),
        ) where {ActivityTup,MB,TapeType,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_tuple_rev(
    activity::Type{Val{ActivityTup}},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    tape::TapeType,
    allargs...,
) where {ActivityTup,MB,Width,TapeType}
    N = div(length(allargs) - (Width - 1), Width)
    primargs, _, primtypes, _, _, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width, :allargs; mixed_or_active = true)
    return body_runtime_tuple_rev(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )
end


function body_runtime_newstruct_augfwd(
    N,
    Width,
    primtypes,
    active_refs,
    primargs,
    batchshadowargs,
)
    body_construct_augfwd(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
        false,
    )
end

function func_runtime_newstruct_augfwd(N, Width)
    primargs, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width; mixed_or_active = true)
    body = body_runtime_newstruct_augfwd(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )

    quote
        function runtime_newstruct_augfwd(
            activity::Type{Val{ActivityTup}},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            ::Type{NewType},
            RT::Val{ReturnType},
            $(allargs...),
        )::ReturnType where {ActivityTup,MB,ReturnType,NewType,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_newstruct_augfwd(
    activity::Type{Val{ActivityTup}},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    ::Type{NewType},
    RT::Val{ReturnType},
    allargs...,
)::ReturnType where {ActivityTup,MB,Width,ReturnType,NewType}
    N = div(length(allargs) + 2, Width + 1) - 1
    primargs, _, primtypes, _, _, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width, :allargs; mixed_or_active = true)
    return body_runtime_newstruct_augfwd(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )
end

function func_runtime_newstruct_rev(N, Width)
    primargs, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width; mixed_or_active = true)
    body = body_runtime_newstruct_rev(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )

    quote
        function runtime_newstruct_rev(
            activity::Type{Val{ActivityTup}},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            ::Type{NewStruct},
            tape::TapeType,
            $(allargs...),
        ) where {ActivityTup,MB,NewStruct,TapeType,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_newstruct_rev(
    activity::Type{Val{ActivityTup}},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    ::Type{NewStruct},
    tape::TapeType,
    allargs...,
) where {ActivityTup,MB,Width,NewStruct,TapeType}
    N = div(length(allargs) - (Width - 1), Width)
    primargs, _, primtypes, _, _, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width, :allargs; mixed_or_active = true)
    return body_runtime_newstruct_rev(
        N,
        Width,
        primtypes,
        active_refs,
        primargs,
        batchshadowargs,
    )
end

for (N, Width) in Iterators.product(0:30, 1:10)
    eval(func_runtime_newstruct_augfwd(N, Width))
    eval(func_runtime_newstruct_rev(N, Width))
    eval(func_runtime_tuple_augfwd(N, Width))
    eval(func_runtime_tuple_rev(N, Width))
end


# returns if legal and completed
function newstruct_common(fwd, run, offset, B, orig, gutils, normalR, shadowR)
    origops = collect(operands(orig))
    width = get_width(gutils)

    world = enzyme_extract_world(LLVM.parent(position(B)))

    @assert is_constant_value(gutils, origops[offset])
    icvs = [is_constant_value(gutils, v) for v in origops[offset+1:end-1]]
    abs_partial = [abs_typeof(v, true) for v in origops[offset+1:end-1]]
    abs = [abs_typeof(v) for v in origops[offset+1:end-1]]

    @assert length(icvs) == length(abs)
    for (icv, (found_partial, typ_partial, byref_partial), (found, typ, byref)) in
        zip(icvs, abs_partial, abs)
        # Constants not handled unless known inactive from type
        if icv
            if !found_partial
                return false
            end
            if !guaranteed_const_nongen(typ_partial, world)
                return false
            end
        end
        # if any active [e.g. ActiveState / MixedState] data could exist
        # err
        if !fwd
            if !found_partial
                return false
            end
            act =
                active_reg_inner(typ_partial, (), world, Val(false), Val(false), Val(true)) #=abstractismixed=#
            if act == MixedState || act == ActiveState
                return false
            end
        end
    end

    if !run
        return true
    end

    shadowsin = LLVM.Value[invert_pointer(gutils, o, B) for o in origops[offset:end-1]]
    if width == 1
        if offset != 1
            pushfirst!(shadowsin, origops[1])
        end
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), shadowsin)
        callconv!(shadowres, callconv(orig))
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            args = LLVM.Value[extract_value!(B, s, idx - 1) for s in shadowsin]
            if offset != 1
                pushfirst!(args, origops[1])
            end
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return true
end


function common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return true
    end

    if !newstruct_common(true, true, offset, B, orig, gutils, normalR, shadowR) #=run=#
        origops = collect(operands(orig))
        abs_partial = [abs_typeof(v, true) for v in origops[offset+1:end-1]]
        icvs = [is_constant_value(gutils, v) for v in origops[offset+1:end-1]]
        emit_error(
            B,
            orig,
            "Enzyme: Not yet implemented, mixed activity for jl_new_struct constants=" *
            string(icvs) *
            " " *
            string(orig) *
            " " *
            string(abs_partial) *
            " " *
            string([v for v in origops[offset+1:end-1]]),
        )
    end

    return false
end

function common_newstructv_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)::Bool
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return true
    end

    if !newstruct_common(false, true, offset, B, orig, gutils, normalR, shadowR) #=run=#
        normal =
            (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) :
            nothing
        shadow =
            (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) :
            nothing

        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)


        width = get_width(gutils)

        sret = generic_setup(
            orig,
            runtime_newstruct_augfwd,
            width == 1 ? Any : AnyArray(Int(width)),
            gutils,
            offset,
            B,
            false;
            firstconst = true,
            endcast = false,
            firstconst_after_tape = true,
            runtime_activity = false,
        ) #=start=#

        if width == 1
            shadow = sret
        else
            AT = LLVM.ArrayType(T_prjlvalue, Int(width))
            llty = convert(LLVMType, AnyArray(Int(width)))
            cal = sret
            cal = LLVM.addrspacecast!(B, cal, LLVM.PointerType(T_jlvalue, Derived))
            cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Derived))
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    cal,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i - 1)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)

        unsafe_store!(tapeR, sret.ref)
        return false
    end

    return false
end

function common_newstructv_rev(offset, B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return true
    end
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

    if !needsShadow
        return
    end

    if !newstruct_common(false, false, offset, B, orig, gutils, nothing, nothing) #=shadowR=#
        @assert tape !== C_NULL
        width = get_width(gutils)
        generic_setup(
            orig,
            runtime_newstruct_rev,
            Nothing,
            gutils,
            offset,
            B,
            true;
            firstconst = true,
            tape,
            firstconst_after_tape = true,
            runtime_activity = false,
        ) #=start=#
    end

    return nothing
end

function common_f_tuple_fwd(offset, B, orig, gutils, normalR, shadowR)
    common_newstructv_fwd(offset, B, orig, gutils, normalR, shadowR)
end

function common_f_tuple_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)::Bool
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if is_constant_value(gutils, orig) || needsShadowP[] == 0
        return true
    end

    if !newstruct_common(false, true, offset, B, orig, gutils, normalR, shadowR) #=run=#
        normal =
            (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) :
            nothing
        shadow =
            (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) :
            nothing

        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)


        width = get_width(gutils)

        sret = generic_setup(
            orig,
            runtime_tuple_augfwd,
            width == 1 ? Any : AnyArray(Int(width)),
            gutils,
            offset + 1,
            B,
            false;
            endcast = false,
            runtime_activity = false,
        ) #=start=#

        if width == 1
            shadow = sret
        else
            AT = LLVM.ArrayType(T_prjlvalue, Int(width))
            llty = convert(LLVMType, AnyArray(Int(width)))
            cal = sret
            cal = LLVM.addrspacecast!(B, cal, LLVM.PointerType(T_jlvalue, Derived))
            cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Derived))
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    cal,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i - 1)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)

        unsafe_store!(tapeR, sret.ref)

    end
    return false
end

function common_f_tuple_rev(offset, B, orig, gutils, tape)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

    if !needsShadow
        return
    end

    if is_constant_value(gutils, orig)
        return true
    end

    if !newstruct_common(false, false, offset, B, orig, gutils, nothing, nothing) #=shadowR=#
        @assert tape !== C_NULL
        width = get_width(gutils)
        tape2 = if width != 1
            res = LLVM.Value[]

            T_jlvalue = LLVM.StructType(LLVMType[])
            T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

            AT = LLVM.ArrayType(T_prjlvalue, Int(width))
            llty = convert(LLVMType, AnyArray(Int(width)))
            cal = tape
            cal = LLVM.addrspacecast!(B, cal, LLVM.PointerType(T_jlvalue, Derived))
            cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Derived))
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))

            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    cal,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i - 1)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                push!(res, ld)
            end
            res
        else
            tape
        end
        generic_setup(
            orig,
            runtime_tuple_rev,
            Nothing,
            gutils,
            offset + 1,
            B,
            true;
            tape = tape2,
            runtime_activity = false,
        ) #=start=#
    end
    return nothing
end


@register_fwd function f_tuple_fwd(B, orig, gutils, normalR, shadowR)
    common_f_tuple_fwd(1, B, orig, gutils, normalR, shadowR)
end

@register_aug function f_tuple_augfwd(B, orig, gutils, normalR, shadowR, tapeR)::Bool
    common_f_tuple_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

@register_rev function f_tuple_rev(B, orig, gutils, tape)
    common_f_tuple_rev(1, B, orig, gutils, tape)
    return nothing
end

@register_fwd function new_structv_fwd(B, orig, gutils, normalR, shadowR)
    common_newstructv_fwd(1, B, orig, gutils, normalR, shadowR)
end

@register_aug function new_structv_augfwd(B, orig, gutils, normalR, shadowR, tapeR)::Bool
    common_newstructv_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

@register_rev function new_structv_rev(B, orig, gutils, tape)
    common_apply_latest_rev(1, B, orig, gutils, tape)
    return nothing
end

@register_fwd function new_structt_fwd(B, orig, gutils, normalR, shadowR)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return true
    end

    origops = collect(operands(orig))
    width = get_width(gutils)

    @assert is_constant_value(gutils, origops[1])
    if is_constant_value(gutils, origops[2])
        emit_error(
            B,
            orig,
            "Enzyme: Not yet implemented, mixed activity for jl_new_struct_t" *
            string(orig),
        )
    end

    shadowsin = invert_pointer(gutils, origops[2], B)
    if width == 1
        vals = [new_from_original(gutils, origops[1]), shadowsin]
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), vals)
        callconv!(shadowres, callconv(orig))
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            vals = [
                new_from_original(gutils, origops[1]),
                extract_value!(B, shadowsin, idx - 1),
            ]
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), vals)
            callconv!(tmp, callconv(orig))
            shadowres = insert_value!(B, shadowres, tmp, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)
    return false
end

@register_aug function new_structt_augfwd(B, orig, gutils, normalR, shadowR, tapeR)::Bool
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        get_mode(gutils),
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return true
    end

    origops = collect(operands(orig))
    width = get_width(gutils)

    @assert is_constant_value(gutils, origops[1])
    if is_constant_value(gutils, origops[2])
        emit_error(
            B,
            orig,
            "Enzyme: Not yet implemented, mixed activity for jl_new_struct_t" *
            string(orig),
        )
    end

    shadowsin = invert_pointer(gutils, origops[2], B)
    if width == 1
        vals = [new_from_original(gutils, origops[1]), val_from_byref_if_mixed(B, gutils, origops[2], shadowsin)]
        shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), vals)
        callconv!(shadowres, callconv(orig))
        shadowres = byref_from_val_if_mixed(B, shadowres)
    else
        shadowres =
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
        for idx = 1:width
            vals = [
                new_from_original(gutils, origops[1]),
                val_from_byref_if_mixed(B, gutils, origops[2], extract_value!(B, shadowsin, idx - 1)),
            ]
            tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), vals)
            callconv!(tmp, callconv(orig))
            tmp = byref_from_val_if_mixed(B, tmp)
            shadowres = insert_value!(B, shadowres, tmp, idx - 1)
        end
    end
    unsafe_store!(shadowR, shadowres.ref)

    legal, TT, _ = abs_typeof(orig)
    if !legal
        unsafe_store!(tapeR, shadowres.ref)
    else
        @assert legal
        world = enzyme_extract_world(LLVM.parent(position(B)))
        act = active_reg_inner(TT, (), world)
        if act == ActiveState || act == MixedState
            unsafe_store!(tapeR, shadowres.ref)
        end
    end

    return false
end

@generated function recursive_tuple(::Val{num}, lhs, rhs) where num
    exprs = Expr[]
    for i in 1:num
        push!(exprs, quote
            recursive_add(lhs[$i], getfield(rhs, $i), identity, guaranteed_nonactive)
        end)
    end
    return quote
        Base.@_inline_meta
        ($(exprs...),)
    end
end

@generated function runtime_newstructt_rev(::Val{Width}, origres::Type{OR}, revres0::RR0, revarg0::RA0, args::Vararg{Any, N}) where {Width, OR, RR0, RA0, N}
    exprs = Expr[]
    for i in 1:Width
        dres = if i == 1
            :revres0
        else
            :(args[$(2*(i-2)+1)])
        end
        darg = if i == 1
            :revarg0
        else
            :(args[$(2*(i-2)+1+1)])
        end
        if OR == Nothing
            push!(exprs, quote
                @assert $dres isa Base.RefValue
                if $darg isa Base.RefValue
                    tmparg = $darg[]
                    tmpres = $dres[]
                    $darg[] = recursive_tuple(Val(length(tmparg)), tmparg, tmpres)
                else
                    error(
                        "Enzyme Mutability Error: Cannot accumulate in place to immutable value " *
                        string($darg),
                    )
                end
            end)
        elseif OR <: Base.RefValue
        else
            push!(exprs, quote
                if $dres isa Base.RefValue
                    if $darg isa Base.RefValue
                        tmparg = $darg[]
                        tmpres = $dres[]
                        $darg[] = recursive_tuple(Val(length(tmparg)), tmparg, tmpres)
                    else
                        error(
                            "Enzyme Mutability Error: Cannot accumulate in place to immutable value " *
                            string($darg),
                        )
                    end
                end
            end)
        end
    end
    expr = quote
        Base.@_inline_meta
        $(exprs...)
        return nothing
    end
    return expr
end

@register_rev function new_structt_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return true
    end
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

    if !needsShadow
        return
    end

    origops = collect(operands(orig))
    width = get_width(gutils)

    legal, TT, _ = abs_typeof(orig)
    torun = false
    if legal
        @assert legal
        world = enzyme_extract_world(LLVM.parent(position(B)))
        act = active_reg_inner(TT, (), world)
        torun = act == ActiveState || act == MixedState
    else
        torun = true
    end

    if torun
        vals = LLVM.Value[
            unsafe_to_llvm(B, runtime_newstructt_rev),
            unsafe_to_llvm(B, Val(Int(width))),
        ]

        if legal
            push!(vals, unsafe_to_llvm(B, Nothing))
        else
            push!(vals, lookup_value(gutils, new_from_original(gutils, origops[1]), B))
        end

        shadowsin = lookup_value(gutils, invert_pointer(gutils, origops[2], B), B)
        if width == 1
            push!(vals, tape)
            push!(vals, shadowsin)
        else
            for i in 1:width  
                push!(vals, extract_value!(B, tape, i - 1))
                push!(vals, extract_value!(B, shadowsin, i - 1))
            end
        end
        emit_apply_generic!(B, vals)
    end

    return nothing
end

function common_jl_getfield_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    origops = collect(operands(orig))[offset:end]
    width = get_width(gutils)
    if !is_constant_value(gutils, origops[2])
        shadowin = invert_pointer(gutils, origops[2], B)
        if width == 1
            args = LLVM.Value[new_from_original(gutils, origops[1]), shadowin]
            for a in origops[3:end-1]
                push!(args, new_from_original(gutils, a))
            end
            if offset != 1
                pushfirst!(args, first(operands(orig)))
            end
            shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(shadowres, callconv(orig))
        else
            shadowres =
                UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx = 1:width
                args = LLVM.Value[
                    new_from_original(gutils, origops[1]),
                    extract_value!(B, shadowin, idx - 1),
                ]
                for a in origops[3:end-1]
                    push!(args, new_from_original(gutils, a))
                end
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
                end
                tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
                callconv!(tmp, callconv(orig))
                shadowres = insert_value!(B, shadowres, tmp, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = new_from_original(gutils, orig)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(
                LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))),
            )
            for idx = 1:width
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end

@generated function ntuple_ref_zero(::Val{N}, ::Type{RT}, res) where {N,RT}
    expr = Vector{Expr}(undef, N)
    fill!(expr, :(Ref{$RT}(make_zero(res))))
    return quote
        Base.@_inline_meta
        ($(expr...),)
    end
end

@generated function ntuple_ref_lookup(::Val{N}, ::Type{RT}, dptrs, symname) where {N,RT}
    expr = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds expr[i] = quote
            begin
                dv = dptrs[$i]
                Ref{RT}(getfield(dv isa Base.RefValue ? dv[] : dv, symname))
            end
        end
    end
    return quote
        Base.@_inline_meta
        ($(expr...),)
    end
end

@generated function ntuple_lookup(::Val{N}, ptrs, symname) where {N}
    expr = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds expr[i] = quote
            begin
                dv = ptrs[$i]
                getfield(dv isa Base.RefValue ? dv[] : dv, symname)
            end
        end
    end
    return quote
        Base.@_inline_meta
        ($(expr...),)
    end
end

function rt_jl_getfield_aug(
    ::Val{NT},
    dptr::T,
    ::Type{Val{symname}},
    ::Val{isconst},
    dptrs::Vararg{T2,Nargs},
) where {NT,T,T2,Nargs,symname,isconst}
    res = if dptr isa Base.RefValue
        Base.getfield(dptr[], symname)
    else
        Base.getfield(dptr, symname)
    end
    RT = Core.Typeof(res)

    actreg = active_reg_nothrow(RT, Val(nothing))
    if actreg == ActiveState || (isconst && actreg == MixedState)
        if length(dptrs) == 0
            return Ref{RT}(make_zero(res))
        else
            return NT(ntuple_ref_zero(Val(1 + length(dptrs)), RT, res))
        end
    elseif actreg == MixedState
        if length(dptrs) == 0
            return Ref{RT}(res)
        else
            fval = NT((
                Ref{RT}(res),
                ntuple_ref_lookup(Val(length(dptrs)), RT, dptrs, symname)...,
            ))
            return fval
        end
    elseif isconst
        if length(dptrs) == 0
            return make_zero(res)
        else
            fval = NT((res, ntuple_ref_zero(Val(length(dptrs)), RT, res)...))
            return fval
        end
    else
        if length(dptrs) == 0
            return res
        else
            fval = NT((res, ntuple_lookup(Val(length(dptrs)), dptrs, symname)...))
            return fval
        end
    end
end

function idx_jl_getfield_aug(
    ::Val{NT},
    dptr::T,
    ::Type{Val{symname}},
    ::Val{isconst},
    dptrs::Vararg{T2,Nargs},
) where {NT,T,T2,Nargs,symname,isconst}
    res = if dptr isa Base.RefValue
        Base.getfield(dptr[], symname + 1)
    else
        Base.getfield(dptr, symname + 1)
    end
    RT = Core.Typeof(res)
    actreg = active_reg_nothrow(RT, Val(nothing))
    if actreg == ActiveState || (isconst && actreg == MixedState)
        if length(dptrs) == 0
            return Ref{RT}(make_zero(res))::Any
        else
            return NT(ntuple_ref_zero(Val(1 + length(dptrs)), RT, res))
        end
    elseif actreg == MixedState
        if length(dptrs) == 0
            return Ref{RT}(res)
        else
            fval = NT((
                Ref{RT}(res),
                ntuple_ref_lookup(Val(length(dptrs)), RT, dptrs, symname + 1)...,
            ))
            return fval
        end
    elseif isconst
        if length(dptrs) == 0
            return make_zero(res)::Any
        else
            fval = NT((res, ntuple_ref_zero(Val(length(dptrs)), RT, res)...))
            return fval
        end
    else
        if length(dptrs) == 0
            return res::Any
        else
            fval = NT((res, ntuple_lookup(Val(length(dptrs)), dptrs, symname + 1)...))
            return fval
        end
    end
end

@generated function recursive_field_add(
    ::Type{dRT},
    vload,
    ::Val{symname},
    dret,
) where {dRT,symname}
    N = fieldcount(dRT)
    exprs = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds exprs[i] = if fieldname(dRT, i) == symname
            :(recursive_add(getfield(vload, $i), dret, identity, guaranteed_nonactive))
        else
            :(getfield(vload, $i))
        end
    end
    res = Expr(:splatnew, dRT, :(($(exprs...)),))
    return quote
        Base.@_inline_meta
        $res
    end
end

function rt_jl_getfield_rev(
    dptr::T,
    dret,
    ::Type{Val{symname}},
    ::Val{isconst},
    dptrs::Vararg{T2,Nargs},
) where {T,T2,Nargs,symname,isconst}
    cur = if dptr isa Base.RefValue
        getfield(dptr[], symname)
    else
        getfield(dptr, symname)
    end

    RT = Core.Typeof(cur)

    actreg = active_reg_nothrow(RT, Val(nothing))
    if (actreg == ActiveState || actreg == MixedState) && !isconst
        if length(dptrs) == 0
            if dptr isa Base.RefValue
                vload = dptr[]
                dRT = Core.Typeof(vload)
                dptr[] = recursive_field_add(dRT, vload, Val(symname), dret[])
            else
                setfield!(
                    dptr,
                    symname,
                    recursive_add(cur, dret[], identity, guaranteed_nonactive),
                )
            end
        else
            if dptr isa Base.RefValue
                vload = dptr[]
                dRT = Core.Typeof(vload)
                dptr[] = recursive_field_add(dRT, vload, Val(symname), dret[1][])
            else
                setfield!(dptr, symname, recursive_add(cur, dret[1][]))
            end
            for i = 1:length(dptrs)
                if dptrs[i] isa Base.RefValue
                    vload = dptrs[i][]
                    dRT = Core.Typeof(vload)
                    dptrs[i][] = recursive_field_add(dRT, vload, Val(symname), dret[1+i][])
                else
                    curi = if dptr isa Base.RefValue
                        Base.getfield(dptrs[i][], symname)
                    else
                        Base.getfield(dptrs[i], symname)
                    end
                    setfield!(
                        dptrs[i],
                        symname,
                        recursive_add(curi, dret[1+i][], identity, guaranteed_nonactive),
                    )
                end
            end
        end
    end
    return nothing
end

@generated function recursive_index_add(
    ::Type{dRT},
    vload,
    ::Val{symname},
    dret,
) where {dRT,symname}
    N = fieldcount(dRT)
    exprs = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds exprs[i] = if i == symname
            :(recursive_add(getfield(vload, $i), dret, identity, guaranteed_nonactive))
        else
            :(getfield(vload, $i))
        end
    end
    res = Expr(:splatnew, dRT, :(($(exprs...)),))
    return quote
        Base.@_inline_meta
        $res
    end
end

function idx_jl_getfield_rev(
    dptr::T,
    dret,
    ::Type{Val{symname}},
    ::Val{isconst},
    dptrs::Vararg{T2,Nargs},
) where {T,T2,Nargs,symname,isconst}
    cur = if dptr isa Base.RefValue
        Base.getfield(dptr[], symname + 1)
    else
        Base.getfield(dptr, symname + 1)
    end

    RT = Core.Typeof(cur)

    actreg = active_reg_nothrow(RT, Val(nothing))
    if (actreg == ActiveState || actreg == MixedState) && !isconst
        if length(dptrs) == 0
            if dptr isa Base.RefValue
                vload = dptr[]
                dRT = Core.Typeof(vload)
                dptr[] = recursive_index_add(dRT, vload, Val(symname + 1), dret[])
            else
                setfield!(
                    dptr,
                    symname + 1,
                    recursive_add(cur, dret[], identity, guaranteed_nonactive),
                )
            end
        else
            if dptr isa Base.RefValue
                vload = dptr[]
                dRT = Core.Typeof(vload)
                dptr[] = recursive_index_add(dRT, vload, Val(symname + 1), dret[1][])
            else
                setfield!(
                    dptr,
                    symname + 1,
                    recursive_add(cur, dret[1][], identity, guaranteed_nonactive),
                )
            end
            for i = 1:length(dptrs)
                if dptrs[i] isa Base.RefValue
                    vload = dptrs[i][]
                    dRT = Core.Typeof(vload)
                    dptrs[i][] =
                        recursive_index_add(dRT, vload, Val(symname + 1), dret[1+i][])
                else
                    curi = if dptr isa Base.RefValue
                        Base.getfield(dptrs[i][], symname + 1)
                    else
                        Base.getfield(dptrs[i], symname + 1)
                    end
                    setfield!(
                        dptrs[i],
                        symname + 1,
                        recursive_add(curi, dret[1+i][], identity, guaranteed_nonactive),
                    )
                end
            end
        end
    end
    return nothing
end

function common_jl_getfield_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)::Bool
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    ops = collect(operands(orig))[offset:end]
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if !is_constant_value(gutils, ops[2])
        inp = invert_pointer(gutils, ops[2], B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w = 1:width
                push!(inps, extract_value!(B, inp, w - 1))
            end
        end
    else
        inps = [new_from_original(gutils, ops[2])]
    end

    AA = Val(AnyArray(Int(width)))
    vals = LLVM.Value[unsafe_to_llvm(B, AA)]
    push!(vals, inps[1])

    sym = new_from_original(gutils, ops[3])
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(B, Val(is_constant_value(gutils, ops[2]))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(B, rt_jl_getfield_aug))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if width == 1
        shadowres = cal
    else
        AT = LLVM.ArrayType(T_prjlvalue, Int(width))

        forgep = cal
        if !is_constant_value(gutils, ops[2])
            forgep = LLVM.addrspacecast!(B, forgep, LLVM.PointerType(T_jlvalue, Derived))
            forgep = LLVM.pointercast!(B, forgep, LLVM.PointerType(AT, Derived))
        end

        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for i = 1:width
            if !is_constant_value(gutils, ops[2])
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    forgep,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i - 1)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
            else
                ld = forgep
            end
            shadow = insert_value!(B, shadow, ld, i - 1)
        end
        shadowres = shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    unsafe_store!(tapeR, cal.ref)
    return false
end

function common_jl_getfield_rev(offset, B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return
    end
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )
    if needsShadowP[] == 0
        return
    end

    ops = collect(operands(orig))[offset:end]
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    if !is_constant_value(gutils, ops[2])
        inp = invert_pointer(gutils, ops[2], B)
        inp = lookup_value(gutils, inp, B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w = 1:width
                push!(inps, extract_value!(B, inp, w - 1))
            end
        end
    else
        inp = new_from_original(gutils, ops[2])
        inp = lookup_value(gutils, inp, B)
        inps = [inp]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    push!(vals, tape)

    sym = new_from_original(gutils, ops[3])
    sym = lookup_value(gutils, sym, B)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(B, Val(is_constant_value(gutils, ops[2]))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(B, rt_jl_getfield_rev))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)
    return nothing
end

@register_fwd function jl_nthfield_fwd(B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end
    origops = collect(operands(orig))
    width = get_width(gutils)
    if !is_constant_value(gutils, origops[1])
        shadowin = invert_pointer(gutils, origops[1], B)
        if width == 1
            args = LLVM.Value[
                shadowin
                new_from_original(gutils, origops[2])
            ]
            shadowres = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
            callconv!(shadowres, callconv(orig))
        else
            shadowres =
                UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig))))
            for idx = 1:width
                args = LLVM.Value[
                    extract_value!(B, shadowin, idx - 1)
                    new_from_original(gutils, origops[2])
                ]
                tmp = LLVM.call!(B, called_type(orig), LLVM.called_operand(orig), args)
                callconv!(tmp, callconv(orig))
                shadowres = insert_value!(B, shadowres, tmp, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    else
        normal = new_from_original(gutils, orig)
        if width == 1
            shadowres = normal
        else
            shadowres = UndefValue(
                LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(normal))),
            )
            for idx = 1:width
                shadowres = insert_value!(B, shadowres, normal, idx - 1)
            end
        end
        unsafe_store!(shadowR, shadowres.ref)
    end
    return false
end
@register_aug function jl_nthfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) || unsafe_load(shadowR) == C_NULL
        return true
    end

    ops = collect(operands(orig))
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if !is_constant_value(gutils, ops[1])
        inp = invert_pointer(gutils, ops[1], B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w = 1:width
                push!(inps, extract_value!(B, inp, w - 1))
            end
        end
    else
        inps = [new_from_original(gutils, ops[1])]
    end

    AA = Val(AnyArray(Int(width)))
    vals = LLVM.Value[unsafe_to_llvm(B, AA)]
    push!(vals, inps[1])

    sym = new_from_original(gutils, ops[2])
    sym = (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, sym)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(B, Val(is_constant_value(gutils, ops[1]))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(B, idx_jl_getfield_aug))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if width == 1
        shadowres = cal
    else
        AT = LLVM.ArrayType(T_prjlvalue, Int(width))
        forgep = cal
        if !is_constant_value(gutils, ops[1])
            forgep = LLVM.addrspacecast!(B, forgep, LLVM.PointerType(T_jlvalue, Derived))
            forgep = LLVM.pointercast!(B, forgep, LLVM.PointerType(AT, Derived))
        end

        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for i = 1:width
            if !is_constant_value(gutils, ops[1])
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    forgep,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i - 1)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
            else
                ld = forgep
            end
            shadow = insert_value!(B, shadow, ld, i - 1)
        end
        shadowres = shadow
    end

    unsafe_store!(shadowR, shadowres.ref)
    unsafe_store!(tapeR, cal.ref)
    return false
end
@register_rev function jl_nthfield_rev(B, orig, gutils, tape)
    if is_constant_value(gutils, orig)
        return
    end

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )
    needsPrimal = needsPrimalP[] != 0
    needsShadow = needsShadowP[] != 0

    if !needsShadow
        return
    end

    ops = collect(operands(orig))
    width = get_width(gutils)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    if !is_constant_value(gutils, ops[1])
        inp = invert_pointer(gutils, ops[1], B)
        inp = lookup_value(gutils, inp, B)
        if width == 1
            inps = [inp]
        else
            inps = LLVM.Value[]
            for w = 1:width
                push!(inps, extract_value!(B, inp, w - 1))
            end
        end
    else
        inp = new_from_original(gutils, ops[1])
        inp = lookup_value(gutils, inp, B)
        inps = [inp]
    end

    vals = LLVM.Value[]
    push!(vals, inps[1])

    push!(vals, tape)

    sym = new_from_original(gutils, ops[2])
    sym = lookup_value(gutils, sym, B)
    sym = (sizeof(Int) == sizeof(Int64) ? emit_box_int64! : emit_box_int32!)(B, sym)
    sym = emit_apply_type!(B, Base.Val, [sym])
    push!(vals, sym)

    push!(vals, unsafe_to_llvm(B, Val(is_constant_value(gutils, ops[1]))))

    for v in inps[2:end]
        push!(vals, v)
    end

    pushfirst!(vals, unsafe_to_llvm(B, idx_jl_getfield_rev))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)
    return nothing
end

@register_fwd function jl_getfield_fwd(B, orig, gutils, normalR, shadowR)
    common_jl_getfield_fwd(1, B, orig, gutils, normalR, shadowR)
end
@register_aug function jl_getfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_jl_getfield_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end
@register_rev function jl_getfield_rev(B, orig, gutils, tape)
    common_jl_getfield_rev(1, B, orig, gutils, tape)
end

function common_setfield_fwd(offset, B, orig, gutils, normalR, shadowR)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    origops = collect(operands(orig))[offset:end]
    if !is_constant_value(gutils, origops[4])
        width = get_width(gutils)

        shadowin = if !is_constant_value(gutils, origops[2])
            invert_pointer(gutils, origops[2], B)
        else
            new_from_original(gutils, origops[2])
        end

        shadowout = invert_pointer(gutils, origops[4], B)
        if width == 1
            args = LLVM.Value[
                new_from_original(gutils, origops[1])
                shadowin
                new_from_original(gutils, origops[3])
                shadowout
            ]
            valTys =
                API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal, API.VT_Shadow]
            if offset != 1
                pushfirst!(args, first(operands(orig)))
                pushfirst!(valTys, API.VT_Primal)
            end

            shadowres =
                call_samefunc_with_inverted_bundles!(B, gutils, orig, args, valTys, false) #=lookup=#
            callconv!(shadowres, callconv(orig))
        else
            for idx = 1:width
                args = LLVM.Value[
                    new_from_original(gutils, origops[1])
                    extract_value!(B, shadowin, idx - 1)
                    new_from_original(gutils, origops[3])
                    extract_value!(B, shadowout, idx - 1)
                ]
                valTys = API.CValueType[
                    API.VT_Primal,
                    API.VT_Shadow,
                    API.VT_Primal,
                    API.VT_Shadow,
                ]
                if offset != 1
                    pushfirst!(args, first(operands(orig)))
                    pushfirst!(valTys, API.VT_Primal)
                end

                tmp = call_samefunc_with_inverted_bundles!(
                    B,
                    gutils,
                    orig,
                    args,
                    valTys,
                    false,
                ) #=lookup=#

                callconv!(tmp, callconv(orig))
            end
        end
    end
    return false
end


function rt_jl_setfield_aug(dptr::T, idx, ::Val{isconst}, val, dval) where {T,isconst}
    RT = Core.Typeof(val)
    if active_reg(RT)
        setfield!(dptr, idx, make_zero(val))
    else
        setfield!(dptr, idx, isconst ? val : dval)
    end
end

function rt_jl_setfield_rev(dptr::T, idx, ::Val{isconst}, val, dval) where {T,isconst}
    RT = Core.Typeof(val)
    if active_reg(RT) && !isconst
        dval[] += getfield(dptr, idx)
        setfield!(dptr, idx, make_zero(val))
    end
end

function common_setfield_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)

    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end

    origops = collect(operands(orig))[offset:end]
    if !is_constant_value(gutils, origops[2])
        width = get_width(gutils)

        shadowstruct = invert_pointer(gutils, origops[2], B)

        shadowval = if !is_constant_value(gutils, origops[2])
            invert_pointer(gutils, origops[4], B)
        else
            nothing
        end

        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

        for idx = 1:width
            vals = LLVM.Value[
                (width == 1) ? shadowstruct : extract_value!(B, shadowstruct, idx - 1),
                new_from_original(gutils, origops[3]),
                unsafe_to_llvm(B, Val(is_constant_value(gutils, origops[4]))),
                new_from_original(gutils, origops[4]),
                is_constant_value(gutils, origops[4]) ? unsafe_to_llvm(B, nothing) :
                ((width == 1) ? shadowval : extract_value!(B, shadowval, idx - 1)),
            ]

            pushfirst!(vals, unsafe_to_llvm(B, rt_jl_setfield_aug))

            cal = emit_apply_generic!(B, vals)

            debug_from_orig!(gutils, cal, orig)
        end
    end

    return false
end

function common_setfield_rev(offset, B, orig, gutils, tape)
    origops = collect(operands(orig))[offset:end]
    if !is_constant_value(gutils, origops[2])
        width = get_width(gutils)

        shadowstruct = invert_pointer(gutils, origops[2], B)

        shadowval = if !is_constant_value(gutils, origops[2])
            invert_pointer(gutils, origops[4], B)
        else
            nothing
        end

        mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

        for idx = 1:width
            vals = LLVM.Value[
                lookup_value(
                    gutils,
                    (width == 1) ? shadowstruct : extract_value!(B, shadowstruct, idx - 1),
                    B,
                ),
                lookup_value(gutils, new_from_original(gutils, origops[3]), B),
                unsafe_to_llvm(B, Val(is_constant_value(gutils, origops[4]))),
                lookup_value(gutils, new_from_original(gutils, origops[4]), B),
                is_constant_value(gutils, origops[4]) ? unsafe_to_llvm(B, nothing) :
                lookup_value(
                    gutils,
                    ((width == 1) ? shadowval : extract_value!(B, shadowval, idx - 1)),
                    B,
                ),
            ]

            pushfirst!(vals, unsafe_to_llvm(B, rt_jl_setfield_rev))

            cal = emit_apply_generic!(B, vals)

            debug_from_orig!(gutils, cal, orig)
        end
    end
    return nothing
end


@register_fwd function setfield_fwd(B, orig, gutils, normalR, shadowR)
    common_setfield_fwd(1, B, orig, gutils, normalR, shadowR)
end

@register_aug function setfield_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_setfield_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

@register_rev function setfield_rev(B, orig, gutils, tape)
    common_setfield_rev(1, B, orig, gutils, tape)
end

function error_if_differentiable(::Type{T}) where {T}
    seen = ()
    areg = active_reg_inner(T, seen, nothing)
    if areg != AnyState
        throw(AssertionError("Found unhandled differentiable variable in jl_f_svec_ref $T"))
    end
    nothing
end

function common_f_svec_ref_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    origmi, origh, origkey = operands(orig)[offset:end-1]

    shadowh = invert_pointer(gutils, origh, B)

    newvals = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal]

    if offset != 1
        pushfirst!(newvals, API.VT_Primal)
    end

    mi = new_from_original(gutils, origmi)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    shadowres = if width == 1
        newops = LLVM.Value[mi, shadowh, new_from_original(gutils, origkey)]
        if offset != 1
            pushfirst!(newops, operands(orig)[1])
        end
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, false) #=lookup=#
        callconv!(cal, callconv(orig))

        if is_constant_value(gutils, origh)
            emit_apply_generic!(
                B,
                LLVM.Value[
                    unsafe_to_llvm(B, error_if_differentiable),
                    emit_jltypeof!(B, cal),
                ],
            )
        end
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j = 1:width
            newops = LLVM.Value[
                mi,
                extract_value!(B, shadowh, j - 1),
                new_from_original(gutils, origkey),
            ]
            if offset != 1
                pushfirst!(newops, operands(orig)[1])
            end
            cal = call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                newops,
                newvals,
                false,
            ) #=lookup=#
            callconv!(cal, callconv(orig))
            if is_constant_value(gutils, origh)
                emit_apply_generic!(
                    B,
                    LLVM.Value[
                        unsafe_to_llvm(B, error_if_differentiable),
                        emit_jltypeof!(B, cal),
                    ],
                )
            end
            shadow = insert_value!(B, shadow, cal, j - 1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)

    return false
end

function common_f_svec_ref_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig)
        return true
    end

    width = get_width(gutils)

    origmi, origh, origkey = operands(orig)[offset:end-1]

    shadowh = invert_pointer(gutils, origh, B)

    newvals = API.CValueType[API.VT_Primal, API.VT_Shadow, API.VT_Primal]

    if offset != 1
        pushfirst!(newvals, API.VT_Primal)
    end

    errfn = if is_constant_value(gutils, origh)
        error_if_differentiable
    else
        error_if_active
    end

    mi = new_from_original(gutils, origmi)

    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    shadowres = if width == 1
        newops = LLVM.Value[mi, shadowh, new_from_original(gutils, origkey)]
        if offset != 1
            pushfirst!(newops, operands(orig)[1])
        end
        cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, false) #=lookup=#
        callconv!(cal, callconv(orig))


        emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(B, errfn), emit_jltypeof!(B, cal)])
        cal
    else
        ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
        shadow = LLVM.UndefValue(ST)
        for j = 1:width
            newops = LLVM.Value[
                mi,
                extract_value!(B, shadowh, j - 1),
                new_from_original(gutils, origkey),
            ]
            if offset != 1
                pushfirst!(newops, operands(orig)[1])
            end
            cal = call_samefunc_with_inverted_bundles!(
                B,
                gutils,
                orig,
                newops,
                newvals,
                false,
            ) #=lookup=#
            callconv!(cal, callconv(orig))
            emit_apply_generic!(
                B,
                LLVM.Value[unsafe_to_llvm(B, errfn), emit_jltypeof!(B, cal)],
            )
            shadow = insert_value!(B, shadow, cal, j - 1)
        end
        shadow
    end

    unsafe_store!(shadowR, shadowres.ref)

    return false
end

function common_f_svec_ref_rev(offset, B, orig, gutils, tape)
    return nothing
end

function common_finalizer_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: unhandled forward for jl_f_finalizer")
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_finalizer_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: unhandled augmented forward for jl_f_finalizer")
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    if shadowR != C_NULL && normal !== nothing
        unsafe_store!(shadowR, normal.ref)
    end
    return false
end

function common_finalizer_rev(offset, B, orig, gutils, tape)
    return nothing
end

@register_fwd function f_svec_ref_fwd(B, orig, gutils, normalR, shadowR)
    common_f_svec_ref_fwd(1, B, orig, gutils, normalR, shadowR)
    return nothing
end

@register_aug function f_svec_ref_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_f_svec_ref_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
    return nothing
end

@register_rev function f_svec_ref_rev(B, orig, gutils, tape)
    common_f_svec_ref_rev(1, B, orig, gutils, tape)
    return nothing
end
