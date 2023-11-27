
function setup_macro_wraps(forwardMode::Bool, N::Int, Width::Int, base=nothing)
    primargs = Union{Symbol,Expr}[]
    shadowargs = Union{Symbol,Expr}[]
    batchshadowargs = Vector{Union{Symbol,Expr}}[]
    primtypes = Union{Symbol,Expr}[]
    allargs = Expr[]
    typeargs = Symbol[]
    dfns = Union{Symbol,Expr}[:df]
    base_idx = 1
    for w in 2:Width
        if base === nothing
            shad = Symbol("df_$w")
            t = Symbol("DF__$w*")
            e = :($shad::$t)
            push!(allargs, e)
            push!(typeargs, t)
        else
            shad = :($base[$base_idx])
            base_idx += 1
        end
        push!(dfns, shad)
    end
    for i in 1:N
        if base === nothing
            prim = Symbol("primal_$i")
            t = Symbol("PT_$i")
            e = :($prim::$t)
            push!(allargs, e)
            push!(typeargs, t)
        else
            prim = :($base[$base_idx])
            base_idx += 1
        end
        t = :(Core.Typeof($prim))
        push!(primargs, prim)
        push!(primtypes, t)
        shadows = Union{Symbol,Expr}[]
        for w in 1:Width
            if base === nothing
                shad = Symbol("shadow_$(i)_$w")
                t = Symbol("ST_$(i)_$w")
                e = :($shad::$t)
                push!(allargs, e)
                push!(typeargs, t)
            else
                shad = :($base[$base_idx])
                base_idx += 1
            end
            push!(shadows, shad)
        end
        push!(batchshadowargs, shadows)
        if Width == 1
            push!(shadowargs, shadows[1])
        else
            push!(shadowargs, :(($(shadows...),)))
        end
    end
    @assert length(primargs) == N
    @assert length(primtypes) == N
    wrapped = Expr[]
    for i in 1:N
        expr = :(
                 if ActivityTup[$i+1] && !guaranteed_const($(primtypes[i]))
                   @assert $(primtypes[i]) !== DataType
                    if !$forwardMode && active_reg($(primtypes[i]))
                    Active($(primargs[i]))
                 else
                     $((Width == 1) ? :Duplicated : :BatchDuplicated)($(primargs[i]), $(shadowargs[i]))
                 end
             else
                 Const($(primargs[i]))
             end

            )
        push!(wrapped, expr)
    end
    return primargs, shadowargs, primtypes, allargs, typeargs, wrapped, batchshadowargs
end

function body_runtime_generic_fwd(N, Width, wrapped, primtypes)
    nnothing = ntuple(i->nothing, Val(Width+1))
    nres = ntuple(i->:(res[1]), Val(Width+1))
    ModifiedBetween = ntuple(i->false, Val(N+1))
    ElTypes = ntuple(i->:(eltype(Core.Typeof(args[$i]))), Val(N))
    Types = ntuple(i->:(Core.Typeof(args[$i])), Val(N))
    return quote
        args = ($(wrapped...),)

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{$(ElTypes...)}
        tt′ = Tuple{$(Types...)}
        rt = Core.Compiler.return_type(f, Tuple{$(ElTypes...)})
        annotation = guess_activity(rt, API.DEM_ForwardMode)

        if annotation <: DuplicatedNoNeed
            annotation = Duplicated{rt}
        end
        if $Width != 1
            if annotation <: Duplicated
                annotation = BatchDuplicated{rt, $Width}
            end
        end

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && guaranteed_const(FT)
            dupClosure = false
        end

        world = codegen_world_age(FT, tt)

        forward = thunk(Val(world), (dupClosure ? Duplicated : Const){FT}, annotation, tt′, Val(API.DEM_ForwardMode), width, #=ModifiedBetween=#Val($ModifiedBetween), #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

        res = forward(dupClosure ? Duplicated(f, df) : Const(f), args...)

        if length(res) == 0
            return ReturnType($nnothing)
        end
        if annotation <: Const
            return ReturnType(($(nres...),))
        end

        if $Width == 1
            return ReturnType((res[1], res[2]))
        else
            return ReturnType((res[1], res[2]...))
        end
    end
end

function func_runtime_generic_fwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _ = setup_macro_wraps(true, N, Width)
    body = body_runtime_generic_fwd(N, Width, wrapped, primtypes)

    quote
        function runtime_generic_fwd(activity::Type{Val{ActivityTup}}, width::Val{$Width}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_fwd(activity::Type{Val{ActivityTup}}, width::Val{Width}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width)-1
    _, _, primtypes, _, _, wrapped, _ = setup_macro_wraps(true, N, Width, :allargs)
    return body_runtime_generic_fwd(N, Width, wrapped, primtypes)
end

function body_runtime_generic_augfwd(N, Width, wrapped, primttypes)
    nnothing = ntuple(i->nothing, Val(Width+1))
    nres = ntuple(i->:(origRet), Val(Width+1))
    nzeros = ntuple(i->:(Ref(zero(resT))), Val(Width))
    nres3 = ntuple(i->:(res[3]), Val(Width))
    ElTypes = ntuple(i->:(eltype(Core.Typeof(args[$i]))), Val(N))
    Types = ntuple(i->:(Core.Typeof(args[$i])), Val(N))

    return quote
        args = ($(wrapped...),)

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt′ = Tuple{$(Types...)}
        rt = Core.Compiler.return_type(f, Tuple{$(ElTypes...)})
        annotation = guess_activity(rt, API.DEM_ReverseModePrimal)

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && guaranteed_const(FT)
            dupClosure = false
        end

        world = codegen_world_age(FT, Tuple{$(ElTypes...)})

        forward, adjoint = thunk(Val(world), (dupClosure ? Duplicated : Const){FT},
                                 annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

        internal_tape, origRet, initShadow = forward(dupClosure ? Duplicated(f, df) : Const(f), args...)
        resT = typeof(origRet)
        if annotation <: Const
            shadow_return = nothing
            tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
            return ReturnType(($(nres...), tape))
        elseif annotation <: Active
            if $Width == 1
                shadow_return = Ref(make_zero(origRet))
            else
                shadow_return = ($(nzeros...),)
            end
            tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
            if $Width == 1
                return ReturnType((origRet, shadow_return, tape))
            else
                return ReturnType((origRet, shadow_return..., tape))
            end
        end

        @assert annotation <: Duplicated || annotation <: DuplicatedNoNeed || annotation <: BatchDuplicated || annotation <: BatchDuplicatedNoNeed

        shadow_return = nothing
        tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
        if $Width == 1
            return ReturnType((origRet, initShadow, tape))
        else
            return ReturnType((origRet, initShadow..., tape))
        end
    end
end

function func_runtime_generic_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _ = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_augfwd(N, Width, wrapped, primtypes)

    quote
        function runtime_generic_augfwd(activity::Type{Val{ActivityTup}}, width::Val{$Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_augfwd(activity::Type{Val{ActivityTup}}, width::Val{Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, _ = setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_augfwd(N, Width, wrapped, primtypes)
end

function body_runtime_generic_rev(N, Width, wrapped, primttypes, shadowargs)
    outs = []
    for i in 1:N
        for w in 1:Width
            expr = if Width == 1
                :(tup[$i])
            else
                :(tup[$i][$w])
            end
            shad = shadowargs[i][w]
            out = :(if $expr === nothing
              elseif $shad isa Base.RefValue
                  $shad[] += $expr
                else
                  ref = shadow_ptr[$(i*(Width)+w)]
                  ref = reinterpret(Ptr{typeof($shad)}, ref)
                  unsafe_store!(ref, $shad+$expr)
                end
               )
            push!(outs, out)
        end
    end
    shadow_ret = nothing
    if Width == 1
        shadowret = :(tape.shadow_return[])
    else
        shadowret = []
        for w in 1:Width
            push!(shadowret, :(tape.shadow_return[$w][]))
        end
        shadowret = :(($(shadowret...),))
    end

    ElTypes = ntuple(i->:(eltype(Core.Typeof(args[$i]))), Val(N))
    Types = ntuple(i->:(Core.Typeof(args[$i])), Val(N))

    quote
        args = ($(wrapped...),)

        # TODO: Annotation of return value
        # tt0 = Tuple{$(primtypes...)}
        tt = Tuple{$(ElTypes...)}
        tt′ = Tuple{$(Types...)}
        rt = Core.Compiler.return_type(f, tt)
        annotation = guess_activity(rt, API.DEM_ReverseModePrimal)

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && guaranteed_const(FT)
            dupClosure = false
        end
        world = codegen_world_age(FT, tt)

        forward, adjoint = thunk(Val(world), (dupClosure ? Duplicated : Const){FT}, annotation, tt′, Val(API.DEM_ReverseModePrimal), width,
                                 ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)
        if tape.shadow_return !== nothing
            args = (args..., $shadowret)
        end

        tup = adjoint(dupClosure ? Duplicated(f, df) : Const(f), args..., tape.internal_tape)[1]

        $(outs...)
        return nothing
    end
end

function func_runtime_generic_rev(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, batchshadowargs = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_rev(N, Width, wrapped, primtypes, batchshadowargs)

    quote
        function runtime_generic_rev(activity::Type{Val{ActivityTup}}, width::Val{$Width}, ModifiedBetween::Val{MB}, tape::TapeType, shadow_ptr, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, TapeType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_rev(activity::Type{Val{ActivityTup}}, width::Val{Width}, ModifiedBetween::Val{MB}, tape::TapeType, shadow_ptr, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, TapeType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, batchshadowargs = setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_rev(N, Width, wrapped, primtypes, batchshadowargs)
end

# Create specializations
for (N, Width) in Iterators.product(0:30, 1:10)
    eval(func_runtime_generic_fwd(N, Width))
    eval(func_runtime_generic_augfwd(N, Width))
    eval(func_runtime_generic_rev(N, Width))
end

function generic_setup(orig, func, ReturnType, gutils, start, B::LLVM.IRBuilder,  lookup; sret=nothing, tape=nothing, firstconst=false)
    width = get_width(gutils)
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    ops = collect(operands(orig))[start+firstconst:end-1]

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    ActivityList = LLVM.Value[]

    to_preserve = LLVM.Value[]

    @assert length(ops) != 0
    fill_val = unsafe_to_llvm(nothing)

    vals = LLVM.Value[]

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if tape !== nothing
        NT = NTuple{length(ops)*Int(width), Ptr{Nothing}}
        SNT = convert(LLVMType, NT)
        shadow_ptr = emit_allocobj!(B, NT)
        shadow = addrspacecast!(B, shadow_ptr, LLVM.PointerType(T_jlvalue, Derived))
        shadow = bitcast!(B, shadow, LLVM.PointerType(SNT, Derived))
    end

    if firstconst
        val = new_from_original(gutils, operands(orig)[start])
        if lookup
            val = lookup_value(gutils, val, B)
        end
        push!(vals, val)
    end

    for (i, op) in enumerate(ops)
        val = new_from_original(gutils, op)
        if lookup
            val = lookup_value(gutils, val, B)
        end

        push!(vals, val)

        inverted = nothing
        active = !is_constant_value(gutils, op)
        
        if !active
            push!(ActivityList, unsafe_to_llvm(false))
        else
            inverted = invert_pointer(gutils, op, B)
            if lookup
                inverted = lookup_value(gutils, inverted, B)
            end
            if API.runtimeActivity()
                inv_0 = if width == 1
                    inverted
                else
                    extract_value!(B, inverted, 0)
                end
                push!(ActivityList, select!(B, icmp!(B, LLVM.API.LLVMIntNE, val, inv_0), unsafe_to_llvm(true), unsafe_to_llvm(false)))
            else
                push!(ActivityList, unsafe_to_llvm(true))
            end
        end

        for w in 1:width
            ev = fill_val
            if inverted !== nothing
                if width == 1
                    ev = inverted
                else
                    ev = extract_value!(B, inverted, w-1)
                end
                if tape !== nothing
                    push!(to_preserve, ev)
                end
            end

            push!(vals, ev)
            if tape !== nothing
                idx = LLVM.Value[LLVM.ConstantInt(0), LLVM.ConstantInt((i-1)*Int(width) + w-1)]
                ev = addrspacecast!(B, ev, is_opaque(value_type(ev)) ? LLVM.PointerType(Derived) : LLVM.PointerType(eltype(value_type(ev)), Derived))
                ev = emit_pointerfromobjref!(B, ev)
                ev = ptrtoint!(B, ev, convert(LLVMType, Int))
                LLVM.store!(B, ev, LLVM.inbounds_gep!(B, SNT, shadow, idx))
            end
        end
    end
    @assert length(ActivityList) == length(ops)

    if tape !== nothing
        pushfirst!(vals, shadow_ptr)
        pushfirst!(vals, tape)
    else
        pushfirst!(vals, unsafe_to_llvm(Val(ReturnType)))
    end

    if mode != API.DEM_ForwardMode
        uncacheable = get_uncacheable(gutils, orig)
        sret = false
        returnRoots = false

        ModifiedBetween = Bool[]

        for idx in 1:(length(ops)+firstconst)
            push!(ModifiedBetween, uncacheable[(start-1)+idx] != 0)
        end
        pushfirst!(vals, unsafe_to_llvm(Val((ModifiedBetween...,))))
    end

    pushfirst!(vals, unsafe_to_llvm(Val(Int(width))))
    etup0 = emit_tuple!(B, ActivityList)
    etup =  emit_apply_type!(B, Base.Val, [etup0])
    if isa(etup, LLVM.Instruction)
        @assert length(collect(LLVM.uses(etup0))) == 1
    end
    pushfirst!(vals, etup)

    @static if VERSION < v"1.7.0-" || true
    else
    mi = emit_methodinstance!(B, func, vals)
    end

    pushfirst!(vals, unsafe_to_llvm(func))

    @static if VERSION < v"1.7.0-" || true
    else
    pushfirst!(vals, mi)
    end

    @static if VERSION < v"1.7.0-" || true
    cal = emit_apply_generic!(B, vals)
    else
    cal = emit_invoke!(B, vals)
    end

    debug_from_orig!(gutils, cal, orig)

    if tape === nothing
        llty = convert(LLVMType, ReturnType)
        cal = LLVM.addrspacecast!(B, cal, LLVM.PointerType(T_jlvalue, Derived))
        cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Derived))
    end

    return cal
end

function common_generic_fwd(offset, B, orig, gutils, normalR, shadowR)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end
    return false
end

function generic_fwd(B, orig, gutils, normalR, shadowR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    common_generic_fwd(1, B, orig, gutils, normalR, shadowR)
end

function common_generic_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)
    return false
end

function generic_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function common_generic_rev(offset, B, orig, gutils, tape)::Cvoid
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)

        @assert tape !== C_NULL
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset, B, true; tape)
    end
    return nothing
end

function generic_rev(B, orig, gutils, tape)::Cvoid
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_apply_latest_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+1, B, false)

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

function common_apply_latest_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))
    # sret = generic_setup(orig, runtime_apply_latest_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, ctx, B, false)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, B, false)

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)
    return false
end

function common_apply_latest_rev(offset, B, orig, gutils, tape)::Cvoid
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, B, true; tape)
    end

    return nothing
end

function apply_latest_fwd(B, orig, gutils, normalR, shadowR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_fwd(1, B, orig, gutils, normalR, shadowR)
end

function apply_latest_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function apply_latest_rev(B, orig, gutils, tape)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_apply_iterate_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    emit_error(B, orig, "Enzyme: Not yet implemented, forward for jl_f__apply_iterate")
    if shadowR != C_NULL
        cal =  new_from_original(gutils, orig)
        width = get_width(gutils)
        if width == 1
            shadow = cal
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                shadow = insert_value!(B, shadow, cal, i-1)
                if i == 1
                    API.moveBefore(cal, shadow, B)
                end
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end
    return false
end

function error_if_active_iter(arg)
    # check if it could contain an active
    for v in arg
        seen = ()
        T = Core.Typeof(v)
        areg = active_reg_inner(T, seen, nothing, #=justActive=#Val(true))
        if areg == ActiveState
            throw(AssertionError("Found unhandled active variable in tuple splat, jl_apply_iterate $T"))
        end
    end
end

function common_apply_iterate_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    v, isiter = absint(operands(orig)[offset+1])
    v2, istup = absint(operands(orig)[offset+2])

    width = get_width(gutils)

    if v && v2 && isiter == Base.iterate && istup == Base.tuple && length(operands(orig)) >= offset+4
        origops = collect(operands(orig)[1:end-1])
        shadowins = [ invert_pointer(gutils, origops[i], B) for i in (offset+3):length(origops) ] 
        shadowres = if width == 1
            newops = LLVM.Value[]
            newvals = API.CValueType[]
            for (i, v) in enumerate(origops)
                if i >= offset + 3
                    shadowin2 = shadowins[i-offset-3+1]
                    emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active_iter), shadowin2])
                    push!(newops, shadowin2)
                    push!(newvals, API.VT_Shadow)
                else
                    push!(newops, new_from_original(gutils, origops[i]))
                    push!(newvals, API.VT_Primal)
                end
            end
            cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
            callconv!(cal, callconv(orig))
            cal
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for j in 1:width
                newops = LLVM.Value[]
                newvals = API.CValueType[]
                for (i, v) in enumerate(origops)
                    if i >= offset + 3
                        shadowin2 = extract_value!(B, shadowins[i-offset-3+1], j-1)
                        emit_apply_generic!(B, LLVM.Value[unsafe_to_llvm(error_if_active_iter), shadowin2])
                        push!(newops, shadowin2)
                        push!(newvals, API.VT_Shadow)
                    else
                        push!(newops, new_from_original(gutils, origops[i]))
                        push!(newvals, API.VT_Primal)
                    end
                end
                cal = call_samefunc_with_inverted_bundles!(B, gutils, orig, newops, newvals, #=lookup=#false)
                callconv!(cal, callconv(orig))
                shadow = insert_value!(B, shadow, cal, j-1)
            end
            shadow
        end

        unsafe_store!(shadowR, shadowres.ref)
        return false
    end

    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate "*string((v, v2, isiter, istup, length(operands(orig)), offset+4)))

    unsafe_store!(shadowR,UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))).ref)
    return false
end

function common_apply_iterate_rev(offset, B, orig, gutils, tape)
    return nothing
end

function apply_iterate_fwd(B, orig, gutils, normalR, shadowR)
    common_apply_iterate_fwd(1, B, orig, gutils, normalR, shadowR)
end

function apply_iterate_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_apply_iterate_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function apply_iterate_rev(B, orig, gutils, tape)
    common_apply_iterate_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_invoke_fwd(offset, B, orig, gutils, normalR, shadowR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end

    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+1, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    return false
end

function common_invoke_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    if is_constant_value(gutils, orig) && is_constant_inst(gutils, orig)
        return true
    end
    normal = (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow = (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing
    
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    conv = LLVM.callconv(orig)

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))

    if shadowR != C_NULL
        if width == 1
            gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i in 1:width
                gep = LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(i)])
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i-1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    end

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)

    return false
end

function common_invoke_rev(offset, B, orig, gutils, tape)
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, B, true; tape)
    end

    return nothing
end

function invoke_fwd(B, orig, gutils, normalR, shadowR)
    common_invoke_fwd(1, B, orig, gutils, normalR, shadowR)
end

function invoke_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_invoke_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function invoke_rev(B, orig, gutils, tape)
    common_invoke_rev(1, B, orig, gutils, tape)
    return nothing
end
