function setup_macro_wraps(
    forwardMode::Bool,
    N::Int,
    Width::Int,
    base = nothing,
    iterate = false;
    func = true,
    mixed_or_active = false,
    reverse = false,
)
    primargs = Union{Symbol,Expr}[]
    shadowargs = Union{Symbol,Expr}[]
    batchshadowargs = Vector{Union{Symbol,Expr}}[]
    primtypes = Union{Symbol,Expr}[]
    allargs = Expr[]
    typeargs = Symbol[]
    dfns = Union{Symbol,Expr}[:df]
    base_idx = 1
    if func
        for w = 2:Width
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
    end
    for i = 1:N
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
        for w = 1:Width
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
    modbetween = Expr[:(MB[1])]
    active_refs = Expr[]
    for i = 1:N
        if iterate
            push!(modbetween, quote
                ntuple(Val(length($(primargs[i])))) do _
                    Base.@_inline_meta
                    MB[$i]
                end
            end)
        end
        aref = Symbol("active_ref_$i")
        push!(active_refs, quote
            $aref = active_reg_nothrow($(primtypes[i]), Val(nothing))
        end)
        expr = if iterate
            if forwardMode
                dupexpr = if Width == 1
                    quote
                        iterate_unwrap_fwd_dup($(primargs[i]), $(shadowargs[i]))
                    end
                else
                    quote
                        iterate_unwrap_fwd_batchdup(
                            Val($Width),
                            $(primargs[i]),
                            $(shadowargs[i]),
                        )
                    end
                end
                :(
                    if ActivityTup[$i+1] && !guaranteed_const($(primtypes[i]))
                        @assert $(primtypes[i]) !== DataType
                        $dupexpr
                    else
                        map(Const, $(primargs[i]))
                    end
                )
            else
                mixexpr = if Width == 1
                    quote
                        iterate_unwrap_augfwd_mix(
                            Val($reverse),
                            refs,
                            $(primargs[i]),
                            $(shadowargs[i]),
                        )
                    end
                else
                    quote
                        iterate_unwrap_augfwd_batchmix(
                            Val($reverse),
                            refs,
                            Val($Width),
                            $(primargs[i]),
                            $(shadowargs[i]),
                        )
                    end
                end
                dupexpr = if Width == 1
                    quote
                        iterate_unwrap_augfwd_dup(
                            Val($reverse),
                            refs,
                            $(primargs[i]),
                            $(shadowargs[i]),
                        )
                    end
                else
                    quote
                        iterate_unwrap_augfwd_batchdup(
                            Val($reverse),
                            refs,
                            Val($Width),
                            $(primargs[i]),
                            $(shadowargs[i]),
                        )
                    end
                end
                :(
                    if ActivityTup[$i+1] && !guaranteed_const($(primtypes[i]))
                        @assert $(primtypes[i]) !== DataType
                        if $aref == ActiveState
                            iterate_unwrap_augfwd_act($(primargs[i])...)
                        elseif $aref == MixedState
                            $mixexpr
                        else
                            $dupexpr
                        end
                    else
                        map(Const, $(primargs[i]))
                    end
                )
            end
        else
            if forwardMode
                quote
                    if ActivityTup[$i+1] && !guaranteed_const($(primtypes[i]))
                        $((Width == 1) ? :Duplicated : :BatchDuplicated)(
                            $(primargs[i]),
                            $(shadowargs[i]),
                        )
                    else
                        Const($(primargs[i]))
                    end
                end
            else
                quote
                    if ActivityTup[$i+1] && $aref != AnyState
                        @assert $(primtypes[i]) !== DataType
                        if $aref == ActiveState
                            Active($(primargs[i]))
                        elseif $aref == MixedState
                            $((Width == 1) ? :MixedDuplicated : :BatchMixedDuplicated)(
                                $(primargs[i]),
                                $(shadowargs[i]),
                            )
                        else
                            $((Width == 1) ? :Duplicated : :BatchDuplicated)(
                                $(primargs[i]),
                                $(shadowargs[i]),
                            )
                        end
                    else
                        Const($(primargs[i]))
                    end
                end
            end
        end
        push!(wrapped, expr)
    end

    any_mixed = quote
        false
    end
    for i = 1:N
        aref = Symbol("active_ref_$i")
        if mixed_or_active
            any_mixed = :($any_mixed || $aref == MixedState || $aref == ActiveState)
        else
            any_mixed = :($any_mixed || $aref == MixedState)
        end
    end
    push!(active_refs, quote
        any_mixed = $any_mixed
    end)
    return primargs,
    shadowargs,
    primtypes,
    allargs,
    typeargs,
    wrapped,
    batchshadowargs,
    modbetween,
    active_refs
end

function body_runtime_generic_fwd(N, Width, wrapped, primtypes)
    nnothing = Vector{Nothing}(undef, Width + 1)
    nres = Vector{Expr}(undef, Width + 1)
    fill!(nnothing, nothing)
    fill!(nres, :(res[1]))
    ModifiedBetween = Vector{Bool}(undef, N + 1)
    fill!(ModifiedBetween, false)
    ElTypes = Vector{Expr}(undef, N)
    Types = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds ElTypes[i] = :(eltype(Core.Typeof(args[$i])))
        @inbounds Types[i] = :(Core.Typeof(args[$i]))
    end

    retres = if Width == 1
        :(return ReturnType((res[2], res[1])))
    else
        :(return ReturnType((res[2], res[1]...)))
    end
    dup = if Width == 1
        :(Duplicated(f, df))
    else
        fargs = [:df]
        for i = 2:Width
            push!(fargs, Symbol("df_$i"))
        end
        :(BatchDuplicated(f, ($(fargs...),)))
    end
    dupty = if Width == 1
        :(Duplicated{FT})
    else
        :(BatchDuplicated{FT,$Width})
    end

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
                annotation = BatchDuplicated{rt,$Width}
            end
        end

        dupClosure = ActivityTup[1]
        FT = Core.Typeof(f)
        if dupClosure && guaranteed_const(FT)
            dupClosure = false
        end

        world = codegen_world_age(FT, tt)
        opt_mi = Val(world)
        forward = thunk(
            opt_mi,
            dupClosure ? $dupty : Const{FT},
            annotation,
            tt′,
            Val(API.DEM_ForwardMode),
            width,
            Val(($(ModifiedBetween...),)),
            Val(true),
            Val(false),
            FFIABI,
            Val(false),
            runtimeActivity,
        ) #=erriffuncwritten=#

        res = forward(dupClosure ? $dup : Const(f), args...)

        if length(res) == 0
            return ReturnType(($(nnothing...),))
        end
        if annotation <: Const
            return ReturnType(($(nres...),))
        end

        $retres
    end
end

function func_runtime_generic_fwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, _, _ = setup_macro_wraps(true, N, Width)
    body = body_runtime_generic_fwd(N, Width, wrapped, primtypes)

    quote
        function runtime_generic_fwd(
            activity::Type{Val{ActivityTup}},
            runtimeActivity::Val{RuntimeActivity},
            width::Val{$Width},
            RT::Val{ReturnType},
            f::F,
            df::DF,
            $(allargs...),
        ) where {ActivityTup,RuntimeActivity,ReturnType,F,DF,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_fwd(
    activity::Type{Val{ActivityTup}},
    runtimeActivity::Val{RuntimeActivity},
    width::Val{Width},
    RT::Val{ReturnType},
    f::F,
    df::DF,
    allargs...,
) where {ActivityTup,RuntimeActivity,Width,ReturnType,F,DF}
    N = div(length(allargs) + 2, Width + 1) - 1
    _, _, primtypes, _, _, wrapped, _, _, _ = setup_macro_wraps(true, N, Width, :allargs)
    return body_runtime_generic_fwd(N, Width, wrapped, primtypes)
end

function body_runtime_generic_augfwd(N, Width, wrapped, primttypes, active_refs)
    nres = Vector{Symbol}(undef, Width + 1)
    fill!(nres, :origRet)
    nzeros = Vector{Expr}(undef, Width)
    fill!(nzeros, :(Ref(make_zero(origRet))))

    ElTypes = Vector{Expr}(undef, N)
    MakeTypes = Vector{Expr}(undef, N)
    Types = Vector{Symbol}(undef, N)
    MixedTypes = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds ElTypes[i] = :(eltype($(Symbol("type_$i"))))
        @inbounds MakeTypes[i] = :($(Symbol("type_$i")) = Core.Typeof(args[$i]))
        @inbounds Types[i] = Symbol("type_$i")
        @inbounds MixedTypes[i] = :(
            $(Symbol("active_ref_$i") == MixedState) ? Ref($(Symbol("type_$i"))) :
            $(Symbol("type_$i"))
        )
    end

    ending = if Width == 1
        quote
            if annotation <: MixedDuplicated
                shadow_return = initShadow
                tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                    internal_tape,
                    shadow_return,
                )
                return ReturnType((origRet, shadow_return, tape))
            else
                shadow_return = nothing
                tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                    internal_tape,
                    shadow_return,
                )
                return ReturnType((origRet, initShadow, tape))
            end
        end
    else
        quote
            if annotation <: BatchMixedDuplicated
                shadow_return = (initShadow...,)
                tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                    internal_tape,
                    shadow_return,
                )
                return ReturnType((origRet, initShadow..., tape))
            else
                shadow_return = nothing
                tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                    internal_tape,
                    shadow_return,
                )
                return ReturnType((origRet, initShadow..., tape))
            end
        end
    end

    shadowretinit = if Width == 1
        :(Ref(make_zero(origRet)))
    else
        :(($(nzeros...),))
    end

    shadowretret = if Width == 1
        :(return ReturnType((origRet, shadow_return, tape)))
    else
        :(return ReturnType((origRet, shadow_return..., tape)))
    end

    dup = if Width == 1
        :(Duplicated(f, df))
    else
        fargs = [:df]
        for i = 2:Width
            push!(fargs, Symbol("df_$i"))
        end
        :(BatchDuplicated(f, ($(fargs...),)))
    end
    dupty = if Width == 1
        :(Duplicated{FT})
    else
        :(BatchDuplicated{FT,$Width})
    end

    return quote
        $(active_refs...)
        args = ($(wrapped...),)
        $(MakeTypes...)

        FT = Core.Typeof(f)
        dupClosure0 = if ActivityTup[1]
            !guaranteed_const(FT)
        else
            false
        end

        tt = Tuple{$(ElTypes...)}
        rt = Core.Compiler.return_type(f, tt)
        annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

        annotationA = if $Width != 1 && annotation0 <: Duplicated
            BatchDuplicated{rt,$Width}
        elseif $Width != 1 && annotation0 <: MixedDuplicated
            BatchMixedDuplicated{rt,$Width}
        else
            annotation0
        end

        internal_tape, origRet, initShadow, annotation = if f isa typeof(Core.getglobal)
            gv = Core.getglobal(args[1].val, args[2].val)
            @assert sizeof(gv) == 0
            (nothing, gv, nothing, Const)
        else
            world = codegen_world_age(FT, tt)

            opt_mi = Val(world)
            forward, adjoint = thunk(
                opt_mi,
                dupClosure0 ? $dupty : Const{FT},
                annotationA,
                Tuple{$(Types...)},
                Val(API.DEM_ReverseModePrimal),
                width,
                ModifiedBetween,
                Val(true),
                Val(false),
                FFIABI,
                Val(false),
                runtimeActivity,
            ) #=erriffuncwritten=#

            (forward(dupClosure0 ? $dup : Const(f), args...)..., annotationA)
        end

        resT = typeof(origRet)
        if annotation <: Const
            shadow_return = nothing
            tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                internal_tape,
                shadow_return,
            )
            return ReturnType(($(nres...), tape))
        elseif annotation <: Active
            shadow_return = $shadowretinit
            tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                internal_tape,
                shadow_return,
            )
            $shadowretret
        end

        $ending
    end
end

function func_runtime_generic_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, _, active_refs =
        setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_augfwd(N, Width, wrapped, primtypes, active_refs)

    quote
        function runtime_generic_augfwd(
            activity::Type{Val{ActivityTup}},
            runtimeActivity::Val{RuntimeActivity},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            RT::Val{ReturnType},
            f::F,
            df::DF,
            $(allargs...),
        )::ReturnType where {ActivityTup,MB,ReturnType,RuntimeActivity,F,DF,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_augfwd(
    activity::Type{Val{ActivityTup}},
    runtimeActivity::Val{RuntimeActivity},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    RT::Val{ReturnType},
    f::F,
    df::DF,
    allargs...,
)::ReturnType where {ActivityTup,MB,RuntimeActivity,Width,ReturnType,F,DF}
    N = div(length(allargs) + 2, Width + 1) - 1
    _, _, primtypes, _, _, wrapped, _, _, active_refs =
        setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_augfwd(N, Width, wrapped, primtypes, active_refs)
end

function nonzero_active_data(x::T) where {T<:AbstractFloat}
    return x != zero(T)
end

nonzero_active_data(::T) where {T<:Base.RefValue} = false
nonzero_active_data(::T) where {T<:Array} = false
nonzero_active_data(::T) where {T<:Ptr} = false

function nonzero_active_data(x::T) where {T}
    if guaranteed_const(T)
        return false
    end
    if ismutable(x)
        return false
    end

    for f in fieldnames(T)
        xi = getfield(x, f)
        if nonzero_active_data(xi)
            return true
        end
    end
    return false
end

function body_runtime_generic_rev(N, Width, wrapped, primttypes, shadowargs, active_refs)
    outs = []
    for i = 1:N
        for w = 1:Width
            expr = if Width == 1
                :(tup[$i])
            else
                :(tup[$i][$w])
            end
            shad = shadowargs[i][w]
            out = quote
                if tup[$i] === nothing
                elseif $shad isa Base.RefValue
                    $shad[] = recursive_add($shad[], $expr)
                else
                    error(
                        "Enzyme Mutability Error: Cannot add one in place to immutable value " *
                        string($shad) *
                        " tup[i]=" *
                        string(tup[$i]) *
                        " i=" *
                        string($i) *
                        " w=" *
                        string($w) *
                        " tup=" *
                        string(tup),
                    )
                end
            end
            push!(outs, out)
        end
    end
    shadow_ret = nothing
    if Width == 1
        shadowret = :(tape.shadow_return[])
    else
        shadowret = []
        for w = 1:Width
            push!(shadowret, :(tape.shadow_return[$w][]))
        end
        shadowret = :(($(shadowret...),))
    end

    ElTypes = Vector{Expr}(undef, N)
    MakeTypes = Vector{Expr}(undef, N)
    Types = Vector{Symbol}(undef, N)
    for i = 1:N
        @inbounds ElTypes[i] = :(eltype($(Symbol("type_$i"))))
        @inbounds MakeTypes[i] = :($(Symbol("type_$i")) = Core.Typeof(args[$i]))
        @inbounds Types[i] = Symbol("type_$i")
    end

    dup = if Width == 1
        :(Duplicated(f, df))
    else
        fargs = [:df]
        for i = 2:Width
            push!(fargs, Symbol("df_$i"))
        end
        :(BatchDuplicated(f, ($(fargs...),)))
    end
    dupty = if Width == 1
        :(Duplicated{FT})
    else
        :(BatchDuplicated{FT,$Width})
    end

    quote
        $(active_refs...)
        args = ($(wrapped...),)
        $(MakeTypes...)

        FT = Core.Typeof(f)
        dupClosure0 = if ActivityTup[1]
            !guaranteed_const(FT)
        else
            false
        end

        tt = Tuple{$(ElTypes...)}
        rt = Core.Compiler.return_type(f, tt)
        annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

        annotation = if $Width != 1 && annotation0 <: Duplicated
            BatchDuplicated{rt,$Width}
        else
            annotation0
        end

        if f isa typeof(Core.getglobal)
        else
            world = codegen_world_age(FT, tt)

            opt_mi = Val(world)
            _, adjoint = thunk(
                opt_mi,
                dupClosure0 ? $dupty : Const{FT},
                annotation,
                Tuple{$(Types...)},
                Val(API.DEM_ReverseModePrimal),
                width,
                ModifiedBetween,
                Val(true),
                Val(false),
                FFIABI,
                Val(false),
                runtimeActivity,
            ) #=erriffuncwritten=#

            tup =
                if annotation0 <: Active ||
                   annotation0 <: MixedDuplicated ||
                   annotation0 <: BatchMixedDuplicated
                    adjoint(
                        dupClosure0 ? $dup : Const(f),
                        args...,
                        $shadowret,
                        tape.internal_tape,
                    )[1]
                else
                    adjoint(dupClosure0 ? $dup : Const(f), args..., tape.internal_tape)[1]
                end

            $(outs...)
        end

        return nothing
    end
end

function func_runtime_generic_rev(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width)
    body =
        body_runtime_generic_rev(N, Width, wrapped, primtypes, batchshadowargs, active_refs)

    quote
        function runtime_generic_rev(
            activity::Type{Val{ActivityTup}},
            runtimeActivity::Val{RuntimeActivity},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            tape::TapeType,
            f::F,
            df::DF,
            $(allargs...),
        ) where {ActivityTup,RuntimeActivity,MB,TapeType,F,DF,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_rev(
    activity::Type{Val{ActivityTup}},
    runtimeActivity::Val{RuntimeActivity},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    tape::TapeType,
    f::F,
    df::DF,
    allargs...,
) where {ActivityTup,MB,RuntimeActivity,Width,TapeType,F,DF}
    N = div(length(allargs) + 2, Width + 1) - 1
    _, _, primtypes, _, _, wrapped, batchshadowargs, _, active_refs =
        setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_rev(
        N,
        Width,
        wrapped,
        primtypes,
        batchshadowargs,
        active_refs,
    )
end

@inline concat() = ()
@inline concat(a) = a
@inline concat(a, b) = (a..., b...)
@inline concat(a, b, c...) = concat(concat(a, b), c...)

@inline iterate_unwrap_inner_fwd(x::Const) = (map(Const, x.val)...,)
@inline iterate_unwrap_inner_fwd(x::Duplicated) = (map(Duplicated, x.val, x.dval)...,)
@inline batch_dup_tuple(x, vals...) = BatchDuplicated(x, (vals...,))
@inline iterate_unwrap_inner_fwd(x::BatchDuplicated) =
    (map(batch_dup_tuple, x.val, x.dval...)...,)

@inline function iterate_unwrap_fwd(args...)
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        iterate_unwrap_inner_fwd(args[i])
    end
end

@inline function iterate_unwrap_augfwd_act(args...)
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        if guaranteed_const(Core.Typeof(arg))
            Const(arg)
        else
            Active(arg)
        end
    end
end

@inline function iterate_unwrap_fwd_dup(args, dargs)
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        if guaranteed_const(ty)
            Const(arg)
        else
            Duplicated(arg, dargs[i])
        end
    end
end


@inline function iterate_unwrap_fwd_batchdup(::Val{Width}, args, dargs) where {Width}
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        if guaranteed_const(ty)
            Const(arg)
        else
            BatchDuplicated(arg, ntuple(Val(Width)) do j
                Base.@_inline_meta
                dargs[j][i]
            end)
        end
    end
end

function push_if_not_ref(::Val{reverse}, vals, darg, ::Type{T2}) where {reverse,T2}
    if reverse
        return popfirst!(vals)
    else
        tmp = Base.RefValue{T2}(darg)
        push!(vals, tmp)
        return tmp
    end
end

function push_if_not_ref(
    ::Val{reverse},
    vals,
    darg::Base.RefValue{T2},
    ::Type{T2},
) where {reverse,T2}
    return darg
end

@inline function iterate_unwrap_augfwd_dup(
    ::Val{reverse},
    vals,
    args,
    dargs,
) where {reverse}
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        actreg = active_reg_nothrow(ty, Val(nothing))
        if actreg == AnyState
            Const(arg)
        elseif actreg == ActiveState
            Active(arg)
        elseif actreg == MixedState
            darg = Base.inferencebarrier(dargs[i])
            MixedDuplicated(
                arg,
                push_if_not_ref(Val(reverse), vals, darg, ty)::Base.RefValue{ty},
            )
        else
            Duplicated(arg, dargs[i])
        end
    end
end

@inline function iterate_unwrap_augfwd_batchdup(
    ::Val{reverse},
    vals,
    ::Val{Width},
    args,
    dargs,
) where {reverse,Width}
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        actreg = active_reg_nothrow(ty, Val(nothing))
        if actreg == AnyState
            Const(arg)
        elseif actreg == ActiveState
            Active(arg)
        elseif actreg == MixedState
            BatchMixedDuplicated(
                arg,
                ntuple(Val(Width)) do j
                    Base.@_inline_meta
                    darg = Base.inferencebarrier(dargs[j][i])
                    push_if_not_ref(Val(reverse), vals, darg, ty)::Base.RefValue{ty}
                end,
            )
        else
            BatchDuplicated(arg, ntuple(Val(Width)) do j
                Base.@_inline_meta
                dargs[j][i]
            end)
        end
    end
end

@inline function iterate_unwrap_augfwd_mix(
    ::Val{reverse},
    vals,
    args,
    dargs0,
) where {reverse}
    dargs = dargs0[]
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        actreg = active_reg_nothrow(ty, Val(nothing))
        if actreg == AnyState
            Const(arg)
        elseif actreg == ActiveState
            Active(arg)
        elseif actreg == MixedState
            darg = Base.inferencebarrier(dargs[i])
            MixedDuplicated(
                arg,
                push_if_not_ref(Val(reverse), vals, darg, ty)::Base.RefValue{ty},
            )
        else
            Duplicated(arg, dargs[i])
        end
    end
end

@inline function iterate_unwrap_augfwd_batchmix(
    ::Val{reverse},
    vals,
    ::Val{Width},
    args,
    dargs,
) where {reverse,Width}
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        actreg = active_reg_nothrow(ty, Val(nothing))
        if actreg == AnyState
            Const(arg)
        elseif actreg == ActiveState
            Active(arg)
        elseif actreg == MixedState
            BatchMixedDuplicated(
                arg,
                ntuple(Val(Width)) do j
                    Base.@_inline_meta
                    darg = Base.inferencebarrier(dargs[j][][i])
                    push_if_not_ref(Val(reverse), vals, darg, ty)::Base.RefValue{ty}
                end,
            )
        else
            BatchDuplicated(arg, ntuple(Val(Width)) do j
                Base.@_inline_meta
                dargs[j][][i]
            end)
        end
    end
end

@inline function allFirst(::Val{Width}, res) where {Width}
    ntuple(Val(Width)) do i
        Base.@_inline_meta
        res[1]
    end
end

@inline function allSame(::Val{Width}, res) where {Width}
    ntuple(Val(Width)) do i
        Base.@_inline_meta
        res
    end
end

@inline function allZero(::Val{Width}, res) where {Width}
    ntuple(Val(Width)) do i
        Base.@_inline_meta
        Ref(make_zero(res))
    end
end

# This is explicitly escaped here to be what is apply generic in total [and thus all the insides are stable]
function fwddiff_with_return(
    runtimeActivity::Val{RuntimeActivity},
    ::Val{width},
    ::Val{dupClosure0},
    ::Type{ReturnType},
    ::Type{FT},
    ::Type{tt′},
    f::FT,
    df::DF,
    args::Vararg{Annotation,Nargs},
)::ReturnType where {RuntimeActivity,width,dupClosure0,ReturnType,FT,tt′,DF,Nargs}
    ReturnPrimal = Val(true)
    ModifiedBetween = Val(Enzyme.falses_from_args(Nargs + 1))

    dupClosure = dupClosure0 && !guaranteed_const(FT)
    FA = dupClosure ? Duplicated{FT} : Const{FT}

    tt = Enzyme.vaEltypes(tt′)

    rt = Core.Compiler.return_type(f, tt)
    annotation0 = guess_activity(rt, API.DEM_ForwardMode)

    annotation = if width != 1
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            BatchDuplicated{rt,width}
        else
            Const{rt}
        end
    else
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            Duplicated{rt}
        else
            Const{rt}
        end
    end

    world = codegen_world_age(FT, tt)
    fa = if dupClosure
        if width == 1
            Duplicated(f, df)
        else
            BatchDuplicated(f, df)
        end
    else
        Const(f)
    end
    opt_mi = Val(world)
    res = thunk(
        opt_mi,
        FA,
        annotation,
        tt′,
        Val(API.DEM_ForwardMode),
        Val(width), #=Mode=#
        ModifiedBetween,
        ReturnPrimal,
        Val(false),
        FFIABI,
        Val(false),
        runtimeActivity,
    )(
        fa,
        args...,
    ) #=erriffuncwritten=#
    return if annotation <: Const
        ReturnType(allFirst(Val(width + 1), res))
    else
        if width == 1
            ReturnType((res[2], res[1]))
        else
            ReturnType((res[2], res[1]...))
        end
    end
end

function body_runtime_iterate_fwd(N, Width, wrapped, primtypes, active_refs)
    wrappedexexpand = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds wrappedexexpand[i] = :($(wrapped[i])...)
    end
    return quote
        $(active_refs...)
        args = ($(wrappedexexpand...),)
        tt′ = Enzyme.vaTypeof(args...)
        FT = Core.Typeof(f)
        fwddiff_with_return(
            runtimeActivity,
            Val($Width),
            Val(ActivityTup[1]),
            ReturnType,
            FT,
            tt′,
            f,
            df,
            args...,
        )::ReturnType
    end
end

function func_runtime_iterate_fwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, _, active_refs =
        setup_macro_wraps(true, N, Width, nothing, true) #=iterate=#
    body = body_runtime_iterate_fwd(N, Width, wrapped, primtypes, active_refs)

    quote
        function runtime_iterate_fwd(
            activity::Type{Val{ActivityTup}},
            runtimeActivity::Val{RuntimeActivity},
            width::Val{$Width},
            RT::Val{ReturnType},
            f::F,
            df::DF,
            $(allargs...),
        ) where {ActivityTup,RuntimeActivity,ReturnType,F,DF,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_iterate_fwd(
    activity::Type{Val{ActivityTup}},
    runtimeActivity::Val{RuntimeActivity},
    width::Val{Width},
    RT::Val{ReturnType},
    f::F,
    df::DF,
    allargs...,
) where {ActivityTup,RuntimeActivity,Width,ReturnType,F,DF}
    N = div(length(allargs) + 2, Width + 1) - 1
    _, _, primtypes, _, _, wrapped, _, _, active_refs =
        setup_macro_wraps(true, N, Width, :allargs, true) #=iterate=#
    return body_runtime_iterate_fwd(N, Width, wrapped, primtypes, active_refs)
end

@generated function primal_tuple(args::Vararg{Annotation,Nargs}) where {Nargs}
    expr = Vector{Expr}(undef, Nargs)
    for i = 1:Nargs
        @inbounds expr[i] = :(args[$i].val)
    end
    return quote
        Base.@_inline_meta
        ($(expr...),)
    end
end

@generated function shadow_tuple(
    ::Type{Ann},
    ::Val{1},
    args::Vararg{Annotation,Nargs},
) where {Ann,Nargs}
    expr = Vector{Expr}(undef, Nargs)
    for i = 1:Nargs
        @inbounds expr[i] = quote
            @assert !(args[$i] isa Active)
            if args[$i] isa Const
                args[$i].val
            elseif args[$i] isa MixedDuplicated
                args[$i].dval[]
            else
                args[$i].dval
            end
        end
    end
    rval = :(($(expr...),))
    if Ann <: MixedDuplicated
        rval = :(Ref($rval))
    end
    return quote
        Base.@_inline_meta
        $rval
    end
end

@generated function shadow_tuple(
    ::Type{Ann},
    ::Val{width},
    args::Vararg{Annotation,Nargs},
) where {Ann,width,Nargs}
    wexpr = Vector{Expr}(undef, width)
    for w = 1:width
        expr = Vector{Expr}(undef, Nargs)
        for i = 1:Nargs
            @inbounds expr[i] = quote
                @assert !(args[$i] isa Active)
                if args[$i] isa Const
                    args[$i].val
                elseif args[$i] isa BatchMixedDuplicated
                    args[$i].dval[$w][]
                else
                    args[$i].dval[$w]
                end
            end
        end
        rval = :(($(expr...),))
        if Ann <: BatchMixedDuplicated
            rval = :(Ref($rval))
        end
        @inbounds wexpr[w] = rval
    end

    return quote
        Base.@_inline_meta
        ($(wexpr...),)
    end
end

# This is explicitly escaped here to be what is apply generic in total [and thus all the insides are stable]
function augfwd_with_return(
    runtimeActivity::Val{RuntimeActivity},
    ::Val{width},
    ::Val{dupClosure0},
    ::Type{ReturnType},
    ::Val{ModifiedBetween0},
    ::Type{FT},
    ::Type{tt′},
    f::FT,
    df::DF,
    args::Vararg{Annotation,Nargs},
)::ReturnType where {
    RuntimeActivity,
    width,
    dupClosure0,
    ReturnType,
    ModifiedBetween0,
    FT,
    tt′,
    DF,
    Nargs,
}
    ReturnPrimal = Val(true)
    ModifiedBetween = Val(ModifiedBetween0)

    tt = Enzyme.vaEltypes(tt′)
    rt = Core.Compiler.return_type(f, tt)
    annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

    annotation = if width != 1
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            BatchDuplicated{rt,width}
        elseif annotation0 <: MixedDuplicated
            BatchMixedDuplicated{rt,width}
        elseif annotation0 <: Active
            Active{rt}
        else
            Const{rt}
        end
    else
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            Duplicated{rt}
        elseif annotation0 <: MixedDuplicated
            MixedDuplicated{rt}
        elseif annotation0 <: Active
            Active{rt}
        else
            Const{rt}
        end
    end

    internal_tape, origRet, initShadow = if f != Base.tuple
        dupClosure = dupClosure0 && !guaranteed_const(FT)
        FA = dupClosure ? Duplicated{FT} : Const{FT}

        fa = if dupClosure
            if width == 1
                Duplicated(f, df)
            else
                BatchDuplicated(f, df)
            end
        else
            Const(f)
        end
        world = codegen_world_age(FT, tt)
        opt_mi = Val(world)
        forward, adjoint = thunk(
            opt_mi,
            FA,
            annotation,
            tt′,
            Val(API.DEM_ReverseModePrimal),
            Val(width),
            ModifiedBetween,
            Val(true),
            Val(false),
            FFIABI,
            Val(false),
            runtimeActivity,
        ) #=erriffuncwritten=#
        forward(fa, args...)
    else
        nothing,
        primal_tuple(args...),
        annotation <: Active ? nothing : shadow_tuple(annotation, Val(width), args...)
    end

    resT = typeof(origRet)

    if annotation <: Const
        shadow_return = nothing
        tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
            internal_tape,
            shadow_return,
        )
        return ReturnType((allSame(Val(width + 1), origRet)..., tape))
    elseif annotation <: Active
        shadow_return = if width == 1
            Ref(make_zero(origRet))
        else
            allZero(Val(width), origRet)
        end
        tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
            internal_tape,
            shadow_return,
        )
        if width == 1
            return ReturnType((origRet, shadow_return, tape))
        else
            return ReturnType((origRet, shadow_return..., tape))
        end
    end

    if width == 1
        if annotation <: MixedDuplicated
            shadow_return = initShadow
            tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                internal_tape,
                shadow_return,
            )
            return ReturnType((origRet, initShadow, tape))
        else
            shadow_return = nothing
            tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                internal_tape,
                shadow_return,
            )
            return ReturnType((origRet, initShadow, tape))
        end
    else
        if annotation <: BatchMixedDuplicated
            shadow_return = initShadow
            tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                internal_tape,
                shadow_return,
            )
            return ReturnType((origRet, initShadow..., tape))
        else
            shadow_return = nothing
            tape = Tape{typeof(internal_tape),typeof(shadow_return),resT}(
                internal_tape,
                shadow_return,
            )
            return ReturnType((origRet, initShadow..., tape))
        end
    end
end

function body_runtime_iterate_augfwd(N, Width, modbetween, wrapped, primtypes, active_refs)
    wrappedexexpand = Vector{Expr}(undef, N)
    for i = 1:N
        @inbounds wrappedexexpand[i] = :($(wrapped[i])...)
    end
    results = Vector{Expr}(undef, Width + 1)
    for i = 1:(Width+1)
        results[i] = :(tmpvals[$i])
    end
    return quote
        refs = Base.RefValue[]
        $(active_refs...)
        args = ($(wrappedexexpand...),)
        tt′ = Enzyme.vaTypeof(args...)
        FT = Core.Typeof(f)
        tmpvals = augfwd_with_return(
            runtimeActivity,
            Val($Width),
            Val(ActivityTup[1]),
            ReturnType,
            Val(concat($(modbetween...))),
            FT,
            tt′,
            f,
            df,
            args...,
        )::ReturnType
        ReturnType(($(results...), (tmpvals[$(Width + 2)], refs)))
    end
end

function func_runtime_iterate_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, modbetween, active_refs =
        setup_macro_wraps(false, N, Width, nothing, true) #=iterate=#
    body =
        body_runtime_iterate_augfwd(N, Width, modbetween, wrapped, primtypes, active_refs)

    quote
        function runtime_iterate_augfwd(
            activity::Type{Val{ActivityTup}},
            runtimeActivity::Val{RuntimeActivity},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            RT::Val{ReturnType},
            f::F,
            df::DF,
            $(allargs...),
        ) where {ActivityTup,RuntimeActivity,MB,ReturnType,F,DF,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_iterate_augfwd(
    activity::Type{Val{ActivityTup}},
    runtimeActivity::Val{RuntimeActivity},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    RT::Val{ReturnType},
    f::F,
    df::DF,
    allargs...,
) where {ActivityTup,RuntimeActivity,MB,Width,ReturnType,F,DF}
    N = div(length(allargs) + 2, Width + 1) - 1
    _, _, primtypes, _, _, wrapped, _, modbetween, active_refs =
        setup_macro_wraps(false, N, Width, :allargs, true) #=iterate=#
    return body_runtime_iterate_augfwd(
        N,
        Width,
        modbetween,
        wrapped,
        primtypes,
        active_refs,
    )
end

function add_into_vec!(val::Base.RefValue, expr, vec, idx_in_vec)
    val[] = recursive_add(val[], expr, identity, guaranteed_nonactive)
    nothing
end

function add_into_vec!(val::T, expr, vec, idx_in_vec) where {T}
    if ismutable(vec)
        @inbounds vec[idx_in_vec] = recursive_add(val, expr, identity, guaranteed_nonactive)
    else
        error(
            "Enzyme Mutability Error: Cannot in place to immutable value vec[$idx_in_vec] = $val, vec=$vec",
        )
    end
    nothing
end

# This is explicitly escaped here to be what is apply generic in total [and thus all the insides are stable]
@generated function rev_with_return(
    runtimeActivity::Val{RuntimeActivity},
    ::Val{width},
    ::Val{dupClosure0},
    ::Val{ModifiedBetween0},
    ::Val{lengths},
    ::Type{FT},
    ::Type{ttp},
    f::FT,
    df::DF,
    tape,
    shadowargs,
    args::Vararg{Annotation,Nargs},
)::Nothing where {
    RuntimeActivity,
    width,
    dupClosure0,
    ModifiedBetween0,
    lengths,
    FT,
    ttp,
    DF,
    Nargs,
}

    nontupexprs = Vector{Expr}(undef, Nargs)
    for i = 1:Nargs
        mid = if width == 1
            :(tape.shadow_return[][$i])
        else
            mexprs = Vector{Expr}(undef, width)
            for w = 1:width
                @inbounds mexprs[w] = :(tape.shadow_return[$w][][$i])
            end
            quote
                ($(mexprs...),)
            end
        end

        @inbounds nontupexprs[i] = quote
            if args[$i] isa Active ||
               args[$i] isa MixedDuplicated ||
               args[$i] isa BatchMixedDuplicated
                $mid
            else
                nothing
            end
        end
    end

    endexprs = Matrix{Expr}(undef, Nargs, width)
    for i = 1:Nargs
        for w = 1:width
            @inbounds endexprs[i, w] = quote
                if args[$i] isa Active ||
                   args[$i] isa MixedDuplicated ||
                   args[$i] isa BatchMixedDuplicated
                    expr = if args[$i] isa Active || f == Base.tuple
                        if $width == 1
                            tup[$i]
                        else
                            tup[$i][$w]
                        end
                    elseif args[$i] isa MixedDuplicated
                        args[$i].dval[]
                    else
                        # if args[$i] isa BatchMixedDuplicated
                        args[$i].dval[$w][]
                    end

                    idx_of_vec, idx_in_vec = $(lengths[i])
                    vec = @inbounds shadowargs[idx_of_vec][$w]
                    if vec isa Base.RefValue
                        vecld = vec[]
                        T = Core.Typeof(vecld)
                        vec[] = recursive_index_add(T, vecld, Val(idx_in_vec), expr)
                    else
                        val = @inbounds vec[idx_in_vec]
                        add_into_vec!(Base.inferencebarrier(val), expr, vec, idx_in_vec)
                    end
                end
            end
        end
    end

    tgen = if FT == typeof(Base.tuple)
        :(tup = ($(nontupexprs...),))
    else
        annotation = if width != 1
            quote
                if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
                    BatchDuplicated{rt,$width}
                elseif annotation0 <: MixedDuplicated
                    BatchMixedDuplicated{rt,$width}
                elseif annotation0 <: Active
                    Active{rt}
                else
                    Const{rt}
                end
            end
        else
            quote
                if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
                    Duplicated{rt}
                elseif annotation0 <: MixedDuplicated
                    MixedDuplicated{rt}
                elseif annotation0 <: Active
                    Active{rt}
                else
                    Const{rt}
                end
            end
        end

        shadadj = if width == 1
            :(adjoint(fa, args..., tape.shadow_return[], tape.internal_tape)[1])
        else
            margs = Vector{Expr}(undef, width)
            for w = 1:width
                @inbounds margs[w] = :(tape.shadow_return[$w][])
            end
            :(adjoint(fa, args..., ($(margs...),), tape.internal_tape)[1])
        end

        tt = Enzyme.vaEltypes(ttp)

        quote
            ReturnPrimal = Val(true)
            ModifiedBetween = Val($ModifiedBetween0)

            dupClosure = $dupClosure0 && !guaranteed_const($FT)
            FA = dupClosure ? Duplicated{$FT} : Const{$FT}

            tt = $tt

            rt = Core.Compiler.return_type(f, tt)
            annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

            annotation = $annotation
            world = codegen_world_age(FT, tt)

            fa = if dupClosure
                $(width == 1 ? :Duplicated : :BatchDuplicated)(f, df)
            else
                Const(f)
            end
            opt_mi = Val(world)
            forward, adjoint = thunk(
                opt_mi,
                FA,
                annotation,
                $ttp,
                Val(API.DEM_ReverseModePrimal),
                Val($width),
                ModifiedBetween,
                Val(true),
                Val(false),
                FFIABI,
                Val(false),
                runtimeActivity,
            ) #=erriffuncwritten=#

            tup = if tape.shadow_return !== nothing
                $shadadj
            else
                adjoint(fa, args..., tape.internal_tape)[1]
            end
        end
    end

    return quote
        $tgen
        $(endexprs...)
        nothing
    end
end

@generated function ntuple_pair(::Val{Len}, ::Val{i}) where {Len,i}
    mexprs = Vector{Expr}(undef, Len)
    for j = 1:Len
        @inbounds mexprs[j] = quote
            ($i, $j)
        end
    end
    quote
        Base.@_inline_meta
        ($(mexprs...),)
    end
end

function body_runtime_iterate_rev(
    N,
    Width,
    modbetween,
    wrapped,
    primargs,
    shadowargs,
    active_refs,
)
    shadow_ret = nothing
    if Width == 1
        shadowret = :(tape.shadow_return[])
    else
        shadowret = Expr[]
        for w = 1:Width
            push!(shadowret, :(tape.shadow_return[$w][]))
        end
        shadowret = :(($(shadowret...),))
    end

    wrappedexexpand = Vector{Expr}(undef, N)
    for i = 1:N
        wrappedexexpand[i] = :($(wrapped[i])...)
    end
    lengths = Vector{Expr}(undef, N)
    for i = 1:N
        lengths[i] = quote
            ntuple_pair(Val(length($(primargs[i]))), Val($i))
        end
    end

    shadowsplat = Expr[]
    for s in shadowargs
        push!(shadowsplat, :(($(s...),)))
    end
    quote
        (tape0, refs) = tape
        $(active_refs...)
        args = ($(wrappedexexpand...),)
        tt′ = Enzyme.vaTypeof(args...)
        FT = Core.Typeof(f)
        rev_with_return(
            runtimeActivity,
            Val($Width),
            Val(ActivityTup[1]),
            Val(concat($(modbetween...))),
            Val(concat($(lengths...))),
            FT,
            tt′,
            f,
            df,
            tape0,
            ($(shadowsplat...),),
            args...,
        )
        return nothing
    end
end

function func_runtime_iterate_rev(N, Width)
    primargs,
    _,
    primtypes,
    allargs,
    typeargs,
    wrapped,
    batchshadowargs,
    modbetween,
    active_refs = setup_macro_wraps(false, N, Width, nothing, true; reverse = true) #=iterate=#
    body = body_runtime_iterate_rev(
        N,
        Width,
        modbetween,
        wrapped,
        primargs,
        batchshadowargs,
        active_refs,
    )

    quote
        function runtime_iterate_rev(
            activity::Type{Val{ActivityTup}},
            runtimeActivity::Val{RuntimeActivity},
            width::Val{$Width},
            ModifiedBetween::Val{MB},
            tape::TapeType,
            f::F,
            df::DF,
            $(allargs...),
        ) where {ActivityTup,RuntimeActivity,MB,TapeType,F,DF,$(typeargs...)}
            $body
        end
    end
end

@generated function runtime_iterate_rev(
    activity::Type{Val{ActivityTup}},
    runtimeActivity::Val{RuntimeActivity},
    width::Val{Width},
    ModifiedBetween::Val{MB},
    tape::TapeType,
    f::F,
    df::DF,
    allargs...,
) where {ActivityTup,RuntimeActivity,MB,Width,TapeType,F,DF}
    N = div(length(allargs) + 2, Width + 1) - 1
    primargs, _, primtypes, _, _, wrapped, batchshadowargs, modbetween, active_refs =
        setup_macro_wraps(false, N, Width, :allargs, true; reverse = true) #=iterate=#
    return body_runtime_iterate_rev(
        N,
        Width,
        modbetween,
        wrapped,
        primargs,
        batchshadowargs,
        active_refs,
    )
end

# Create specializations
for (N, Width) in Iterators.product(0:30, 1:10)
    eval(func_runtime_generic_fwd(N, Width))
    eval(func_runtime_generic_augfwd(N, Width))
    eval(func_runtime_generic_rev(N, Width))
    eval(func_runtime_iterate_fwd(N, Width))
    eval(func_runtime_iterate_augfwd(N, Width))
    eval(func_runtime_iterate_rev(N, Width))
end

function generic_setup(
    orig,
    func,
    ReturnType,
    gutils,
    start,
    B::LLVM.IRBuilder,
    lookup;
    sret = nothing,
    tape = nothing,
    firstconst = false,
    endcast = true,
    firstconst_after_tape = true,
    runtime_activity = true,
)
    width = get_width(gutils)
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    ops = collect(operands(orig))[start+firstconst:end-1]

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    ActivityList = LLVM.Value[]

    @assert length(ops) != 0
    fill_val = unsafe_to_llvm(B, nothing)

    vals = LLVM.Value[]

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    if firstconst && !firstconst_after_tape
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
            push!(ActivityList, unsafe_to_llvm(B, false))
        else
            inverted = invert_pointer(gutils, op, B)
            if lookup
                inverted = lookup_value(gutils, inverted, B)
            end
            if get_runtime_activity(gutils)
                inv_0 = if width == 1
                    inverted
                else
                    extract_value!(B, inverted, 0)
                end
                push!(
                    ActivityList,
                    select!(
                        B,
                        icmp!(B, LLVM.API.LLVMIntNE, val, inv_0),
                        unsafe_to_llvm(B, true),
                        unsafe_to_llvm(B, false),
                    ),
                )
            else
                push!(ActivityList, unsafe_to_llvm(B, true))
            end
        end

        for w = 1:width
            ev = fill_val
            if inverted !== nothing
                if width == 1
                    ev = inverted
                else
                    ev = extract_value!(B, inverted, w - 1)
                end
            end

            push!(vals, ev)
        end
    end
    @assert length(ActivityList) == length(ops)

    if tape !== nothing
        if tape isa Vector
            for t in reverse(tape)
                pushfirst!(vals, t)
            end
        else
            pushfirst!(vals, tape)
        end
    else
        pushfirst!(vals, unsafe_to_llvm(B, Val(ReturnType)))
    end

    if firstconst && firstconst_after_tape
        val = new_from_original(gutils, operands(orig)[start])
        if lookup
            val = lookup_value(gutils, val, B)
        end
        pushfirst!(vals, val)
    end

    if mode != API.DEM_ForwardMode
        uncacheable = get_uncacheable(gutils, orig)
        sret = false
        returnRoots = false

        ModifiedBetween = Bool[]

        for idx = 1:(length(ops)+firstconst)
            push!(ModifiedBetween, uncacheable[(start-1)+idx] != 0)
        end
        pushfirst!(vals, unsafe_to_llvm(B, Val((ModifiedBetween...,))))
    end

    pushfirst!(vals, unsafe_to_llvm(B, Val(Int(width))))
    if runtime_activity
        pushfirst!(vals, unsafe_to_llvm(B, Val(get_runtime_activity(gutils))))
    end
    etup0 = emit_tuple!(B, ActivityList)
    etup = emit_apply_type!(B, Base.Val, [etup0])
    if isa(etup, LLVM.Instruction)
        @assert length(collect(LLVM.uses(etup0))) == 1
    end
    pushfirst!(vals, etup)

    pushfirst!(vals, unsafe_to_llvm(B, func))

    cal = emit_apply_generic!(B, vals)

    debug_from_orig!(gutils, cal, orig)

    if tape === nothing && endcast
        llty = convert(LLVMType, ReturnType)
        cal = LLVM.addrspacecast!(B, cal, LLVM.PointerType(T_jlvalue, Derived))
        cal = LLVM.pointercast!(B, cal, LLVM.PointerType(llty, Derived))
    end

    return cal
end

function common_generic_fwd(offset, B, orig, gutils, normalR, shadowR)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

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

    width = get_width(gutils)

    sret = generic_setup(
        orig,
        runtime_generic_fwd,
        AnyArray(1 + Int(width)),
        gutils,
        offset,
        B,
        false,
    ) #=start=#
    AT = LLVM.ArrayType(T_prjlvalue, 1 + Int(width))
    if unsafe_load(shadowR) != C_NULL
        if width == 1
            gep =
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end
    return false
end

@register_fwd function generic_fwd(B, orig, gutils, normalR, shadowR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37
    common_generic_fwd(1, B, orig, gutils, normalR, shadowR)
end

function common_generic_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

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

    width = get_width(gutils)
    sret = generic_setup(
        orig,
        runtime_generic_augfwd,
        AnyArray(2 + Int(width)),
        gutils,
        offset,
        B,
        false,
    ) #=start=#
    AT = LLVM.ArrayType(T_prjlvalue, 2 + Int(width))

    if unsafe_load(shadowR) != C_NULL
        if width == 1
            gep =
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    tape = LLVM.load!(
        B,
        T_prjlvalue,
        LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1 + width)]),
    )
    unsafe_store!(tapeR, tape.ref)

    if normalR != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end
    return false
end

@register_aug function generic_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function common_generic_rev(offset, B, orig, gutils, tape)::Cvoid
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return nothing
    end

    @assert tape !== C_NULL
    width = get_width(gutils)
    generic_setup(orig, runtime_generic_rev, Nothing, gutils, offset, B, true; tape) #=start=#
    return nothing
end

@register_rev function generic_rev(B, orig, gutils, tape)::Cvoid
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_apply_latest_fwd(offset, B, orig, gutils, normalR, shadowR)
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
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 1 + Int(width))
    sret = generic_setup(
        orig,
        runtime_generic_fwd,
        AnyArray(1 + Int(width)),
        gutils,
        offset + 1,
        B,
        false,
    ) #=start=#

    if unsafe_load(shadowR) != C_NULL
        if width == 1
            gep =
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end

    return false
end

function common_apply_latest_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
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

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 2 + Int(width))
    # sret = generic_setup(orig, runtime_apply_latest_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, ctx, B, false)
    sret = generic_setup(
        orig,
        runtime_generic_augfwd,
        AnyArray(2 + Int(width)),
        gutils,
        offset + 1,
        B,
        false,
    ) #=start=#

    if unsafe_load(shadowR) != C_NULL
        if width == 1
            gep =
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    tape = LLVM.load!(
        B,
        T_prjlvalue,
        LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1 + width)]),
    )
    unsafe_store!(tapeR, tape.ref)

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end
    return false
end

function common_apply_latest_rev(offset, B, orig, gutils, tape)::Cvoid
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return nothing
    end
    if !is_constant_value(gutils, orig) || !is_constant_inst(gutils, orig)
        width = get_width(gutils)
        generic_setup(orig, runtime_generic_rev, Nothing, gutils, offset + 1, B, true; tape) #=start=#
    end

    return nothing
end

@register_fwd function apply_latest_fwd(B, orig, gutils, normalR, shadowR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_fwd(1, B, orig, gutils, normalR, shadowR)
end

@register_aug function apply_latest_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

@register_rev function apply_latest_rev(B, orig, gutils, tape)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20
    @assert conv == 37

    common_apply_latest_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_apply_iterate_fwd(offset, B, orig, gutils, normalR, shadowR)
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

    v, isiter = absint(operands(orig)[offset+1])
    v2, istup = absint(operands(orig)[offset+2])

    width = get_width(gutils)

    if v &&
       v2 &&
       isiter == Base.iterate &&
       istup == Base.tuple &&
       length(operands(orig)) >= offset + 4
        origops = collect(operands(orig)[1:end-1])
        shadowins =
            [invert_pointer(gutils, origops[i], B) for i = (offset+3):length(origops)]
        shadowres = if width == 1
            newops = LLVM.Value[]
            newvals = API.CValueType[]
            for (i, v) in enumerate(origops)
                if i >= offset + 3
                    shadowin2 = shadowins[i-offset-3+1]
                    push!(newops, shadowin2)
                    push!(newvals, API.VT_Shadow)
                else
                    push!(newops, new_from_original(gutils, origops[i]))
                    push!(newvals, API.VT_Primal)
                end
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
            cal
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for j = 1:width
                newops = LLVM.Value[]
                newvals = API.CValueType[]
                for (i, v) in enumerate(origops)
                    if i >= offset + 3
                        shadowin2 = extract_value!(B, shadowins[i-offset-3+1], j - 1)
                        push!(newops, shadowin2)
                        push!(newvals, API.VT_Shadow)
                    else
                        push!(newops, new_from_original(gutils, origops[i]))
                        push!(newvals, API.VT_Primal)
                    end
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
                shadow = insert_value!(B, shadow, cal, j - 1)
            end
            shadow
        end

        unsafe_store!(shadowR, shadowres.ref)
        return false
    end

    if v && isiter == Base.iterate
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        sret = generic_setup(
            orig,
            runtime_iterate_fwd,
            AnyArray(1 + Int(width)),
            gutils,
            offset + 2,
            B,
            false,
        ) #=start=#
        AT = LLVM.ArrayType(T_prjlvalue, 1 + Int(width))
        if unsafe_load(shadowR) != C_NULL
            if width == 1
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(1)],
                )
                shadow = LLVM.load!(B, T_prjlvalue, gep)
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
                shadow = LLVM.UndefValue(ST)
                for i = 1:width
                    gep = LLVM.inbounds_gep!(
                        B,
                        AT,
                        sret,
                        [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                    )
                    ld = LLVM.load!(B, T_prjlvalue, gep)
                    shadow = insert_value!(B, shadow, ld, i - 1)
                end
            end
            unsafe_store!(shadowR, shadow.ref)
        end

        if unsafe_load(normalR) != C_NULL
            normal = LLVM.load!(
                B,
                T_prjlvalue,
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
            )
            unsafe_store!(normalR, normal.ref)
        else
            # Delete the primal code
            ni = new_from_original(gutils, orig)
            erase_with_placeholder(gutils, ni, orig)
        end
        return false
    end

    emit_error(
        B,
        orig,
        "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate " *
        string((v, v2, isiter, istup, length(operands(orig)), offset + 4)),
    )

    return false
end

function common_apply_iterate_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
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

    v, isiter = absint(operands(orig)[offset+1])
    v2, istup = absint(operands(orig)[offset+2])

    width = get_width(gutils)

    if v && isiter == Base.iterate
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        sret = generic_setup(
            orig,
            runtime_iterate_augfwd,
            AnyArray(2 + Int(width)),
            gutils,
            offset + 2,
            B,
            false,
        ) #=start=#
        AT = LLVM.ArrayType(T_prjlvalue, 2 + Int(width))

        if unsafe_load(shadowR) != C_NULL
            if width == 1
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(1)],
                )
                shadow = LLVM.load!(B, T_prjlvalue, gep)
            else
                ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
                shadow = LLVM.UndefValue(ST)
                for i = 1:width
                    gep = LLVM.inbounds_gep!(
                        B,
                        AT,
                        sret,
                        [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                    )
                    ld = LLVM.load!(B, T_prjlvalue, gep)
                    shadow = insert_value!(B, shadow, ld, i - 1)
                end
            end
            unsafe_store!(shadowR, shadow.ref)
        end

        tape = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(
                B,
                AT,
                sret,
                [LLVM.ConstantInt(0), LLVM.ConstantInt(1 + width)],
            ),
        )
        unsafe_store!(tapeR, tape.ref)

        if normalR != C_NULL
            normal = LLVM.load!(
                B,
                T_prjlvalue,
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
            )
            unsafe_store!(normalR, normal.ref)
        else
            # Delete the primal code
            ni = new_from_original(gutils, orig)
            erase_with_placeholder(gutils, ni, orig)
        end
        return false
        return false
    end

    emit_error(
        B,
        orig,
        "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate " *
        string((v, v2, isiter, istup, length(operands(orig)), offset + 4)),
    )

    unsafe_store!(
        shadowR,
        UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))).ref,
    )
    return false
end

function common_apply_iterate_rev(offset, B, orig, gutils, tape)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return nothing
    end

    @assert tape !== C_NULL
    width = get_width(gutils)
    generic_setup(orig, runtime_iterate_rev, Nothing, gutils, offset + 2, B, true; tape) #=start=#
    return nothing
end

@register_fwd function apply_iterate_fwd(B, orig, gutils, normalR, shadowR)
    common_apply_iterate_fwd(1, B, orig, gutils, normalR, shadowR)
end

@register_aug function apply_iterate_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_apply_iterate_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

@register_rev function apply_iterate_rev(B, orig, gutils, tape)
    common_apply_iterate_rev(1, B, orig, gutils, tape)
    return nothing
end

function common_invoke_fwd(offset, B, orig, gutils, normalR, shadowR)
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

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    sret = generic_setup(
        orig,
        runtime_generic_fwd,
        AnyArray(1 + Int(width)),
        gutils,
        offset + 1,
        B,
        false,
    ) #=start=#
    AT = LLVM.ArrayType(T_prjlvalue, 1 + Int(width))

    if unsafe_load(shadowR) != C_NULL
        if width == 1
            gep =
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end

    return false
end

function common_invoke_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
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
    normal =
        (unsafe_load(normalR) != C_NULL) ? LLVM.Instruction(unsafe_load(normalR)) : nothing
    shadow =
        (unsafe_load(shadowR) != C_NULL) ? LLVM.Instruction(unsafe_load(shadowR)) : nothing

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    conv = LLVM.callconv(orig)

    width = get_width(gutils)
    sret = generic_setup(
        orig,
        runtime_generic_augfwd,
        AnyArray(2 + Int(width)),
        gutils,
        offset + 1,
        B,
        false,
    ) #=start=#
    AT = LLVM.ArrayType(T_prjlvalue, 2 + Int(width))

    if unsafe_load(shadowR) != C_NULL
        if width == 1
            gep =
                LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1)])
            shadow = LLVM.load!(B, T_prjlvalue, gep)
        else
            ST = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
            shadow = LLVM.UndefValue(ST)
            for i = 1:width
                gep = LLVM.inbounds_gep!(
                    B,
                    AT,
                    sret,
                    [LLVM.ConstantInt(0), LLVM.ConstantInt(i)],
                )
                ld = LLVM.load!(B, T_prjlvalue, gep)
                shadow = insert_value!(B, shadow, ld, i - 1)
            end
        end
        unsafe_store!(shadowR, shadow.ref)
    end

    tape = LLVM.load!(
        B,
        T_prjlvalue,
        LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1 + width)]),
    )
    unsafe_store!(tapeR, tape.ref)

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(
            B,
            T_prjlvalue,
            LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]),
        )
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end

    return false
end

function common_invoke_rev(offset, B, orig, gutils, tape)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(
        gutils,
        orig,
        needsPrimalP,
        needsShadowP,
        API.DEM_ReverseModePrimal,
    )

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0) &&
       is_constant_inst(gutils, orig)
        return nothing
    end

    width = get_width(gutils)
    generic_setup(orig, runtime_generic_rev, Nothing, gutils, offset + 1, B, true; tape) #=start=#

    return nothing
end

@register_fwd function invoke_fwd(B, orig, gutils, normalR, shadowR)
    common_invoke_fwd(1, B, orig, gutils, normalR, shadowR)
end

@register_aug function invoke_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    common_invoke_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

@register_rev function invoke_rev(B, orig, gutils, tape)
    common_invoke_rev(1, B, orig, gutils, tape)
    return nothing
end
