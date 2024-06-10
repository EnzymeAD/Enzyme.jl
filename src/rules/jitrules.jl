function func_mixed_call(N)
    allargs = Expr[]
    typeargs = Union{Symbol,Expr}[]
    exprs2 = Union{Symbol,Expr}[]
    for i in 1:N
        arg = Symbol("arg_$i")
        targ = Symbol("T$i")
        e = :($arg::$targ)
        push!(allargs, e)
        push!(typeargs, targ)

        inarg = quote
            if RefTypes[1+$i]
                $arg[]
            else
                $arg
            end
        end
        push!(exprs2, inarg)
    end
    
    quote
        @generated function runtime_mixed_call(::Val{RefTypes}, f::F, $(allargs...)) where {RefTypes, F, $(typeargs...)}
            fexpr = :f
            if RefTypes[1]
                fexpr = :(($fexpr)[])
            end
            exprs2 = Union{Symbol,Expr}[]
            for i in 1:$N
                arg = Symbol("arg_$i")
                inarg = if RefTypes[1+i]
                    :($arg[])
                else
                    :($arg)
                end
                push!(exprs2, inarg)
            end
            @static if VERSION ≥ v"1.8-"
                return quote
                    Base.@_inline_meta
                    @inline $fexpr($(exprs2...))
                end
            else
                return quote
                    Base.@_inline_meta
                    $fexpr($(exprs2...))
                end
            end
        end
    end
end

@generated function runtime_mixed_call(::Val{RefTypes}, f::F, allargs::Vararg{Any, N}) where {RefTypes, F, N}
    fexpr = :f
    if RefTypes[1]
        fexpr = :(($fexpr)[])
    end
    exprs2 = Union{Symbol,Expr}[]
    for i in 1:N
        inarg = if RefTypes[1+i]
            :(allargs[$i][])
        else
            :(allargs[$i])
        end
        push!(exprs2, inarg)
    end
    @static if VERSION ≥ v"1.8-"
        return quote
            Base.@_inline_meta
            @inline $fexpr($(exprs2...))
        end
    else
        return quote
            Base.@_inline_meta
            $fexpr($(exprs2...))
        end
    end
end

for N in 0:10
    eval(func_mixed_call(N))
end

function setup_macro_wraps(forwardMode::Bool, N::Int, Width::Int, base=nothing, iterate=false; func=true, mixed_or_active = false)
    primargs = Union{Symbol,Expr}[]
    shadowargs = Union{Symbol,Expr}[]
    batchshadowargs = Vector{Union{Symbol,Expr}}[]
    primtypes = Union{Symbol,Expr}[]
    allargs = Expr[]
    typeargs = Symbol[]
    dfns = Union{Symbol,Expr}[:df]
    base_idx = 1
    if func
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
    modbetween = Expr[:(MB[1])]
    active_refs = Expr[]
    for i in 1:N
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
            $aref = active_reg_nothrow($(primtypes[i]), Val(nothing));
        end)
        expr = if iterate
            :(
                 if ActivityTup[$i+1] && !guaranteed_const($(primtypes[i]))
                   @assert $(primtypes[i]) !== DataType
                    if !$forwardMode && active_reg($(primtypes[i]))
                        iterate_unwrap_augfwd_act($(primargs[i])...)
                 else
                     $((Width == 1) ? quote
                        iterate_unwrap_augfwd_dup(Val($forwardMode), $(primargs[i]), $(shadowargs[i]))
                    end : quote
                        iterate_unwrap_augfwd_batchdup(Val($forwardMode), Val($Width), $(primargs[i]), $(shadowargs[i]))
                    end
                    )
                 end
             else
                 map(Const, $(primargs[i]))
             end
            )
        else
            if forwardMode
                quote
                    if ActivityTup[$i+1] && !guaranteed_const($(primtypes[i]))
                        $((Width == 1) ? :Duplicated : :BatchDuplicated)($(primargs[i]), $(shadowargs[i]))
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
                            $((Width == 1) ? :Duplicated : :BatchDuplicated)(Ref($(primargs[i])), $(shadowargs[i]))
                        else
                            $((Width == 1) ? :Duplicated : :BatchDuplicated)($(primargs[i]), $(shadowargs[i]))
                        end
                    else
                        Const($(primargs[i]))
                    end
                end
            end
        end
        push!(wrapped, expr)
    end

    any_mixed = quote false end
    for i in 1:N
        aref = Symbol("active_ref_$i")
        if mixed_or_active
            any_mixed = :($any_mixed || $aref == MixedState || $aref == ActiveState)
        else
            any_mixed = :($any_mixed || $aref == MixedState)
        end
    end

    if mixed_or_active
        push!(active_refs, quote
            active_refs = (false, $(collect(:($(Symbol("active_ref_$i")) == MixedState || $(Symbol("active_ref_$i")) == ActiveState) for i in 1:N)...))
        end)
    else
        push!(active_refs, quote
            active_refs = (false, $(collect(:($(Symbol("active_ref_$i")) == MixedState) for i in 1:N)...))
        end)
    end
    push!(active_refs, quote
        any_mixed = $any_mixed
    end)
    return primargs, shadowargs, primtypes, allargs, typeargs, wrapped, batchshadowargs, modbetween, active_refs
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
    _, _, primtypes, allargs, typeargs, wrapped, _, _, _ = setup_macro_wraps(true, N, Width)
    body = body_runtime_generic_fwd(N, Width, wrapped, primtypes)

    quote
        function runtime_generic_fwd(activity::Type{Val{ActivityTup}}, width::Val{$Width}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_fwd(activity::Type{Val{ActivityTup}}, width::Val{Width}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, _, _, _ = setup_macro_wraps(true, N, Width, :allargs)
    return body_runtime_generic_fwd(N, Width, wrapped, primtypes)
end

function body_runtime_generic_augfwd(N, Width, wrapped, primttypes, active_refs)
    nnothing = ntuple(i->nothing, Val(Width+1))
    nres = ntuple(i->:(origRet), Val(Width+1))
    nzeros = ntuple(i->:(Ref(make_zero(origRet))), Val(Width))
    nres3 = ntuple(i->:(res[3]), Val(Width))
    ElTypes = ntuple(i->:(eltype($(Symbol("type_$i")))), Val(N))

    MakeTypes = ntuple(i->:($(Symbol("type_$i")) = Core.Typeof(args[$i])), Val(N))

    Types = ntuple(i->Symbol("type_$i"), Val(N))

    MixedTypes = ntuple(i->:($(Symbol("active_ref_$i") == MixedState) ? Ref($(Symbol("type_$i"))) : $(Symbol("type_$i"))), Val(N))

    ending = if Width == 1
        quote
            if active_reg_nothrow(resT, Val(nothing)) == MixedState && !(initShadow isa Base.RefValue)
                shadow_return = Ref(initShadow)
                tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
                return ReturnType((origRet, shadow_return, tape))
            else
                shadow_return = nothing
                tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
                return ReturnType((origRet, initShadow, tape))
            end
        end
    else
        expr = :()
        shads = Expr[]
        for i in 1:Width
            if i == 1
                expr = quote !(initShadow[$i] isa Base.RefValue) end
            else
                expr = quote $expr || !(initShadow[$i] isa Base.RefValue) end
            end
            push!(shads, quote
                Ref(initShadow[$i])
            end)
        end
        quote
            if active_reg_nothrow(resT, Val(nothing)) == MixedState && ($expr)
                shadow_return = ($(shads...),)
                tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
                return ReturnType((origRet, shadow_return..., tape))
            else
                shadow_return = nothing
                tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
                return ReturnType((origRet, initShadow..., tape))
            end
        end
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


        internal_tape, origRet, initShadow, annotation = if any_mixed
            ttM = Tuple{Val{active_refs}, FT, $(ElTypes...)}
            rtM = Core.Compiler.return_type(runtime_mixed_call, ttM)
            annotation0M = guess_activity(rtM, API.DEM_ReverseModePrimal)

            annotationM = if $Width != 1 && annotation0M <: Duplicated
                BatchDuplicated{rt, $Width}
            else
                annotation0M
            end
            worldM = codegen_world_age(typeof(runtime_mixed_call), ttM)
            ModifiedBetweenM = Val((false, false, element(ModifiedBetween)...))

            forward, adjoint = thunk(Val(worldM), 
                                     Const{typeof(runtime_mixed_call)},
                                     annotationM, Tuple{Const{Val{active_refs}}, dupClosure0 ? Duplicated{FT} : Const{FT}, $(Types...)}, Val(API.DEM_ReverseModePrimal), width,
                                     ModifiedBetweenM, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

            forward(Const(runtime_mixed_call), Const(Val(active_refs)), dupClosure0 ? Duplicated(f, df) : Const(f), args...)..., annotationM

        else
            tt = Tuple{$(ElTypes...)}
            rt = Core.Compiler.return_type(f, tt)
            annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

            annotationA = if $Width != 1 && annotation0 <: Duplicated
                BatchDuplicated{rt, $Width}
            else
                annotation0
            end
            world = codegen_world_age(FT, tt)

            forward, adjoint = thunk(Val(world), dupClosure0 ? Duplicated{FT} : Const{FT},
                                     annotationA, Tuple{$(Types...)}, Val(API.DEM_ReverseModePrimal), width,
                                     ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

            forward(dupClosure0 ? Duplicated(f, df) : Const(f), args...)..., annotationA
        end

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

        $ending
    end
end

function func_runtime_generic_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, _, active_refs = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_augfwd(N, Width, wrapped, primtypes, active_refs)

    quote
        function runtime_generic_augfwd(activity::Type{Val{ActivityTup}}, width::Val{$Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...))::ReturnType where {ActivityTup, MB, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_augfwd(activity::Type{Val{ActivityTup}}, width::Val{Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, allargs...)::ReturnType where {ActivityTup, MB, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, _, _, active_refs = setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_augfwd(N, Width, wrapped, primtypes, active_refs)
end

function nonzero_active_data(x::T) where T<: AbstractFloat
    return x != zero(T)
end

nonzero_active_data(::T) where T<: Base.RefValue = false
nonzero_active_data(::T) where T<: Array = false
nonzero_active_data(::T) where T<: Ptr = false

function nonzero_active_data(x::T) where T
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
    for i in 1:N
        for w in 1:Width
            expr = if Width == 1
                :(tup[$i])
            else
                :(tup[$i][$w])
            end
            shad = shadowargs[i][w]
            out = :(if tup[$i] === nothing
              elseif $shad isa Base.RefValue
                  $shad[] = recursive_add($shad[], $expr)
                else
                    error("Enzyme Mutability Error: Cannot add one in place to immutable value "*string($shad)*" tup[i]="*string(tup[$i])*" i="*string($i)*" w="*string($w)*" tup="*string(tup))
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

    ElTypes = ntuple(i->:(eltype($(Symbol("type_$i")))), Val(N))

    MakeTypes = ntuple(i->:($(Symbol("type_$i")) = Core.Typeof(args[$i])), Val(N))

    Types = ntuple(i->Symbol("type_$i"), Val(N))

    MixedTypes = ntuple(i->:($(Symbol("active_ref_$i") == MixedState) ? Ref($(Symbol("type_$i"))) : $(Symbol("type_$i"))), Val(N))

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

        if any_mixed
            ttM = Tuple{Val{active_refs}, FT, $(ElTypes...)}
            rtM = Core.Compiler.return_type(runtime_mixed_call, ttM)
            annotation0M = guess_activity(rtM, API.DEM_ReverseModePrimal)

            annotationM = if $Width != 1 && annotation0M <: Duplicated
                BatchDuplicated{rt, $Width}
            else
                annotation0M
            end
            worldM = codegen_world_age(typeof(runtime_mixed_call), ttM)
            ModifiedBetweenM = Val((false, false, element(ModifiedBetween)...))

            _, adjoint = thunk(Val(worldM), 
                                     Const{typeof(runtime_mixed_call)},
                                     annotationM, Tuple{Const{Val{active_refs}}, dupClosure0 ? Duplicated{FT} : Const{FT}, $(Types...)}, Val(API.DEM_ReverseModePrimal), width,
                                     ModifiedBetweenM, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

            if tape.shadow_return !== nothing
                if !(annotation0M <: Active) && nonzero_active_data(($shadowret,))
                    ET = ($(ElTypes...),)
                    throw(AssertionError("Shadow value "*string(($shadowret,))*" returned from type unstable call to $f($(ET...)) has mixed internal activity types. See https://enzyme.mit.edu/julia/stable/faq/#Mixed-activity for more information"))
                end
            end
            if annotation0M <: Active
                adjoint(Const(runtime_mixed_call), Const(Val(active_refs)), dupClosure0 ? Duplicated(f, df) : Const(f), args..., $shadowret, tape.internal_tape)
            else
                adjoint(Const(runtime_mixed_call), Const(Val(active_refs)), dupClosure0 ? Duplicated(f, df) : Const(f), args..., tape.internal_tape)
            end
            nothing
        else

            annotation = if $Width != 1 && annotation0 <: Duplicated
                BatchDuplicated{rt, $Width}
            else
                annotation0
            end

            world = codegen_world_age(FT, tt)

            _, adjoint = thunk(Val(world), dupClosure0 ? Duplicated{FT} : Const{FT},
                                     annotation, Tuple{$(Types...)}, Val(API.DEM_ReverseModePrimal), width,
                                     ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

            if tape.shadow_return !== nothing
                if !(annotation0 <: Active) && nonzero_active_data(($shadowret,))
                    ET = ($(ElTypes...),)
                    throw(AssertionError("Shadow value "*string(($shadowret,))*" returned from type unstable call to $f($(ET...)) has mixed internal activity types. See https://enzyme.mit.edu/julia/stable/faq/#Mixed-activity for more information"))
                end
            end
            tup = if annotation0 <: Active
                adjoint(dupClosure0 ? Duplicated(f, df) : Const(f), args..., $shadowret, tape.internal_tape)[1]
            else
                adjoint(dupClosure0 ? Duplicated(f, df) : Const(f), args..., tape.internal_tape)[1]
            end

            $(outs...)
        end
        return nothing
    end
end

function func_runtime_generic_rev(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, _, active_refs = setup_macro_wraps(false, N, Width)
    body = body_runtime_generic_rev(N, Width, wrapped, primtypes, batchshadowargs, active_refs)

    quote
        function runtime_generic_rev(activity::Type{Val{ActivityTup}}, width::Val{$Width}, ModifiedBetween::Val{MB}, tape::TapeType, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, TapeType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_generic_rev(activity::Type{Val{ActivityTup}}, width::Val{Width}, ModifiedBetween::Val{MB}, tape::TapeType, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, TapeType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, batchshadowargs, _, active_refs = setup_macro_wraps(false, N, Width, :allargs)
    return body_runtime_generic_rev(N, Width, wrapped, primtypes, batchshadowargs, active_refs)
end

@inline concat() = ()
@inline concat(a) = a
@inline concat(a, b) = (a..., b...)
@inline concat(a, b, c...) = concat(concat(a, b), c...)

@inline iterate_unwrap_inner_fwd(x::Const) = (map(Const, x.val)...,)
@inline iterate_unwrap_inner_fwd(x::Duplicated) = (map(Duplicated, x.val, x.dval)...,)
@inline batch_dup_tuple(x, vals...) = BatchDuplicated(x, (vals...,))
@inline iterate_unwrap_inner_fwd(x::BatchDuplicated) = (map(batch_dup_tuple, x.val, x.dval...)...,)

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

@inline function iterate_unwrap_augfwd_dup(::Val{forwardMode}, args, dargs) where forwardMode
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        if guaranteed_const(ty)
            Const(arg)
        elseif !forwardMode && active_reg(ty)
            Active(arg)
        else
            Duplicated(arg, dargs[i])
        end
    end
end

@inline function iterate_unwrap_augfwd_batchdup(::Val{forwardMode}, ::Val{Width}, args, dargs) where {forwardMode, Width}
    ntuple(Val(length(args))) do i
        Base.@_inline_meta
        arg = args[i]
        ty = Core.Typeof(arg)
        if guaranteed_const(ty)
            Const(arg)
        elseif !forwardMode && active_reg(ty)
            Active(arg)
        else
            BatchDuplicated(arg, ntuple(Val(Width)) do j
                Base.@_inline_meta
                dargs[j][i]
            end)
        end
    end
end

@inline function allFirst(::Val{Width}, res) where Width
    ntuple(Val(Width)) do i
        Base.@_inline_meta
        res[1]
    end
end

@inline function allSame(::Val{Width}, res) where Width
    ntuple(Val(Width)) do i
        Base.@_inline_meta
        res
    end
end

@inline function allZero(::Val{Width}, res) where Width
    ntuple(Val(Width)) do i
        Base.@_inline_meta
        Ref(make_zero(res))
    end
end

# This is explicitly escaped here to be what is apply generic in total [and thus all the insides are stable]
function fwddiff_with_return(::Val{width}, ::Val{dupClosure0}, ::Type{ReturnType}, ::Type{FT}, ::Type{tt′}, f::FT, df::DF, args::Vararg{Annotation, Nargs})::ReturnType where {width, dupClosure0, ReturnType, FT, tt′, DF, Nargs}
    ReturnPrimal = Val(true)
    ModifiedBetween = Val(Enzyme.falses_from_args(Nargs+1))

    dupClosure = dupClosure0 && !guaranteed_const(FT)
    FA = dupClosure ? Duplicated{FT} : Const{FT}

    tt    = Enzyme.vaEltypes(tt′)

    rt = Core.Compiler.return_type(f, tt)
    annotation0 = guess_activity(rt, API.DEM_ForwardMode)

    annotation = if width != 1
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            BatchDuplicated{rt, width}
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
    res = thunk(Val(world), FA, annotation, tt′, #=Mode=# Val(API.DEM_ForwardMode), Val(width),
                                     ModifiedBetween, ReturnPrimal, #=ShadowInit=#Val(false), FFIABI)(fa, args...)
    return if annotation <: Const
        ReturnType(allFirst(Val(width+1), res))
    else
        if width == 1
            ReturnType((res[1], res[2]))
        else
            ReturnType((res[1], res[2]...))
        end
    end
end

function body_runtime_iterate_fwd(N, Width, wrapped, primtypes)
    wrappedexexpand = ntuple(i->:($(wrapped[i])...), Val(N))
    return quote
        args = ($(wrappedexexpand...),)
        tt′    = Enzyme.vaTypeof(args...)
        FT = Core.Typeof(f)
        fwddiff_with_return(Val($Width), Val(ActivityTup[1]), ReturnType, FT, tt′, f, df, args...)::ReturnType
    end
end

function func_runtime_iterate_fwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, _, active_refs = setup_macro_wraps(true, N, Width, #=base=#nothing, #=iterate=#true)
    body = body_runtime_iterate_fwd(N, Width, wrapped, primtypes)

    quote
        function runtime_iterate_fwd(activity::Type{Val{ActivityTup}}, width::Val{$Width}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_iterate_fwd(activity::Type{Val{ActivityTup}}, width::Val{Width}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, _, _, active_refs = setup_macro_wraps(true, N, Width, :allargs, #=iterate=#true)
    return body_runtime_iterate_fwd(N, Width, wrapped, primtypes)
end

function primal_tuple(args::Vararg{Annotation, Nargs}) where Nargs
    ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        args[i].val
    end
end

function shadow_tuple(::Val{1}, args::Vararg{Annotation, Nargs}) where Nargs
    ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        @assert !(args[i] isa Active)
        if args[i] isa Const
            args[i].val
        else 
            args[i].dval
        end
    end
end

function shadow_tuple(::Val{width}, args::Vararg{Annotation, Nargs}) where {width, Nargs}
    ntuple(Val(width)) do w
    ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        @assert !(args[i] isa Active)
        if args[i] isa Const
            args[i].val
        else 
            args[i].dval[w]
        end
    end
    end
end

# This is explicitly escaped here to be what is apply generic in total [and thus all the insides are stable]
function augfwd_with_return(::Val{width}, ::Val{dupClosure0}, ::Type{ReturnType}, ::Val{ModifiedBetween0}, ::Type{FT}, ::Type{tt′}, f::FT, df::DF, args::Vararg{Annotation, Nargs})::ReturnType where {width, dupClosure0, ReturnType, ModifiedBetween0, FT, tt′, DF, Nargs}
    ReturnPrimal = Val(true)
    ModifiedBetween = Val(ModifiedBetween0)

    tt    = Enzyme.vaEltypes(tt′)
    rt = Core.Compiler.return_type(f, tt)
    annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

    annotation = if width != 1
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            BatchDuplicated{rt, width}
        elseif annotation0 <: Active
            Active{rt}
        else
            Const{rt}
        end
    else
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            Duplicated{rt}
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
        forward, adjoint = thunk(Val(world), FA,
                                 annotation, tt′, Val(API.DEM_ReverseModePrimal), Val(width),
                                 ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)
        forward(fa, args...)
    else
        nothing, primal_tuple(args...), annotation <: Active ? nothing : shadow_tuple(Val(width), args...)
    end

    resT = typeof(origRet)
    if annotation <: Const
        shadow_return = nothing
        tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
        return ReturnType((allSame(Val(width+1), origRet)..., tape))
    elseif annotation <: Active
        if width == 1
            shadow_return = Ref(make_zero(origRet))
        else
            shadow_return = allZero(Val(width), origRet)
        end
        tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
        if width == 1
            return ReturnType((origRet, shadow_return, tape))
        else
            return ReturnType((origRet, shadow_return..., tape))
        end
    end

    @assert annotation <: Duplicated || annotation <: DuplicatedNoNeed || annotation <: BatchDuplicated || annotation <: BatchDuplicatedNoNeed

    shadow_return = nothing
    tape = Tape{typeof(internal_tape), typeof(shadow_return), resT}(internal_tape, shadow_return)
    if width == 1
        return ReturnType((origRet, initShadow, tape))
    else
        return ReturnType((origRet, initShadow..., tape))
    end
end

function body_runtime_iterate_augfwd(N, Width, modbetween, wrapped, primtypes)
    wrappedexexpand = ntuple(i->:($(wrapped[i])...), Val(N))
    return quote
        args = ($(wrappedexexpand...),)
        tt′    = Enzyme.vaTypeof(args...)
        FT = Core.Typeof(f)
        augfwd_with_return(Val($Width), Val(ActivityTup[1]), ReturnType, Val(concat($(modbetween...))), FT, tt′, f, df, args...)::ReturnType
    end
end

function func_runtime_iterate_augfwd(N, Width)
    _, _, primtypes, allargs, typeargs, wrapped, _, modbetween, active_refs = setup_macro_wraps(false, N, Width, #=base=#nothing, #=iterate=#true)
    body = body_runtime_iterate_augfwd(N, Width, modbetween, wrapped, primtypes)

    quote
        function runtime_iterate_augfwd(activity::Type{Val{ActivityTup}}, width::Val{$Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, ReturnType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_iterate_augfwd(activity::Type{Val{ActivityTup}}, width::Val{Width}, ModifiedBetween::Val{MB}, RT::Val{ReturnType}, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, ReturnType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    _, _, primtypes, _, _, wrapped, _ , modbetween, active_refs = setup_macro_wraps(false, N, Width, :allargs, #=iterate=#true)
    return body_runtime_iterate_augfwd(N, Width, modbetween, wrapped, primtypes)
end



# This is explicitly escaped here to be what is apply generic in total [and thus all the insides are stable]
function rev_with_return(::Val{width}, ::Val{dupClosure0}, ::Val{ModifiedBetween0}, ::Val{lengths}, ::Type{FT}, ::Type{tt′}, f::FT, df::DF, tape, shadowargs, args::Vararg{Annotation, Nargs})::Nothing where {width, dupClosure0, ModifiedBetween0, lengths, FT, tt′, DF, Nargs}
    ReturnPrimal = Val(true)
    ModifiedBetween = Val(ModifiedBetween0)

    dupClosure = dupClosure0 && !guaranteed_const(FT)
    FA = dupClosure ? Duplicated{FT} : Const{FT}

    tt    = Enzyme.vaEltypes(tt′)

    rt = Core.Compiler.return_type(f, tt)
    annotation0 = guess_activity(rt, API.DEM_ReverseModePrimal)

    annotation = if width != 1
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            BatchDuplicated{rt, width}
        elseif annotation0 <: Active
            Active{rt}
        else
            Const{rt}
        end
    else
        if annotation0 <: DuplicatedNoNeed || annotation0 <: Duplicated
            Duplicated{rt}
        elseif annotation0 <: Active
            Active{rt}
        else
            Const{rt}
        end
    end

    tup = if f != Base.tuple
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
        forward, adjoint = thunk(Val(world), FA,
                                 annotation, tt′, Val(API.DEM_ReverseModePrimal), Val(width),
                                 ModifiedBetween, #=returnPrimal=#Val(true), #=shadowInit=#Val(false), FFIABI)

        args2 = if tape.shadow_return !== nothing
            if width == 1
                (args..., tape.shadow_return[])
            else
                (args..., ntuple(Val(width)) do w
                    Base.@_inline_meta
                    tape.shadow_return[w][]
                end)
            end
        else
            args
        end

        adjoint(fa, args2..., tape.internal_tape)[1]
    else
        ntuple(Val(Nargs)) do i
            Base.@_inline_meta
            if args[i] isa Active
                if width == 1
                    tape.shadow_return[][i]
                else
                    ntuple(Val(width)) do w
                        Base.@_inline_meta
                        tape.shadow_return[w][][i]
                    end
                end
            else
                nothing
            end
        end
    end

    ntuple(Val(Nargs)) do i
        Base.@_inline_meta

        ntuple(Val(width)) do w
            Base.@_inline_meta

            if tup[i] == nothing
            else
                expr = if width == 1
                    tup[i]
                else
                    tup[i][w]
                end
                idx_of_vec, idx_in_vec = lengths[i]
                vec = @inbounds shadowargs[idx_of_vec][w]
                if vec isa Base.RefValue
                    vecld = vec[]                    
                    T = Core.Typeof(vecld)
                    vec[] = splatnew(T, ntuple(Val(fieldcount(T))) do i
                        Base.@_inline_meta
                        prev = getfield(vecld, i)
                        if i == idx_in_vec
                            recursive_add(prev, expr)
                        else
                            prev
                        end
                    end)
                else
                    val = @inbounds vec[idx_in_vec]
                    if val isa Base.RefValue
                        val[] = recursive_add(val[], expr)
                    elseif ismutable(vec)
                        @inbounds vec[idx_in_vec] = recursive_add(val, expr)
                    else
                      error("Enzyme Mutability Error: Cannot in place to immutable value vec[$idx_in_vec] = $val, vec=$vec")
                    end
                end
            end

            nothing
        end

        nothing
    end
    nothing
end

function body_runtime_iterate_rev(N, Width, modbetween, wrapped, primargs, shadowargs)
    outs = []
    for i in 1:N
        for w in 1:Width
            expr = if Width == 1
                :(tup[$i])
            else
                :(tup[$i][$w])
            end
            shad = shadowargs[i][w]
            out = :(if tup[$i] === nothing
              elseif $shad isa Base.RefValue
                  $shad[] = recursive_add($shad[], $expr)
                else
                  error("Enzyme Mutability Error: Cannot add in place to immutable value "*string($shad))
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

    wrappedexexpand = ntuple(i->:($(wrapped[i])...), Val(N))
    lengths = ntuple(i->quote
        (ntuple(Val(length($(primargs[i])))) do j
            Base.@_inline_meta
            ($i, j)
        end)
    end, Val(N))

    shadowsplat = Expr[]
    for s in shadowargs
        push!(shadowsplat, :(($(s...),)))
    end
    quote
        args = ($(wrappedexexpand...),)
        tt′    = Enzyme.vaTypeof(args...)
        FT = Core.Typeof(f)
        rev_with_return(Val($Width), Val(ActivityTup[1]), Val(concat($(modbetween...))), Val(concat($(lengths...))), FT, tt′, f, df, tape, ($(shadowsplat...),), args...)
        return nothing
    end
end

function func_runtime_iterate_rev(N, Width)
    primargs, _, primtypes, allargs, typeargs, wrapped, batchshadowargs, modbetween = setup_macro_wraps(false, N, Width, #=body=#nothing, #=iterate=#true)
    body = body_runtime_iterate_rev(N, Width, modbetween, wrapped, primargs, batchshadowargs)

    quote
        function runtime_iterate_rev(activity::Type{Val{ActivityTup}}, width::Val{$Width}, ModifiedBetween::Val{MB}, tape::TapeType, f::F, df::DF, $(allargs...)) where {ActivityTup, MB, TapeType, F, DF, $(typeargs...)}
            $body
        end
    end
end

@generated function runtime_iterate_rev(activity::Type{Val{ActivityTup}}, width::Val{Width}, ModifiedBetween::Val{MB}, tape::TapeType, f::F, df::DF, allargs...) where {ActivityTup, MB, Width, TapeType, F, DF}
    N = div(length(allargs)+2, Width+1)-1
    primargs, _, primtypes, _, _, wrapped, batchshadowargs, modbetween, active_refs = setup_macro_wraps(false, N, Width, :allargs, #=iterate=#true)
    return body_runtime_iterate_rev(N, Width, modbetween, wrapped, primargs, batchshadowargs)
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

function generic_setup(orig, func, ReturnType, gutils, start, B::LLVM.IRBuilder,  lookup; sret=nothing, tape=nothing, firstconst=false, endcast=true)
    width = get_width(gutils)
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    ops = collect(operands(orig))[start+firstconst:end-1]

    T_int8 = LLVM.Int8Type()
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    ActivityList = LLVM.Value[]

    @assert length(ops) != 0
    fill_val = unsafe_to_llvm(nothing)

    vals = LLVM.Value[]

    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

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
    
    if tape === nothing && endcast
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

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)

    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
    if unsafe_load(shadowR) != C_NULL
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

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
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

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return true
    end

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))
    
     if unsafe_load(shadowR) != C_NULL
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

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)

    if normalR != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
        unsafe_store!(normalR, normal.ref)
    else
        # Delete the primal code
        ni = new_from_original(gutils, orig)
        erase_with_placeholder(gutils, ni, orig)
    end
    return false
end

function generic_augfwd(B, orig, gutils, normalR, shadowR, tapeR)
    conv = LLVM.callconv(orig)
    # https://github.com/JuliaLang/julia/blob/5162023b9b67265ddb0bbbc0f4bd6b225c429aa0/src/codegen_shared.h#L20

    @assert conv == 37

    common_generic_augfwd(1, B, orig, gutils, normalR, shadowR, tapeR)
end

function common_generic_rev(offset, B, orig, gutils, tape)::Cvoid
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return nothing
    end

    @assert tape !== C_NULL
    width = get_width(gutils)
    generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset, B, true; tape)
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
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return true
    end
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+1, B, false)

    if unsafe_load(shadowR) != C_NULL
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

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
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
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return true
    end

    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))
    # sret = generic_setup(orig, runtime_apply_latest_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, ctx, B, false)
    sret = generic_setup(orig, runtime_generic_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+1, B, false)

    if unsafe_load(shadowR) != C_NULL
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

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
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
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return nothing
    end
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
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
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

    if v && isiter == Base.iterate
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        sret = generic_setup(orig, runtime_iterate_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+2, B, false)
        AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))
        if unsafe_load(shadowR) != C_NULL
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

        if unsafe_load(normalR) != C_NULL
            normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
            unsafe_store!(normalR, normal.ref)
        else
            # Delete the primal code
            ni = new_from_original(gutils, orig)
            erase_with_placeholder(gutils, ni, orig)
        end
        return false
    end

    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate "*string((v, v2, isiter, istup, length(operands(orig)), offset+4)))

    return false
end

function common_apply_iterate_augfwd(offset, B, orig, gutils, normalR, shadowR, tapeR)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return true
    end

    v, isiter = absint(operands(orig)[offset+1])
    v2, istup = absint(operands(orig)[offset+2])

    width = get_width(gutils)

    if v && isiter == Base.iterate
        T_jlvalue = LLVM.StructType(LLVMType[])
        T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

        sret = generic_setup(orig, runtime_iterate_augfwd, AnyArray(2+Int(width)), gutils, #=start=#offset+2, B, false)
        AT = LLVM.ArrayType(T_prjlvalue, 2+Int(width))

         if unsafe_load(shadowR) != C_NULL
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

        tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
        unsafe_store!(tapeR, tape.ref)

        if normalR != C_NULL
            normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
            unsafe_store!(normalR, normal.ref)
        else
            # Delete the primal code
            ni = new_from_original(gutils, orig)
            erase_with_placeholder(gutils, ni, orig)
        end
        return false
        return false
    end

    emit_error(B, orig, "Enzyme: Not yet implemented augmented forward for jl_f__apply_iterate "*string((v, v2, isiter, istup, length(operands(orig)), offset+4)))

    unsafe_store!(shadowR,UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))).ref)
    return false
end

function common_apply_iterate_rev(offset, B, orig, gutils, tape)
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return nothing
    end

    @assert tape !== C_NULL
    width = get_width(gutils)
    generic_setup(orig, runtime_iterate_rev, Nothing, gutils, #=start=#offset+2, B, true; tape)
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
    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return true
    end
    
    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    width = get_width(gutils)
    sret = generic_setup(orig, runtime_generic_fwd, AnyArray(1+Int(width)), gutils, #=start=#offset+1, B, false)
    AT = LLVM.ArrayType(T_prjlvalue, 1+Int(width))

    if unsafe_load(shadowR) != C_NULL
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

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
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
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
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

    if unsafe_load(shadowR) != C_NULL
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

    tape = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(1+width)]))
    unsafe_store!(tapeR, tape.ref)

    if unsafe_load(normalR) != C_NULL
        normal = LLVM.load!(B, T_prjlvalue, LLVM.inbounds_gep!(B, AT, sret, [LLVM.ConstantInt(0), LLVM.ConstantInt(0)]))
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
    activep = API.EnzymeGradientUtilsGetReturnDiffeType(gutils, orig, needsPrimalP, needsShadowP, get_mode(gutils))

    if (is_constant_value(gutils, orig) || needsShadowP[] == 0 ) && is_constant_inst(gutils, orig)
        return nothing
    end
    
    width = get_width(gutils)
    generic_setup(orig, runtime_generic_rev, Nothing, gutils, #=start=#offset+1, B, true; tape)

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
