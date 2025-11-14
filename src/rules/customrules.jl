import LinearAlgebra

@inline add_fwd(prev, post) = recursive_add(prev, post)

@generated function EnzymeCore.EnzymeRules.multiply_fwd_into(prev, partial::Union{AbstractArray,Number}, dx::Union{AbstractArray,Number})
    if partial <: Number || dx isa Number
        if !(prev <: Type)
            return quote
                Base.@_inline_meta
                add_fwd(prev, EnzymeCore.EnzymeRules.multiply_fwd_into(Core.Typeof(prev), partial, dx))
            end
        end
        return quote
            Base.@_inline_meta
            prev(partial * dx)
        end
    end

    @assert partial <: AbstractArray
    if dx <: Number
        if !(prev <: Type)
    	    return quote
    		    Base.@_inline_meta
    		    LinearAlgebra.axpy!(dx, partial, prev)
    		    prev
    	    end
    	else
    	    return quote
    		    Base.@_inline_meta
    		    prev(partial * dx)
    	    end
    	end
    end
    @assert dx <: AbstractArray
    N = ndims(partial)
    M = ndims(dx)

    if N == M
        if !(prev <: Type)
            return quote
                Base.@_inline_meta
                add_fwd(prev, EnzymeCore.EnzymeRules.multiply_fwd_into(typeof(prev), partial, dx))
            end
        end

        res = if partial <: AbstractFloat || partial <: AbstractArray{<:AbstractFloat}
            :(LinearAlgebra.dot(partial,dx))
        elseif dx <: AbstractFloat || dx <: AbstractArray{<:AbstractFloat}
            :(LinearAlgebra.dot(dx, partial))
        elseif partial <: AbstractVector
            :(LinearAlgebra.dot(adjoint(partial),dx))
        else
            :(LinearAlgebra.dot(conj(partial),dx))
        end
        return quote
            Base.@_inline_meta
            prev($res)
        end
    end

    if N < M
        return quote
            Base.@_inline_meta
            throw(MethodError(EnzymeCore.EnzymeRules.multiply_fwd_into, (prev, partial, dx)))
        end
    end

    init = if prev <: Type
        :(prev = similar(prev, size(partial)[1:$(N-M)]...))
    end

    idxs = Symbol[]
    for i in 1:(N-M)
        push!(idxs, Symbol("i_$i"))
    end
    others = Symbol[]
    for i in 1:M
        push!(others, :(:))
    end

    outp = :prev
    if N-M != 1
        outp = Expr(:call, Base.reshape, outp, Expr(:call, Base.length, outp))
    end
    inp = :dx
    if M != 1
        inp = Expr(:call, Base.reshape, inp, Expr(:call, Base.length, inp))
    end

    matp = :partial
    if N-M != 1 || M != 1
        matp = Expr(:call, Base.reshape, matp, Expr(:call, Base.length, outp), Expr(:call, Base.length, inp))
    end

    outexpr = if prev <: Type
        Expr(:call, LinearAlgebra.mul!, outp, matp, inp)
    else
        Expr(:call, LinearAlgebra.mul!, outp, matp, inp, true, true)
    end

    quote
        Base.@_inline_meta
        @assert size(partial)[$(N-M+1):end] == size(dx)
        $init
        @inbounds $outexpr
        return prev
    end
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::Real, dx)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, partial, dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::Complex, dx)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, conj(partial), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractArray{<:Real}, dx::Number)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, partial, dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractArray{<:Complex}, dx::Number)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, conj(partial), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractArray{<:Real, N}, dx::AbstractArray{<:Any, N}) where N
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, partial, dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractArray{<:Complex, N}, dx::AbstractArray{<:Any, N}) where N
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, conj(partial), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractVector{<:Complex}, dx::AbstractVector{<:Any})
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, adjoint(partial), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractMatrix{<:Real}, dx::AbstractVector)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, transpose(partial), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractMatrix{<:Complex}, dx::AbstractVector)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, adjoint(partial), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractArray{<:Real}, dx::AbstractArray)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, Base.permutedims(partial, (((ndims(dx)+1):ndims(partial))..., Base.OneTo(ndims(dx))...)), dx)
end

@inline function EnzymeCore.EnzymeRules.multiply_rev_into(prev, partial::AbstractArray{<:Complex}, dx::AbstractArray)
    pd = Base.permutedims(partial, (((ndims(dx)+1):ndims(partial))..., Base.OneTo(ndims(dx))...))
    Base.conj!(pd)
    EnzymeCore.EnzymeRules.multiply_fwd_into(prev, pd, dx)
end

function enzyme_custom_setup_args(
    @nospecialize(B::Union{Nothing, LLVM.IRBuilder}),
    orig::LLVM.CallInst,
    gutils::GradientUtils,
    mi::Core.MethodInstance,
    @nospecialize(RT::Type),
    reverse::Bool,
    isKWCall::Bool,
)
    ops = collect(operands(orig))
    called = ops[end]
    ops = ops[1:end-1]
    width = get_width(gutils)
    kwtup = nothing

    args = LLVM.Value[]
    activity = Type[]
    overwritten = Bool[]

    actives = LLVM.Value[]

    mixeds = Tuple{LLVM.Value,Type,LLVM.Value}[]
    uncacheable = get_uncacheable(gutils, orig)
    mode = get_mode(gutils)

    retRemoved, parmsRemoved = removed_ret_parms(orig)

    @assert length(parmsRemoved) == 0

    _, sret, returnRoots = get_return_info(RT)
    sret = sret !== nothing
    returnRoots = returnRoots !== nothing

    cv = LLVM.called_operand(orig)
    swiftself = has_swiftself(cv)
    jlargs = classify_arguments(
        mi.specTypes,
        called_type(orig),
        sret,
        returnRoots,
        swiftself,
        parmsRemoved,
    )

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    ofn = LLVM.parent(LLVM.parent(orig))
    world = enzyme_extract_world(ofn)

    for arg in jlargs
        @assert arg.cc != RemovedParam
        if arg.cc == GPUCompiler.GHOST
            @assert guaranteed_const_nongen(arg.typ, world)
            if isKWCall && arg.arg_i == 2
                Ty = arg.typ
                kwtup = Ty
                continue
            end
            push!(activity, Const{arg.typ})
            # Don't push overwritten for Core.kwcall
            if !(isKWCall && arg.arg_i == 1)
                push!(overwritten, false)
            end
            if B !== nothing
                if Core.Compiler.isconstType(arg.typ) &&
                   !Core.Compiler.isconstType(Const{arg.typ})
                    llty = convert(LLVMType, Const{arg.typ})
                    al0 = al = emit_allocobj!(B, Const{arg.typ})
                    al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                    al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

                    ptr = inbounds_gep!(
                        B,
                        llty,
                        al,
                        [
                            LLVM.ConstantInt(LLVM.IntType(64), 0),
                            LLVM.ConstantInt(LLVM.IntType(32), 0),
                        ],
                    )
                    val = unsafe_to_llvm(B, arg.typ.parameters[1])
                    store!(B, val, ptr)

                    if any_jltypes(llty)
                        emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
                    end
                    push!(args, al)
                else
                    @assert isghostty(Const{arg.typ}) ||
                            Core.Compiler.isconstType(Const{arg.typ})
                end
            end
            continue
        end
        @assert !(isghostty(arg.typ) || Core.Compiler.isconstType(arg.typ))

        op = ops[arg.codegen.i]
        # Don't push the keyword args to uncacheable
        if !(isKWCall && arg.arg_i == 2)
            push!(overwritten, uncacheable[arg.codegen.i] != 0)
        end

        val = new_from_original(gutils, op)
        if reverse && B !== nothing
            val = lookup_value(gutils, val, B)
        end

        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, op, false) #=isforeign=#

        if isKWCall && arg.arg_i == 2
            Ty = arg.typ

            if EnzymeRules.is_inactive_kwarg_from_sig(Interpreter.simplify_kw(mi.specTypes); world)
                activep = API.DFT_CONSTANT
            end

            push!(args, val)

            # Only constant kw arg tuple's are currently supported
            if activep == API.DFT_CONSTANT
                kwtup = Ty
            else
                @assert activep == API.DFT_DUP_ARG
                kwtup = Duplicated{Ty}
            end
            continue
        end

        # TODO type analysis deduce if duplicated vs active
        if activep == API.DFT_CONSTANT
            Ty = Const{arg.typ}
            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed = true)
            if B !== nothing
                al0 = al = emit_allocobj!(B, Ty)
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

                ptr = inbounds_gep!(
                    B,
                    llty,
                    al,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), 0),
                    ],
                )
                
                if !is_opaque(value_type(ptr))
                    @assert eltype(value_type(ptr)) == arty
                end

                if value_type(val) != arty
                    val = load!(B, arty, val)
                end
                store!(B, val, ptr)

                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
                end

                push!(args, al)
            end

            push!(activity, Ty)

        elseif activep == API.DFT_OUT_DIFF || (
            mode != API.DEM_ForwardMode &&
            active_reg(arg.typ, world) == ActiveState
        )
            Ty = Active{arg.typ}
            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed = true)
            if B !== nothing
                al0 = al = emit_allocobj!(B, Ty)
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

                ptr = inbounds_gep!(
                    B,
                    llty,
                    al,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), 0),
                    ],
                )

                if !is_opaque(value_type(ptr))
                    @assert eltype(value_type(ptr)) == arty
                end

                if value_type(val) != arty
                    if overwritten[end]
                        bt = GPUCompiler.backtrace(orig)
                        msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
                        emit_error(
                            B,
                            orig,
                            "Enzyme: active by ref type $Ty is overwritten in application of custom rule for $mi val=$(string(val)) ptr=$(string(ptr)). " *
                            "As a workaround until support for this is added, try passing values as separate arguments rather than as an aggregate of type $Ty.\n"*msg2,
                        )
                    end

                    if !LLVM.is_opaque(value_type(val))
                        if arty != eltype(value_type(val))
                            msg = sprint() do io
                                println(io, "Enzyme: active by ref type $Ty is wrong type in application of custom rule for $mi val=$(string(val)) ptr=$(string(ptr)) arty=$arty")
                            end

                            EnzymeInternalError(msg, ir, bt)
                        end
                    end

                    val = load!(B, arty, val)
                end

                if arty == value_type(val)
                    store!(B, val, ptr)
                    if any_jltypes(llty)
                        emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
                    end
                else
                    bt = GPUCompiler.backtrace(orig)
                    msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
                    emit_error(
                        B,
                        orig,
                        "Enzyme: active by ref type $Ty is wrong store type in application of custom rule for $mi val=$(string(val)) ptr=$(string(ptr))\n"*msg2,
                    )
                end

                push!(args, al)
            end

            push!(activity, Ty)
            push!(actives, op)
        else
            if B !== nothing
                ival = invert_pointer(gutils, op, B)
                if reverse
                    ival = lookup_value(gutils, ival, B)
                end
            end
            shadowty = arg.typ
            mixed = false
            if width == 1

                if active_reg(arg.typ, world) == MixedState
                    # TODO mixedupnoneed
                    shadowty = Base.RefValue{shadowty}
                    Ty = MixedDuplicated{arg.typ}
                    mixed = true
                else
                    if activep == API.DFT_DUP_ARG
                        Ty = Duplicated{arg.typ}
                    else
                        @assert activep == API.DFT_DUP_NONEED
                        Ty = DuplicatedNoNeed{arg.typ}
                    end
                end
            else
                if active_reg(arg.typ, world) == MixedState
                    # TODO batchmixedupnoneed
                    shadowty = Base.RefValue{shadowty}
                    Ty = BatchMixedDuplicated{arg.typ,Int(width)}
                    mixed = true
                else
                    if activep == API.DFT_DUP_ARG
                        Ty = BatchDuplicated{arg.typ,Int(width)}
                    else
                        @assert activep == API.DFT_DUP_NONEED
                        Ty = BatchDuplicatedNoNeed{arg.typ,Int(width)}
                    end
                end
            end

            llty = convert(LLVMType, Ty)
            arty = convert(LLVMType, arg.typ; allow_boxed = true)
            iarty = convert(LLVMType, shadowty; allow_boxed = true)
            sarty = LLVM.LLVMType(API.EnzymeGetShadowType(width, arty))
            siarty = LLVM.LLVMType(API.EnzymeGetShadowType(width, iarty))
            if B !== nothing
                al0 = al = emit_allocobj!(B, Ty)
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

                ptr = inbounds_gep!(
                    B,
                    llty,
                    al,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), 0),
                    ],
                )
                needsload = false

                if !is_opaque(value_type(ptr))
                    @assert eltype(value_type(ptr)) == arty
                end

                if value_type(val) != arty
                    val = load!(B, arty, val)
                    if !mixed
                        ptr_val = ival
                        ival = UndefValue(siarty)
                        for idx = 1:width
                            ev =
                                (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                            ld = load!(B, iarty, ev)
                            ival = (width == 1) ? ld : insert_value!(B, ival, ld, idx - 1)
                        end
                    end
                    needsload = true
                end
                store!(B, val, ptr)

                iptr = inbounds_gep!(
                    B,
                    llty,
                    al,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), 1),
                    ],
                )

                if mixed
                    RefTy = arg.typ
                    if width != 1
                        RefTy = NTuple{Int(width),RefTy}
                    end
                    llrty = convert(LLVMType, RefTy)
                    RefTy = Base.RefValue{RefTy}
                    refal0 = refal = emit_allocobj!(B, RefTy)
                    refal = bitcast!(
                        B,
                        refal,
                        LLVM.PointerType(llrty, addrspace(value_type(refal))),
                    )

                    @assert needsload
                    ptr_val = ival
                    ival = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llrty)))
                    for idx = 1:width
                        ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                        ld = load!(B, llrty, ev)
                        ival = (width == 1) ? ld : insert_value!(B, ival, ld, idx - 1)
                    end
                    store!(B, ival, refal)
                    emit_writebarrier!(B, get_julia_inner_types(B, refal0, ival))
                    ival = refal0
                    push!(mixeds, (ptr_val, arg.typ, refal))
                end

                store!(B, ival, iptr)

                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, val, ival))
                end

                push!(args, al)
            end
            push!(activity, Ty)
        end

    end
    return args, activity, (overwritten...,), actives, kwtup, mixeds
end

function enzyme_custom_setup_ret(
    gutils::GradientUtils,
    orig::LLVM.CallInst,
    mi::Core.MethodInstance,
    @nospecialize(RealRt::Type),
    @nospecialize(B::Union{LLVM.IRBuilder,Nothing})
)
    width = get_width(gutils)
    mode = get_mode(gutils)

    world = enzyme_extract_world(LLVM.parent(LLVM.parent(orig)))

    needsShadowP = Ref{UInt8}(0)
    needsPrimalP = Ref{UInt8}(0)

    # Conditionally use the get return. This is done because EnzymeGradientUtilsGetReturnDiffeType
    # calls differential use analysis to determine needsprimal/shadow. However, since now this function
    # is used as part of differential use analysis, we need to avoid an ininite recursion. Thus use
    # the version without differential use if actual unreachable results are not available anyways.
    uncacheable = Vector{UInt8}(undef, length(collect(LLVM.operands(orig))) - 1)
    cmode = mode
    if cmode == API.DEM_ReverseModeGradient
        cmode = API.DEM_ReverseModePrimal
    end
    activep =
        if mode == API.DEM_ForwardMode ||
           API.EnzymeGradientUtilsGetUncacheableArgs(
            gutils,
            orig,
            uncacheable,
            length(uncacheable),
        ) == 1
            API.EnzymeGradientUtilsGetReturnDiffeType(
                gutils,
                orig,
                needsPrimalP,
                needsShadowP,
                cmode,
            )
        else
            actv = API.EnzymeGradientUtilsGetDiffeType(gutils, orig, false)
            if !isghostty(RealRt)
                needsPrimalP[] = 1
                if actv == API.DFT_DUP_ARG || actv == API.DFT_DUP_NONEED
                    needsShadowP[] = 1
                end
            end
            actv
        end
    needsPrimal = needsPrimalP[] != 0
    origNeedsPrimal = needsPrimal
    _, sret, _ = get_return_info(RealRt)
    if sret !== nothing
        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, operands(orig)[1], false) #=isforeign=#
        needsPrimal = activep == API.DFT_DUP_ARG || activep == API.DFT_CONSTANT
        needsShadowP[] = activep == API.DFT_DUP_ARG || activep == API.DFT_DUP_NONEED
    end

    if !needsPrimal && activep == API.DFT_DUP_ARG
        activep = API.DFT_DUP_NONEED
    end

    if activep == API.DFT_CONSTANT
        RT = Const{RealRt}

    elseif activep == API.DFT_OUT_DIFF || (
        mode != API.DEM_ForwardMode &&
        !guaranteed_nonactive(RealRt, world)
    )
        if active_reg(RealRt, world) == MixedState && B !== nothing        
            bt = GPUCompiler.backtrace(orig)
            msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))            
            mi, _ = enzyme_custom_extract_mi(orig)
            emit_error(
                B,
                orig,
                (msg2, mi, world),
                MixedReturnException{RealRt}
            )
        end
        RT = Active{RealRt}

    elseif activep == API.DFT_DUP_ARG
        if width == 1
            RT = Duplicated{RealRt}
        else
            RT = BatchDuplicated{RealRt,Int(width)}
        end
    else
        @assert activep == API.DFT_DUP_NONEED
        if width == 1
            RT = DuplicatedNoNeed{RealRt}
        else
            RT = BatchDuplicatedNoNeed{RealRt,Int(width)}
        end
    end
    return RT, needsPrimal, needsShadowP[] != 0, origNeedsPrimal
end

function custom_rule_method_error(world::UInt, @nospecialize(fn), @nospecialize(args::Vararg))
    throw(MethodError(fn, (args...,), world))
end

@register_fwd function enzyme_custom_fwd(B::LLVM.IRBuilder, orig::LLVM.CallInst, gutils::GradientUtils, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef})
    if is_constant_value(gutils, orig) &&
       is_constant_inst(gutils, orig) &&
       !has_rule(orig, gutils)
        return false
    end

    width = get_width(gutils)

    if shadowR != C_NULL
        unsafe_store!(
            shadowR,
            UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))).ref,
        )
    end

    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)

    # TODO: don't inject the code multiple times for multiple calls

    fmi, (args, TT, fwd_RT, kwtup, RT, needsPrimal, RealRt, origNeedsPrimal, activity, C) = fwd_mi(orig, gutils, B)

    if kwtup !== nothing && kwtup <: Duplicated
        mi, _ = enzyme_custom_extract_mi(orig)

        bt = GPUCompiler.backtrace(orig)
        msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
        emit_error(B, orig, (msg2, mi, world), NonConstantKeywordArgException)
        return false
    end
    
    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))
    mode = get_mode(gutils)
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))
    width = get_width(gutils)


    llvmf = nested_codegen!(mode, mod, fmi, world)

    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))

    swiftself = has_swiftself(llvmf)
    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end
    _, sret, returnRoots = get_return_info(enzyme_custom_extract_mi(llvmf)[2])
    if sret !== nothing
        sret = alloca!(alloctx, convert(LLVMType, eltype(sret)))
        pushfirst!(args, sret)
        if returnRoots !== nothing
            returnRoots = alloca!(alloctx, convert(LLVMType, eltype(returnRoots)))
            insert!(args, 2, returnRoots)
        else
            returnRoots = nothing
        end
    else
        sret = nothing
    end

    if length(args) != length(parameters(llvmf))
        bt = GPUCompiler.backtrace(orig)
        msg2 = sprint() do io
            if startswith(LLVM.name(llvmf), "japi3") || startswith(LLVM.name(llvmf), "japi1")
                Base.println(io, "Function uses the japi convention, which is not supported yet: ", LLVM.name(llvmf))
            else
                Base.println(io, "args = ", args)
                Base.println(io, "llvmf = ", string(llvmf))
                Base.println(io, "value_type(llvmf) = ", string(value_type(llvmf)))
                Base.println(io, "orig = ", string(orig))
                Base.println(io, "isKWCall = ", string(isKWCall))
                Base.println(io, "kwtup = ", string(kwtup))
                Base.println(io, "TT = ", string(TT))
                Base.println(io, "sret = ", string(sret))
                Base.println(io, "returnRoots = ", string(returnRoots))
            end
            Base.show_backtrace(io, bt)
        end
        emit_error(B, orig, (msg2, fmi, world), CallingConventionMismatchError{Cstring})
        return false
    end

    for i in eachindex(args)
        party = value_type(parameters(llvmf)[i])
        if value_type(args[i]) == party
            continue
        end
        # Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
        args[i] = calling_conv_fixup(B, args[i], party)
        # GPUCompiler.@safe_error "Calling convention mismatch", party, args[i], i, llvmf, fn, args, sret, returnRoots
        return false
    end

    res = LLVM.call!(B, LLVM.function_type(llvmf), llvmf, args)
    debug_from_orig!(gutils, res, orig)
    callconv!(res, callconv(llvmf))

    hasNoRet = has_fn_attr(llvmf, EnumAttribute("noreturn"))

    if hasNoRet
        return false
    end

    if sret !== nothing
        sty = sret_ty(llvmf, 1)
        if LLVM.version().major >= 12
            attr = TypeAttribute("sret", sty)
        else
            attr = EnumAttribute("sret")
        end
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1), attr)
        res = load!(B, sty, sret)
    end
    if swiftself
        attr = EnumAttribute("swiftself")
        LLVM.API.LLVMAddCallSiteAttribute(
            res,
            LLVM.API.LLVMAttributeIndex(1 + (sret !== nothing)),
            attr,
        )
    end

    shadowV = C_NULL
    normalV = C_NULL

    ExpRT = EnzymeRules.forward_rule_return_type(C, RT)
    if ExpRT != fwd_RT
        bt = GPUCompiler.backtrace(orig)
        msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))            
        emit_error(
            B,
            orig,
            (msg2, fmi, world),
            ForwardRuleReturnError{C, RT, fwd_RT}
        )
        return false
    end

    if RT <: Const
        if needsPrimal
            @assert RealRt == fwd_RT
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, res, val)
            else
                normalV = res.ref
            end
        else
            @assert Nothing == fwd_RT
        end
    else
        if !needsPrimal
            ST = RealRt
            if width != 1
                ST = NTuple{Int(width),ST}
            end
            @assert ST == fwd_RT
            if get_return_info(RealRt)[2] !== nothing
                dval_ptr = invert_pointer(gutils, operands(orig)[1], B)
                for idx = 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx - 1)
                    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx - 1)
                    store!(B, res, pev)
                end
            else
                shadowV = res.ref
            end
        else
            ST = if width == 1
                Duplicated{RealRt}
            else
                BatchDuplicated{RealRt,Int(width)}
            end
            @assert ST == fwd_RT
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, extract_value!(B, res, 0), val)

                dval_ptr = invert_pointer(gutils, operands(orig)[1], B)
                dval = extract_value!(B, res, 1)
                for idx = 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx - 1)
                    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx - 1)
                    store!(B, ev, pev)
                end
            else
                normalV = extract_value!(B, res, 0).ref
                shadowV = extract_value!(B, res, 1).ref
            end
        end
    end

    if shadowR != C_NULL
        unsafe_store!(shadowR, shadowV)
    end

    # Delete the primal code
    if origNeedsPrimal
        unsafe_store!(normalR, normalV)
    else
        ni = new_from_original(gutils, orig)
        if value_type(ni) != LLVM.VoidType()
            API.EnzymeGradientUtilsReplaceAWithB(
                gutils,
                ni,
                LLVM.UndefValue(value_type(ni)),
            )
        end
        API.EnzymeGradientUtilsErase(gutils, ni)
    end

    return false
end

@inline function aug_fwd_mi(
    orig::LLVM.CallInst,
    gutils::GradientUtils,
    forward::Bool = false,
    @nospecialize(B::Union{Nothing, LLVM.IRBuilder}) = nothing,
)
    width = get_width(gutils)

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)
    isKWCall = isKWCallSignature(mi.specTypes)

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup, mixeds =
        enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, !forward, isKWCall) #=reverse=#
    RT, needsPrimal, needsShadow, origNeedsPrimal =
        enzyme_custom_setup_ret(gutils, orig, mi, RealRt, B)

    needsShadowJL = if RT <: Active
        false
    else
        needsShadow
    end

    fn = LLVM.parent(LLVM.parent(orig))
    world = enzyme_extract_world(fn)

    C = EnzymeRules.RevConfig{
        Bool(needsPrimal),
        Bool(needsShadowJL),
        Int(width),
        overwritten,
        get_runtime_activity(gutils),
        get_strong_zero(gutils),
    }

    mode = get_mode(gutils)


    augprimal_tt = copy(activity)
    functy = if isKWCall
        popfirst!(augprimal_tt)
        @assert kwtup !== nothing
        insert!(augprimal_tt, 1, kwtup)
        insert!(augprimal_tt, 2, Core.typeof(EnzymeRules.augmented_primal))
        insert!(augprimal_tt, 3, C)
        insert!(augprimal_tt, 5, Type{RT})

        augprimal_TT = Tuple{augprimal_tt...}
        Core.Typeof(Core.kwfunc(EnzymeRules.augmented_primal))
    else
        @assert kwtup === nothing
        insert!(augprimal_tt, 1, C)
        insert!(augprimal_tt, 3, Type{RT})

        augprimal_TT = Tuple{augprimal_tt...}
        typeof(EnzymeRules.augmented_primal)
    end

    ami = my_methodinstance(Reverse, functy, augprimal_TT, world)
    if ami === nothing
        augprimal_TT = Tuple{typeof(world),functy,augprimal_TT.parameters...}
        ami = my_methodinstance(
            Reverse,
            typeof(custom_rule_method_error),
            augprimal_TT,
            world,
        )
        if forward
            pushfirst!(args, LLVM.ConstantInt(world))
        end
        ami
    end

    ami = ami::Core.MethodInstance
    @safe_debug "Applying custom augmented_primal rule" TT = augprimal_TT, functy=functy
    return ami,
    augprimal_TT,
    (
        args,
        activity,
        overwritten,
        actives,
        kwtup,
        RT,
        needsPrimal,
        needsShadow,
        origNeedsPrimal,
        mixeds,
    )
end

@inline function fwd_mi(
    orig::LLVM.CallInst,
    gutils::GradientUtils,
    @nospecialize(B::Union{Nothing, LLVM.IRBuilder}) = nothing,
)
    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)

    kwfunc = nothing

    isKWCall = isKWCallSignature(mi.specTypes)
    if isKWCall
        kwfunc = Core.kwfunc(EnzymeRules.forward)
    end

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup, _ =
        enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, false, isKWCall) #=reverse=#
    RT, needsPrimal, needsShadow, origNeedsPrimal =
        enzyme_custom_setup_ret(gutils, orig, mi, RealRt, B)
    width = get_width(gutils)

    C = EnzymeRules.FwdConfig{
        Bool(needsPrimal),
        Bool(needsShadow),
        Int(width),
        get_runtime_activity(gutils),
        get_strong_zero(gutils),
    }

    tt = copy(activity)
    if isKWCall
        popfirst!(tt)
        @assert kwtup !== nothing
        insert!(tt, 1, kwtup)
        insert!(tt, 2, Core.typeof(EnzymeRules.forward))
        insert!(tt, 3, C)
        insert!(tt, 5, Type{RT})
    else
        @assert kwtup === nothing
        insert!(tt, 1, C)
        insert!(tt, 3, Type{RT})
    end
    TT = Tuple{tt...}

    fn = LLVM.parent(LLVM.parent(orig))
    world = enzyme_extract_world(fn)
    @safe_debug "Trying to apply custom forward rule" TT isKWCall
        
    functy = if isKWCall
        rkwfunc = typeof(Core.kwfunc(EnzymeRules.forward))
    else
        typeof(EnzymeRules.forward)
    end
    @safe_debug "Applying custom forward rule" TT = TT, functy = functy
    fmi = my_methodinstance(Forward, functy, TT, world)
    if fmi === nothing
        TT = Tuple{typeof(world),functy,TT.parameters...}
        fmi = my_methodinstance(Forward, typeof(custom_rule_method_error), TT, world)
        pushfirst!(args, LLVM.ConstantInt(world))
        fwd_RT = Union{}
    else
        fwd_RT = primal_return_type_world(Forward, world, fmi)
    end
    
    fmi = fmi::Core.MethodInstance
    fwd_RT = fwd_RT::Type
    return fmi, (args, TT, fwd_RT, kwtup, RT, needsPrimal, RealRt, origNeedsPrimal, activity, C)
end

@inline function has_easy_rule_from_call(orig::LLVM.CallInst, gutils::GradientUtils)::Bool
    fn = LLVM.parent(LLVM.parent(orig))
    world = enzyme_extract_world(fn)
    mi, RealRt = enzyme_custom_extract_mi(orig)
    specTypes = Interpreter.simplify_kw(mi.specTypes)
    return EnzymeRules.has_easy_rule_from_sig(specTypes; world)
end

@inline function has_rule(orig::LLVM.CallInst, gutils::GradientUtils)::Bool
    if get_mode(gutils) == API.DEM_ForwardMode
       tup = fwd_mi(orig, gutils)
        if tup[1] === nothing
           return false
        end
    else
       if aug_fwd_mi(orig, gutils)[1] === nothing
            return false
        end
    end

    # Having an easy rule for a constant instruction -> no rule override
    if has_easy_rule_from_call(orig, gutils) && is_constant_inst(gutils, orig)
        return false
    end

    return true
end

function enzyme_custom_common_rev(
    forward::Bool,
    B::LLVM.IRBuilder,
    orig::LLVM.CallInst,
    gutils::GradientUtils,
    normalR::Ptr{LLVM.API.LLVMValueRef},
    shadowR::Ptr{LLVM.API.LLVMValueRef},
    tape::Union{Nothing, LLVM.Value},
)::LLVM.API.LLVMValueRef

    ctx = LLVM.context(orig)

    width = get_width(gutils)

    shadowType = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(orig)))
    if shadowR != C_NULL
        unsafe_store!(shadowR, UndefValue(shadowType).ref)
    end

    # TODO: don't inject the code multiple times for multiple calls

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)
    isKWCall = isKWCallSignature(mi.specTypes)

    # 2) Create activity, and annotate function spec
    ami, augprimal_TT, setup = aug_fwd_mi(orig, gutils, forward, B)
    args,
    activity,
    overwritten,
    actives,
    kwtup,
    RT,
    needsPrimal,
    needsShadow,
    origNeedsPrimal,
    mixeds = setup

    needsShadowJL = if RT <: Active
        false
    else
        needsShadow
    end

    C = EnzymeRules.RevConfig{
        Bool(needsPrimal),
        Bool(needsShadowJL),
        Int(width),
        overwritten,
        get_runtime_activity(gutils),
        get_strong_zero(gutils),
    }

    alloctx = LLVM.IRBuilder()
    position!(alloctx, LLVM.BasicBlock(API.EnzymeGradientUtilsAllocationBlock(gutils)))

    curent_bb = position(B)
    fn = LLVM.parent(curent_bb)
    world = enzyme_extract_world(fn)

    mode = get_mode(gutils)

    @assert ami !== nothing
    target = DefaultCompilerTarget()
    params = PrimalCompilerParams(mode)
    interp = GPUCompiler.get_interpreter(
        CompilerJob(ami, CompilerConfig(target, params; kernel = false), world),
    )
    aug_RT = return_type(interp, ami)
    if kwtup !== nothing && kwtup <: Duplicated
        mi, _ = enzyme_custom_extract_mi(orig)
        bt = GPUCompiler.backtrace(orig)
        msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
        emit_error(B, orig, (msg2, mi, world), NonConstantKeywordArgException)
        return C_NULL
    end

    rev_TT = nothing

    TapeT = Nothing


    if (
           aug_RT <: EnzymeRules.AugmentedReturn ||
           aug_RT <: EnzymeRules.AugmentedReturnFlexShadow
       ) &&
       !(aug_RT isa UnionAll) &&
       !(aug_RT isa Union) &&
       !(aug_RT === Union{})
        TapeT = EnzymeRules.tape_type(aug_RT)
    elseif (aug_RT isa UnionAll) &&
           (aug_RT <: EnzymeRules.AugmentedReturn) && hasfield(typeof(aug_RT.body), :name) &&
           aug_RT.body.name == EnzymeCore.EnzymeRules.AugmentedReturn.body.body.body.name
        if aug_RT.body.parameters[3] isa TypeVar
            TapeT = aug_RT.body.parameters[3].ub
        else
            TapeT = Any
        end
    elseif (aug_RT isa UnionAll) &&
           (aug_RT <: EnzymeRules.AugmentedReturnFlexShadow) && hasfield(typeof(aug_RT.body), :name) &&
           aug_RT.body.name ==
           EnzymeCore.EnzymeRules.AugmentedReturnFlexShadow.body.body.body.name
        if aug_RT.body.parameters[3] isa TypeVar
            TapeT = aug_RT.body.parameters[3].ub
        else
            TapeT = Any
        end
    else
        TapeT = Any
    end
    
    mod = LLVM.parent(LLVM.parent(LLVM.parent(orig)))

    llvmf = nothing
    applicablefn = true

    final_mi = nothing

    if forward
        llvmf = nested_codegen!(mode, mod, ami, world)
        @assert llvmf !== nothing
        rev_RT = nothing
        final_mi = ami
    else
        tt = copy(activity)
        if isKWCall
            popfirst!(tt)
            @assert kwtup !== nothing
            insert!(tt, 1, kwtup)
            insert!(tt, 2, Core.typeof(EnzymeRules.reverse))
            insert!(tt, 3, C)
            insert!(tt, 5, RT <: Active ? (width == 1 ? RT : NTuple{Int(width), RT}) : Type{RT})
            insert!(tt, 6, TapeT)
        else
            @assert kwtup === nothing
            insert!(tt, 1, C)
            insert!(tt, 3, RT <: Active ? (width == 1 ? RT : NTuple{Int(width), RT}) : Type{RT})
            insert!(tt, 4, TapeT)
        end
        rev_TT = Tuple{tt...}

        functy = if isKWCall
            rkwfunc = typeof(Core.kwfunc(EnzymeRules.reverse))
        else
            typeof(EnzymeRules.reverse)
        end

        @safe_debug "Applying custom reverse rule" TT = rev_TT, functy=functy
        rmi = my_methodinstance(Reverse, functy, rev_TT, world)

        if rmi === nothing
            rev_TT = Tuple{typeof(world),functy,rev_TT.parameters...}
            rmi = my_methodinstance(Reverse, typeof(custom_rule_method_error), rev_TT, world)
            pushfirst!(args, LLVM.ConstantInt(world))
            rev_RT = Union{}
            applicablefn = false
        else
            rev_RT = return_type(interp, rmi)
        end
        
        rmi = rmi::Core.MethodInstance
        rev_RT = rev_RT::Type
        llvmf = nested_codegen!(mode, mod, rmi, world)
        final_mi = rmi
    end

    push!(function_attributes(llvmf), EnumAttribute("alwaysinline", 0))

    needsTape = !isghostty(TapeT) && !Core.Compiler.isconstType(TapeT)

    tapeV = C_NULL
    if forward && needsTape
        tapeV = LLVM.UndefValue(convert(LLVMType, TapeT; allow_boxed = true)).ref
    end

    # if !forward
    #     argTys = copy(activity)
    #     if RT <: Active
    #         if width == 1
    #             push!(argTys, RealRt)
    #         else
    #             push!(argTys, NTuple{RealRt, (Int)width})
    #         end
    #     end
    #     push!(argTys, tapeType)
    #     llvmf = nested_codegen!(mode, mod, rev_func, Tuple{argTys...}, world)
    # end

    swiftself = has_swiftself(llvmf)

    miRT = enzyme_custom_extract_mi(llvmf)[2]
    _, sret, returnRoots = get_return_info(miRT)
    sret_union = is_sret_union(miRT)

    if sret_union
        bt = GPUCompiler.backtrace(orig)
        msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
        emit_error(B, orig, (msg2, final_mi, world), UnionSretReturnException{miRT})
        return tapeV
    end

    if !forward
        funcTy = rev_TT.parameters[isKWCall ? 4 : 2]
        if needsTape
            @assert tape isa LLVM.Value
            tape_idx =
                1 +
                (kwtup !== nothing && !isghostty(kwtup)) +
                !isghostty(funcTy) +
                (!applicablefn)
            trueidx =
                tape_idx +
                (sret !== nothing) +
                (returnRoots !== nothing) +
                swiftself +
                (RT <: Active)
            innerTy = value_type(parameters(llvmf)[trueidx])
            if innerTy != value_type(tape)
                if isabstracttype(TapeT) ||
                   TapeT isa UnionAll ||
                   TapeT == Tuple ||
                   TapeT.layout == C_NULL ||
                   TapeT == Array
                    msg = sprint() do io
                        println(
                            io,
                            "Enzyme : mismatch between innerTy $innerTy and tape type $(value_type(tape))",
                        )
                        println(io, "tape_idx=", tape_idx)
                        println(io, "true_idx=", trueidx)
                        println(io, "isKWCall=", isKWCall)
                        println(io, "kwtup=", kwtup)
                        println(io, "funcTy=", funcTy)
                        println(io, "isghostty(funcTy)=", isghostty(funcTy))
                        println(io, "miRT=", miRT)
                        println(io, "sret=", sret)
                        println(io, "returnRoots=", returnRoots)
                        println(io, "swiftself=", swiftself)
                        println(io, "RT=", RT)
                        println(io, "rev_RT=", rev_RT)
                        println(io, "applicablefn=", applicablefn)
                        println(io, "tape=", tape)
                        println(io, "llvmf=", string(LLVM.function_type(llvmf)))
                        println(io, "TapeT=", TapeT)
                        println(io, "mi=", mi)
                        println(io, "ami=", ami)
                        println(io, "rev_TT =", rev_TT)
                    end
                    throw(AssertionError(msg))
                end
                llty = convert(LLVMType, TapeT; allow_boxed = true)
                al0 = al = emit_allocobj!(B, TapeT)
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                store!(B, tape, al)
                if any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, tape))
                end
                tape = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))
            end
            insert!(args, tape_idx, tape)
        end
        if RT <: Active
            nRT = if width == 1
                RT
            else
                NTuple{Int(width), RT}
            end

            llty = convert(LLVMType, nRT)

            if API.EnzymeGradientUtilsGetDiffeType(gutils, orig, false) == API.DFT_OUT_DIFF #=isforeign=#
                val = LLVM.Value(API.EnzymeGradientUtilsDiffe(gutils, orig, B))
                API.EnzymeGradientUtilsSetDiffe(gutils, orig, LLVM.null(value_type(val)), B)
            else
                llety = convert(LLVMType, eltype(RT); allow_boxed = true)
                ptr_val = invert_pointer(gutils, operands(orig)[1+!isghostty(funcTy)], B)
                ptr_val = lookup_value(gutils, ptr_val, B)
                val = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llety)))
                for idx = 1:width
                    ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                    ld = load!(B, llety, ev)
                    store!(B, LLVM.null(llety), ev)
                    val = (width == 1) ? ld : insert_value!(B, val, ld, idx - 1)
                end
            end

            al0 = al = emit_allocobj!(B, nRT)
            al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
            al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

            if width == 1
                ptr = inbounds_gep!(
                    B,
                    llty,
                    al,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), 0),
                    ],
                )
            else
                llety = convert(LLVMType, eltype(RT); allow_boxed = true)
                pty = LLVM.LLVMType(API.EnzymeGetShadowType(width, llety))
                ptr = bitcast!(B, al, LLVM.PointerType(pty, Derived))
            end
            store!(B, val, ptr)

            if any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end
            insert!(
                args,
                1 +
                (!isghostty(funcTy)) +
                (kwtup !== nothing && !isghostty(kwtup)) +
                (!applicablefn),
                al,
            )
        end
    end

    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end

    if sret !== nothing
        sret = alloca!(alloctx, convert(LLVMType, eltype(sret)))
        pushfirst!(args, sret)
        if returnRoots !== nothing
            returnRoots = alloca!(alloctx, convert(LLVMType, eltype(returnRoots)))
            insert!(args, 2, returnRoots)
        else
            returnRoots = nothing
        end
    else
        sret = nothing
    end

    if length(args) != length(parameters(llvmf))
        bt = GPUCompiler.backtrace(orig)
        msg2 = sprint() do io
            if startswith(LLVM.name(llvmf), "japi3") || startswith(LLVM.name(llvmf), "japi1")
                Base.println(io, "Function uses the japi convention, which is not supported yet: ", LLVM.name(llvmf))
            else
                Base.println(io, "args = ", args)
                Base.println(io, "llvmf = ", string(llvmf))
                Base.println(io, "value_type(llvmf) = ", string(value_type(llvmf)))
                Base.println(io, "orig = ", string(orig))
                Base.println(io, "isKWCall = ", string(isKWCall))
                Base.println(io, "kwtup = ", string(kwtup))
                Base.println(io, "augprimal_TT = ", string(augprimal_TT))
                Base.println(io, "rev_TT = ", string(rev_TT))
                Base.println(io, "fn = ", string(fn))
                Base.println(io, "sret = ", string(sret))
                Base.println(io, "returnRoots = ", string(returnRoots))
            end
            Base.show_backtrace(io, bt)
        end
        emit_error(B, orig, (msg2, final_mi, world), CallingConventionMismatchError{Cstring})
        return tapeV
    end


    T_jlvalue = LLVM.StructType(LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

    for i = 1:length(args)
        party = value_type(parameters(llvmf)[i])
        if value_type(args[i]) != party
            if party == T_prjlvalue
                while true
                    if isa(args[i], LLVM.BitCastInst)
                        args[i] = operands(args[i])[1]
                        continue
                    end
                    if isa(args[i], LLVM.AddrSpaceCastInst)
                        args[i] = operands(args[i])[1]
                        continue
                    end
                    break
                end
            end
        end

        if value_type(args[i]) == party
            continue
        end
        # Fix calling convention within julia that Tuple{Float,Float} ->[2 x float] rather than {float, float}
        function msg(io)
            println(io, string(llvmf))
            println(io, "args = ", args)
            println(io, "i = ", i)
            println(io, "args[i] = ", args[i])
            println(io, "party = ", party)
        end
        args[i] = calling_conv_fixup(
            B,
            args[i],
            party,
            LLVM.UndefValue(party),
            Cuint[],
            Cuint[],
            msg,
        )
    end

    res = LLVM.call!(B, LLVM.function_type(llvmf), llvmf, args)
    ncall = res
    debug_from_orig!(gutils, res, orig)
    callconv!(res, callconv(llvmf))

    hasNoRet = has_fn_attr(llvmf, EnumAttribute("noreturn"))

    if hasNoRet
        return tapeV
    end

    if sret !== nothing
        sty = sret_ty(llvmf, 1+swiftself)
        if LLVM.version().major >= 12
            attr = TypeAttribute("sret", sty)
        else
            attr = EnumAttribute("sret")
        end
        LLVM.API.LLVMAddCallSiteAttribute(
            res,
            LLVM.API.LLVMAttributeIndex(1 + swiftself),
            attr,
        )
        res = load!(B, sty, sret)
        API.SetMustCache!(res)
    end
    if swiftself
        attr = EnumAttribute("swiftself")
        LLVM.API.LLVMAddCallSiteAttribute(
            res,
            LLVM.API.LLVMAttributeIndex(1 + (sret !== nothing) + (returnRoots !== nothing)),
            attr,
        )
    end

    shadowV = C_NULL
    normalV = C_NULL


    if forward
        ShadT = RealRt
        if width != 1
            ShadT = NTuple{Int(width),RealRt}
        end
        ST = EnzymeRules.AugmentedReturn{
            needsPrimal ? RealRt : Nothing,
            needsShadowJL ? ShadT : Nothing,
            TapeT,
        }
        if ST != EnzymeRules.augmented_rule_return_type(C, RT, TapeT)
            throw(AssertionError("Unexpected augmented rule return computation\nST = $ST\nER = $(EnzymeRules.augmented_rule_return_type(C, RT, TapeT))\nC = $C\nRT = $RT\nTapeT = $TapeT"))
        end
        if !(aug_RT <: EnzymeRules.AugmentedReturnFlexShadow) && !(aug_RT <: EnzymeRules.AugmentedReturn{
            needsPrimal ? RealRt : Nothing,
            needsShadowJL ? ShadT : Nothing})

            bt = GPUCompiler.backtrace(orig)
            msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
            emit_error(B, orig, (msg2, ami, world), AugmentedRuleReturnError{C, RT, aug_RT})
            return tapeV
        end


        if aug_RT != ST
            if aug_RT <: EnzymeRules.AugmentedReturnFlexShadow
                if convert(LLVMType, EnzymeRules.shadow_type(aug_RT); allow_boxed = true) !=
                   convert(LLVMType, EnzymeRules.shadow_type(ST); allow_boxed = true)
                    emit_error(
                        B,
                        orig,
                        "Enzyme: Augmented forward pass custom rule " *
                        string(augprimal_TT) *
                        " flex shadow ABI return type mismatch, expected " *
                        string(ST) *
                        " found " *
                        string(aug_RT),
                    )
                    return tapeV
                end
                ST = EnzymeRules.AugmentedReturnFlexShadow{
                    needsPrimal ? RealRt : Nothing,
                    needsShadowJL ? EnzymeRules.shadow_type(aug_RT) : Nothing,
                    TapeT,
                }
            end
        end
        abstract = false
        if aug_RT != ST
            abs = (
                EnzymeRules.AugmentedReturn{
                    needsPrimal ? RealRt : Nothing,
                    needsShadowJL ? ShadT : Nothing,
                    T,
                } where {T}
            )
            if aug_RT <: abs
                abstract = true
            else
                @assert false
            end
        end

        resV = if abstract
            StructTy = convert(
                LLVMType,
                EnzymeRules.AugmentedReturn{
                    needsPrimal ? RealRt : Nothing,
                    needsShadowJL ? ShadT : Nothing,
                    Nothing,
                },
            )
            if StructTy != LLVM.VoidType()
                load!(
                    B,
                    StructTy,
                    bitcast!(
                        B,
                        res,
                        LLVM.PointerType(StructTy, addrspace(value_type(res))),
                    ),
                )
            else
                res
            end
        else
            res
        end

        idx = 0
        if needsPrimal
            @assert !isghostty(RealRt)
            normalV = extract_value!(B, resV, idx)
            if get_return_info(RealRt)[2] !== nothing
                val = new_from_original(gutils, operands(orig)[1])
                store!(B, normalV, val)
            else
                @assert value_type(normalV) == value_type(orig)
                normalV = normalV.ref
            end
            idx += 1
        end
        if needsShadow
            if needsShadowJL
                @assert !isghostty(RealRt)
                shadowV = extract_value!(B, resV, idx)
                if get_return_info(RealRt)[2] !== nothing
                    dval = invert_pointer(gutils, operands(orig)[1], B)

                    for idx = 1:width
                        to_store =
                            (width == 1) ? shadowV : extract_value!(B, shadowV, idx - 1)

                        store_ptr = (width == 1) ? dval : extract_value!(B, dval, idx - 1)

                        store!(B, to_store, store_ptr)
                    end
                    shadowV = C_NULL
                else
                    @assert value_type(shadowV) == shadowType
                    shadowV = shadowV.ref
                end
                idx += 1
            end
        end
        if needsTape
            tapeV = if abstract
                emit_nthfield!(B, res, LLVM.ConstantInt(2)).ref
            else
                extract_value!(B, res, idx).ref
            end
            idx += 1
        end
    else
        Tys = (
            A <: Active ? (width == 1 ? eltype(A) : NTuple{Int(width),eltype(A)}) : Nothing for A in activity[2+isKWCall:end]
        )
        ST = Tuple{Tys...}
        if rev_RT != ST
            bt = GPUCompiler.backtrace(orig)
            msg2 = sprint(Base.Fix2(Base.show_backtrace, bt))
            emit_error(B, orig, (msg2, rmi, world), ReverseRuleReturnError{C, Tuple{activity[2+isKWCall:end]...,}, rev_RT})
            return tapeV
        end
        if length(actives) >= 1 &&
           !isa(value_type(res), LLVM.StructType) &&
           !isa(value_type(res), LLVM.ArrayType)
            GPUCompiler.@safe_error "Shadow arg calling convention mismatch found return ",
            res
            return tapeV
        end

        idx = 0
        dl = string(LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(orig)))))
        Tys2 = (eltype(A) for A in activity[(2+isKWCall):end] if A <: Active)
        seen = TypeTreeTable()
        for (v, Ty) in zip(actives, Tys2)
            TT = typetree(Ty, ctx, dl, seen)
            Typ = C_NULL
            ext = extract_value!(B, res, idx)
            shadowVType = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(v)))
            if value_type(ext) != shadowVType
                size = sizeof(Ty)
                align = 0
                premask = C_NULL
                API.EnzymeGradientUtilsAddToInvertedPointerDiffeTT(
                    gutils,
                    orig,
                    C_NULL,
                    TT,
                    size,
                    v,
                    ext,
                    B,
                    align,
                    premask,
                )
            else
                @assert value_type(ext) == shadowVType
                API.EnzymeGradientUtilsAddToDiffe(gutils, v, ext, B, Typ)
            end
            idx += 1
        end

        for (ptr_val, argTyp, refal) in mixeds
            RefTy = argTyp
            if width != 1
                RefTy = NTuple{Int(width),RefTy}
            end
            curs = load!(B, convert(LLVMType, RefTy), refal)

            for idx = 1:width
                evp = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                evcur = (width == 1) ? curs : extract_value!(B, curs, idx - 1)
                store_nonjl_types!(B, evcur, evp)
            end
        end
    end

    if forward
        if shadowR != C_NULL && shadowV != C_NULL
            unsafe_store!(shadowR, shadowV)
        end

        # Delete the primal code
        if origNeedsPrimal
            unsafe_store!(normalR, normalV)
        else
            ni = new_from_original(gutils, orig)
            erase_with_placeholder(gutils, ni, orig)
        end
    end

    return tapeV
end


@register_aug function enzyme_custom_augfwd(B::LLVM.IRBuilder, orig::LLVM.CallInst, gutils::GradientUtils, normalR::Ptr{LLVM.API.LLVMValueRef}, shadowR::Ptr{LLVM.API.LLVMValueRef}, tapeR::Ptr{LLVM.API.LLVMValueRef})
    if is_constant_value(gutils, orig) &&
       is_constant_inst(gutils, orig) &&
       !has_rule(orig, gutils)
        return true
    end
    tape = enzyme_custom_common_rev(true, B, orig, gutils, normalR, shadowR, nothing) #=tape=#
    if tape != C_NULL
        unsafe_store!(tapeR, tape)
    end
    return false
end

@register_rev function enzyme_custom_rev(B::LLVM.IRBuilder, orig::LLVM.CallInst, gutils::GradientUtils, @nospecialize(tape::Union{Nothing, LLVM.Value}))
    if is_constant_value(gutils, orig) &&
       is_constant_inst(gutils, orig) &&
       !has_rule(orig, gutils)
        return
    end
    enzyme_custom_common_rev(false, B, orig, gutils, reinterpret(Ptr{LLVM.API.LLVMValueRef}, C_NULL), reinterpret(Ptr{LLVM.API.LLVMValueRef}, C_NULL), tape) #=tape=#
    return nothing
end

@register_diffuse function enzyme_custom_diffuse(orig::LLVM.CallInst, gutils::GradientUtils, @nospecialize(val::LLVM.Value), isshadow::Bool, mode::API.CDerivativeMode)
    # use default
    if is_constant_value(gutils, orig) &&
       is_constant_inst(gutils, orig) &&
       !has_rule(orig, gutils)
        return (false, true)
    end
    non_rooting_use = false
    fop = called_operand(orig)::LLVM.Function
    for (i, v) in enumerate(operands(orig)[1:end-1])
        if v == val
            if !has_arg_attr(fop, i, StringAttribute("enzymejl_returnRoots"))
                non_rooting_use = true
                break
            end
        end
    end

    # If the operand is just rooting, we don't need it and should override defaults
    if !non_rooting_use
        return (false, false)
    end

    # don't use default and always require the arg
    return (true, false)
end
