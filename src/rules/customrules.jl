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

function push_box_for_argument!(@nospecialize(B::LLVM.IRBuilder),
                          @nospecialize(Ty::Type),
                          @nospecialize(val::Union{LLVM.Value, Nothing}),
                          @nospecialize(roots_val::Union{Nothing, LLVM.Value}),
                          arg,
                          args::Vector{LLVM.Value},
                          overwritten::Vector{UInt8},
                          activity_wrap::Bool,
                          ogval::LLVM.Value,
                          @nospecialize(roots_cache::Union{LLVM.Value, Nothing}), 
                          @nospecialize(shadow_roots::Union{Nothing, LLVM.Value}) = nothing,
			  just_primal_rooting::Bool = false
                          )::Union{Nothing, Tuple{LLVM.Value, LLVM.Value}}

    if !activity_wrap
        @assert arg.typ == Ty
    else
        @assert Ty <: Annotation
        @assert arg.typ == eltype(Ty)
    end

    num_inline_roots = inline_roots_type(Ty)
    if num_inline_roots != 0
        @assert roots_val isa LLVM.Value
    else
        @assert roots_cache === nothing
        @assert roots_val === nothing
    end

    if shadow_roots !== nothing
        if inline_roots_type(Ty) < inline_roots_type(arg.typ)
            throw(AssertionError("inline_roots_type(Ty = $Ty) [ $(inline_roots_type(Ty)) ] < inline_roots_type(arg.typ = $(arg.typ))  [ $(inline_roots_type(arg.typ)) ]"))
        end
    else
        @assert inline_roots_type(Ty) == inline_roots_type(arg.typ)
    end

    arty = convert(LLVMType, arg.typ; allow_boxed = true)

    # if either not a bits ref, or the data was not overwritten, the data is left
    # in the primal pointer.
    if val isa Nothing
    
        # If we just want the primal pointer, we can simply push the old data as per usual.
    
        if !activity_wrap
            push!(args, ogval)

            if roots_val !== nothing
                if roots_cache !== nothing
                    ral = alloca!(B, convert(LLVMType, AnyArray(num_inline_roots)))
                    store!(B, roots_cache, ral)
                    push!(args, ral)
                else
                    push!(args, roots_val)
                end
            end
            @assert shadow_roots === nothing
            return nothing
        else
            val = load!(B, arty, ogval)
        end

    end

    root_ptr = nothing

    if roots_cache !== nothing
        root_ty = convert(LLVMType, AnyArray(num_inline_roots))
        if shadow_roots === nothing
            ral = alloca!(B, root_ty)
            store!(B, roots_cache, ral)
            root_ptr = ral
        else
            sr2 = bitcast!(B, shadow_roots, LLVM.PointerType(root_ty))
            store!(B, roots_cache, sr2)
            root_ptr = shadow_roots
        end
    elseif roots_val !== nothing
        if shadow_roots === nothing
            root_ptr = roots_val
        else
	    cur_inline_roots, eTy = if just_primal_rooting
		@assert activity_wrap
		inline_roots_type(eltype(Ty)), "primal.$Ty"
	    else
	        num_inline_roots, string(Ty)
	    end
	    
	    if cur_inline_roots != 0
		    root_ty = convert(LLVMType, AnyArray(cur_inline_roots))
		    ld = load!(B, root_ty, roots_val, "loaded.roots.$eTy")
		    sr2 = bitcast!(B, shadow_roots, LLVM.PointerType(root_ty))
		    store!(B, ld, sr2)
	    end
            root_ptr = shadow_roots
        end
    end

    # Val now currently contains the reverse pass version of BITS_VALUE data.
    # We form the boxed object to contain it.

    llty = convert(LLVMType, Ty; allow_boxed = true)

    al0 = al = emit_allocobj!(B, Ty, "arg.$Ty")
    al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
    al = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))

    ptr = if activity_wrap
        inbounds_gep!(
            B,
            llty,
            al,
            [
                LLVM.ConstantInt(LLVM.IntType(64), 0),
                LLVM.ConstantInt(LLVM.IntType(32), 0),
            ],
        )
    else
        @assert llty == arty
        al
    end

    store!(B, val, ptr)

    push!(args, al)

    if root_ptr !== nothing
        push!(args, root_ptr)
    end

    if num_inline_roots == 0
        if any_jltypes(llty)
            emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
        end
    end

    return al0, al
end

function enzyme_custom_setup_args(
    @nospecialize(B::Union{Nothing, LLVM.IRBuilder}),
    orig::LLVM.CallInst,
    gutils::GradientUtils,
    mi::Core.MethodInstance,
    @nospecialize(RT::Type),
    reverse::Bool,
    isKWCall::Bool,
    @nospecialize(tape::Union{Nothing, LLVM.Value}),
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

    byval_tapes = LLVM.Value[]

    for arg in jlargs
        @assert arg.cc != RemovedParam
        
        if arg.rooted_typ !== nothing
            continue
        end
        
        if arg.cc == GPUCompiler.GHOST
            @assert inline_roots_type(arg.typ) == 0
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
                    val = unsafe_to_llvm(B, arg.typ.parameters[1])
                    roots_val = nothing
                    push_box_for_argument!(B, Const{arg.typ}, val, roots_val, arg, args, uncacheable, true, val, nothing)
                else
                    @assert isghostty(Const{arg.typ}) ||
                            Core.Compiler.isconstType(Const{arg.typ})
                end
            end
            continue
        end

        @assert !(isghostty(arg.typ) || Core.Compiler.isconstType(arg.typ))

        op = ops[arg.codegen.i]
        roots_op = nothing

        activity_state = active_reg(arg.typ, world)

        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, op, false) #=isforeign=#
	orig_activep = activep
	any_active_data = mode != API.DEM_ForwardMode && (activity_state == ActiveState || activity_state == MixedState)

        roots_activep = nothing

        if inline_roots_type(arg.typ) != 0
            roots_op = ops[arg.codegen.i + 1]
            roots_activep = API.EnzymeGradientUtilsGetDiffeType(gutils, roots_op, false)
	    any_active = false
	    for ty in non_rooted_types(arg.typ)
	        if active_reg(ty, world) != Compiler.AnyState
		   any_active = true
		   break
		end
	    end

	    if activep == API.DFT_CONSTANT && !any_active
	        @assert roots_activep == API.DFT_DUP_ARG
		activep = roots_activep
		any_active_data = false
	    end

            if roots_activep != activep
		    throw("roots_activep ($roots_activep) != activep ($activep) arg.typ=$(arg.typ) equivalent_rooted_type=$(equivalent_rooted_type(arg.typ)) non_rooted_types=$(non_rooted_types(arg.typ))")
            end
        end

        # Don't push the keyword args to uncacheable
        if !(isKWCall && arg.arg_jl_i == 2)
            uncache_arg = uncacheable[arg.codegen.i] != 0
            if roots_op !== nothing
                uncache_arg |= uncacheable[arg.codegen.i + 1] != 0
            end
            push!(overwritten, uncache_arg)
        end

        val = new_from_original(gutils, op)

        root_ty = nothing
        roots_val = if roots_op !== nothing
            root_ty = convert(LLVMType, AnyArray(inline_roots_type(arg.typ)))
            new_from_original(gutils, roots_op)
        end


        arty = convert(LLVMType, arg.typ; allow_boxed = true)

        # Val will contain the literal data inside (aka bits_value), properly
        # cached from fwd to reverse.
        # Val may also contain nothing, if it can be equally recreated by loading from ogval.



        # In the case that the argument is a bits_ref whose data is not overwritten,
        # val will contain nothing
        ogval = val
        roots_cache = nothing
        if arg.cc == GPUCompiler.BITS_REF
            @assert value_type(val) == LLVM.PointerType(arty, Derived)

            if uncacheable[arg.codegen.i] != 0
                # If is overwritten
                if !reverse
                    if B !== nothing
                        val = load!(B, arty, val)

                        # Since we will be caching this value (and thus GC pointers need to be valid),
                        # if the roots aren't here, we need to recombine before we stash on tape.
                        # However, as an optimization, if the roots aren't overwritten we don't need to actually
                        # recombine, we can get away with filling with any valid gc pointer
                        if roots_op != nothing
                            if uncacheable[arg.codegen.i + 1] != 0
                                # Roots are overwritten, recombine with root
                                val = recombine_value!(B, val, roots_op)
                            else
                                # Roots are not overwritten, put placeholder valid GC value
                                val = nullify_rooted_values!(B, val)
                            end
                        end

                        push!(byval_tapes, val)
                    end
                else
                    if B !== nothing
                        @assert tape isa LLVM.Value
                        val = extract_value!(B, tape, length(byval_tapes))
                        @assert value_type(val) == arty
                        push!(byval_tapes, val)
                    end
                end
            else
                # Don't perform the lookup here as we may be able to optimize it away
                # The corresponding load would look like:
                #
                # val = load!(B, arty, val)
                #
                # However we still have the same rooting issue re caching from forward to reverse
                # As a result, we may need to take care of rooting before performing lookups
                #
                # Let's also check if we need to cache the rooting, storing it if so.
                if roots_op !== nothing
                    if uncacheable[arg.codegen.i + 1] != 0
                        # Roots are is overwritten

                        if !reverse
                            if B !== nothing
                                root_cache = load!(B, root_ty, roots_op)
                                push!(byval_tapes, root_cache)
                            end
                        else
                            if B !== nothing
                                @assert tape isa LLVM.Value
                                root_cache = extract_value!(B, tape, length(byval_tapes))
                                @assert value_type(root_cache) == root_ty
                                push!(byval_tapes, root_cache)

                                al = alloca!(B, root_ty)
                                store!(B, root_cache, al)
                                roots_op = al
                            end
                        end
                    end
                end

                # We still need to ensure the pointer of val is itself accessible in the reverse pass location,
                # even if val itself is not
                val = nothing
                if reverse && B !== nothing
                    ogval = lookup_value(gutils, ogval, B)
                end
            end
        else
            @assert value_type(val) == arty
            if reverse && B !== nothing
                val = lookup_value(gutils, val, B)
            end
        end

        if isKWCall && arg.arg_jl_i == 2

            if EnzymeRules.is_inactive_kwarg_from_sig(Interpreter.simplify_kw(mi.specTypes); world)
                activep = API.DFT_CONSTANT
            end

            # Only constant kw arg tuple's are currently supported
            if activep == API.DFT_CONSTANT
                kwtup0 = arg.typ
                if B !== nothing
                    push_box_for_argument!(B, kwtup0, val, roots_val, arg, args, uncacheable, false, ogval, roots_cache)
                end
            else
                @assert activep == API.DFT_DUP_ARG
                kwtup0 = Duplicated{arg.typ}
                if B !== nothing
                    push_box_for_argument!(B, kwtup0, val, roots_val, arg, args, uncacheable, true, ogval, roots_cache)
                end
            end

            kwtup = kwtup0


            continue
        end

        # TODO type analysis deduce if duplicated vs active
        if activep == API.DFT_CONSTANT 
            Ty = Const{arg.typ}

            if B !== nothing
                push_box_for_argument!(B, Ty, val, roots_val, arg, args, uncacheable, true, ogval, roots_cache)
            end

            push!(activity, Ty)

	elseif activep == API.DFT_OUT_DIFF || (any_active_data && activity_state == ActiveState)
            Ty = Active{arg.typ}

            if B !== nothing
                push_box_for_argument!(B, Ty, val, roots_val, arg, args, uncacheable, true, ogval, roots_cache)
            end

            push!(activity, Ty)
            push!(actives, op)
        else

            ival = nothing
            roots_ival = nothing
            if B !== nothing
                ival = if is_constant_value(gutils, op)
                    @assert orig_activep != activep
                    @assert orig_activep == API.DFT_CONSTANT
                    if val == nothing
                        load!(B, iarty, ogval)
                    else
                        val
                    end
                else
                    invert_pointer(gutils, op, B)
                end
                @assert ival !== nothing

                uncache_arg = uncacheable[arg.codegen.i] != 0
                if roots_op !== nothing
                    uncache_arg |= uncacheable[arg.codegen.i + 1] != 0
                end
                if uncache_arg
                    # TODO we will are not restoring the bits_ref data of the
                    # shadow value (though now we are at least doing so properly for primal)
                    # x/ref https://github.com/EnzymeAD/Enzyme.jl/issues/2304
                end

                if reverse && !is_constant_value(gutils, op)
                    ival = lookup_value(gutils, ival, B)
                end
                if roots_op !== nothing
                    roots_ival = invert_pointer(gutils, roots_op, B)
                    if reverse
                        roots_ival = lookup_value(gutils, roots_ival, B)
                    end
                end
                @assert ival !== nothing
            end


            shadowty = arg.typ
            mixed = false
            if width == 1
                if any_active_data && activity_state == MixedState
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
                if any_active_data && activity_state == MixedState
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

            if mixed
                @assert arg.cc == GPUCompiler.BITS_REF
            end

            if B !== nothing
                @assert ival !== nothing

                n_shadow_roots = inline_roots_type(Ty)
                n_primal_roots = inline_roots_type(arg.typ)

                sroots_ty = nothing
                shadow_roots = if n_shadow_roots != 0
                    sroots_ty = convert(LLVMType, AnyArray(n_shadow_roots))
                    alloca!(B, sroots_ty, "roots.arg.$Ty")
                end


                T_jlvalue = LLVM.StructType(LLVMType[])
                T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

                if arg.cc == GPUCompiler.BITS_REF && !mixed
                    @assert n_shadow_roots == (width + 1) * n_primal_roots

                    ptr_val = if !is_constant_value(gutils, op)
                        @assert ival !== nothing
                         ival
                    else
                        if val == nothing
                            load!(B, iarty, ogval)
                        else
                            val
                        end
                    end
                    @assert ptr_val !== nothing
                    ival = UndefValue(siarty)
                    @assert ival !== nothing

                    for idx = 1:width
                        if !is_constant_value(gutils, op)
                            ev =
                                (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                            ld = load!(B, iarty, ev)
                            ival = (width == 1) ? ld : insert_value!(B, ival, ld, idx - 1)
                            @assert ival !== nothing
                        else
                            ival = (width == 1) ? ptr_val : insert_value!(B, ival, ptr_val, idx - 1)
                            @assert ival !== nothing
                        end

                        local_shadow_root = if roots_ival !== nothing
                            (width == 1) ? roots_ival : extract_value!(B, roots_ival, idx - 1)
                        end

                        if shadow_roots !== nothing

                            for r = 1:n_primal_roots
                                rptr = inbounds_gep!(
                                    B,
                                    sroots_ty,
                                    shadow_roots,
                                    [
                                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                                        LLVM.ConstantInt(LLVM.IntType(32), idx * n_primal_roots + r - 1 ),
                                    ],
                                )

                                ld = load!(B, T_prjlvalue, inbounds_gep!(
                                    B,
                                    LLVM.ArrayType(T_prjlvalue, n_primal_roots),
                                    local_shadow_root,
                                    [
                                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                                        LLVM.ConstantInt(LLVM.IntType(32), r - 1),
                                    ]
                                ))
                                stv = store!(B, ld, rptr)
                            end
                        end

                    end
                    @assert ival !== nothing
                elseif !mixed && is_constant_value(gutils, op)
                    @assert n_shadow_roots == (width + 1) * n_primal_roots

                    ptr_val = val
                    ival = UndefValue(siarty)

                    for idx = 1:width
                        ival = (width == 1) ? ptr_val : insert_value!(B, ptr_val, ld, idx - 1)
                    end
                    @assert ival !== nothing
                end
                @assert ival !== nothing
                

                if mixed
                    @assert !is_constant_value(gutils, op)
                    @assert arg.cc == GPUCompiler.BITS_REF
                    RefTy = arg.typ
                    if width != 1
                        RefTy = NTuple{Int(width),RefTy}
                    end
                    llrty = convert(LLVMType, RefTy)
                    RefTy = Base.RefValue{RefTy}
                    refal0 = refal = emit_allocobj!(B, RefTy, "mixed.$RefTy")
                    refal = bitcast!(
                        B,
                        refal,
                        LLVM.PointerType(llrty, addrspace(value_type(refal))),
                    )

                    ptr_val = ival
                    ival = UndefValue(LLVM.LLVMType(API.EnzymeGetShadowType(width, llrty)))
                    for idx = 1:width
                        ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                        ld = load!(B, llrty, ev)
                        if n_primal_roots > 0
                            sroots = (width == 1) ? roots_ival : extract_value!(B, roots_ival, idx - 1)
                            ld = recombine_value!(B, ld, sroots)
                        end
                        ival = (width == 1) ? ld : insert_value!(B, ival, ld, idx - 1)
                    end
                    store!(B, ival, refal)

                    emit_writebarrier!(B, get_julia_inner_types(B, refal0, ival))

                    if n_shadow_roots != 0
                        @assert n_shadow_roots == n_primal_roots + 1

                        rptr = inbounds_gep!(
                            B,
                            LLVM.ArrayType(T_prjlvalue, n_primal_roots),
                            shadow_roots,
                            [
                                LLVM.ConstantInt(LLVM.IntType(64), 0),
                                LLVM.ConstantInt(LLVM.IntType(32), n_shadow_roots - 1),
                            ]
                        )
                        store!(B, refal0, rptr)
                    end

                    ival = refal0
                    push!(mixeds, (ptr_val, arg.typ, refal))
                end

                @assert ival !== nothing
                just_primal_rooting = true
                al0, al = push_box_for_argument!(B, Ty, val, roots_val, arg, args, uncacheable, true, ogval, roots_cache, shadow_roots, just_primal_rooting)

                iptr = inbounds_gep!(
                    B,
                    llty,
                    al,
                    [
                        LLVM.ConstantInt(LLVM.IntType(64), 0),
                        LLVM.ConstantInt(LLVM.IntType(32), 1),
                    ],
                )

                store!(B, ival, iptr)

                if n_shadow_roots == 0 && any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, ival))
                end


            end
            push!(activity, Ty)
        end

    end
    return args, activity, (overwritten...,), actives, kwtup, mixeds, byval_tapes
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
    _, sret, returnRoots = get_return_info(RealRt)
    cv = LLVM.called_operand(orig)
    swiftself = has_swiftself(cv)

    may_have_active_reg = mode != API.DEM_ForwardMode && !guaranteed_nonactive(RealRt, world)

    if sret !== nothing
        activep = API.EnzymeGradientUtilsGetDiffeType(gutils, operands(orig)[1+swiftself], false) #=isforeign=#
        needsPrimal = activep == API.DFT_DUP_ARG || activep == API.DFT_CONSTANT
        needsShadowP[] = activep == API.DFT_DUP_ARG || activep == API.DFT_DUP_NONEED
	if returnRoots !== nothing && VERSION >= v"1.12"
        	roots_activep = API.EnzymeGradientUtilsGetDiffeType(gutils, operands(orig)[2+swiftself], false) #=isforeign=#
		may_have_active_reg = false
		if activep == API.DFT_CONSTANT
		    activep = roots_activep
        	    needsPrimal |= activep == API.DFT_DUP_ARG || activep == API.DFT_CONSTANT
		    needsShadowP[] = activep == API.DFT_DUP_ARG || activep == API.DFT_DUP_NONEED
		end
            	if roots_activep != activep
			throw("Returned roots_activep ($roots_activep) != activep ($activep) arg.typ=$(RealRt) equivalent_rooted_type=$(equivalent_rooted_type(RealRt)) non_rooted_types=$(non_rooted_types(RealRt))")
		end
		roots_needsPrimal = roots_activep == API.DFT_DUP_ARG || roots_activep == API.DFT_CONSTANT
		roots_needsShadowP = roots_activep == API.DFT_DUP_ARG || roots_activep == API.DFT_DUP_NONEED
	end
    end

    if !needsPrimal && activep == API.DFT_DUP_ARG
        activep = API.DFT_DUP_NONEED
    end

    if activep == API.DFT_CONSTANT
        RT = Const{RealRt}

    elseif activep == API.DFT_OUT_DIFF || may_have_active_reg
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

    orig_swiftself = has_swiftself(LLVM.called_operand(orig))

    swiftself = has_swiftself(llvmf)
    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end
    _, sret, returnRoots0 = get_return_info(enzyme_custom_extract_mi(llvmf)[2])
    returnRoots = returnRoots0
    if sret !== nothing
	sret_lty = convert(LLVMType, eltype(sret))
	esret = eltype(sret)
	if VERSION >= v"1.12" && returnRoots !== nothing
	     dl = LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(orig))))
	     sret_lty = LLVM.ArrayType(LLVM.Int8Type(), LLVM.sizeof(dl, sret_lty))
	end
        sret = alloca!(alloctx, sret_lty)
	metadata(sret)["enzymejl_allocart"] = MDNode(LLVM.Metadata[MDString(string(convert(UInt, unsafe_to_pointer(esret))))])
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
        LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(1 + swiftself), attr)
	if returnRoots !== nothing
	    LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(2 + swiftself), StringAttribute("enzymejl_returnRoots", string(length(eltype(returnRoots0).parameters[1]))))
	end

	if returnRoots !== nothing && VERSION >= v"1.12"
	   res = recombine_value_ptr!(B, sty, sret, returnRoots)
	else
	   res = load!(B, sty, sret)
	end
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
	    _, prim_sret, prim_roots = get_return_info(RealRt)
            if prim_sret !== nothing
                val = new_from_original(gutils, operands(orig)[1+orig_swiftself])
		
		if prim_roots !== nothing && VERSION >= v"1.12"
                    extract_nonjlvalues_into!(B, value_type(res), val, res)

                    rval = new_from_original(gutils, operands(orig)[2+orig_swiftself])

		    extract_roots_from_value!(B, res, rval)
		else
                    store!(B, res, val)
		end
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
	    _, prim_sret, prim_roots = get_return_info(RealRt)
            if prim_sret !== nothing
	        dval_ptr = if !is_constant_value(gutils, operands(orig)[1+orig_swiftself])
		    @assert prim_roots !== nothing && VERSION >= v"1.12"
		    @assert !is_constant_value(gutils, operands(orig)[2+orig_swiftself])
		    nothing
		else
		    invert_pointer(gutils, operands(orig)[1+orig_swiftself], B)
		end
                dval = extract_value!(B, res, 1)
		
		droots = if prim_roots !== nothing && VERSION >= v"1.12"
		    @assert !is_constant_value(gutils, operands(orig)[2+orig_swiftself])
		    invert_pointer(gutils, operands(orig)[2], B)
	        end
                
		for idx = 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx - 1)
			
		    if prim_roots !== nothing && VERSION >= v"1.12"
                    	if !is_constant_value(gutils, operands(orig)[1+orig_swiftself])
			   pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx - 1)
		           extract_nonjlvalues_into!(B, value_type(ev), pev, ev)
			end

		        rval = (width == 1) ? droots : extract_value!(B, droots, idx - 1)

		        extract_roots_from_value!(B, ev, rval)
		    else
			@assert dval_ptr !== nothing
                        pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx - 1)
                        store!(B, ev, pev)
		    end
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
	    
	    _, prim_sret, prim_roots = get_return_info(RealRt)
            if prim_sret !== nothing
                val = new_from_original(gutils, operands(orig)[1+orig_swiftself])
                
		res0 = extract_value!(B, res, 0)
		if prim_roots !== nothing && VERSION >= v"1.12"
                    extract_nonjlvalues_into!(B, value_type(res0), val, res0)

                    rval = new_from_original(gutils, operands(orig)[2+orig_swiftself])

		    extract_roots_from_value!(B, res0, rval)
		else
                    store!(B, res0, val)
		end

	        dval_ptr = if is_constant_value(gutils, operands(orig)[1+orig_swiftself])
		    @assert prim_roots !== nothing && VERSION >= v"1.12"
		    @assert !is_constant_value(gutils, operands(orig)[2+orig_swiftself])
		    nothing
		else
		    invert_pointer(gutils, operands(orig)[1+orig_swiftself], B)
		end
                dval = extract_value!(B, res, 1)
		
		droots = if prim_roots !== nothing && VERSION >= v"1.12"
		    @assert !is_constant_value(gutils, operands(orig)[2+orig_swiftself])
		    invert_pointer(gutils, operands(orig)[2+orig_swiftself], B)
	        end
                
		for idx = 1:width
                    ev = (width == 1) ? dval : extract_value!(B, dval, idx - 1)
		    if prim_roots !== nothing && VERSION >= v"1.12"
		        if !is_constant_value(gutils, operands(orig)[1+orig_swiftself])
			    pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx - 1)
			    extract_nonjlvalues_into!(B, value_type(ev), pev, ev)
			end

		        rval = (width == 1) ? droots : extract_value!(B, droots, idx - 1)

		        extract_roots_from_value!(B, ev, rval)
		    else
			@assert dval_ptr !== nothing
                    	pev = (width == 1) ? dval_ptr : extract_value!(B, dval_ptr, idx - 1)
                        store!(B, ev, pev)
		    end
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
    @nospecialize(tape::Union{Nothing, LLVM.Value}) = nothing,
)
    width = get_width(gutils)

    # 1) extract out the MI from attributes
    mi, RealRt = enzyme_custom_extract_mi(orig)
    isKWCall = isKWCallSignature(mi.specTypes)

    # 2) Create activity, and annotate function spec
    args, activity, overwritten, actives, kwtup, mixeds, byval_tapes =
        enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, !forward, isKWCall, tape) #=reverse=#
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
        byval_tapes
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
    args, activity, overwritten, actives, kwtup, _, byval_tapes =
        enzyme_custom_setup_args(B, orig, gutils, mi, RealRt, false, isKWCall, nothing) #=reverse=#
    @assert length(byval_tapes) == 0
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

function sret_union_tape_type(@nospecialize(aug_RT))
    InnerTypes = Type[]
    for_each_uniontype_small(aug_RT) do T
        TapeT = EnzymeRules.tape_type(T)
        push!(InnerTypes, TapeT)
    end
    return Union{InnerTypes...}
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
    ami, augprimal_TT, setup = aug_fwd_mi(orig, gutils, forward, B, tape)
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
    byval_tapes = setup

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
    elseif (
           aug_RT <: EnzymeRules.AugmentedReturn ||
           aug_RT <: EnzymeRules.AugmentedReturnFlexShadow
       ) && is_sret_union(aug_RT)
        TapeT = sret_union_tape_type(aug_RT)
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
    if forward
        if length(byval_tapes) == 0
            if needsTape
                tapeV = LLVM.UndefValue(convert(LLVMType, TapeT; allow_boxed = true)).ref
            else
                tapeV = C_NULL
            end
        else
            tapetys = LLVM.LLVMType[]
            for v in byval_tapes
                push!(tapetys, value_type(v))
            end
            if needsTape
                jltapeType = convert(LLVMType, TapeT; allow_boxed = true)
                push!(tapetys, jltapeType)
            end
            tapeV = LLVM.UndefValue(LLVM.StructType(tapetys))
            for (i, v) in enumerate(byval_tapes)
                tapeV = LLVM.insert_value!(B, tapeV, v, i - 1)
            end
            tapeV = tapeV.ref
        end
    elseif tape isa LLVM.Value && length(byval_tapes) != 0
        if needsTape
            @assert length(LLVM.elements(value_type(tape))) ==  length(byval_tapes) + 1
            tape = extract_value!(B, tape, length(byval_tapes))
        else
            tape = nothing
        end
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

    orig_swiftself = has_swiftself(LLVM.called_operand(orig))
    swiftself = has_swiftself(llvmf)

    miRT = enzyme_custom_extract_mi(llvmf)[2]
    _, sret, returnRoots0 = get_return_info(miRT)
    returnRoots = returnRoots0
    sret_union = is_sret_union(miRT)

    if sret_union
        @assert sret !== nothing
        @assert returnRoots === nothing
    end

    if !forward
        funcTy = rev_TT.parameters[isKWCall ? 4 : 2]
        if needsTape
            @assert tape isa LLVM.Value
            tape_idx = 1

            if kwtup !== nothing && !isghostty(kwtup)
                tape_idx += 1
                if inline_roots_type(kwtup) != 0
                    tape_idx += 1
                end
            end

            if !isghostty(funcTy)
                tape_idx += 1
                if inline_roots_type(funcTy) != 0
                    tape_idx += 1
                end
            end

            if !applicablefn
                tape_idx += 1
            end

            trueidx = tape_idx +
                (sret !== nothing) +
                (returnRoots !== nothing) +
                swiftself

            if (RT <: Active)
                trueidx += 1
                if inline_roots_type(RT) != 0
                    trueidx += 1
                end
            end

            innerTy = value_type(parameters(llvmf)[trueidx])
            tape_al = nothing
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

                tape_roots = inline_roots_type(TapeT)
                if tape_roots != 0
                    roots_ty = convert(LLVMType, AnyArray(tape_roots))
                    tape_al = alloca!(B, roots_ty)
                    extract_roots_from_value!(B, tape, tape_al)
                end

                al0 = al = emit_allocobj!(B, TapeT, "tape.$TapeT")
                al = bitcast!(B, al, LLVM.PointerType(llty, addrspace(value_type(al))))
                store!(B, tape, al)
                if tape_roots == 0 && any_jltypes(llty)
                    emit_writebarrier!(B, get_julia_inner_types(B, al0, tape))
                end
                tape = addrspacecast!(B, al, LLVM.PointerType(llty, Derived))
            end
            insert!(args, tape_idx, tape)
            if tape_al !== nothing
                insert!(args, tape_idx + 1, tape_al)
            end
        end
        if RT <: Active
            nRT = if width == 1
                RT
            else
                NTuple{Int(width), RT}
            end

            llty = convert(LLVMType, nRT)
            
	    active_roots = inline_roots_type(RT)

	    ral = nothing

            if API.EnzymeGradientUtilsGetDiffeType(gutils, orig, false) == API.DFT_OUT_DIFF #=isforeign=#
		@assert active_roots == 0
                val = LLVM.Value(API.EnzymeGradientUtilsDiffe(gutils, orig, B))
                API.EnzymeGradientUtilsSetDiffe(gutils, orig, LLVM.null(value_type(val)), B)
            else
                llety = convert(LLVMType, eltype(RT); allow_boxed = true)
        	if active_roots != 0
		   msg2 = "Unimplemented in 1.12 (use 1.10 or 1.11): Active Return with rooted types, RT=$RT"
		   emit_error(B, orig, (msg2, final_mi, world), CallingConventionMismatchError{Cstring})
		   return tapeV
		end
		@assert !is_constant_value(gutils,  operands(orig)[1+!isghostty(funcTy)+orig_swiftself]) "Handle constant RT, but active roots"
		ptr_val = invert_pointer(gutils, operands(orig)[1+!isghostty(funcTy)+orig_swiftself], B)
           
		if active_roots != 0
		    ptr_val = nullify_rooted_values!(ptr_val, B) # TODO this should be fwdB
		    @assert !is_constant_value(gutils,  operands(orig)[1+!isghostty(funcTy)+orig_swiftself+1])
		    roots_ty = convert(LLVMType, AnyArray(width * active_roots))
		    nroots_ty = convert(LLVMType, AnyArray(active_roots))
		    ral = alloca!(alloctx, nroots_ty)
		    rptr_val = invert_pointer(gutils, operands(orig)[1+!isghostty(funcTy)+orig_swiftself+1], B)
                    rptr_val = lookup_value(gutils, rptr_val, B)
		    # TODO actually cache the roots in the forward for use in the reverse here
                    for idx = 1:width
                       ev = (width == 1) ? rptr_val : extract_value!(B, rptr_val, idx - 1)
		       ld = load!(B, roots_ty, ev)
		       pv = gep!(B, nroots_ty, ral, [LLVM.ConstantInt(Int32(0)), LLVM.ConstantInt(Int32((idx-1)*active_roots))]) 
		       store!(B, ld, pv)
		    end
		end

		shadow_type = LLVM.LLVMType(API.EnzymeGetShadowType(width, llety))
                ptr_val = lookup_value(gutils, ptr_val, B)
                val = UndefValue(shadow_type)
                for idx = 1:width
                    ev = (width == 1) ? ptr_val : extract_value!(B, ptr_val, idx - 1)
                    ld = load!(B, llety, ev)
		    extract_nonjlvalues_into!(B, llety, ev, LLVM.null(llety))
                    val = (width == 1) ? ld : insert_value!(B, val, ld, idx - 1)
                end
            end

            al0 = al = emit_allocobj!(B, nRT, "activeRT.$RT")
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

            if active_roots == 0 && any_jltypes(llty)
                emit_writebarrier!(B, get_julia_inner_types(B, al0, val))
            end

            active_idx = 1

            if kwtup !== nothing && !isghostty(kwtup)
                active_idx += 1
                if inline_roots_type(kwtup) != 0
                    active_idx += 1
                end
            end

            if !isghostty(funcTy)
                active_idx += 1
                if inline_roots_type(funcTy) != 0
                    active_idx += 1
                end
            end

            if !applicablefn
                active_idx += 1
            end

            insert!(args, active_idx, al)

            if ral !== nothing
                insert!(args, active_idx + 1, ral)
            end

        end
    end

    if swiftself
        pushfirst!(reinsert_gcmarker!(fn, B))
    end

    if sret !== nothing
    	sret_lty, esret = if sret_union
            LLVM.ArrayType(LLVM.Int8Type(), union_alloca_type(miRT)), miRT
        else
            convert(LLVMType, eltype(sret)), eltype(sret)
        end
    	if VERSION >= v"1.12" && returnRoots !== nothing
    	     dl = LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(orig))))
    	     sret_lty = LLVM.ArrayType(LLVM.Int8Type(), LLVM.sizeof(dl, sret_lty))
    	end
        sret = alloca!(alloctx, sret_lty)
	metadata(sret)["enzymejl_allocart"] = MDNode(LLVM.Metadata[MDString(string(convert(UInt, unsafe_to_pointer(esret))))])
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

    sret_union_tape = nothing
    
    if sret_union && forward

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

        if ST != EnzymeRules.augmented_rule_return_type(C, RT, TapeT)
            throw(AssertionError("Unexpected augmented rule return computation\nST = $ST\nER = $(EnzymeRules.augmented_rule_return_type(C, RT, TapeT))\nC = $C\nRT = $RT\nTapeT = $TapeT"))
        end

        cur = nothing
        cur_size = nothing
        cur_offset = nothing

        counter = 1

        idxv = extract_value!(B, res, 1)

        function inner(@nospecialize(aug_RT::Type))
            jlrettype = EnzymeRules.tape_type(aug_RT)
            if cur_size == nothing
                cur_size = sizeof(jlrettype)
            elseif cur_size != sizeof(jlrettype)
                same_size = false
            end

            if cur === nothing
                cur = unsafe_to_llvm(B, jlrettype)
                cur_size = LLVM.ConstantInt(sizeof(jlrettype))
                cur_offset = LLVM.ConstantInt(fieldoffset(aug_RT, 3))
            else
                cmpv = icmp!(B, LLVM.API.LLVMIntEQ, idxv, LLVM.ConstantInt(value_type(idxv), counter))
                cur = select!(B, cmpv, unsafe_to_llvm(B, jlrettype), cur)
                cur_size = select!(B, cmpv, LLVM.ConstantInt(sizeof(jlrettype)), cur_size)
                cur_offset = select!(B, cmpv, LLVM.ConstantInt(fieldoffset(aug_RT, 3)), cur_offset)
            end

            counter += 1
            return
        end
        for_each_uniontype_small(inner, miRT)

        sret_union_tape = emit_allocobj!(B, cur, cur_size, false)
        T_int8 = LLVM.Int8Type()
        memcpy!(B, bitcast!(B, sret_union_tape, LLVM.PointerType(T_int8, Tracked)), 0, gep!(B, T_int8, bitcast!(B, sret, LLVM.PointerType(T_int8)), LLVM.Value[cur_offset]), 0, cur_size)

        res = sret

    elseif sret !== nothing
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
    	if returnRoots !== nothing
    	    LLVM.API.LLVMAddCallSiteAttribute(res, LLVM.API.LLVMAttributeIndex(2 + swiftself), StringAttribute("enzymejl_returnRoots", string(length(eltype(returnRoots0).parameters[1]))))
    	end
    	if returnRoots !== nothing && VERSION >= v"1.12"
    	    res = recombine_value_ptr!(B, sty, sret, returnRoots; must_cache=true)
    	else
            res = load!(B, sty, sret)
            API.SetMustCache!(res)
    	end
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
                lresV = load!(
                    B,
                    StructTy,
                    bitcast!(
                        B,
                        res,
                        LLVM.PointerType(StructTy, addrspace(value_type(res))),
                    ),
                )
                API.SetMustCache!(lresV)
                lresV
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
	        _, prim_sret, prim_roots = get_return_info(RealRt)
            if prim_sret !== nothing
                val = new_from_original(gutils, operands(orig)[1+orig_swiftself])
		
    		    if prim_roots !== nothing && VERSION >= v"1.12"
                    extract_nonjlvalues_into!(B, value_type(normalV), val, normalV)

                    rval = new_from_original(gutils, operands(orig)[2+orig_swiftself])

        		    extract_roots_from_value!(B, normalV, rval)
        		else
                    store!(B, normalV, val)
        		end
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
	        _, prim_sret, prim_roots = get_return_info(RealRt)
                if prim_sret !== nothing
                    dval = if is_constant_value(gutils, operands(orig)[1+orig_swiftself])
                        @assert prim_roots !== nothing && VERSION >= v"1.12"
                        @assert !is_constant_value(gutils, operands(orig)[2+orig_swiftself])
		    	nothing
		    else
			invert_pointer(gutils, operands(orig)[1+orig_swiftself], B)
		    end

		    droots = if prim_roots !== nothing && VERSION >= v"1.12"
			@assert !is_constant_value(gutils, operands(orig)[2+orig_swiftself])
                    	invert_pointer(gutils, operands(orig)[2+orig_swiftself], B)
		    end

		    for idx = 1:width
                        to_store =
                            (width == 1) ? shadowV : extract_value!(B, shadowV, idx - 1)


			if prim_roots !== nothing && VERSION >= v"1.12"
			    if !is_constant_value(gutils, operands(orig)[1+orig_swiftself])
			        store_ptr = (width == 1) ? dval : extract_value!(B, dval, idx - 1)
				extract_nonjlvalues_into!(B, value_type(to_store), store_ptr, to_store)
			    end

                            rval = (width == 1) ? droots : extract_value!(B, droots, idx - 1)

			    extract_roots_from_value!(B, to_store, rval)
			else
			    @assert dval !== nothing
                            store_ptr = (width == 1) ? dval : extract_value!(B, dval, idx - 1)
                            store!(B, to_store, store_ptr)
			end
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

            tapeV0 = if sret_union
                sret_union_tape
            elseif abstract
                emit_nthfield!(B, res, LLVM.ConstantInt(2))
            else
                extract_value!(B, res, idx)
            end
            if length(byval_tapes) == 0
                tapeV = tapeV0.ref
            else
                tapeV = insert_value!(B, LLVM.Value(tapeV), tapeV0, length(byval_tapes)).ref
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
            if true || !has_arg_attr(fop, i, StringAttribute("enzymejl_returnRoots"))
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
