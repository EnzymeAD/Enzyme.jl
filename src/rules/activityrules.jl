
function inactive_arg_indices(@nospecialize(specTypes::Type), world::UInt, method_table = nothing)
    simp = Interpreter.simplify_kw(specTypes)
    if !EnzymeRules.is_inactive_arg_from_sig(simp; world, method_table)
        return ()
    end
    rt = primal_return_type_world(Forward, world, typeof(EnzymeRules.inactive_arg), simp)
    if !(rt isa DataType && rt <: Tuple)
        return ()
    end
    idxs = Int[]
    for p in rt.parameters
        if p isa DataType && p <: Val && length(p.parameters) == 1 && p.parameters[1] isa Int
            push!(idxs, p.parameters[1]::Int)
        end
    end
    return Tuple(idxs)
end

function julia_activity_rule(f::LLVM.Function, method_table)
    if startswith(LLVM.name(f), "japi3") || startswith(LLVM.name(f), "japi1") || startswith(LLVM.name(f), "jlcapi")
        return
    end
    mi, RT = enzyme_custom_extract_mi(f)

    llRT, sret, returnRoots = get_return_info(RT)
    retRemoved, parmsRemoved = removed_ret_parms(f)

    dl = string(LLVM.datalayout(LLVM.parent(f)))

    ftype = function_type(f)
    swiftself = has_swiftself(f)

    # Unsupported calling conv
    # also wouldn't have any type info for this [would for earlier args though]
    if mi.specTypes.parameters[end] === Vararg{Any}
        return
    end
    world = enzyme_world()

    jlargs = classify_arguments(
        mi.specTypes,
        ftype,
        sret !== nothing,
        returnRoots !== nothing,
        swiftself,
        parmsRemoved,
        mi,
        world,
    )

    kwarg_inactive = false
    isKW = isKWCallSignature(mi.specTypes)

    if isKW
        if EnzymeRules.is_inactive_kwarg_from_sig(Interpreter.simplify_kw(mi.specTypes); world, method_table)
            kwarg_inactive = true
        end
    end
    inactive_args = inactive_arg_indices(mi.specTypes, world, method_table)



    if !Enzyme.Compiler.no_type_setting(mi.specTypes; world)[1]
        any_active = false
        for arg in jlargs
            if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
                continue
            end

            op_idx = arg.codegen.i

            typ, _ = enzyme_extract_parm_type(f, arg.codegen.i)
            @assert typ == arg.typ

            simp_idx = isKW ? (arg.arg_jl_i >= 3 ? arg.arg_jl_i - 2 : 0) : arg.arg_jl_i
            if (kwarg_inactive && arg.arg_i == 2) || in(simp_idx, inactive_args) || guaranteed_const_nongen(arg.typ, world) || (arg.rooted_typ !== nothing && guaranteed_const_nongen(arg.rooted_typ, world))
                push!(
                    parameter_attributes(f, arg.codegen.i),
                    StringAttribute("enzyme_inactive"),
                )
    	    else
        		any_active = true
            end
        end
        if sret !== nothing
            idx = 0
            if !in(0, parmsRemoved)
                if guaranteed_const_nongen(RT, world)
                    push!(
                        parameter_attributes(f, idx + 1),
                        StringAttribute("enzyme_inactive"),
                    )
                end
                idx += 1
            end
            if returnRoots !== nothing
	        if !in(idx, parmsRemoved)
		    if (VERSION < v"1.12" || guaranteed_const_nongen(RT, world))
                    push!(
                        parameter_attributes(f, idx + 1),
                        StringAttribute("enzyme_inactive"),
                    )
		    end
                end
            end
        end

        if llRT !== nothing && LLVM.return_type(function_type(f)) != LLVM.VoidType()
            if guaranteed_const_nongen(RT, world)
                push!(return_attributes(f), StringAttribute("enzyme_inactive"))
            end
        end

	if !any_active && guaranteed_const_nongen(RT, world)
            push!(
		function_attributes(f),
		StringAttribute("enzyme_inactive"),
	    )
            push!(
		function_attributes(f),
		StringAttribute("enzyme_nofree"),
	    )
            push!(
		function_attributes(f),
		StringAttribute("enzyme_no_escaping_allocation"),
	    )
	end
    end
end
