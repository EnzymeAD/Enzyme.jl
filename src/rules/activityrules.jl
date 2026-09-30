
function julia_activity_rule(f::LLVM.Function, method_table)
    if startswith(f.name, "japi3") || startswith(f.name, "japi1") || startswith(f.name, "jlcapi")
        return
    end
    mi, RT = enzyme_custom_extract_mi(f)

    llRT, sret, returnRoots = get_return_info(RT)
    retRemoved, parmsRemoved = removed_ret_parms(f)

    dl = string(f.parent.datalayout)

    ftype = f.function_type
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

    if isKWCallSignature(mi.specTypes)
        if EnzymeRules.is_inactive_kwarg_from_sig(Interpreter.simplify_kw(mi.specTypes); world, method_table)
            kwarg_inactive = true
        end
    end



    if !Enzyme.Compiler.no_type_setting(mi.specTypes; world)[1]
        any_active = false
        for arg in jlargs
            if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
                continue
            end

            op_idx = arg.codegen.i

            typ, _ = enzyme_extract_parm_type(f, arg.codegen.i)
            @assert typ == arg.typ

	    if (kwarg_inactive && arg.arg_i == 2) || guaranteed_const_nongen(arg.typ, world) || (arg.rooted_typ !== nothing && guaranteed_const_nongen(arg.rooted_typ, world))
                push!(
                    f.parameter_attributes[arg.codegen.i],
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
                        f.parameter_attributes[ idx + 1],
                        StringAttribute("enzyme_inactive"),
                    )
                end
                idx += 1
            end
            if returnRoots !== nothing
	        if !in(idx, parmsRemoved)
		    if (VERSION < v"1.12" || guaranteed_const_nongen(RT, world))
                    push!(
                        f.parameter_attributes[ idx + 1],
                        StringAttribute("enzyme_inactive"),
                    )
		    end
                end
            end
        end

        if llRT !== nothing && f.function_type.return_type != LLVM.VoidType()
            if guaranteed_const_nongen(RT, world)
                push!(f.return_attributes, StringAttribute("enzyme_inactive"))
            end
        end

	if !any_active && guaranteed_const_nongen(RT, world)
            push!(
		f.function_attributes,
		StringAttribute("enzyme_inactive"),
	    )
            push!(
		f.function_attributes,
		StringAttribute("enzyme_nofree"),
	    )
            push!(
		f.function_attributes,
		StringAttribute("enzyme_no_escaping_allocation"),
	    )
	end
    end
end
