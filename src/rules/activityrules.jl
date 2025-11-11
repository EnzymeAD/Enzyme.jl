
function julia_activity_rule(f::LLVM.Function, method_table)
    if startswith(LLVM.name(f), "japi3") || startswith(LLVM.name(f), "japi1")
        return
    end
    mi, RT = enzyme_custom_extract_mi(f)

    llRT, sret, returnRoots = get_return_info(RT)
    retRemoved, parmsRemoved = removed_ret_parms(f)

    dl = string(LLVM.datalayout(LLVM.parent(f)))

    expectLen = (sret !== nothing) + (returnRoots !== nothing)
    for source_typ in mi.specTypes.parameters
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end
        expectLen += 1
    end
    expectLen -= length(parmsRemoved)

    swiftself = has_swiftself(f)

    if swiftself
        expectLen += 1
    end

    # Unsupported calling conv
    # also wouldn't have any type info for this [would for earlier args though]
    if mi.specTypes.parameters[end] === Vararg{Any}
        return
    end
    world = enzyme_extract_world(f)

    # TODO fix the attributor inlining such that this can assert always true
    if expectLen != length(parameters(f))
        msg = sprint() do io::IO
            println(io, "Enzyme Internal Error (expectLen != length(parameters(f)))")
            println(io, string(f))
            println(io, "expectLen=", string(expectLen))
            println(io, "swiftself=", string(swiftself))
            println(io, "sret=", string(sret))
            println(io, "returnRoots=", string(returnRoots))
            println(io, "mi.specTypes.parameters=", string(mi.specTypes.parameters))
            println(io, "retRemoved=", string(retRemoved))
            println(io, "parmsRemoved=", string(parmsRemoved))
        end
        throw(AssertionError(msg))
    end

    jlargs = classify_arguments(
        mi.specTypes,
        function_type(f),
        sret !== nothing,
        returnRoots !== nothing,
        swiftself,
        parmsRemoved,
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

            if (kwarg_inactive && arg.arg_i == 2) || guaranteed_const_nongen(arg.typ, world)
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
                    push!(
                        parameter_attributes(f, idx + 1),
                        StringAttribute("enzyme_inactive"),
                    )
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
