
function julia_activity_rule(f::LLVM.Function)
    mi, RT = enzyme_custom_extract_mi(f)

    llRT, sret, returnRoots =  get_return_info(RT)
    retRemoved, parmsRemoved = removed_ret_parms(f)
    
    dl = string(LLVM.datalayout(LLVM.parent(f)))

    expectLen = (sret !== nothing) + (returnRoots !== nothing)
    for source_typ in mi.specTypes.parameters
        if isghostty(source_typ) || Core.Compiler.isconstType(source_typ)
            continue
        end
        expectLen+=1
    end
    expectLen -= length(parmsRemoved)

    swiftself = any(any(map(k->kind(k)==kind(EnumAttribute("swiftself")), collect(parameter_attributes(f, i)))) for i in 1:length(collect(parameters(f))))

    if swiftself
        expectLen += 1
    end

    # Unsupported calling conv
    # also wouldn't have any type info for this [would for earlier args though]
    if mi.specTypes.parameters[end] === Vararg{Any}
        return
    end

    world = enzyme_extract_world(f)

    if  expectLen != length(parameters(f))
        println(string(f))
        @show expectLen, swiftself, sret, returnRoots, mi.specTypes.parameters, retRemoved, parmsRemoved
    end
    # TODO fix the attributor inlining such that this can assert always true
    @assert expectLen == length(parameters(f))

    jlargs = classify_arguments(mi.specTypes, function_type(f), sret !== nothing, returnRoots !== nothing, swiftself, parmsRemoved)

    for arg in jlargs
        if arg.cc == GPUCompiler.GHOST || arg.cc == RemovedParam
            continue
        end

        op_idx = arg.codegen.i

        if guaranteed_const_nongen(arg.typ, world)
            push!(parameter_attributes(f, arg.codegen.i), StringAttribute("enzyme_inactive"))
        end
    end

    if sret !== nothing
        idx = 0
        if !in(0, parmsRemoved)
            if guaranteed_const_nongen(RT, world)
                push!(parameter_attributes(f, idx+1), StringAttribute("enzyme_inactive"))
            end
            idx+=1
        end
        if returnRoots !== nothing
            if !in(idx, parmsRemoved)
                push!(parameter_attributes(f, idx+1), StringAttribute("enzyme_inactive"))
            end
        end
    end

    if llRT !== nothing && LLVM.return_type(function_type(f)) != LLVM.VoidType()
        if guaranteed_const_nongen(RT, world)
            push!(return_attributes(f), StringAttribute("enzyme_inactive"))
        end
    end
end