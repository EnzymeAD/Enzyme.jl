# Abstractly interpret julia from LLVM

# Return (bool if could interpret, julia object interpreted to)
function absint(arg::LLVM.Value, partial::Bool = false)
    if isa(arg, LLVM.BitCastInst) || isa(arg, LLVM.AddrSpaceCastInst)
        return absint(operands(arg)[1], partial)
    end
    if isa(arg, ConstantExpr)
        if opcode(arg) == LLVM.API.LLVMAddrSpaceCast || opcode(arg) == LLVM.API.LLVMBitCast
            return absint(operands(arg)[1], partial)
        end
    end
    if isa(arg, LLVM.CallInst)
        fn = LLVM.called_operand(arg)
        nm = ""
        if isa(fn, LLVM.Function)
            nm = LLVM.name(fn)
        end
        for (fname, ty) in (
            ("jl_box_int64", Int64),
            ("ijl_box_int64", Int64),
            ("jl_box_uint64", UInt64),
            ("ijl_box_uint64", UInt64),
            ("jl_box_int32", Int32),
            ("ijl_box_int32", Int32),
            ("jl_box_uint32", UInt32),
            ("ijl_box_uint32", UInt32),
            ("jl_box_char", Char),
            ("ijl_box_char", Char),
        )
            if nm == fname
                v = first(operands(arg))
                if isa(v, ConstantInt)
                    if ty == Char
                        return (true, Char(convert(Int, v)))
                    else
                        return (true, convert(ty, v))
                    end
                end
            end
        end
        if nm == "julia.pointer_from_objref"
            return absint(operands(arg)[1], partial)
        end
        if nm == "jl_typeof" || nm == "ijl_typeof"
            vals = abs_typeof(operands(arg)[1], partial)
            return (vals[1], vals[2])
        end
        if LLVM.callconv(arg) == 37 || nm == "julia.call"
            index = 1
            if LLVM.callconv(arg) != 37
                fn = first(operands(arg))
                nm = LLVM.name(fn)
                index += 1
            end
            if nm == "jl_f_apply_type" || nm == "ijl_f_apply_type"
                index += 1
                found = []
                legal, Ty = absint(operands(arg)[index], partial)
                unionalls = []
                for sarg in operands(arg)[index+1:end-1]
                    slegal, foundv = absint(sarg, partial)
                    if slegal
                        push!(found, foundv)
                    elseif partial
                        foundv = TypeVar(Symbol("sarg" * string(sarg)))
                        push!(found, foundv)
                        push!(unionalls, foundv)
                    else
                        legal = false
                        break
                    end
                end

                if legal
                    res = Ty{found...}
                    for u in unionalls
                        res = UnionAll(u, res)
                    end
                    return (true, res)
                end
            end
            if nm == "jl_f_tuple" || nm == "ijl_f_tuple"
                index += 1
                found = []
                legal = true
                for sarg in operands(arg)[index:end-1]
                    slegal, foundv = absint(sarg, partial)
                    if slegal
                        push!(found, foundv)
                    else
                        legal = false
                        break
                    end
                end
                if legal
                    res = (found...,)
                    return (true, res)
                end
            end
        end
    end
    if isa(arg, ConstantExpr)
        ce = arg
        if opcode(ce) == LLVM.API.LLVMIntToPtr
            ce = operands(ce)[1]
            if isa(ce, LLVM.ConstantInt)
                ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
                typ = Base.unsafe_pointer_to_objref(ptr)
                return (true, typ)
            end
        end
    end

    if isa(arg, GlobalVariable)
        gname = LLVM.name(arg)
        for (k, v) in JuliaGlobalNameMap
            if gname == k || gname == "ejl_" * k
                return (true, v)
            end
        end
        for (k, v) in JuliaEnzymeNameMap
            if gname == k || gname == "ejl_" * k
                return (true, v)
            end
        end
    end

    if isa(arg, LLVM.LoadInst) &&
       value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Tracked)
        ptr = operands(arg)[1]
        ce = ptr
        while isa(ce, ConstantExpr)
            if opcode(ce) == LLVM.API.LLVMAddrSpaceCast ||
               opcode(ce) == LLVM.API.LLVMBitCast ||
               opcode(ce) == LLVM.API.LLVMIntToPtr
                ce = operands(ce)[1]
            else
                break
            end
        end
        if !isa(ce, LLVM.ConstantInt)
            return (false, nothing)
        end
        ptr = unsafe_load(reinterpret(Ptr{Ptr{Cvoid}}, convert(UInt, ce)))
        if ptr == C_NULL
            # bt = GPUCompiler.backtrace(arg)
            # btstr = sprint() do io
            #     Base.show_backtrace(io, bt)
            # end
            # @error "Found null pointer at\n $btstr" arg
            return (false, nothing)
        end
        typ = Base.unsafe_pointer_to_objref(ptr)
        return (true, typ)
    end
    return (false, nothing)
end

function actual_size(@nospecialize(typ2))
    if typ2 <: Array || typ2 <: AbstractString
        return sizeof(Int)
    elseif Base.isconcretetype(typ2)
        return sizeof(typ2)
    else
        return sizeof(Int)
    end
end

function abs_typeof(
    arg::LLVM.Value,
    partial::Bool = false,
)::Union{Tuple{Bool,Type,GPUCompiler.ArgumentCC},Tuple{Bool,Nothing,Nothing}}
    if isa(arg, LLVM.BitCastInst) || isa(arg, LLVM.AddrSpaceCastInst)
        return abs_typeof(operands(arg)[1], partial)
    end
    if isa(arg, ConstantExpr)
        if opcode(arg) == LLVM.API.LLVMAddrSpaceCast || opcode(arg) == LLVM.API.LLVMBitCast
            return abs_typeof(operands(arg)[1], partial)
        end
    end

    if isa(arg, LLVM.CallInst)
        fn = LLVM.called_operand(arg)
        nm = ""
        if isa(fn, LLVM.Function)
            nm = LLVM.name(fn)
        end

        if nm == "julia.pointer_from_objref"
            return abs_typeof(operands(arg)[1], partial)
        end

        for (fname, ty) in (
            ("jl_box_int64", Int64),
            ("ijl_box_int64", Int64),
            ("jl_box_uint64", UInt64),
            ("ijl_box_uint64", UInt64),
            ("jl_box_int32", Int32),
            ("ijl_box_int32", Int32),
            ("jl_box_uint32", UInt32),
            ("ijl_box_uint32", UInt32),
            ("jl_box_float32", Float32),
            ("ijl_box_float32", Float32),
            ("jl_box_char", Char),
            ("ijl_box_char", Char),
            ("jl_specializations_get_linfo", Core.MethodInstance),
            ("ijl_specializations_get_linfo", Core.MethodInstance),
        )
            if nm == fname
                return (true, ty, GPUCompiler.MUT_REF)
            end
        end

        # Type tag is arg 3
        if nm == "julia.gc_alloc_obj" ||
           nm == "jl_gc_alloc_typed" ||
           nm == "ijl_gc_alloc_typed"
            vals = absint(operands(arg)[3], partial)
            return (vals[1], vals[2], vals[1] ? GPUCompiler.BITS_REF : nothing)
        end
        # Type tag is arg 1
        if nm == "jl_alloc_array_1d" ||
           nm == "ijl_alloc_array_1d" ||
           nm == "jl_alloc_array_2d" ||
           nm == "ijl_alloc_array_2d" ||
           nm == "jl_alloc_array_3d" ||
           nm == "ijl_alloc_array_3d" ||
           nm == "jl_new_array" ||
           nm == "ijl_new_array"
            vals = absint(operands(arg)[1], partial)
            return (vals[1], vals[2], vals[1] ? GPUCompiler.MUT_REF : nothing)
        end

        if nm == "jl_new_structt" || nm == "ijl_new_structt"
            vals = absint(operands(arg)[1], partial)
            return (vals[1], vals[2], vals[1] ? GPUCompiler.MUT_REF : nothing)
        end

        if LLVM.callconv(arg) == 37 || nm == "julia.call"
            index = 1
            if LLVM.callconv(arg) != 37
                fn = first(operands(arg))
                nm = LLVM.name(fn)
                index += 1
            end

            if nm == "jl_f_isdefined" || nm == "ijl_f_isdefined"
                return (true, Bool, GPUCompiler.MUT_REF)
            end

            if nm == "jl_new_structv" || nm == "ijl_new_structv"
                @assert index == 2
                vals = absint(operands(arg)[index], partial)
                return (vals[1], vals[2], vals[1] ? GPUCompiler.MUT_REF : nothing)
            end

            if nm == "jl_f_tuple" || nm == "ijl_f_tuple"
                index += 1
                found = []
                unionalls = []
                legal = true
                for sarg in operands(arg)[index:end-1]
                    slegal, foundv, _ = abs_typeof(sarg, partial)
                    if slegal
                        push!(found, foundv)
                    elseif partial
                        foundv = TypeVar(Symbol("sarg" * string(sarg)))
                        push!(found, foundv)
                        push!(unionalls, foundv)
                    else
                        legal = false
                        break
                    end
                end
                if legal
                    res = Tuple{found...}
                    for u in unionalls
                        res = UnionAll(u, res)
                    end
                    return (true, res, GPUCompiler.BITS_REF)
                end
            end
        end

        if nm == "julia.call"
            fn = operands(arg)[1]
            nm = ""
            if isa(fn, LLVM.Function)
                nm = LLVM.name(fn)
            end

        end

        if nm == "jl_array_copy" || nm == "ijl_array_copy"
            legal, RT, _ = abs_typeof(operands(arg)[1], partial)
            if legal
                @assert RT <: Array
                return (legal, RT, GPUCompiler.MUT_REF)
            end
            return (legal, RT, nothing)
        end

        _, RT = enzyme_custom_extract_mi(arg, false)
        if RT !== nothing
            llrt, sret, returnRoots = get_return_info(RT)
            if sret !== nothing
                if llrt == RT
                    return (true, RT, GPUCompiler.BITS_VALUE)
                elseif llrt == Ptr{RT}
                    return (true, RT, GPUCompiler.MUT_REF)
                elseif llrt == Any
                    return (true, RT, GPUCompiler.BITS_REF)
                end
            end
        end
    end

    if isa(arg, LLVM.LoadInst)
        larg = operands(arg)[1]
        offset = nothing
        error = false
        while true
            if isa(larg, LLVM.BitCastInst) || isa(larg, LLVM.AddrSpaceCastInst)
                larg = operands(larg)[1]
                continue
            end
            if offset === nothing &&
               isa(larg, LLVM.GetElementPtrInst) &&
               all(x -> isa(x, LLVM.ConstantInt), operands(larg)[2:end])
                b = LLVM.IRBuilder()
                position!(b, larg)
                offty = LLVM.IntType(8 * sizeof(Int))
                offset = API.EnzymeComputeByteOffsetOfGEP(b, larg, offty)
                @assert isa(offset, LLVM.ConstantInt)
                offset = convert(Int, offset)
                larg = operands(larg)[1]
                continue
            end
            if isa(larg, LLVM.Argument)
                break
            end
            error = true
            break
        end

        if !error
            legal, typ, byref = abs_typeof(larg)
            if legal && (byref == GPUCompiler.MUT_REF || byref == GPUCompiler.BITS_REF)
                @static if VERSION < v"1.11-"
                    if typ <: Array && Base.isconcretetype(typ)
                        T = eltype(typ)
                        if offset === nothing || offset == 0
                            return (true, Ptr{T}, GPUCompiler.BITS_VALUE)
                        else
                            return (true, Int, GPUCompiler.BITS_VALUE)
                        end
                    end
                end
                if byref == GPUCompiler.BITS_REF || byref == GPUCompiler.MUT_REF
                    dl = LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(arg))))
                    if offset === nothing
                        byref = GPUCompiler.BITS_VALUE
                        legal = true
                        typ2 = typ
                        while actual_size(typ2) != sizeof(dl, value_type(arg))
                            if fieldcount(typ2) > 0
                                typ2 = fieldtype(typ, 1)
                                if !Base.allocatedinline(typ2)
                                    # @assert byref == GPUCompiler.BITS_VALUE
                                    byref = GPUCompiler.MUT_REF
                                    continue
                                end
                            end
                            legal = false
                            break
                        end
                        if legal
                            return (true, typ2, byref)
                        end
                    else
                        @assert Base.isconcretetype(typ)
                        for i = 1:fieldcount(typ)
                            if fieldoffset(typ, i) == offset
                                subT = fieldtype(typ, i)
                                fsize = if i == fieldcount(typ)
                                    sizeof(typ)
                                else
                                    fieldoffset(typ, i + 1)
                                end - offset
                                if fsize == sizeof(dl, value_type(arg))
                                    if Base.isconcretetype(subT) &&
                                       is_concrete_tuple(subT) &&
                                       length(subT.parameters) == 1
                                        subT = subT.parameters[1]
                                    end
                                    if Base.allocatedinline(subT)
                                        return (true, subT, GPUCompiler.BITS_VALUE)
                                    else
                                        return (true, subT, GPUCompiler.MUT_REF)
                                    end
                                end
                            end
                        end
                    end
                end
            elseif legal && if typ <: Ptr && Base.isconcretetype(typ)
                return (true, eltype(typ), GPUCompiler.BITS_VALUE)
            end
            end
        end
    end

    if isa(arg, LLVM.ExtractValueInst)
        larg = operands(arg)[1]
        indptrs = LLVM.API.LLVMGetIndices(arg)
        numind = LLVM.API.LLVMGetNumIndices(arg)
        offset = Cuint[unsafe_load(indptrs, i) for i = 1:numind]
        found, typ, byref = abs_typeof(larg, partial)
        if !found
            return (false, nothing, nothing)
        end
        if byref == GPUCompiler.BITS_VALUE
            for ind in offset
                @assert Base.isconcretetype(typ)
                cnt = 0
                for i = 1:fieldcount(typ)
                    styp = fieldtype(typ, i)
                    if isghostty(styp)
                        continue
                    end
                    if cnt == ind
                        typ = styp
                        break
                    end
                    cnt += 1
                end
            end
            if Base.allocatedinline(typ)
                return (true, typ, GPUCompiler.BITS_VALUE)
            else
                return (true, typ, GPUCompiler.MUT_REF)
            end
        end
    end


    if isa(arg, LLVM.Argument)
        f = LLVM.Function(LLVM.API.LLVMGetParamParent(arg))
        idx = only([i for (i, v) in enumerate(LLVM.parameters(f)) if v == arg])
        typ, byref = enzyme_extract_parm_type(f, idx, false) #=error=#
        if typ !== nothing
            return (true, typ, byref)
        end
    end

    legal, val = absint(arg, partial)
    if legal
        return (true, Core.Typeof(val), GPUCompiler.BITS_REF)
    end
    return (false, nothing, nothing)
end
# 
# function abs_cstring(arg::LLVM.Value)::Tuple{Bool,String}
#     if isa(arg, ConstantExpr)
#         ce = arg
# 	    while isa(ce, ConstantExpr)
# 	        if opcode(ce) == LLVM.API.LLVMAddrSpaceCast || opcode(ce) == LLVM.API.LLVMBitCast ||  opcode(ce) == LLVM.API.LLVMIntToPtr
# 	            ce = operands(ce)[1]
#             elseif opcode(ce) == LLVM.API.LLVMGetElementPtr
#                 if all(x -> isa(x, LLVM.ConstantInt) && convert(UInt, x) == 0, operands(ce)[2:end])
#                     ce = operands(ce)[1]
#                 else
#                     break
#                 end
# 	        else
# 	            break
# 	        end
# 	    end
# 	    if isa(ce, LLVM.GlobalVariable)
# 	        ce = LLVM.initializer(ce)
# 	        if (isa(ce, LLVM.ConstantArray) || isa(ce, LLVM.ConstantDataArray)) && eltype(value_type(ce)) == LLVM.IntType(8)
# 	        	return (true, String(map((x)->convert(UInt8, x), collect(ce)[1:(end-1)])))
# 		    end
# 
# 	    end
# 	end
# 	return (false, "")
# end
