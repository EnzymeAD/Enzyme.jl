# Abstractly interpret julia from LLVM

# Return (bool if could interpret, julia object interpreted to)


const JL_MAX_TAGS = 64 # see `enum jl_small_typeof_tags` in julia.h

function unbind(@nospecialize(val))
   if val isa Core.Binding
       return val.value
   else
       return val
   end
end

function absint(@nospecialize(arg::LLVM.Value), partial::Bool = false, istracked::Bool=false, typetag::Bool=false)::Tuple{Bool, Any}
    if (value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Tracked)) || (value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Derived)) || istracked
        ce, _ = get_base_and_offset(arg; offsetAllowed = false, inttoptr = true)
        if isa(ce, GlobalVariable)
            gname = LLVM.name(ce)
            for (k, v) in JuliaGlobalNameMap
                if gname == k
                    return (true, v)
                end
            end
            for (k, v) in JuliaEnzymeNameMap
                if gname == "ejl_" * k
                    return (true, v)
                end
            end
	    @assert !startswith(gname, "ejl_inserted") "Could not find ejl_inserted variable in map $gname"
        end
        if isa(ce, LLVM.LoadInst)
            gv = operands(ce)[1]
            if isa(gv, LLVM.GlobalVariable)
                init = LLVM.initializer(gv)
                if init !== nothing
                    just_load = true
                    for u in LLVM.uses(gv)
                        u = LLVM.user(u)
                        if !isa(u, LLVM.LoadInst)
                            just_load = false
                            break
                        end
                    end
                    if just_load
                        ce, _ = get_base_and_offset(init; offsetAllowed = false, inttoptr = true)
                    end
                end
            end
        end
        if isa(ce, LLVM.ConstantInt)
          ce = convert(UInt, ce)
          # "small" type tags are indices into a special array
	  ptr = if typetag && ce < (JL_MAX_TAGS << 4)
            jl_small_typeof = Ptr{Ptr{Cvoid}}(cglobal(:jl_small_typeof))
            type_idx = ce ÷ Core.sizeof(Ptr{Cvoid})
	    unsafe_load(jl_small_typeof, type_idx + 1)
          else
	    reinterpret(Ptr{Cvoid}, ce)
	  end
            val = Base.unsafe_pointer_to_objref(ptr)
            return (true, val)
        end
    end

    if isa(arg, ConstantExpr)
        if opcode(arg) == LLVM.API.LLVMAddrSpaceCast || opcode(arg) == LLVM.API.LLVMBitCast
            return absint(operands(arg)[1], partial, false, typetag)
        end
    end
    if isa(arg, LLVM.BitCastInst) || isa(arg, LLVM.AddrSpaceCastInst) || isa(arg, LLVM.IntToPtrInst)
        return absint(operands(arg)[1], partial, false, typetag)
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
        if nm == "julia.gc_loaded"
            return absint(operands(arg)[2], partial)
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
                found = Any[]
                legal, Ty = absint(operands(arg)[index], partial)
                unionalls = TypeVar[]
                for sarg in operands(arg)[(index + 1):(end - 1)]
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
                found = Any[]
                legal = true
                for sarg in operands(arg)[index:(end - 1)]
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

    if isa(arg, GlobalVariable)
        gname = LLVM.name(arg)
        for (k, v) in JuliaGlobalNameMap
            if gname == "ejl_" * k
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
            ((value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Tracked)) || (value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Derived)))
        ptr = operands(arg)[1]
        ce, _ = get_base_and_offset(ptr; offsetAllowed = false, inttoptr = true)
        if isa(ce, GlobalVariable)
            gname = LLVM.name(ce)
            for (k, v) in JuliaGlobalNameMap
                if gname == k
                    return (true, v)
                end
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

function actual_size(@nospecialize(typ2))::Int
    @static if VERSION < v"1.11-"
        if typ2 <: Array
            return sizeof(Ptr{Cvoid}) + 2 + 2 + 4 + 2 * sizeof(Csize_t) + sizeof(Csize_t)
        end
    else
        if typ2 <: GenericMemory
            return sum(map(sizeof, fieldtypes(typ2)))
        end
    end
    if typ2 <: String || typ2 <: Symbol || typ2 <: Core.SimpleVector
        return sizeof(Int)
    elseif Base.isconcretetype(typ2)
        return sizeof(typ2)
    else
        return sizeof(Int)
    end
end

@inline function first_non_ghost(@nospecialize(typ2))::Tuple{Int, Int}
    @static if VERSION < v"1.11-"
        if typ2 <: Array
            return (1, 0)
        end
    end
    fc = fieldcount(typ2)
    for i in 1:fc
        if i == fc
            return (i, sizeof(typ2))
        else
            fo = fieldoffset(typ2, i + 1)
            if fo != 0
                return (i, fo)
            end
        end
    end
    return (-1, 0)
end

function should_recurse(@nospecialize(typ2), @nospecialize(arg_t::LLVM.LLVMType), byref::GPUCompiler.ArgumentCC, dl::LLVM.DataLayout)::Bool
    sz = if arg_t == LLVM.IntType(1)
        1
    else
        sizeof(dl, arg_t)
    end
    if byref != GPUCompiler.BITS_VALUE
        if sz != sizeof(Int)
            throw(AssertionError("non bits type $byref of $typ2 has size $sz != sizeof(Int) from arg type $arg_t"))
        end
        return false
    else
        if actual_size(typ2) != sz
            return true
        else
            if Base.isconcretetype(typ2)
                idx, sz2 = first_non_ghost(typ2)
                if idx != -1
                    if sz2 == sz
                        return true
                    end
                end
            end
            return false
        end
    end
end

function get_base_and_offset(@nospecialize(larg::LLVM.Value); offsetAllowed::Bool = true, inttoptr::Bool = false, inst::Union{LLVM.Instruction, Nothing} = nothing, addrcast::Bool=true)::Tuple{LLVM.Value, Int}
    offset = 0
    pinst = isa(larg, LLVM.Instruction) ? larg::LLVM.Instruction : inst
    while true
        if isa(larg, LLVM.ConstantExpr)
            if opcode(larg) == LLVM.API.LLVMBitCast || opcode(larg) == LLVM.API.LLVMPtrToInt
                larg = operands(larg)[1]
                continue
            end
            if addrcast && opcode(larg) == LLVM.API.LLVMAddrSpaceCast
                larg = operands(larg)[1]
                continue
            end
            if inttoptr && opcode(larg) == LLVM.API.LLVMIntToPtr
                larg = operands(larg)[1]
                continue
            end
	    if opcode(larg) == LLVM.API.LLVMGetElementPtr && pinst isa LLVM.Instruction
		    b = LLVM.IRBuilder()
		    position!(b, pinst)
		    offty = LLVM.IntType(8 * sizeof(Int))
		    offset2 = API.EnzymeComputeByteOffsetOfGEP(b, larg, offty)
		    if isa(offset2, LLVM.ConstantInt)
			val = convert(Int, offset2)
			if offsetAllowed || val == 0
			    offset += val
			    larg = operands(larg)[1]
			    continue
			else
			    break
			end
		    else
			break
		    end
		end
        end
        if isa(larg, LLVM.BitCastInst) || isa(larg, LLVM.IntToPtrInst)
            larg = operands(larg)[1]
            continue
        end
        if addrcast && isa(larg, LLVM.AddrSpaceCastInst)
            larg = operands(larg)[1]
            continue
        end
        if inttoptr && isa(larg, LLVM.PtrToIntInst)
            larg = operands(larg)[1]
            continue
        end
        if LLVM.API.LLVMGetValueKind(larg) == LLVM.API.LLVMGlobalAliasValueKind
            larg = LLVM.Value(ccall((:LLVMAliasGetAliasee, LLVM.API.libllvm), LLVM.API.LLVMValueRef, (LLVM.API.LLVMValueRef,), larg))
            continue
        end
        if isa(larg, LLVM.GetElementPtrInst) &&
                all(Base.Fix2(isa, LLVM.ConstantInt), operands(larg)[2:end])
            b = LLVM.IRBuilder()
            position!(b, larg)
            offty = LLVM.IntType(8 * sizeof(Int))
            offset2 = API.EnzymeComputeByteOffsetOfGEP(b, larg, offty)
            if isa(offset2, LLVM.ConstantInt)
                val = convert(Int, offset2)
                if offsetAllowed || val == 0
                    offset += val
                    larg = operands(larg)[1]
                    continue
                else
                    break
                end
            else
                break
            end
        end
        if isa(larg, LLVM.Argument)
            break
        end
        break
    end
    return larg, offset
end

const TypesNotToDisect = Set{Type}([BigFloat])

function abs_typeof(
        @nospecialize(arg::LLVM.Value),
        partial::Bool = false, seenphis = Set{LLVM.PHIInst}()
    )::Union{Tuple{Bool, Type, GPUCompiler.ArgumentCC}, Tuple{Bool, Nothing, Nothing}}
    if (value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Tracked)) || (value_type(arg) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Derived))
        ce, _ = get_base_and_offset(arg; offsetAllowed = false, inttoptr = true)
	if isa(ce, GlobalVariable)
            gname = LLVM.name(ce)
            for (k, v) in JuliaGlobalNameMap
                if gname == k
                    return (true, Core.Typeof(v), GPUCompiler.BITS_REF)
                end
            end
            for (k, v) in JuliaEnzymeNameMap
                if gname == "ejl_" * k
		    return (true, Core.Typeof(unbind(v)), GPUCompiler.BITS_REF)
                end
            end
        end
        if isa(ce, LLVM.LoadInst)
            gv = operands(ce)[1]
            if isa(gv, LLVM.GlobalVariable)
                init = LLVM.initializer(gv)
                if init !== nothing
                    just_load = true
                    for u in LLVM.uses(gv)
                        u = LLVM.user(u)
                        if !isa(u, LLVM.LoadInst)
                            just_load = false
                            break
                        end
                    end
                    if just_load
                        ce, _ = get_base_and_offset(init; offsetAllowed = false, inttoptr = true)
                    end
                end
            end
        end
        if isa(ce, LLVM.ConstantInt)
            ptr = reinterpret(Ptr{Cvoid}, convert(UInt, ce))
            val = Base.unsafe_pointer_to_objref(ptr)
            return (true, Core.Typeof(val), GPUCompiler.BITS_REF)
        end
    end

    if isa(arg, ConstantExpr)
        if opcode(arg) == LLVM.API.LLVMAddrSpaceCast || opcode(arg) == LLVM.API.LLVMBitCast
            return abs_typeof(operands(arg)[1], partial, seenphis)
        end
    end
    if isa(arg, LLVM.BitCastInst) || isa(arg, LLVM.AddrSpaceCastInst) || isa(arg, LLVM.IntToPtrInst)
        return abs_typeof(operands(arg)[1], partial, seenphis)
    end

    if isa(arg, LLVM.AllocaInst) || isa(arg, LLVM.CallInst)
	for mdname in ("enzymejl_gc_alloc_rt", "enzymejl_allocart")
        if haskey(metadata(arg), mdname)
            mds = operands(metadata(arg)[mdname])[1]::MDString
            mds = Base.convert(String, mds)
            ptr = reinterpret(Ptr{Cvoid}, parse(UInt, mds))
            RT = Base.unsafe_pointer_to_objref(ptr)
            return (true, RT, GPUCompiler.MUT_REF)
        end
	end
    end

    if isa(arg, LLVM.CallInst)
        fn = LLVM.called_operand(arg)
        nm = ""
        if isa(fn, LLVM.Function)
            nm = LLVM.name(fn)
        end

        if nm == "julia.pointer_from_objref"
            return abs_typeof(operands(arg)[1], partial, seenphis)
        end

        if nm == "julia.gc_loaded"
            legal, res, byref = abs_typeof(operands(arg)[2], partial, seenphis)
            return legal, res, byref
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
            vals = absint(operands(arg)[3], partial, false, #=typetag=#true)
	    @assert !(vals[2] isa Core.Binding)
            return (vals[1], vals[2], vals[1] ? GPUCompiler.BITS_REF : nothing)
        end
        # Type tag is arg 3
        if nm == "jl_alloc_genericmemory_unchecked" ||
		nm == "ijl_alloc_genericmemory_unchecked"
	    vals = absint(operands(arg)[3], partial, true, #=typetag=#true)
	    @assert !(vals[2] isa Core.Binding)
            return (vals[1], vals[2], vals[1] ? GPUCompiler.MUT_REF : nothing)
        end
        # Type tag is arg 1
        if nm == "jl_alloc_array_1d" ||
                nm == "ijl_alloc_array_1d" ||
                nm == "jl_alloc_array_2d" ||
                nm == "ijl_alloc_array_2d" ||
                nm == "jl_alloc_array_3d" ||
                nm == "ijl_alloc_array_3d" ||
                nm == "jl_new_array" ||
                nm == "ijl_new_array" ||
                nm == "jl_alloc_genericmemory" ||
                nm == "ijl_alloc_genericmemory"
            vals = absint(operands(arg)[1], partial, false, #=typetag=#true)
	    @assert !(vals[2] isa Core.Binding)
            return (vals[1], vals[2], vals[1] ? GPUCompiler.MUT_REF : nothing)
        end

        if nm == "jl_new_structt" || nm == "ijl_new_structt"
            vals = absint(operands(arg)[1], partial, false, #=typetag=#true)
	    @assert !(vals[2] isa Core.Binding)
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
                vals = absint(operands(arg)[index], partial, false, #=typetag=#true)
	    	@assert !(vals[2] isa Core.Binding)
                return (vals[1], vals[2], vals[1] ? GPUCompiler.MUT_REF : nothing)
            end

            if nm == "jl_f_tuple" || nm == "ijl_f_tuple"
                index += 1
                found = Union{Type, TypeVar}[]
                unionalls = TypeVar[]
                legal = true
                for sarg in operands(arg)[index:(end - 1)]
                    slegal, foundv, _ = abs_typeof(sarg, partial, seenphis)
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

            if nm == "jl_f__apply_iterate" || nm == "ijl_f__apply_iterate"
                index += 1
                legal, iterfn = absint(operands(arg)[index])
	    	iterfn = unbind(iterfn)
                index += 1
                if legal && iterfn == Base.iterate
                    legal0, combfn = absint(operands(arg)[index])
		    combfn = unbind(combfn)
                    index += 1
                    if legal0 && combfn == Core.apply_type && partial
                        return (true, Type, GPUCompiler.BITS_REF)
                    end
                    resvals = Type[]
                    while index != length(operands(arg))
                        legal, pval, _ = abs_typeof(operands(arg)[index], partial, seenphis)
                        if !legal
                            break
                        end
                        push!(resvals, pval)
                        index += 1
                    end
                    if legal0 && legal && combfn == Base.tuple && partial && length(resvals) == 1
                        if resvals[1] <: Vector
                            return (true, Tuple{Vararg{eltype(resvals[1])}}, GPUCompiler.BITS_REF)
                        end
                    end
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
            legal, RT, _ = abs_typeof(operands(arg)[1], partial, seenphis)
            if legal
                if !(RT <: Array)
                    return (false, nothing, nothing)
                end
                return (legal, RT, GPUCompiler.MUT_REF)
            end
            return (legal, RT, nothing)
        end
        @static if VERSION < v"1.11-"
        else
            if nm == "jl_genericmemory_copy_slice" || nm == "ijl_genericmemory_copy_slice"
                legal, RT, _ = abs_typeof(operands(arg)[1], partial, seenphis)
                if legal
                    @assert RT <: Memory
                    return (legal, RT, GPUCompiler.MUT_REF)
                end
                return (legal, RT, nothing)
            end
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
        ce, _ = get_base_and_offset(operands(arg)[1]; offsetAllowed = false, inttoptr = true)
        if isa(ce, GlobalVariable)
            gname = LLVM.name(ce)
            for (k, v) in JuliaGlobalNameMap
                if gname == k
                    return (true, Core.Typeof(v), GPUCompiler.BITS_REF)
                end
            end
        end
        larg, offset = get_base_and_offset(operands(arg)[1])
        legal, typ, byref = abs_typeof(larg, false, seenphis)

        dl = LLVM.datalayout(LLVM.parent(LLVM.parent(LLVM.parent(arg))))

        shouldLoad = true

        if legal && typ <: Ptr && Base.isconcretetype(typ) && byref == GPUCompiler.BITS_VALUE
            ET = eltype(typ)
            byref = GPUCompiler.MUT_REF
            typ = ET
            # We currently make the assumption that Ptr{T} either represents a ptr which could be generated by
            # julia code (for example pointer(x) ), or is a storage container for an array / memory
            # in the latter case, it may need an extra level of indirection because of boxing. It is semantically
            # consistent here to consider Ptr{T} to represent the ptr to the boxed value in that case [and we essentially
            # add the extra pointer offset when loading here]. However for pointers constructed by ccall outside julia
            # to a julia object, which are not inline by type but appear so, like SparseArrays, this is a problem
            # and merits further investigation. x/ref https://github.com/EnzymeAD/Enzyme.jl/issues/2085
            @static if Base.USE_GPL_LIBS
                cholmod_exception = typ != SparseArrays.cholmod_dense_struct && typ != SparseArrays.cholmod_sparse_struct && typ != SparseArrays.cholmod_factor_struct
            else
                cholmod_exception = true
            end
            if !Base.allocatedinline(typ) && cholmod_exception
                shouldLoad = false
                offset %= sizeof(Int)
            else
                sz = max(1, actual_size(ET))
                offset %= sz
            end
        end

        if legal && (byref == GPUCompiler.MUT_REF || byref == GPUCompiler.BITS_REF) && Base.isconcretetype(typ)
            if shouldLoad
                byref = GPUCompiler.BITS_VALUE
            end

            legal = true

            while offset != 0 && legal
                @assert Base.isconcretetype(typ)
                seen = false
                lasti = 1

                for i in 1:typed_fieldcount(typ)
                    fo = typed_fieldoffset(typ, i)
                    if fo == offset && (i == typed_fieldcount(typ) || typed_fieldoffset(typ, i + 1) != offset)
                        offset = 0
			if in(typ, TypesNotToDisect)
			  legal = false
			end
                        typ = typed_fieldtype(typ, i)
                        if !Base.allocatedinline(typ)
                            if byref != GPUCompiler.BITS_VALUE
                                legal = false
                            end
                            byref = GPUCompiler.MUT_REF
                        end
                        seen = true
                        break
                    elseif fo > offset
                        offset = offset - typed_fieldoffset(typ, lasti)
			if in(typ, TypesNotToDisect)
			  legal = false
			end
                        typ = typed_fieldtype(typ, lasti)
                        if offset == 0
                            if !Base.allocatedinline(typ)
                                if byref != GPUCompiler.BITS_VALUE
                                    legal = false
                                end
                                byref = GPUCompiler.MUT_REF
                            end
                        else
                            if !Base.isconcretetype(typ) || !Base.allocatedinline(typ)
                                legal = false
                            end
                        end
                        seen = true
                        break
                    end

                    if (i != typed_fieldcount(typ) && fo != typed_fieldoffset(typ, i + 1)) ||
                            (i == typed_fieldcount(typ) && fo != actual_size(typ))
                        lasti = i
                    end
                end
                if !seen && typed_fieldcount(typ) > 0
                    offset = offset - typed_fieldoffset(typ, lasti)
			if in(typ, TypesNotToDisect)
			  legal = false
			end
                    typ = typed_fieldtype(typ, lasti)
                    if offset == 0
                        if !Base.allocatedinline(typ)
                            if byref != GPUCompiler.BITS_VALUE
                                legal = false
                            end
                            byref = GPUCompiler.MUT_REF
                        end
                    else
                        if !Base.isconcretetype(typ) || !Base.allocatedinline(typ)
                            legal = false
                        end
                    end
                    seen = true
                end
                if !seen
                    legal = false
                end
            end

            typ2 = typ
            while legal && should_recurse(typ2, value_type(arg), byref, dl)
		if !Base.isconcretetype(typ2)
                    legal = false
                    break
                end
                idx, _ = first_non_ghost(typ2)
                if idx != -1
			if in(typ2, TypesNotToDisect)
			  legal = false
			end
                    typ2 = typed_fieldtype(typ2, idx)
                    if Base.allocatedinline(typ2)
                        if byref == GPUCompiler.BITS_VALUE
                            continue
                        end
                        legal = false
                        break
                    else
                        if byref != GPUCompiler.BITS_VALUE
                            legal = false
                            break
                        end
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
        end
    end

    if isa(arg, LLVM.ExtractValueInst)
        larg = operands(arg)[1]
        indptrs = LLVM.API.LLVMGetIndices(arg)
        numind = LLVM.API.LLVMGetNumIndices(arg)
        offset = Cuint[unsafe_load(indptrs, i) for i in 1:numind]
        found, typ, byref = abs_typeof(larg, partial, seenphis)
        if !found
            return (false, nothing, nothing)
        end
        if byref == GPUCompiler.BITS_VALUE
            ltyp = typ
            for ind in offset
                if !Base.isconcretetype(typ)
                    throw(AssertionError("Illegal absint of $(string(arg)) ltyp=$ltyp, typ=$typ, offset=$offset, ind=$ind"))
                end
                cnt = 0
                desc = Base.DataTypeFieldDesc(typ)
		if in(typ, TypesNotToDisect)
		   return (false, nothing, nothing)
		end
                for i in 1:fieldcount(typ)
                    styp = typed_fieldtype(typ, i)
                    if isghostty(styp)
                        continue
                    end

                    # Extra i8 at the end of an inline union type
                    inline_union = !desc[i].isptr && styp isa Union
                    if cnt == ind
                        typ = styp
                        if inline_union
                            typ = remove_nothing_from_union_type(typ)
                        end
                        break
                    end
                    cnt += 1
                    if inline_union
                        if cnt == ind
                            typ = UInt8
                            break
                        end
                        cnt += 1
                    end
                end
            end
            if Base.allocatedinline(typ)
                return (true, typ, GPUCompiler.BITS_VALUE)
            else
                return (true, typ, GPUCompiler.MUT_REF)
            end
        end
    end

    if isa(arg, LLVM.PHIInst)
        if arg in seenphis
            return (false, nothing, nothing)
        end
        todo = LLVM.PHIInst[arg]
        ops = LLVM.Value[]
        seen = Set{LLVM.PHIInst}()
        legal = true
        while length(todo) > 0
            cur = pop!(todo)
            if cur in seen
                continue
            end
            push!(seen, cur)
            for (v, _) in LLVM.incoming(cur)
                v2, off = get_base_and_offset(v, inttoptr=false, addrcast=false)
                if off != 0
                    if isa(v, LLVM.Instruction) && arg in collect(operands(v))
                        legal = false
                        break
                    end
                    push!(ops, v)
                elseif v2 isa LLVM.PHIInst
                    push!(todo, v2)
                else
                    if isa(v2, LLVM.Instruction) && arg in collect(operands(v2))
                        legal = false
                        break
                    end
                    push!(ops, v2)
                end
            end
        end
        if legal
            resvals = nothing
            seenphis2 = copy(seenphis)
            push!(seenphis2, arg)
            for op in ops
                tmp = abs_typeof(op, partial, seenphis2)
                if resvals == nothing
                    resvals = tmp
                else
                    if tmp[1] == false || resvals[1] == false
                        resvals = (false, nothing, nothing)
                        break
                    elseif tmp[2] == resvals[2] && (tmp[3] == resvals[3] || (in(tmp[3], (GPUCompiler.BITS_REF, GPUCompiler.MUT_REF)) && in(resvals[3], (GPUCompiler.BITS_REF, GPUCompiler.MUT_REF))))

                        continue
                    elseif partial
                        resvals = (true, Union{resvals[2], tmp[2]}, GPUCompiler.BITS_REF)
                    else
                        resvals = (false, nothing, nothing)
                        break
                    end
                end
            end
            if resvals != nothing
                return resvals
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
	val = unbind(val)
        return (true, Core.Typeof(val), GPUCompiler.BITS_REF)
    end
    return (false, nothing, nothing)
end

@inline function is_zero(@nospecialize(x::LLVM.Value))::Bool
    if x isa LLVM.ConstantInt
        return convert(UInt, x) == 0
    end
    return false
end

function abs_cstring(@nospecialize(arg::LLVM.Value))::Tuple{Bool, String}
        ce = arg
        while isa(ce, ConstantExpr)
            if opcode(ce) == LLVM.API.LLVMAddrSpaceCast || opcode(ce) == LLVM.API.LLVMBitCast ||  opcode(ce) == LLVM.API.LLVMIntToPtr
                ce = operands(ce)[1]
            elseif opcode(ce) == LLVM.API.LLVMGetElementPtr
                if all(is_zero, operands(ce)[2:end])
                    ce = operands(ce)[1]
                else
                    break
                end
            else
                break
            end
        end
        
        larg = nothing
        if LLVM.API.LLVMGetValueKind(ce) == LLVM.API.LLVMGlobalAliasValueKind
            larg = LLVM.Value(ccall((:LLVMAliasGetAliasee, LLVM.API.libllvm), LLVM.API.LLVMValueRef, (LLVM.API.LLVMValueRef,), ce))
        elseif isa(ce, LLVM.GlobalVariable)
            larg = LLVM.initializer(ce)
        end

        if larg !== nothing
            if (isa(larg, LLVM.ConstantArray) || isa(larg, LLVM.ConstantDataArray)) && eltype(value_type(larg)) == LLVM.IntType(8)
                return (true, String(map(Base.Fix1(convert, UInt8), collect(larg)[1:(end - 1)])))
            end

        end
    return (false, "")
end
