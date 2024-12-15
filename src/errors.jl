const VERBOSE_ERRORS = Ref(false)

abstract type CompilationException <: Base.Exception end

struct EnzymeRuntimeException <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeRuntimeException)
    print(io, "Enzyme execution failed.\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct NoDerivativeException <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::NoDerivativeException)
    print(io, "Enzyme compilation failed.\n")
    if ece.ir !== nothing
    	if VERBOSE_ERRORS[]
            print(io, "Current scope: \n")
            print(io, ece.ir)
    	else
	    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
	end
    end
    if occursin("cannot handle unknown binary operator", ece.msg)
      for msg in split(ece.msg, '\n')
        if occursin("cannot handle unknown binary operator", msg)
          print('\n', msg, '\n')
        end
      end
    else
      print(io, '\n', ece.msg, '\n')
    end
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct IllegalTypeAnalysisException <: CompilationException
    msg::String
    sval::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalTypeAnalysisException)
    print(io, "Enzyme compilation failed due to illegal type analysis.\n")
    print(io, " This usually indicates the use of a Union type, which is not fully supported with Enzyme.API.strictAliasing set to true [the default].\n")
    print(io, " Ideally, remove the union (which will also make your code faster), or try setting Enzyme.API.strictAliasing!(false) before any autodiff call.\n")
    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    if VERBOSE_ERRORS[]
        if ece.ir !== nothing
            print(io, "Current scope: \n")
            print(io, ece.ir)
        end
        print(io, "\n Type analysis state: \n")
        write(io, ece.sval)
        print(io, '\n', ece.msg, '\n')
    end
    if ece.bt !== nothing
        print(io, "\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct IllegalFirstPointerException <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalFirstPointerException)
    print(io, "Enzyme compilation failed due to an internal error (first pointer exception).\n")
    print(io, " Please open an issue with the code to reproduce and full error log on github.com/EnzymeAD/Enzyme.jl\n")
    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    if VERBOSE_ERRORS[]
      if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
      end
    end
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct EnzymeInternalError <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::EnzymeInternalError)
    print(io, "Enzyme compilation failed due to an internal error.\n")
    print(io, " Please open an issue with the code to reproduce and full error log on github.com/EnzymeAD/Enzyme.jl\n")
    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    if VERBOSE_ERRORS[]
      if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
      end
      print(io, '\n', ece.msg, '\n')
    else
      for msg in split(ece.msg, '\n')
        if occursin("Illegal replace ficticious phi for", msg)
          print('\n', msg, '\n')
        end
      end
    end
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct EnzymeMutabilityException <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeMutabilityException)
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeRuntimeActivityError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeRuntimeActivityError)
    println(io, "Constant memory is stored (or returned) to a differentiable variable.")
    println(
        io,
        "As a result, Enzyme cannot provably ensure correctness and throws this error.",
    )
    println(
        io,
        "This might be due to the use of a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/faq/#Runtime-Activity).",
    )
    println(
        io,
        "If Enzyme should be able to prove this use non-differentable, open an issue!",
    )
    println(io, "To work around this issue, either:")
    println(
        io,
        " a) rewrite this variable to not be conditionally active (fastest, but requires a code change), or",
    )
    println(
        io,
        " b) set the Enzyme mode to turn on runtime activity (e.g. autodiff(set_runtime_activity(Reverse), ...) ). This will maintain correctness, but may slightly reduce performance.",
    )
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoTypeError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoTypeError)
    print(io, "Enzyme cannot deduce type\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoShadowError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoShadowError)
    print(io, "Enzyme could not find shadow for value\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoDerivativeError <: Base.Exception
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoDerivativeError)
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

parent_scope(val::LLVM.Function, depth = 0) = depth == 0 ? LLVM.parent(val) : val
parent_scope(val::LLVM.Module, depth = 0) = val
parent_scope(@nospecialize(val::LLVM.Value), depth = 0) = parent_scope(LLVM.parent(val), depth + 1)
parent_scope(val::LLVM.Argument, depth = 0) =
    parent_scope(LLVM.Function(LLVM.API.LLVMGetParamParent(val)), depth + 1)

function julia_error(
    cstr::Cstring,
    val::LLVM.API.LLVMValueRef,
    errtype::API.ErrorType,
    data::Ptr{Cvoid},
    data2::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
)::LLVM.API.LLVMValueRef
    msg = Base.unsafe_string(cstr)
    julia_error(msg, val, errtype, data, data2, B)
end

function julia_error(
    msg::String,
    val::LLVM.API.LLVMValueRef,
    errtype::API.ErrorType,
    data::Ptr{Cvoid},
    data2::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
)::LLVM.API.LLVMValueRef
    bt = nothing
    ir = nothing
    if val != C_NULL
        val = LLVM.Value(val)
        if isa(val, LLVM.Instruction)
            dbgval = val
            while !haskey(metadata(dbgval), LLVM.MD_dbg)
                dbgval = LLVM.API.LLVMGetNextInstruction(dbgval)
                if dbgval == C_NULL
                    dbgval = nothing
                    break
                else
                    dbgval = LLVM.Instruction(dbgval)
                end
            end
            if dbgval !== nothing
                bt = GPUCompiler.backtrace(dbgval)
            end
        end
        if isa(val, LLVM.ConstantExpr)
            for u in LLVM.uses(val)
                u = LLVM.user(u)
                if isa(u, LLVM.Instruction)
                    bt = GPUCompiler.backtrace(val)
                end
            end
        else
            # Need to convert function to string, since when the error is going to be printed
            # the module might have been destroyed
            ir = string(parent_scope(val))
        end
    end

    if errtype == API.ET_NoDerivative
        if occursin("No create nofree of empty function", msg) ||
           occursin("No forward mode derivative found for", msg) ||
           occursin("No augmented forward pass", msg) ||
           occursin("No reverse pass found", msg)
            ir = nothing
        end
        if B != C_NULL
            B = IRBuilder(B)
            msg2 = sprint() do io
                if ir !== nothing
                    print(io, "Current scope: \n")
                    print(io, ir)
                end
                print(io, '\n', msg, '\n')
                if bt !== nothing
                    Base.show_backtrace(io, bt)
                    println(io)
                end
            end
            emit_error(B, nothing, msg2, EnzymeNoDerivativeError)
            return C_NULL
        end
        throw(NoDerivativeException(msg, ir, bt))
    elseif errtype == API.ET_NoShadow
        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))

        msgN = sprint() do io::IO
            if isa(val, LLVM.Argument)
                fn = parent_scope(val)
                ir = string(LLVM.name(fn)) * string(function_type(fn))
                print(io, "Current scope: \n")
                print(io, ir)
            end
            if !isa(val, LLVM.Argument)
                print(io, "\n Inverted pointers: \n")
                ip = API.EnzymeGradientUtilsInvertedPointersToString(gutils)
                sval = Base.unsafe_string(ip)
                write(io, sval)
                API.EnzymeStringFree(ip)
            end
            print(io, '\n', msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(IRBuilder(B), nothing, msgN, EnzymeNoShadowError)
        return LLVM.null(get_shadow_type(gutils, value_type(val))).ref
    elseif errtype == API.ET_IllegalTypeAnalysis
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)

        if isa(val, LLVM.Instruction)
            mi, rt = enzyme_custom_extract_mi(
                LLVM.parent(LLVM.parent(val))::LLVM.Function,
                false,
            ) #=error=#
            if mi !== nothing
                msg *= "\n" * string(mi) * "\n"
            end
        end
        throw(IllegalTypeAnalysisException(msg, sval, ir, bt))
    elseif errtype == API.ET_NoType
        @assert B != C_NULL
        B = IRBuilder(B)

        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)

        msg2 = sprint() do io::IO
            if !occursin("Cannot deduce single type of store", msg)
                if ir !== nothing
                    print(io, "Current scope: \n")
                    print(io, ir)
                end
                print(io, "\n Type analysis state: \n")
                write(io, sval)
            end
            print(io, '\n', msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
            pscope = parent_scope(val)
            mi, rt = enzyme_custom_extract_mi(pscope, false) #=error=#
            if mi !== nothing
                println(io, "within ", mi)
            end
        end
        emit_error(B, nothing, msg2, EnzymeNoTypeError)
        return C_NULL
    elseif errtype == API.ET_IllegalFirstPointer
        throw(IllegalFirstPointerException(msg, ir, bt))
    elseif errtype == API.ET_InternalError
        throw(EnzymeInternalError(msg, ir, bt))
    elseif errtype == API.ET_TypeDepthExceeded
        msg2 = sprint() do io
            print(io, msg)
            println(io)

            if val != C_NULL
                println(io, val)
            end

            st = API.EnzymeTypeTreeToString(data)
            println(io, Base.unsafe_string(st))
            API.EnzymeStringFree(st)

            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        GPUCompiler.@safe_warn msg2
        return C_NULL
    elseif errtype == API.ET_IllegalReplaceFicticiousPHIs
        data2 = LLVM.Value(data2)
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            println(io, string(LLVM.parent(LLVM.parent(data2))))
            println(io, val)
            println(io, data2)
        end
        throw(EnzymeInternalError(msg2, ir, bt))
    elseif errtype == API.ET_MixedActivityError
        data2 = LLVM.Value(data2)
        badval = nothing
        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))
        # Ignore mismatched activity if phi/store of ghost
        seen = Dict{LLVM.Value,LLVM.Value}()
        illegal = false
        created = LLVM.Instruction[]
        world = enzyme_extract_world(LLVM.parent(position(IRBuilder(B))))
        width = get_width(gutils)
        function make_batched(@nospecialize(cur::LLVM.Value), B::LLVM.IRBuilder)::LLVM.Value
            if width == 1
                return cur
            else
                shadowres = UndefValue(
                    LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur))),
                )
                for idx = 1:width
                    shadowres = insert_value!(B, shadowres, cur, idx - 1)
                    if isa(shadowres, LLVM.Instruction)
                        push!(created, shadowres)
                    end
                end
                return shadowres
            end
        end

        illegalVal = nothing
        mode = get_mode(gutils)

        function make_replacement(@nospecialize(cur::LLVM.Value), prevbb::LLVM.IRBuilder)::LLVM.Value
            ncur = new_from_original(gutils, cur)
            if cur in keys(seen)
                return seen[cur]
            end

            legal, TT, byref = abs_typeof(cur, true)

            if legal
                if guaranteed_const_nongen(TT, world)
                    return make_batched(ncur, prevbb)
                end

                legal2, obj = absint(cur)

                # Only do so for the immediate operand/etc to a phi, since otherwise we will make multiple
                if legal2
                   if active_reg_inner(TT, (), world) == ActiveState &&
                   isa(cur, LLVM.ConstantExpr) &&
                   cur == data2
                    if width == 1
                        if mode == API.DEM_ForwardMode
                            instance = make_zero(obj)
                            return unsafe_to_llvm(prevbb, instance)
                        else
                            res = emit_allocobj!(prevbb, Base.RefValue{TT})
                            push!(created, res)
                            return res
                        end
                    else
                        shadowres = UndefValue(
                            LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur))),
                        )
                        for idx = 1:width
                            res = if mode == API.DEM_ForwardMode
                                instance = make_zero(obj)
                                unsafe_to_llvm(prevbb, instance)
                            else
                                sres = emit_allocobj!(prevbb, Base.RefValue{TT})
                                push!(created, sres)
                                sres
                            end
                            shadowres = insert_value!(prevbb, shadowres, res, idx - 1)
                            if shadowres isa LLVM.Instruction
				push!(created, shadowres)
			    end
                        end
                        return shadowres
                    end
                    end

@static if VERSION < v"1.11-"
else    
                    if obj isa Memory && obj == typeof(obj).instance
                        return make_batched(ncur, prevbb)
                    end
end
                end

@static if VERSION < v"1.11-"
else   
                if isa(cur, LLVM.LoadInst)
                    larg, off = get_base_and_offset(operands(cur)[1])
                    if isa(larg, LLVM.LoadInst)
                        legal2, obj = absint(larg)
                        if legal2 && obj isa Memory && obj == typeof(obj).instance
                            return make_batched(ncur, prevbb)
                        end
                    end
                end
end

                badval = if legal2
                    string(obj) * " of type" * " " * string(TT)
                else
                    "Unknown object of type" * " " * string(TT)
                end
                @assert !illegal
                illegalVal = cur
                illegal = true
                return make_batched(ncur, prevbb)
            end

            if isa(cur, LLVM.PointerNull)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.UndefValue)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.PoisonValue)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantAggregateZero)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantAggregate)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantInt)
                if convert(UInt64, cur) == 0
                    return make_batched(ncur, prevbb)
                end
            end
            if isa(cur, LLVM.ConstantFP)
                return make_batched(ConstantFP(value_type(cur), 0), prevbb)
            end
            if isa(cur, LLVM.ConstantDataSequential)
                cvals = LLVM.Value[]
                changed = false
                for v in collect(cur)
                    tmp = make_replacement(v, prevbb)
                    if illegal
                        return ncur
                    end
                    if v != tmp
                        changed = true
                    end
                    push!(cvals, tmp)
                end

                cur2 = if changed
                    @assert !illegal
                    illegalVal = cur
                    illegal = true
                    # TODO replace with correct insertions/splats
                    ncur
                else
                    make_batched(ncur, prevbb)
                end
                return cur2
            end
            if isa(cur, LLVM.ConstantInt)
                if LLVM.width(value_type(cur)) <= sizeof(Int) * 8
                    return make_batched(ncur, prevbb)
                end
                if LLVM.width(value_type(cur)) == sizeof(Int) * 8 &&
                   abs(convert(Int, cur)) < 10000
                    return make_batched(ncur, prevbb)
                end
                # if storing a constant int as a non-pointer, presume it is not a GC'd var and is safe
                # for activity state to mix
                if isa(val, LLVM.StoreInst)
                    operands(val)[1] == cur &&
                        !isa(value_type(operands(val)[1]), LLVM.PointerType)
                    return make_batched(ncur, prevbb)
                end
            end

            if isa(cur, LLVM.SelectInst)
                lhs = make_replacement(operands(cur)[2], prevbb)
                if illegal
                    return ncur
                end
                rhs = make_replacement(operands(cur)[3], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(cur)[2] && rhs == operands(cur)[3]
                    return make_batched(ncur, prevbb)
                end
                if width == 1
                    nv = select!(
                        prevbb,
                        new_from_original(gutils, operands(cur)[1]),
                        lhs,
                        rhs,
                    )
                    push!(created, nv)
                    seen[cur] = nv
                    return nv
                else
                    shadowres = LLVM.UndefValue(value_type(lhs))
                    for idx = 1:width
                        shadowres = insert_value!(
                            prevbb,
                            shadowres,
                            select!(
                                prevbb,
                                new_from_original(gutils, operands(cur)[1]),
                                extract_value!(prevbb, lhs, idx - 1),
                                extract_value!(prevbb, rhs, idx - 1),
                            ),
                            idx - 1,
                        )
                        if isa(shadowres, LLVM.Instruction)
                            push!(created, shadowres)
                        end
                    end
                    return shadowres
                end
            end

            if isa(cur, LLVM.InsertValueInst)
                lhs = make_replacement(operands(cur)[1], prevbb)
                if illegal
                    return ncur
                end
                rhs = make_replacement(operands(cur)[2], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(cur)[1] && rhs == operands(cur)[2]
                    return make_batched(ncur, prevbb)
                end
                inds = LLVM.API.LLVMGetIndices(cur.ref)
                ninds = LLVM.API.LLVMGetNumIndices(cur.ref)
                jinds = Cuint[unsafe_load(inds, i) for i = 1:ninds]
                if width == 1
                    nv = API.EnzymeInsertValue(prevbb, lhs, rhs, jinds)
                    push!(created, nv)
                    seen[cur] = nv
                    return nv
                else
                    shadowres = lhs
                    for idx = 1:width
                        jindsv = copy(jinds)
                        pushfirst!(jindsv, idx - 1)
                        shadowres = API.EnzymeInsertValue(
                            prevbb,
                            shadowres,
                            extract_value!(prevbb, rhs, idx - 1),
                            jindsv,
                        )
                        if isa(shadowres, LLVM.Instruction)
                            push!(created, shadowres)
                        end
                    end
                    return shadowres
                end
            end
           
            if isa(cur, LLVM.LoadInst) || isa(cur, LLVM.BitCastInst) || isa(cur, LLVM.AddrSpaceCastInst) || (isa(cur, LLVM.GetElementPtrInst) && all(Base.Fix2(isa, LLVM.ConstantInt), operands(cur)[2:end]))
                lhs = make_replacement(operands(cur)[1], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(ncur)[1]
                    return make_batched(ncur, prevbb)
                elseif width != 1 && isa(lhs, LLVM.InsertValueInst) && operands(lhs)[2] == operands(ncur)[1]
                    return make_batched(ncur, prevbb)
                end
            end

            if isa(cur, LLVM.PHIInst)
                Bphi = IRBuilder()
                position!(Bphi, ncur)
                shadowty = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur)))
                phi2 = phi!(Bphi, shadowty, "tempphi" * LLVM.name(cur))
                seen[cur] = phi2
                changed = false
                recsize = length(created) + 1
                for (v, bb) in LLVM.incoming(cur)
                    B2 = IRBuilder()
                    position!(B2, new_from_original(gutils, last(instructions(bb))))
                    tmp = make_replacement(v, B2)
                    if illegal
                        changed = true
                        break
                    end
                    @assert value_type(tmp) == shadowty
                    if tmp != new_from_original(gutils, v) && v != cur
                        changed = true
                    end
                    push!(LLVM.incoming(phi2), (tmp, new_from_original(gutils, bb)))
                end
                if !changed || illegal
                    LLVM.API.LLVMInstructionEraseFromParent(phi2)
                    seen[cur] = ncur
                    plen = length(created)
                    for i = recsize:plen
                        u = created[i]
                        replace_uses!(u, LLVM.UndefValue(value_type(u)))
                    end
                    for i = recsize:plen
                        u = created[i]
                        LLVM.API.LLVMInstructionEraseFromParent(u)
                    end
                    for i = recsize:plen
                        pop!(created)
                    end
                    return illegal ? ncur : make_batched(ncur, prevbb)
                end
                push!(created, phi2)
                return phi2
            end
            
            tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, cur))
            st = API.EnzymeTypeTreeToString(tt)
            st2 = Base.unsafe_string(st)
            API.EnzymeStringFree(st)
            if st2 == "{[-1]:Integer}"
                return make_batched(ncur, prevbb)
            end

            if !illegal
                illegal = true
                illegalVal = cur
            end
            return ncur
        end

        b = IRBuilder(B)
        replacement = make_replacement(data2, b)

        if !illegal
            return replacement.ref
        end
        for u in created
            replace_uses!(u, LLVM.UndefValue(value_type(u)))
        end
        for u in created
            LLVM.API.LLVMInstructionEraseFromParent(u)
        end
        if LLVM.API.LLVMIsAReturnInst(val) != C_NULL
            mi, rt = enzyme_custom_extract_mi(
                LLVM.parent(LLVM.parent(val))::LLVM.Function,
                false,
            ) #=error=#
            if mi !== nothing && isghostty(rt)
                return C_NULL
            end
        end
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            if badval !== nothing
                println(io, " value=" * badval)
            else
                ttval = val
                if isa(ttval, LLVM.StoreInst)
                    ttval = operands(ttval)[1]
                end
                tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, ttval))
                st = API.EnzymeTypeTreeToString(tt)
                print(io, "Type tree: ")
                println(io, Base.unsafe_string(st))
                API.EnzymeStringFree(st)
            end
            if illegalVal !== nothing
                println(io, " llvalue=" * string(illegalVal))
            end
            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        emit_error(b, nothing, msg2, EnzymeRuntimeActivityError)
        return C_NULL
    elseif errtype == API.ET_GetIndexError
        @assert B != C_NULL
        B = IRBuilder(B)
        msg5 = sprint() do io::IO
            print(io, "Enzyme internal error\n")
            print(io, msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(B, nothing, msg5)
        return C_NULL
    end
    throw(AssertionError("Unknown errtype"))
end

