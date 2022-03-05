using LLVM
using ObjectFile
using libblastrampoline_jll
using Libdl
import GPUCompiler: IRError, InvalidIRError

const ptr_map = Dict{Ptr{Cvoid},String}()

function restore_lookups(mod::LLVM.Module)
    i64 = LLVM.IntType(64; ctx=context(mod))
    for (v, k) in ptr_map
        if haskey(functions(mod), k)
            f = functions(mod)[k]
            replace_uses!(f, LLVM.Value(LLVM.API.LLVMConstIntToPtr(ConstantInt(i64, convert(Int, v)), llvmtype(f))))
            unsafe_delete!(mod, f)
        end
    end
end

function check_ir(job, mod::LLVM.Module)
    errors = check_ir!(job, IRError[], mod)
    unique!(errors)
    if !isempty(errors)
        throw(InvalidIRError(job, errors))
    end
end

function check_ir!(job, errors, mod::LLVM.Module)
    imported = Set(String[])
    if haskey(functions(mod), "malloc")
        f = functions(mod)["malloc"]
        name!(f, "")
        ctx = context(mod)
        ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))

        prev_ft = eltype(llvmtype(f)::LLVM.PointerType)::LLVM.FunctionType

        mfn = LLVM.API.LLVMAddFunction(mod, "malloc", LLVM.FunctionType(ptr8, parameters(prev_ft)))
        replace_uses!(f, LLVM.Value(LLVM.API.LLVMConstPointerCast(mfn, llvmtype(f))))
        unsafe_delete!(mod, f)
    end
    for f in collect(functions(mod))
        check_ir!(job, errors, imported, f)
    end

    return errors
end

function check_ir!(job, errors, imported, f::LLVM.Function)
    calls = []
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            push!(calls, inst)
        end
    end

    while length(calls) > 0
        inst = pop!(calls)
        check_ir!(job, errors, imported, inst, calls)
    end
    return errors
end

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

import GPUCompiler: DYNAMIC_CALL, DELAYED_BINDING, RUNTIME_FUNCTION, UNKNOWN_FUNCTION, POINTER_FUNCTION
import GPUCompiler: backtrace, isintrinsic
function check_ir!(job, errors, imported, inst::LLVM.CallInst, calls)
    bt = backtrace(inst)
    dest = called_value(inst)
    if isa(dest, LLVM.Function)
        fn = LLVM.name(dest)

        # some special handling for runtime functions that we don't implement
        if fn == "jl_get_binding_or_error"
        elseif fn == "jl_invoke"
        elseif fn == "jl_apply_generic"
        elseif fn == "gpu_malloc"
            ofn = LLVM.parent(LLVM.parent(inst))
            mod = LLVM.parent(ofn)
            ctx = context(mod)

            b = Builder(ctx)
            position!(b, inst)

            mfn = LLVM.API.LLVMGetNamedFunction(mod, "malloc")
            if mfn == C_NULL
                ptr8 = LLVM.PointerType(LLVM.IntType(8; ctx))
                mfn = LLVM.API.LLVMAddFunction(mod, "malloc", LLVM.FunctionType(ptr8, [llvmtype(LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0)))]))
            end
            mfn2 = LLVM.Function(mfn)
            nval = ptrtoint!(b, call!(b, mfn2, [LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0))]), llvmtype(inst))
            replace_uses!(inst, nval)
            LLVM.API.LLVMInstructionEraseFromParent(inst)
        elseif fn == "jl_load_and_lookup"
            ofn = LLVM.parent(LLVM.parent(inst))
            mod = LLVM.parent(ofn)
            ctx = context(mod)

            flib = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0))
            if isa(flib, LLVM.ConstantExpr)
                flib = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(flib, 0))
            end
            if isa(flib, LLVM.GlobalVariable)
                flib = LLVM.initializer(flib)
            end
            if isa(flib, LLVM.ConstantArray) && eltype(llvmtype(flib)) == LLVM.IntType(8; ctx)
                flib = String(map((x)->convert(UInt8, x), collect(flib)[1:(end-1)]))
            end

            fname = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 1))
            if isa(fname, LLVM.ConstantExpr)
                fname = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(fname, 0))
            end
            if isa(fname, LLVM.GlobalVariable)
                fname = LLVM.initializer(fname)
            end
            if isa(fname, LLVM.ConstantArray) && eltype(llvmtype(fname)) == LLVM.IntType(8; ctx)
                fname = String(map((x)->convert(UInt8, x), collect(fname)[1:(end-1)]))
            end
            hnd = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 2))
            if isa(hnd, LLVM.GlobalVariable)
                hnd = LLVM.name(hnd)
            end

            if !isa(hnd, String) || !isa(fname, String) || !isa(flib, String)
                push!(errors, ("jl_load_and_lookup", bt, nothing))
                return
            end
            # res = ccall(:jl_load_and_lookup, Ptr{Cvoid}, (Cstring, Cstring, Ptr{Cvoid}), flib, fname, cglobal(Symbol(hnd)))
            push!(errors, ("jl_load_and_lookup", bt, nothing))
        elseif fn == "jl_lazy_load_and_lookup" || fn == "ijl_lazy_load_and_lookup"
            ofn = LLVM.parent(LLVM.parent(inst))
            mod = LLVM.parent(ofn)
            ctx = context(mod)

            flib = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0))
            if isa(flib, LLVM.LoadInst)
                op = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(flib, 0))
                if isa(op, LLVM.ConstantExpr)
                    op1 = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(op, 0))
                    if isa(op1, LLVM.ConstantExpr)
                        op2 = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(op1, 0))
                        if isa(op2, ConstantInt)
                            rep = reinterpret(Ptr{Cvoid}, convert(Csize_t, op2)+8)
                            ld = unsafe_load(convert(Ptr{Ptr{Cvoid}}, rep))
                            flib = Base.unsafe_pointer_to_objref(ld)
                        end
                    end
                end
            end

            fname = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 1))
            if isa(fname, LLVM.ConstantExpr)
                fname = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(fname, 0))
            end
            if isa(fname, LLVM.GlobalVariable)
                fname = LLVM.initializer(fname)
            end
            if isa(fname, LLVM.ConstantArray) && eltype(llvmtype(fname)) == LLVM.IntType(8; ctx)
                fname = String(map((x)->convert(UInt8, x), collect(fname)[1:(end-1)]))
            end

            if !isa(fname, String) || !isa(flib, String)
                push!(errors, ("jl_lazy_load_and_lookup", bt, nothing))
                return
            end

            data = open(flib, "r") do io
                lib = readmeta(io)
                sections = Sections(lib)
                if !(".llvmbc" in sections)
                    return nothing
                end
                llvmbc = read(findfirst(sections, ".llvmbc"))
                return llvmbc
            end

            found = false
            if data !== nothing
                inmod = parse(LLVM.Module, data; ctx)
                found = haskey(functions(inmod), fname)
            end

            if found
                if !(fn in imported)
                    internalize = String[]
                    for fn in functions(inmod)
                        if !isempty(LLVM.blocks(fn))
                            push!(internalize, name(fn))
                        end
                    end
                    for g in globals(inmod)
                        linkage!(g, LLVM.API.LLVMExternalLinkage)
                    end
                    # override libdevice's triple and datalayout to avoid warnings
                    triple!(inmod, triple(mod))
                    datalayout!(inmod, datalayout(mod))
                    GPUCompiler.link_library!(mod, inmod)
                    for n in internalize
                        linkage!(functions(mod)[n], LLVM.API.LLVMInternalLinkage)
                    end
                    push!(imported, fn)
                end
                replaceWith = functions(mod)[fname]

                for u in LLVM.uses(inst)
                    st = LLVM.user(u)
                    if isa(st, LLVM.StoreInst) && LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(st, 0)) == inst
                        ptr = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(st, 1))
                        for u in LLVM.uses(ptr)
                            ld = LLVM.user(u)
                            if isa(ld, LLVM.LoadInst)
                                b = Builder(ctx)
                                position!(b, ld)
                                replace_uses!(ld, LLVM.pointercast!(b, replaceWith, llvmtype(inst)))
                            end
                        end
                    end
                end

                b = Builder(ctx)

                position!(b, inst)
                replace_uses!(inst, LLVM.pointercast!(b, replaceWith, llvmtype(inst)))
                LLVM.API.LLVMInstructionEraseFromParent(inst)

            else
                if fn == "jl_lazy_load_and_lookup"
                    res = ccall(:jl_lazy_load_and_lookup, Ptr{Cvoid}, (Any, Cstring), flib, fname)
                else
                    res = ccall(:ijl_lazy_load_and_lookup, Ptr{Cvoid}, (Any, Cstring), flib, fname)
                end
                replaceWith = LLVM.ConstantInt(LLVM.IntType(64; ctx), reinterpret(UInt64, res))
                for u in LLVM.uses(inst)
                    st = LLVM.user(u)
                    if isa(st, LLVM.StoreInst) && LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(st, 0)) == inst
                        ptr = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(st, 1))
                        for u in LLVM.uses(ptr)
                            ld = LLVM.user(u)
                            if isa(ld, LLVM.LoadInst)
                                b = Builder(ctx)
                                position!(b, ld)
                                for u in LLVM.uses(ld)
                                    u = LLVM.user(u)
                                    if isa(u, LLVM.CallInst)
                                        push!(calls, u)
                                    end
                                end
                                replace_uses!(ld, LLVM.inttoptr!(b, replaceWith, llvmtype(inst)))
                            end
                        end
                    end
                end

                b = Builder(ctx)
                position!(b, inst)
                replacement = LLVM.inttoptr!(b, replaceWith, llvmtype(inst))
                            for u in LLVM.uses(inst)
                                u = LLVM.user(u)
                                if isa(u, LLVM.CallInst)
                                    push!(calls, u)
                                end
                                if isa(u, LLVM.PHIInst)
                                    if all(x->first(x) == inst || first(x) == replacement, LLVM.incoming(u))

                                        for u in LLVM.uses(u)
                                            u = LLVM.user(u)
                                            if isa(u, LLVM.CallInst)
                                                push!(calls, u)
                                            end
                                            if isa(u, LLVM.BitCastInst)
                                                for u1 in LLVM.uses(u)
                                                    u1 = LLVM.user(u1)
                                                    if isa(u1, LLVM.CallInst)
                                                        push!(calls, u1)
                                                    end
                                                end
                                                replace_uses!(u, LLVM.inttoptr!(b, replaceWith, llvmtype(u)))
                                            end
                                        end
                                    end
                                end
                            end
                replace_uses!(inst, replacement)
                LLVM.API.LLVMInstructionEraseFromParent(inst)
            end
        end

    elseif isa(dest, InlineAsm)
        # let's assume it's valid ASM

    elseif isa(dest, ConstantExpr)
        # Enzyme should be able to handle these
        # detect calls to literal pointers and replace with function name, if possible
        if occursin("inttoptr", string(dest))
            # extract the literal pointer
            ptr_arg = first(operands(dest))
            GPUCompiler.@compiler_assert isa(ptr_arg, ConstantInt) job
            ptr_val = convert(Int, ptr_arg)
            ptr = Ptr{Cvoid}(ptr_val)

            # look it up in the Julia JIT cache
            frames = ccall(:jl_lookup_code_address, Any, (Ptr{Cvoid}, Cint,), ptr, 0)

            if length(frames) >= 1
                if VERSION >= v"1.4.0-DEV.123"
                    fn, file, line, linfo, fromC, inlined = last(frames)
                else
                    fn, file, line, linfo, fromC, inlined, ip = last(frames)
                end

                known_names = ("jl_alloc_array_1d", "jl_alloc_array_2d", "jl_alloc_array_3d","jl_new_array","jl_array_copy","jl_alloc_string",
                                "jl_in_threaded_region","jl_enter_threaded_region","jl_exit_threaded_region","jl_set_task_tid","jl_new_task",
                                "malloc","memmove","memcpy","jl_array_grow_beg","jl_array_grow_end","jl_array_grow_at","jl_array_del_beg",
                                "jl_array_del_end","jl_array_del_at","jl_array_ptr","jl_value_ptr","jl_get_ptls_states","jl_gc_add_finalizer_th",
                                "jl_symbol_n")
                fn = string(fn)
                if length(fn) == 0
                    if length(ptr_map) == 0
                        for name in known_names
                            ptr_map[LLVM.find_symbol(name)] = name
                        end
                        if libblastrampoline_jll.is_available()
                            for s in Symbols(readmeta(open(libblastrampoline_jll.libblastrampoline_path,"r")))
                                name = symbol_name(s)
                                if name != ""
                                    found = Libdl.dlsym(libblastrampoline_jll.libblastrampoline_handle,name; throw_error=false)
                                    if found !== nothing
                                        ptr_map[found] = name
                                    end
                                end
                            end
                        end
                    end
                    fn = get(ptr_map, ptr, "")
                else
                    ptr_map[ptr] = fn
                end


                if length(fn) > 1 && fromC
                    mod = LLVM.parent(LLVM.parent(LLVM.parent(inst)))
                    lfn = LLVM.API.LLVMGetNamedFunction(mod, fn)
                    if lfn == C_NULL
                        lfn = LLVM.API.LLVMAddFunction(mod, fn, LLVM.API.LLVMGetCalledFunctionType(inst))
                    else
                        lfn = LLVM.API.LLVMConstBitCast(lfn, LLVM.PointerType(LLVM.FunctionType(LLVM.API.LLVMGetCalledFunctionType(inst))))
                    end
                    LLVM.API.LLVMSetOperand(inst, LLVM.API.LLVMGetNumOperands(inst)-1, lfn)
                end
            end
        end
    end

    return errors
end
