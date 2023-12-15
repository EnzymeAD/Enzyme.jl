using LLVM
using ObjectFile
using Libdl

module FFI
    using LLVM
    module BLASSupport
        # TODO: LAPACK handling
        using LinearAlgebra
        using ObjectFile
        using Libdl
        if VERSION >= v"1.7"
            function __init__()
                if VERSION > v"1.8"
                  global blas_handle = Libdl.dlopen(BLAS.libblastrampoline)
                else
                  global blas_handle = Libdl.dlopen(BLAS.libblas)
                end
            end
            function get_blas_symbols()
                symbols = BLAS.get_config().exported_symbols
                if BLAS.USE_BLAS64
                    return map(n->n*"64_", symbols)
                end
                return symbols
            end

            function lookup_blas_symbol(name)
                Libdl.dlsym(blas_handle::Ptr{Cvoid}, name; throw_error=false)
            end
        else
            function __init__()
                global blas_handle = Libdl.dlopen(BLAS.libblas)
            end
            function get_blas_symbols()
                symbols = Set{String}()
                path = Libdl.dlpath(BLAS.libblas)
                ignoreSymbols = Set(String["", "edata", "_edata", "end", "_end", "_bss_start", "__bss_start", ".text", ".data"])
                for meta in readmeta(open(path, "r"))
                    for s in Symbols(meta)
                        name = symbol_name(s)
                        if !Sys.iswindows() && BLAS.vendor() == :openblas64
                            endswith(name, "64_") || continue
                        else
                            endswith(name, "_") || continue
                        end
                        if !in(name, ignoreSymbols)
                            push!(symbols, name)
                        end
                    end
                end
                symbols = collect(symbols)
                if Sys.iswindows() &&  BLAS.vendor() == :openblas64
                    return map(n->n*"64_", symbols)
                end
                return symbols
            end

            function lookup_blas_symbol(name)
                Libdl.dlsym(blas_handle::Ptr{Cvoid}, name; throw_error=false)
            end
        end
    end

    const ptr_map = Dict{Ptr{Cvoid},String}()

    function __init__()
        known_names = (
            "jl_alloc_array_1d", "jl_alloc_array_2d", "jl_alloc_array_3d", 
            "ijl_alloc_array_1d", "ijl_alloc_array_2d", "ijl_alloc_array_3d", 
            "jl_new_array", "ijl_new_array",
            "jl_array_copy", "ijl_array_copy",
            "jl_alloc_string",
            "jl_in_threaded_region", "jl_enter_threaded_region", "jl_exit_threaded_region", "jl_set_task_tid", "jl_new_task",
            "malloc", "memmove", "memcpy", "memset",
            "jl_array_grow_beg", "ijl_array_grow_beg",
            "jl_array_grow_end", "ijl_array_grow_end",
            "jl_array_grow_at", "ijl_array_grow_at",
            "jl_array_del_beg", "ijl_array_del_beg",
            "jl_array_del_end", "ijl_array_del_end",
            "jl_array_del_at", "ijl_array_del_at",
            "jl_array_ptr", "ijl_array_ptr",
            "jl_value_ptr", "jl_get_ptls_states", "jl_gc_add_finalizer_th",
            "jl_symbol_n", "jl_", "jl_object_id",
            "jl_reshape_array","ijl_reshape_array",
            "jl_matching_methods", "ijl_matching_methods",
            "jl_array_sizehint", "ijl_array_sizehint",
            "jl_get_keyword_sorter", "ijl_get_keyword_sorter",
            "jl_ptr_to_array",
            "jl_box_float32", 
            "ijl_box_float32", 
            "jl_box_float64", 
            "ijl_box_float64", 
            "jl_ptr_to_array_1d",
            "jl_eqtable_get", "ijl_eqtable_get",
            "memcmp","memchr",
            "jl_get_nth_field_checked", "ijl_get_nth_field_checked",
            "jl_stored_inline",
            "ijl_stored_inline",
            "jl_array_isassigned", "ijl_array_isassigned",
            "jl_array_ptr_copy", "ijl_array_ptr_copy",
            "jl_array_typetagdata", "ijl_array_typetagdata",
        )
        for name in known_names
            sym = LLVM.find_symbol(name)
            if sym == C_NULL
                continue
            end
            if haskey(ptr_map, sym)
                # On MacOS memcpy and memmove seem to collide?
                if name == "memcpy"
                    continue
                end
            end
            @assert !haskey(ptr_map, sym)
            ptr_map[sym] = name
        end
        for sym in BLASSupport.get_blas_symbols()
            ptr = BLASSupport.lookup_blas_symbol(sym)
            if ptr !== nothing
                if haskey(ptr_map, ptr)
                    if ptr_map[ptr] != sym
                        @warn "Duplicated symbol in ptr_map" ptr, sym, ptr_map[ptr]
                    end
                    continue
                end
                ptr_map[ptr] = sym
            end
        end
    end

    function memoize!(ptr, fn)
        fn = get(ptr_map, ptr, fn)
        if !haskey(ptr_map, ptr)
            ptr_map[ptr] = fn
        else
            @assert ptr_map[ptr] == fn
        end
        return fn
    end
end

import GPUCompiler: IRError, InvalidIRError

function restore_lookups(mod::LLVM.Module)
    T_size_t = convert(LLVM.LLVMType, Int)
    for (v, k) in FFI.ptr_map
        if haskey(functions(mod), k)
            f = functions(mod)[k]
            replace_uses!(f, LLVM.Value(LLVM.API.LLVMConstIntToPtr(ConstantInt(T_size_t, convert(UInt, v)), value_type(f))))
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
        ptr8 = LLVM.PointerType(LLVM.IntType(8))

        prev_ft = eltype(value_type(f)::LLVM.PointerType)::LLVM.FunctionType

        mfn = LLVM.API.LLVMAddFunction(mod, "malloc", LLVM.FunctionType(ptr8, parameters(prev_ft)))
        replace_uses!(f, LLVM.Value(LLVM.API.LLVMConstPointerCast(mfn, value_type(f))))
        unsafe_delete!(mod, f)
    end
    for f in collect(functions(mod))
        check_ir!(job, errors, imported, f)
    end
    for f in collect(functions(mod))
        check_ir!(job, errors, imported, f)
    end

    return errors
end

function check_ir!(job, errors, imported, f::LLVM.Function)
    calls = []
    isInline = API.EnzymeGetCLBool(cglobal((:EnzymeInline, API.libEnzyme))) != 0
    for bb in blocks(f), inst in instructions(bb)
        if isa(inst, LLVM.CallInst)
            push!(calls, inst)
        # remove illegal invariant.load and jtbaa_const invariants
        elseif isInline && isa(inst, LLVM.LoadInst)
            md = metadata(inst)
            if haskey(md, LLVM.MD_tbaa)
                modified = LLVM.Metadata(ccall((:EnzymeMakeNonConstTBAA, API.libEnzyme), LLVM.API.LLVMMetadataRef, (LLVM.API.LLVMMetadataRef,), md[LLVM.MD_tbaa]))
                setindex!(md, modified, LLVM.MD_tbaa)
            end
            if haskey(md, LLVM.MD_invariant_load)
                delete!(md, LLVM.MD_invariant_load)
            end
        end
    end

    while length(calls) > 0
        inst = pop!(calls)
        check_ir!(job, errors, imported, inst, calls)
    end
    return errors
end

const libjulia = Ref{Ptr{Cvoid}}(C_NULL)

# List of methods to location of arg which is the mi/function, then start of args
const generic_method_offsets = Dict{String, Tuple{Int,Int}}(("jl_f__apply_latest" => (2,3), "ijl_f__apply_latest" => (2,3), "jl_f__call_latest" => (2,3), "ijl_f__call_latest" => (2,3), "jl_f_invoke" => (2,3), "jl_invoke" => (1,3), "jl_apply_generic" => (1,2), "ijl_f_invoke" => (2,3), "ijl_invoke" => (1,3), "ijl_apply_generic" => (1,2)))

import GPUCompiler: DYNAMIC_CALL, DELAYED_BINDING, RUNTIME_FUNCTION, UNKNOWN_FUNCTION, POINTER_FUNCTION
import GPUCompiler: backtrace, isintrinsic
function check_ir!(job, errors, imported, inst::LLVM.CallInst, calls)
    world = job.world
    interp = GPUCompiler.get_interpreter(job)
    method_table = Core.Compiler.method_table(interp)
    bt = backtrace(inst)
    dest = called_operand(inst)
    if isa(dest, LLVM.Function)
        fn = LLVM.name(dest)

        # some special handling for runtime functions that we don't implement
        if fn == "jl_get_binding_or_error"
        elseif fn == "jl_invoke"
        elseif fn == "jl_apply_generic"
        elseif fn == "gpu_malloc"
            ofn = LLVM.parent(LLVM.parent(inst))
            mod = LLVM.parent(ofn)
            b = IRBuilder()
            position!(b, inst)

            mfn = LLVM.API.LLVMGetNamedFunction(mod, "malloc")
            if mfn == C_NULL
                ptr8 = LLVM.PointerType(LLVM.IntType(8))
                mfn = LLVM.API.LLVMAddFunction(mod, "malloc", LLVM.FunctionType(ptr8, [value_type(LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0)))]))
            end
            mfn2 = LLVM.Function(mfn)
            nval = ptrtoint!(b, call!(b, LLVM.function_type(mfn2), mfn2, [LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0))]), value_type(inst))
            replace_uses!(inst, nval)
            LLVM.API.LLVMInstructionEraseFromParent(inst)   
        elseif fn == "jl_load_and_lookup"
            ofn = LLVM.parent(LLVM.parent(inst))
            mod = LLVM.parent(ofn)

            legal, flib = abs_cstring(operands(inst)[1])
            legal2, fname = abs_cstring(operands(inst)[2])
            legal &= legal2

            hnd = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 2))
            if isa(hnd, LLVM.GlobalVariable)
                hnd = LLVM.name(hnd)
            else
                legal = false
            end

            if !legal
                return
            end

            # res = ccall(:jl_load_and_lookup, Ptr{Cvoid}, (Cstring, Cstring, Ptr{Cvoid}), flib, fname, cglobal(Symbol(hnd)))
            push!(errors, ("jl_load_and_lookup", bt, nothing))
            
        elseif fn == "jl_lazy_load_and_lookup" || fn == "ijl_lazy_load_and_lookup"
            ofn = LLVM.parent(LLVM.parent(inst))
            mod = LLVM.parent(ofn)

            ops = collect(operands(inst))[1:end-1]
            @assert length(ops) == 2
            flib = ops[1]
            fname = ops[2]

            if isa(flib, LLVM.LoadInst)
                op = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(flib, 0))
                while isa(op, LLVM.ConstantExpr)
                    op = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(op, 0))
                end
                if isa(op, ConstantInt)
                    rep = reinterpret(Ptr{Cvoid}, convert(Csize_t, op)+8)
                    ld = unsafe_load(convert(Ptr{Ptr{Cvoid}}, rep))
                    flib = Base.unsafe_pointer_to_objref(ld)
                end
            end
            if isa(flib, GlobalRef)
                flib = getfield(flib.mod, flib.name)
            end

            fname = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 1))
            if isa(fname, LLVM.ConstantExpr)
                fname = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(fname, 0))
            end
            if isa(fname, LLVM.GlobalVariable)
                fname = LLVM.initializer(fname)
            end
            if (isa(fname, LLVM.ConstantArray)  || isa(fname, LLVM.ConstantDataArray)) && eltype(value_type(fname)) == LLVM.IntType(8)
                fname = String(map((x)->convert(UInt8, x), collect(fname)[1:(end-1)]))
            end

            if !isa(fname, String) || !isa(flib, String)
                return
            end

            found = false
            
            try
                data = open(flib, "r") do io
                    lib = readmeta(io)
                    sections = Sections(lib)
                    if !(".llvmbc" in sections)
                        return nothing
                    end
                    llvmbc = read(findfirst(sections, ".llvmbc"))
                    return llvmbc
                end

                if data !== nothing
                    inmod = parse(LLVM.Module, data)
                    found = haskey(functions(inmod), fname)
                end
            catch e
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
                                b = IRBuilder()
                                position!(b, ld)
                                replace_uses!(ld, LLVM.pointercast!(b, replaceWith, value_type(inst)))
                            end
                        end
                    end
                end

                b = IRBuilder()

                position!(b, inst)
                replace_uses!(inst, LLVM.pointercast!(b, replaceWith, value_type(inst)))
                LLVM.API.LLVMInstructionEraseFromParent(inst)

            else
                if fn == "jl_lazy_load_and_lookup"
                    res = ccall(:jl_lazy_load_and_lookup, Ptr{Cvoid}, (Any, Cstring), flib, fname)
                else
                    res = ccall(:ijl_lazy_load_and_lookup, Ptr{Cvoid}, (Any, Cstring), flib, fname)
                end
                replaceWith = LLVM.ConstantInt(LLVM.IntType(8*sizeof(Int)), reinterpret(UInt, res))
                for u in LLVM.uses(inst)
                    st = LLVM.user(u)
                    if isa(st, LLVM.StoreInst) && LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(st, 0)) == inst
                        ptr = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(st, 1))
                        for u in LLVM.uses(ptr)
                            ld = LLVM.user(u)
                            if isa(ld, LLVM.LoadInst)
                                b = IRBuilder()
                                position!(b, ld)
                                for u in LLVM.uses(ld)
                                    u = LLVM.user(u)
                                    if isa(u, LLVM.CallInst)
                                        push!(calls, u)
                                    end
                                end
                                replace_uses!(ld, LLVM.inttoptr!(b, replaceWith, value_type(inst)))
                            end
                        end
                    end
                end

                b = IRBuilder()
                position!(b, inst)
                replacement = LLVM.inttoptr!(b, replaceWith, value_type(inst))
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
                                                replace_uses!(u, LLVM.inttoptr!(b, replaceWith, value_type(u)))
                                            end
                                        end
                                    end
                                end
                            end
                replace_uses!(inst, replacement)
                LLVM.API.LLVMInstructionEraseFromParent(inst)
            end
        elseif fn == "julia.call" || fn == "julia.call2"
            dest = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(inst, 0))

            if isa(dest, LLVM.Function) && LLVM.name(dest) == "jl_f__apply_iterate"
                # Add 1 to account for function being first arg
                iteroff = 2
                
                legal, iterlib = absint(operands(inst)[iteroff+1])
                if legal && iterlib == Base.iterate
                    legal, GT = abs_typeof(operands(inst)[4+1], true)
                    if legal && GT <: Vector
                        funcoff = 3
                        funclib = operands(inst)[funcoff+1]
                        while isa(funclib, LLVM.ConstantExpr)
                            funclib = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(funclib, 0))
                        end
                        if isa(funclib, ConstantInt)
                            rep = reinterpret(Ptr{Cvoid}, convert(Csize_t, funclib))
                            funclib = Base.unsafe_pointer_to_objref(rep)
                            tys = [typeof(funclib), Vararg{Any}]
                            if EnzymeRules.is_inactive_from_sig(Tuple{tys...}; world, method_table) || EnzymeRules.is_inactive_noinl_from_sig(Tuple{tys...}; world, method_table)
                                inactive = LLVM.StringAttribute("enzyme_inactive", "")
                                LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, inactive)
                                nofree = LLVM.EnumAttribute("nofree")
                                LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, nofree)
                            end
                            if funclib == Base.tuple && length(operands(inst)) == 4+1+1 && Base.isconcretetype(GT) && Enzyme.Compiler.guaranteed_const_nongen(GT, world)
                                inactive = LLVM.StringAttribute("enzyme_inactive", "")
                                LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, inactive)
                                nofree = LLVM.EnumAttribute("nofree")
                                LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, nofree)
                            end
                        end
                    end
                end
            end

            if isa(dest, LLVM.Function) && in(LLVM.name(dest), keys(generic_method_offsets))
                offset, start = generic_method_offsets[LLVM.name(dest)]
                # Add 1 to account for function being first arg
                legal, flibty = abs_typeof(operands(inst)[offset+1])
                if legal
                    tys = Type[flibty]
                    for op in collect(operands(inst))[start+1:end-1]
                        legal, typ = abs_typeof(op, true)
                        if !legal
                            typ = Any
                        end
                        push!(tys, typ)
                    end
                    legal, flib = absint(operands(inst)[offset+1])
                    if legal && isa(flib, Core.MethodInstance)
                        if !Base.isvarargtype(flib.specTypes.parameters[end])
                            @assert length(tys) == length(flib.specTypes.parameters)
                        end
                        tys = flib.specTypes.parameters
                    end
                    if EnzymeRules.is_inactive_from_sig(Tuple{tys...}; world, method_table) || EnzymeRules.is_inactive_noinl_from_sig(Tuple{tys...}; world, method_table)
                        inactive = LLVM.StringAttribute("enzyme_inactive", "")
                        LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, inactive)
                        nofree = LLVM.EnumAttribute("nofree")
                        LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, nofree)
                    end
                end
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

                # Remember pointer in our global map
                fn = FFI.memoize!(ptr, string(fn))

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
        dest = LLVM.Value(LLVM.LLVM.API.LLVMGetOperand(dest, 0))
        if isa(dest, LLVM.Function) && in(LLVM.name(dest), keys(generic_method_offsets))
            offset, start = generic_method_offsets[LLVM.name(dest)]

            legal, flibty = abs_typeof(operands(inst)[offset])
            if legal
                tys = Type[flibty]
                for op in collect(operands(inst))[start:end-1]
                    legal, typ = abs_typeof(op, true)
                    if !legal
                        typ = Any
                    end
                    push!(tys, typ)
                end
                legal, flib = absint(operands(inst)[offset+1])
                if legal && isa(flib, Core.MethodInstance)
                    if !Base.isvarargtype(flib.specTypes.parameters[end])
                        if length(tys) != length(flib.specTypes.parameters)
                            @show tys, flib, inst, offset, start
                        end
                        @assert length(tys) == length(flib.specTypes.parameters)
                    end
                    tys = flib.specTypes.parameters
                end
                if EnzymeRules.is_inactive_from_sig(Tuple{tys...}; world, method_table) || EnzymeRules.is_inactive_noinl_from_sig(Tuple{tys...}; world, method_table) 
                    ofn = LLVM.parent(LLVM.parent(inst))
                    mod = LLVM.parent(ofn)
                    inactive = LLVM.StringAttribute("enzyme_inactive", "")
                    LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, inactive)
                    nofree = LLVM.EnumAttribute("nofree")
                    LLVM.API.LLVMAddCallSiteAttribute(inst, LLVM.API.LLVMAttributeFunctionIndex, nofree)
                end
            end
        end
    end

    return errors
end


function rewrite_union_returns_as_ref(enzymefn::LLVM.Function, off, world, width)
    todo = Tuple{LLVM.Value, Tuple}[]
    for b in blocks(enzymefn)
        term = terminator(b)
        if LLVM.API.LLVMIsAReturnInst(term) != C_NULL
            if width == 1
                push!(todo, (operands(term)[1], off == -1 ? () : (off,)))
            else
                for i in 1:width
                    push!(todo, (operands(term)[1], off == -1 ? (i,) : (off,i)))
                end
            end
        end
    end

    seen = Dict{LLVM.Value,Tuple}()
    while length(todo) != 0
        cur, off = pop!(todo)

        while isa(cur, LLVM.AddrSpaceCastInst) # || isa(cur, LLVM.BitCastInst)
            cur = operands(cur)[1]
        end

        if cur in keys(seen)
            @assert seen[cur] == off
            continue
        end
        seen[cur] = off

        if isa(cur, LLVM.PHIInst)
            for (v, _) in LLVM.incoming(cur)
                push!(todo, (v, off))
            end
            continue
        end

        if isa(cur, LLVM.ExtractValueInst)
            noff = off
            for i in 1:LLVM.API.LLVMGetNumIndices(cur)
                noff = (noff..., convert(Int, unsafe_load(LLVM.API.LLVMGetIndices(cur), i)))
            end
            push!(todo, (operands(cur)[1], noff))
            continue
        end

        if isa(cur, LLVM.InsertValueInst)
            @assert length(off) != 0
            @assert LLVM.API.LLVMGetNumIndices(cur) == 1

            ind = unsafe_load(LLVM.API.LLVMGetIndices(cur))

            # if inserting at the current desired offset, we have found the value we need
            if ind == off[1]
                push!(todo, (operands(cur)[2], -1))
            # otherwise it must be inserted at a different point
            else
                push!(todo, (operands(cur)[1], off))
            end
            continue
        end

        if isa(cur, LLVM.CallInst)
            fn = LLVM.called_operand(cur)
            nm = ""
            if isa(fn, LLVM.Function)
                nm = LLVM.name(fn)
            end

            # Type tag is arg 3
            if nm == "julia.gc_alloc_obj"
                legal, Ty = abs_typeof(cur)
                @assert legal
                reg = active_reg_inner(Ty, (), world)
                if reg == ActiveState || reg == MixedState
                    NTy = Base.RefValue{Ty}
                    @assert sizeof(Ty) == sizeof(NTy)
                    LLVM.API.LLVMSetOperand(cur, 2, unsafe_to_llvm(NTy))
                end
                continue
            end
        end

        undefpoisonornull = isa(cur, LLVM.UndefValue) || isa(cur, LLVM.PointerNull)
        @static if LLVM.version() >= v"12"
            undefpoisonornull |= isa(cur, LLVM.PoisonValue)
        end
        if undefpoisonornull
            continue
        end

        if isa(cur, LLVM.LoadInst)
            al = operands(cur)[1]
            if isa(al, LLVM.AllocaInst)
                atodo = Tuple{LLVM.Value, Tuple, LLVM.Value}[]
                for u in LLVM.uses(al)
                    push!(atodo, (LLVM.user(u), off, al))
                end
                while length(atodo) > 0
                    acur, aoff, prev = pop!(atodo)
                    if isa(acur, LLVM.LoadInst)
                        continue
                    end
                    if isa(acur, LLVM.StoreInst)
                        @assert operands(acur)[2] == prev
                        push!(todo, (operands(acur)[1], aoff))
                        continue
                    end
                    if isa(acur, LLVM.GetElementPtrInst)
                        aoff2 = aoff
                        @assert convert(Int, operands(acur)[2]) == 0
                        match = true
                        for val in (convert(Int, op) for op in operands(acur)[3:end])
                            @assert length(aoff) > 0
                            if val == aoff2[1]
                                aoff2 = (aoff2[2:end]...,)
                            else
                                match = false
                                break
                            end
                        end
                        if match
                            for u in LLVM.uses(acur)
                                push!(atodo, (LLVM.user(u), aoff2, acur))
                            end
                        end
                        continue
                    end

                    println(string(enzymefn))
                    @show "BAD", acur, aoff, prev
                    @assert false
                end
                continue
            end
        end

        if length(off) == 0 && value_type(cur) == LLVM.PointerType(LLVM.StructType(LLVMType[]), Tracked)
            legal, typ = abs_typeof(cur)
            if legal
                reg = active_reg_inner(typ, (), world)
                if !(reg == ActiveState || reg == MixedState)
                    continue
                end
            end
        end

        println(string(enzymefn))

        @show cur, off
        @assert false
    end
end