"""
    unsafe_to_pointer

!!! warning
    Assumes that `val` is globally rooted and pointer to it can be leaked. Prefer `pointer_from_objref`.
    Only use inside Enzyme.jl should be for Types.
"""
function unsafe_to_pointer end
export unsafe_to_pointer

if VERSION >= v"1.12-"
    @inline function unsafe_to_pointer(@nospecialize(val::Type))
        return Core.Intrinsics.llvmcall((
            """
            declare nonnull ptr @julia.pointer_from_objref(ptr addrspace(11))

            define ptr @f(ptr addrspace(10) %obj) readnone alwaysinline {
                %c = addrspacecast ptr addrspace(10) %obj to ptr addrspace(11)
                %r = call ptr @julia.pointer_from_objref(ptr addrspace(11) %c)
                ret ptr %r
            }
            """, "f"), Ptr{Cvoid}, Tuple{Any}, val)
    end
elseif Int == Int64
    @inline function unsafe_to_pointer(@nospecialize(val::Type))
        return Base.llvmcall((
            """
            declare nonnull {}* @julia.pointer_from_objref({} addrspace(11)*)

            define i64 @f({} addrspace(10)* %obj) readnone alwaysinline {
                %c = addrspacecast {} addrspace(10)* %obj to {} addrspace(11)*
                %r = call {}* @julia.pointer_from_objref({} addrspace(11)* %c)
                %e = ptrtoint {}* %r to i64
                ret i64 %e
            }
            """, "f"), Ptr{Cvoid}, Tuple{Any}, val)
    end
else
    @inline function unsafe_to_pointer(@nospecialize(val::Type))
        return Base.llvmcall((
            """
            declare nonnull {}* @julia.pointer_from_objref({} addrspace(11)*)

            define i32 @f({} addrspace(10)* %obj) readnone alwaysinline {
                %c = addrspacecast {} addrspace(10)* %obj to {} addrspace(11)*
                %r = call {}* @julia.pointer_from_objref({} addrspace(11)* %c)
                %e = ptrtoint {}* %r to i32
                ret i32 %e
            }
            """, "f"), Ptr{Cvoid}, Tuple{Any}, val)
    end
end


@inline is_concrete_tuple(x::Type{T2}) where {T2} =
    (T2 <: Tuple) && !(T2 === Tuple) && !(T2 isa UnionAll)
export is_concrete_tuple

const Tracked = 10
const Derived = 11
export Tracked, Derived

const captured_constants = Base.IdSet{Any}()

function unsafe_nothing_to_llvm(mod::LLVM.Module)
    globs = LLVM.globals(mod)
    k = "jl_nothing"
    if Base.haskey(globs, "ejl_" * k)
        return globs["ejl_"*k]
    end
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    gv = LLVM.GlobalVariable(mod, T_jlvalue, "ejl_" * k, Tracked)

    API.SetMD(gv, "enzyme_ta_norecur", LLVM.MDNode(LLVM.Metadata[]))
    API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
    return gv
end

function unsafe_to_ptr(@nospecialize(val))
    if !Base.ismutable(val)
        val = Core.Box(val) # FIXME many objects could be leaked here
        @assert Base.ismutable(val)
        push!(captured_constants, val) # Globally root
        ptr = unsafe_load(Base.reinterpret(Ptr{Ptr{Cvoid}}, Base.pointer_from_objref(val)))
    else
        @assert Base.ismutable(val)
        push!(captured_constants, val) # Globally root
        ptr = Base.pointer_from_objref(val)
    end
    return ptr
end
export unsafe_to_ptr

# This mimicks literal_pointer_val / literal_pointer_val_slot
function unsafe_to_llvm(B::LLVM.IRBuilder, @nospecialize(val))::LLVM.Value
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    world = nothing
    for fattr in collect(LLVM.function_attributes(LLVM.parent(LLVM.position(B))))
        if isa(fattr, LLVM.StringAttribute)
            if LLVM.kind(fattr) == "enzymejl_world"
                world = parse(UInt, LLVM.value(fattr))
                break
            end
        end
    end

    for (k, v) in Compiler.JuliaGlobalNameMap
        if v === val
            mod = LLVM.parent(LLVM.parent(LLVM.position(B)))
            globs = LLVM.globals(mod)
            if Base.haskey(globs, "ejl_" * k)
                return globs["ejl_"*k]
            end
            gv = LLVM.GlobalVariable(mod, T_jlvalue, "ejl_" * k, Tracked)

            API.SetMD(gv, "enzyme_ta_norecur", LLVM.MDNode(LLVM.Metadata[]))
            if world isa UInt
                legal, jTy, byref = Compiler.abs_typeof(gv, true)
                if legal
                    curent_bb = position(B)
                    fn = LLVM.parent(curent_bb)
                    if Compiler.guaranteed_const_nongen(jTy, world)
                        API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
                    end
                end
            end
            return gv
        end
    end

    for (k, v) in Compiler.JuliaEnzymeNameMap
        if v === val
            mod = LLVM.parent(LLVM.parent(LLVM.position(B)))
            globs = LLVM.globals(mod)
            if Base.haskey(globs, "ejl_" * k)
                return globs["ejl_"*k]
            end
            gv = LLVM.GlobalVariable(mod, T_jlvalue, "ejl_" * k, Tracked)
            API.SetMD(gv, "enzyme_ta_norecur", LLVM.MDNode(LLVM.Metadata[]))
            if world isa UInt
                legal, jTy, byref = Compiler.abs_typeof(gv, true)
                if legal
                    curent_bb = position(B)
                    fn = LLVM.parent(curent_bb)
                    if Compiler.guaranteed_const_nongen(jTy, world)
                        API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
                    end
                end
            end
            return gv
        end
    end

    # XXX: This prevents code from being runtime relocatable
    #      We likely should emit global variables and use something
    #      like `absolute_symbol_materialization` and write out cache-files
    #      that have relocation tables.
    ptr = unsafe_to_ptr(val)

    fill_val = LLVM.ConstantInt(convert(UInt, ptr))
    fill_val = LLVM.const_inttoptr(fill_val, T_prjlvalue_UT)
    LLVM.const_addrspacecast(fill_val, T_prjlvalue)
end
export unsafe_to_llvm, unsafe_nothing_to_llvm

function makeInstanceOf(B::LLVM.IRBuilder, @nospecialize(T::Type))
    if !Core.Compiler.isconstType(T)
        throw(AssertionError("Tried to make instance of non constant type $T"))
    end
    @assert T <: Type
    return unsafe_to_llvm(B, T.parameters[1])
end

export makeInstanceOf

function hasfieldcount(@nospecialize(dt))::Bool
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

import Base: allocatedinline

using Core: MethodInstance
using GPUCompiler: tls_world_age, MethodError, methodinstance
using Core.Compiler: CodeInfo, SSAValue, ReturnNode
using Base: _methods_by_ftype

# Julia compiler integration

@inline function has_method(@nospecialize(sig::Type), world::UInt, mt::Union{Nothing,Core.MethodTable})
    return ccall(:jl_gf_invoke_lookup, Any, (Any, Any, UInt), sig, mt, world) !== nothing
end

@inline function has_method(@nospecialize(sig::Type), world::UInt, mt::Core.Compiler.InternalMethodTable)
    return has_method(sig, mt.world, nothing)
end

@inline function has_method(@nospecialize(sig::Type), world::UInt, mt::Core.Compiler.OverlayMethodTable)
    return has_method(sig, mt.world, mt.mt) || has_method(sig, mt.world, nothing)
end

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Union{Nothing,Core.MethodTable},
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
    res = ccall(
        :jl_gf_invoke_lookup_worlds,
        Any,
        (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
        sig,
        mt,
        world,
        min_world,
        max_world,
    )
    return res
end

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Core.Compiler.InternalMethodTable,
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
    res = lookup_world(sig, mt.world, nothing, min_world, max_world)
    return res
end

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Core.Compiler.CachedMethodTable,
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
    res = lookup_world(sig, world, mt.table, min_world, max_world)
    return res
end

@inline function lookup_world(
    @nospecialize(sig::Type),
    world::UInt,
    mt::Core.Compiler.OverlayMethodTable,
    min_world::Ref{UInt},
    max_world::Ref{UInt},
)
    res = lookup_world(sig, mt.world, mt.mt, min_world, max_world)
    if res !== nothing
        return res
    else
        return lookup_world(sig, mt.world, nothing, min_world, max_world)
    end
end


"""
    create_fresh_codeinfo(fn, source, world)

Create a fresh `Core.CodeInfo` for a generated function `fn` with source location `source` and world age `world`.
Callers are responsible for setting `ci.edges`
"""
function create_fresh_codeinfo(fn, source, world, slotnames, code)
    ci = ccall(:jl_new_code_info_uninit, Ref{Core.CodeInfo}, ())
    
    @static if isdefined(Core, :DebugInfo)
        # TODO: Add proper debug info
        ci.debuginfo = Core.DebugInfo(:none)
    else
        ci.codelocs = Int32[]
        ci.linetable = [
            Core.Compiler.LineInfoNode(@__MODULE__, fn, source.file, Int32(source.line), Int32(0))
        ]
    end

    ci.min_world = world
    ci.max_world = typemax(UInt)

    ci.slotnames = Symbol[sym for sym in slotnames]
    ci.slotflags = UInt8[0x00 for i = 1:length(slotnames)]
    if VERSION >= v"1.12-"
        ci.nargs = length(slotnames)
        ci.isva = false
    end

    ci.code = code
    ci.ssaflags = UInt8[0x00 for i = 1:length(code)]   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    @static if isdefined(Core, :DebugInfo)
    else
        for _ in 1:length(code)
            push!(ci.codelocs, 1)   # all instructions map to the first line
        end
    end

    ci.ssavaluetypes = length(code)

    return ci
end

function add_edge!(edges::Vector{Any}, @nospecialize(invoke_sig::Type))
    mt = ccall(:jl_method_table_for, Any, (Any,), invoke_sig)
    if VERSION < v"1.12-"
        push!(edges, mt)
        push!(edges, invoke_sig)
    else
        push!(edges, invoke_sig)
        push!(edges, mt)
    end
    return edges
end

function add_edge!(edges::Vector{Any}, mi::Core.MethodInstance)
    push!(edges, mi)
    return edges
end

@inline function my_methodinstance(@nospecialize(method_table::Union{Core.Compiler.MethodTableView, Nothing}), @nospecialize(ft::Type), @nospecialize(tt::Type), world::UInt, min_world::Union{Nothing, Base.RefValue{UInt}}=nothing, max_world::Union{Nothing, Base.RefValue{UInt}}=nothing)::Union{Core.MethodInstance, Nothing}

    if min_world === nothing
        min_world = Ref{UInt}(typemin(UInt))
    end
    if max_world === nothing
        max_world = Ref{UInt}(typemax(UInt))
    end

    sig = Tuple{ft, tt.parameters...}
    
    lookup_result = lookup_world(
        sig, world, method_table, min_world, max_world
    )
    if lookup_result === nothing
        return nothing
    end

    match = lookup_result::Core.MethodMatch
    
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance},
               (Any, Any, Any), match.method, match.spec_types, match.sparams)
    return mi::Core.MethodInstance
end

@inline function my_methodinstance(@nospecialize(interp::Core.Compiler.AbstractInterpreter), @nospecialize(ft::Type), @nospecialize(tt::Type),  min_world::Union{Nothing, Base.RefValue{UInt}}=nothing, max_world::Union{Nothing, Base.RefValue{UInt}}=nothing)::Union{Core.MethodInstance, Nothing}
    my_methodinstance(Core.Compiler.method_table(interp), ft, tt, interp.world, min_world, max_world)
end

@inline function my_methodinstance(@nospecialize(mode::Union{Missing, EnzymeCore.ForwardMode, EnzymeCore.ReverseMode}), @nospecialize(ft::Type), @nospecialize(tt::Type), world::UInt, min_world::Union{Nothing, Base.RefValue{UInt}}=nothing, max_world::Union{Nothing, Base.RefValue{UInt}}=nothing)::Union{Core.MethodInstance, Nothing}
    interp = if mode === missing # TODO: PrimalMode
        Core.Compiler.NativeInterpreter(world)
    else
        @assert mode == Forward || mode == Reverse
        Compiler.primal_interp_world(mode, world)
    end
    my_methodinstance(interp, ft, tt, min_world, max_world)
end

function prevmethodinstance end

function methodinstance_generator(world::UInt, source, self, @nospecialize(mode::Type), @nospecialize(ft::Type), @nospecialize(tt::Type))
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    slotnames = Core.svec(Symbol("#self#"), :mode, :ft, :tt)
    stub = Core.GeneratedFunctionStub(identity, slotnames, Core.svec())

    # look up the method match
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    mi = my_methodinstance(mode.instance, ft, tt, world, min_world, max_world)
    
    mi === nothing && return stub(world, source, :(throw(MethodError(ft, tt, $world))))
    
    code = Any[Core.Compiler.ReturnNode(mi)]

    ci = create_fresh_codeinfo(prevmethodinstance, source, world, slotnames, code)
    ci.min_world = min_world[]
    ci.max_world = max_world[]
    ci.edges = Any[mi]

    # TODO: Missing edge handling
    return ci
end

@eval function prevmethodinstance(mode, ft, tt)::Core.MethodInstance
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, methodinstance_generator))
end

# XXX: version of Base.method_instance that uses a function type
@inline function my_methodinstance(@nospecialize(mode::Union{Nothing, EnzymeCore.ForwardMode, EnzymeCore.ReverseMode}), @nospecialize(ft::Type), @nospecialize(tt::Type))::Core.MethodInstance
    sig = GPUCompiler.signature_type_by_tt(ft, tt)
    return prevmethodinstance(mode, ft, tt)::Core.MethodInstance
end

export my_methodinstance

@static if VERSION < v"1.11-"

# JL_EXTENSION typedef struct {
#     JL_DATA_TYPE
#     void *data;
# #ifdef STORE_ARRAY_LEN (just true new newer versions)
# 	size_t length;
# #endif
#     jl_array_flags_t flags;
#     uint16_t elsize;  // element size including alignment (dim 1 memory stride)
#     uint32_t offset;  // for 1-d only. does not need to get big.
#     size_t nrows;
#     union {
#         // 1d
#         size_t maxsize;
#         // Nd
#         size_t ncols;
#     };
#     // other dim sizes go here for ndims > 2
#
#     // followed by alignment padding and inline data, or owner pointer
# } jl_array_t;
@inline function typed_fieldtype(@nospecialize(T::Type), i::Int)::Type
    if T <: Array
        eT = eltype(T)
        PT = Ptr{eT}
        return (PT, Csize_t, UInt16, UInt16, UInt32, Csize_t, Csize_t)[i]
    else
        fieldtype(T, i)
    end
end

@inline function typed_fieldcount(@nospecialize(T::Type))::Int
    if T <: Array
        return 7
    else
        fieldcount(T)
    end
end

@inline function typed_fieldoffset(@nospecialize(T::Type), i::Int)::Int
    if T <: Array
        tys = (Ptr, Csize_t, UInt16, UInt16, UInt32, Csize_t, Csize_t)
        sum = 0
        idx = 1
        while idx < i
            sum += sizeof(tys[idx])
            idx+=1
        end
        return sum 
    else
        fieldoffset(T, i)
    end
end

else

function is_memory_ref_field2_an_offset(@nospecialize(T::Type{<:GenericMemoryRef}))
    ET = eltype(T)

    # 0 = inlinealloc
    # 1 = isboxed
    # 2 = isbitsunion
    return (Base.datatype_arrayelem(T.types[2]) == 2) || Compiler.datatype_layoutsize(T.types[2]) == 0
end

@inline function typed_fieldtype(@nospecialize(T::Type), i::Int)::Type
    if T <: GenericMemoryRef && i == 1 || T <: GenericMemory && i == 2
        if T <: GenericMemoryRef && i == 1 && is_memory_ref_field2_an_offset(T)
            Int
        else
            eT = eltype(T)
            Ptr{eT}
        end
    else
        fieldtype(T, i)
    end
end

@inline function typed_fieldcount(@nospecialize(T::Type))::Int
    fieldcount(T)
end

@inline function typed_fieldoffset(@nospecialize(T::Type), i::Int)::Int
    fieldoffset(T, i)
end

end

export typed_fieldtype
export typed_fieldcount
export typed_fieldoffset

# returns the inner type of an sret/enzyme_sret/enzyme_sret_v
function sret_ty(fn::LLVM.Function, idx::Int)::LLVM.LLVMType

    vt = LLVM.value_type(LLVM.parameters(fn)[idx])

    sretkind = LLVM.kind(if LLVM.version().major >= 12
        LLVM.TypeAttribute("sret", LLVM.Int32Type())
    else
        LLVM.EnumAttribute("sret")
    end)


    enzymejl_parmtype_ref = nothing
    enzymejl_parmtype = nothing

    for attr in collect(LLVM.parameter_attributes(fn, idx))
        ekind = LLVM.kind(attr)

        if ekind == sretkind
            res = LLVM.value(attr)
            if !LLVM.is_opaque(vt)
                @assert eltype(vt) == res
            end
            return res
        end

        if ekind == "enzymejl_sret_union_bytes"
            nbytes = parse(Int, LLVM.value(attr))
            i8 = LLVM.IntType(8)

            res = LLVM.ArrayType(i8, nbytes)
            if !LLVM.is_opaque(vt)
                @assert eltype(vt) == res
            end
            return res
        end

        if ekind == "enzymejl_returnRoots"
            nroots = parse(Int, LLVM.value(attr))
    
            T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
            T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)

            res = LLVM.ArrayType(T_prjlvalue, nroots)
            if !LLVM.is_opaque(vt)
                @assert eltype(vt) == res
            end
            return res
        end

        if ekind == "enzyme_sret"
	    ety = parse(UInt, LLVM.value(attr))
	    ety = Base.reinterpret(LLVM.API.LLVMTypeRef, ety)
	    ety = LLVM.LLVMType(ety)
            if !LLVM.is_opaque(vt)
		@assert ety == eltype(vt)
            end
        
            return ety
        end

        if ekind == "enzymejl_parmtype_ref"
            enzymejl_parmtype_ref = GPUCompiler.ArgumentCC(parse(UInt, LLVM.value(attr)))
            continue
        end

        if ekind == "enzymejl_parmtype"
            ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(attr)))
            enzymejl_parmtype = Base.unsafe_pointer_to_objref(ptr)::Type
        end
    end

    if enzymejl_parmtype_ref == GPUCompiler.BITS_REF && enzymejl_parmtype !== nothing
        res = convert(LLVM.LLVMType, enzymejl_parmtype)
        if !LLVM.is_opaque(vt)
            @assert eltype(vt) == res
        end
        return res
    end

    throw(AssertionError("Function requesting sret type was not an sret\nidx=$idx\nfn=$(string(fn)) enzymejl_parmtype=$enzymejl_parmtype enzymejl_parmtype_ref=$enzymejl_parmtype_ref"))
end

export sret_ty

function get_rooted_typ(fn::LLVM.Function, idx::Int)::LLVM.LLVMType
    for attr in collect(LLVM.parameter_attributes(fn, idx))
        ekind = LLVM.kind(attr)

        if ekind == "enzymejl_rooted_typ"
            ptr = reinterpret(Ptr{Cvoid}, parse(UInt, LLVM.value(attr)))
            return Base.unsafe_pointer_to_objref(ptr)
        end
    end

    throw(AssertionError("Function requesting rooted type was not an sret\nidx=$idx\nfn=$(string(fn))"))
end

export get_rooted_typ

function remove_nothing_from_union_type(@nospecialize(T::Type))::Type
    if T isa Union
        if T.a === Nothing
            return remove_nothing_from_union_type(T.b)
        elseif T.b === Nothing
            return remove_nothing_from_union_type(T.a)
        else
            return Union{remove_nothing_from_union_type(T.a), remove_nothing_from_union_type(T.b)}
        end
    else
        return T
    end
end

export remove_nothing_from_union_type
