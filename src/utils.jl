"""
    unsafe_to_pointer

!!! warning
    Assumes that `val` is globally rooted and pointer to it can be leaked. Prefer `pointer_from_objref`.
    Only use inside Enzyme.jl should be for Types.
"""
@inline Base.@nospecializeinfer unsafe_to_pointer(@nospecialize(val::Type)) = @static if sizeof(Int) == sizeof(Int64)
    Base.llvmcall((
"""
declare nonnull {}* @julia.pointer_from_objref({} addrspace(11)*)

define i64 @f({} addrspace(10)* %obj) readnone alwaysinline {
  %c = addrspacecast {} addrspace(10)* %obj to {} addrspace(11)*
  %r = call {}* @julia.pointer_from_objref({} addrspace(11)* %c)
  %e = ptrtoint {}* %r to i64
  ret i64 %e
}
""", "f"),
    Ptr{Cvoid},
    Tuple{Any},
    val,
)
else
    Base.llvmcall((
"""
declare nonnull {}* @julia.pointer_from_objref({} addrspace(11)*)

define i32 @f({} addrspace(10)* %obj) readnone alwaysinline {
  %c = addrspacecast {} addrspace(10)* %obj to {} addrspace(11)*
  %r = call {}* @julia.pointer_from_objref({} addrspace(11)* %c)
  %e = ptrtoint {}* %r to i32
  ret i32 %e
}
""", "f"),
    Ptr{Cvoid},
    Tuple{Any},
    val,
)
end

export unsafe_to_pointer

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

Base.@nospecializeinfer function unsafe_to_ptr(@nospecialize(val))
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
Base.@nospecializeinfer function unsafe_to_llvm(B::LLVM.IRBuilder, @nospecialize(val))::LLVM.Value
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[])
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)

    for (k, v) in Compiler.JuliaGlobalNameMap
        if v === val
            mod = LLVM.parent(LLVM.parent(LLVM.position(B)))
            globs = LLVM.globals(mod)
            if Base.haskey(globs, "ejl_" * k)
                return globs["ejl_"*k]
            end
            gv = LLVM.GlobalVariable(mod, T_jlvalue, "ejl_" * k, Tracked)

            API.SetMD(gv, "enzyme_ta_norecur", LLVM.MDNode(LLVM.Metadata[]))
            legal, jTy, byref = Compiler.abs_typeof(gv, true)
            if legal
                curent_bb = position(B)
                fn = LLVM.parent(curent_bb)
                if Compiler.guaranteed_const_nongen(jTy, nothing)
                    API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
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
            legal, jTy, byref = Compiler.abs_typeof(gv, true)
            if legal
                curent_bb = position(B)
                fn = LLVM.parent(curent_bb)
                if Compiler.guaranteed_const_nongen(jTy, nothing)
                    API.SetMD(gv, "enzyme_inactive", LLVM.MDNode(LLVM.Metadata[]))
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

Base.@nospecializeinfer function makeInstanceOf(B::LLVM.IRBuilder, @nospecialize(T::Type))
    if !Core.Compiler.isconstType(T)
        throw(AssertionError("Tried to make instance of non constant type $T"))
    end
    @assert T <: Type
    return unsafe_to_llvm(B, T.parameters[1])
end

export makeInstanceOf

Base.@nospecializeinfer function hasfieldcount(@nospecialize(dt))::Bool
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
using Core.Compiler: retrieve_code_info, CodeInfo, SSAValue, ReturnNode
using Base: _methods_by_ftype

# Julia compiler integration


if VERSION >= v"1.11.0-DEV.1552"


const prevmethodinstance = GPUCompiler.generic_methodinstance

function methodinstance_generator(world::UInt, source, self, @nospecialize(ft::Type), @nospecialize(tt::Type))
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :ft, :tt), Core.svec())

    # look up the method match
    method_error = :(throw(MethodError(ft, tt, $world)))
    sig = Tuple{ft, tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    match = ccall(:jl_gf_invoke_lookup_worlds, Any,
                  (Any, Any, Csize_t, Ref{Csize_t}, Ref{Csize_t}),
                  sig, #=mt=# nothing, world, min_world, max_world)
    match === nothing && return stub(world, source, method_error)

    # look up the method and code instance
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance},
               (Any, Any, Any), match.method, match.spec_types, match.sparams)
    ci = Core.Compiler.retrieve_code_info(mi, world)

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    empty!(new_ci.linetable)
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0

    # propagate edge metadata
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]
    new_ci.edges = MethodInstance[mi]

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:3]

    # return the method instance
    push!(new_ci.code, Core.Compiler.ReturnNode(mi))
    push!(new_ci.ssaflags, 0x00)
    push!(new_ci.linetable, GPUCompiler.@LineInfoNode(methodinstance))
    push!(new_ci.codelocs, 1)
    new_ci.ssavaluetypes += 1

    return new_ci
end

@eval function prevmethodinstance(ft, tt)::Core.MethodInstance
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, methodinstance_generator))
end

# XXX: version of Base.method_instance that uses a function type
@inline function my_methodinstance(@nospecialize(ft::Type), @nospecialize(tt::Type),
                                world::Integer=tls_world_age())::Core.MethodInstance
    sig = GPUCompiler.signature_type_by_tt(ft, tt)
    if Base.isdispatchtuple(sig)   # JuliaLang/julia#52233
        return GPUCompiler.methodinstance(ft, tt, world)::Core.MethodInstance
    else
        return prevmethodinstance(ft, tt, world)::Core.MethodInstance
    end
end
else
    import GPUCompiler: methodinstance as my_methodinstance
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

@inline function typed_fieldtype(@nospecialize(T::Type), i::Int)::Type
    if T <: GenericMemoryRef && i == 1 || T <: GenericMemory && i == 2
        eT = eltype(T)
        Ptr{eT}
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
    return eltype(LLVM.value_type(LLVM.parameters(fn)[idx]))
end

export sret_ty

