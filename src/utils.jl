"""
    unsafe_to_pointer

!!! warning
    Assumes that `val` is globally rooted and pointer to it can be leaked. Prefer `pointer_from_objref`.
    Only use inside Enzyme.jl should be for Types.
"""
@inline unsafe_to_pointer(val::Type{T}) where {T} = ccall(
    Base.@cfunction(Base.identity, Ptr{Cvoid}, (Ptr{Cvoid},)),
    Ptr{Cvoid},
    (Any,),
    val,
)
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
function unsafe_to_llvm(B::LLVM.IRBuilder, @nospecialize(val))
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

function makeInstanceOf(B::LLVM.IRBuilder, @nospecialize(T))
    if !Core.Compiler.isconstType(T)
        throw(AssertionError("Tried to make instance of non constant type $T"))
    end
    @assert T <: Type
    return unsafe_to_llvm(B, T.parameters[1])
end

export makeInstanceOf

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

import Base: allocatedinline

#Excerpt from https://github.com/JuliaGPU/GPUCompiler.jl/blob/v0.19.4/src/jlgen.jl
# !!! warning "codegen_world_age below is fundamentally unsound."
#     It was removed from GPUCompiler since it can produce incorrect results. 

using Core: MethodInstance
using GPUCompiler: tls_world_age, MethodError, methodinstance
using Core.Compiler: retrieve_code_info, CodeInfo, SSAValue, ReturnNode
using Base: _methods_by_ftype

# Julia compiler integration


## world age lookups

# `tls_world_age` should be used to look up the current world age. in most cases, this is
# what you should use to invoke the compiler with.
#
# `codegen_world_age` is a special function that returns the world age in which the passed
# method instance (identified by its function and argument types) is to be compiled. the
# returned constant is automatically invalidated when the method is redefined, and as such
# can be used to drive cached compilation. it is unlikely that you should use this function
# directly, instead use `cached_compilation` which handles invalidation for you.


# on 1.10 (JuliaLang/julia#48611) the generated function knows which world it was invoked in

function _generated_ex(world, source, ex)
    stub = Core.GeneratedFunctionStub(
        identity,
        Core.svec(:methodinstance, :ft, :tt),
        Core.svec(),
    )
    stub(world, source, ex)
end

function codegen_world_age_generator(world::UInt, source, self, ft::Type, tt::Type)
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    # validation
    ft <: Core.Builtin &&
        error("$(GPUCompiler.unsafe_function_from_type(ft)) is not a generic function")

    # look up the method
    method_error = :(throw(MethodError(ft, tt, $world)))
    sig = Tuple{ft,tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL)  # don't care about ambiguous results
    mthds = Base._methods_by_ftype(
        sig,
        nothing,
        -1, #=lim=#
        world,
        false, #=ambig=#
        min_world,
        max_world,
        has_ambig,
    )
    mthds === nothing && return _generated_ex(world, source, method_error)
    length(mthds) == 1 || return _generated_ex(world, source, method_error)

    # look up the method and code instance
    mtypes, msp, m = mthds[1]
    mi = ccall(
        :jl_specializations_get_linfo,
        Ref{MethodInstance},
        (Any, Any, Any),
        m,
        mtypes,
        msp,
    )
    ci = retrieve_code_info(mi, world)::CodeInfo

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    empty!(new_ci.codelocs)
    resize!(new_ci.linetable, 1)                # see note below
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]
    new_ci.edges = MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:3]

    # return the codegen world age
    push!(new_ci.code, ReturnNode(world))
    push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    push!(new_ci.codelocs, 1)   # see note below
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

@eval function codegen_world_age(ft, tt)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, codegen_world_age_generator))
end

export codegen_world_age


if VERSION >= v"1.11.0-DEV.1552"

# XXX: version of Base.method_instance that uses a function type
@inline function my_methodinstance(@nospecialize(ft::Type), @nospecialize(tt::Type),
                                world::Integer=tls_world_age())
    sig = GPUCompiler.signature_type_by_tt(ft, tt)
    # @assert Base.isdispatchtuple(sig)   # JuliaLang/julia#52233

    mi = ccall(:jl_method_lookup_by_tt, Any,
               (Any, Csize_t, Any),
               sig, world, #=method_table=# nothing)
    mi === nothing && throw(MethodError(ft, tt, world))
    mi = mi::MethodInstance

    # `jl_method_lookup_by_tt` and `jl_method_lookup` can return a unspecialized mi
    if !Base.isdispatchtuple(mi.specTypes)
        mi = Core.Compiler.specialize_method(mi.def, sig, mi.sparam_vals)::MethodInstance
    end

    return mi
end
else
    import GPUCompiler: methodinstance as my_methodinstance
end

export my_methodinstance


@static if VERSION < v"1.11-"

@inline function typed_fieldtype(@nospecialize(T::Type), i::Int)
    fieldtype(T, i)
end

else

@inline function typed_fieldtype(@nospecialize(T::Type), i::Int)
    if T <: GenericMemoryRef && i == 1 || T <: GenericMemory && i == 2
        eT = eltype(T)
        if !allocatedinline(eT) && Base.isconcretetype(eT)
            Ptr{Ptr{eT}}
        else
            Ptr{eT}
        end
    else
        fieldtype(T, i)
    end
end

end

export typed_fieldtype
