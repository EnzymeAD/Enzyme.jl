unsafe_to_pointer(ptr) = ccall(Base.@cfunction(x->x, Ptr{Cvoid}, (Ptr{Cvoid},)), Ptr{Cvoid}, (Any,), ptr)
export unsafe_to_pointer

const Tracked = 10
export Tracked

function unsafe_to_llvm(val, ctx)
    T_jlvalue = LLVM.StructType(LLVM.LLVMType[]; ctx)
    T_prjlvalue = LLVM.PointerType(T_jlvalue, Tracked)
    T_prjlvalue_UT = LLVM.PointerType(T_jlvalue)
    fill_val = unsafe_to_pointer(val)
    fill_val = LLVM.ConstantInt(convert(UInt, fill_val); ctx)
    fill_val = LLVM.const_inttoptr(fill_val, T_prjlvalue_UT)
    LLVM.const_addrspacecast(fill_val, T_prjlvalue)
end
export unsafe_to_llvm

function makeInstanceOf(@nospecialize(T), ctx)
    @assert Core.Compiler.isconstType(T)
    @assert T <: Type
    return unsafe_to_llvm(T.parameters[1], ctx)
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

if VERSION <= v"1.6"
    allocatedinline(@nospecialize(T)) = T.isinlinealloc
else
    import Base: allocatedinline
end

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


if VERSION >= v"1.10.0-DEV.873"

# on 1.10 (JuliaLang/julia#48611) the generated function knows which world it was invoked in

function _generated_ex(world, source, ex)
    stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :ft, :tt), Core.svec())
    stub(world, source, ex)
end

function codegen_world_age_generator(world::UInt, source, self, ft::Type, tt::Type)
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    # validation
    ft <: Core.Builtin && error("$(unsafe_function_from_type(ft)) is not a generic function")

    # look up the method
    method_error = :(throw(MethodError(ft, tt, $world)))
    sig = Tuple{ft, tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL)  # don't care about ambiguous results
    mthds = if VERSION >= v"1.7.0-DEV.1297"
        Base._methods_by_ftype(sig, #=mt=# nothing, #=lim=# -1,
                               world, #=ambig=# false,
                               min_world, max_world, has_ambig)
        # XXX: use the correct method table to support overlaying kernels
    else
        Base._methods_by_ftype(sig, #=lim=# -1,
                               world, #=ambig=# false,
                               min_world, max_world, has_ambig)
    end
    mthds === nothing && return _generated_ex(world, source, method_error)
    length(mthds) == 1 || return _generated_ex(world, source, method_error)

    # look up the method and code instance
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
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

else

# on older versions of Julia we fall back to looking up the current world. this may be wrong
# when the generator is invoked in a different world (TODO: when does this happen?)

function codegen_world_age_generator(self, ft::Type, tt::Type)
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    ft = ft.parameters[1]
    tt = tt.parameters[1]

    # validation
    ft <: Core.Builtin && error("$(unsafe_function_from_type(ft)) is not a generic function")

    # look up the method
    method_error = :(throw(MethodError(ft, tt)))
    sig = Tuple{ft, tt.parameters...}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL)  # don't care about ambiguous results
    mthds = if VERSION >= v"1.7.0-DEV.1297"
        Base._methods_by_ftype(sig, #=mt=# nothing, #=lim=# -1,
                               #=world=# typemax(UInt), #=ambig=# false,
                               min_world, max_world, has_ambig)
        # XXX: use the correct method table to support overlaying kernels
    else
        Base._methods_by_ftype(sig, #=lim=# -1,
                               #=world=# typemax(UInt), #=ambig=# false,
                               min_world, max_world, has_ambig)
    end
    # XXX: using world=-1 is wrong, but the current world isn't exposed to this generator
    mthds === nothing && return method_error
    length(mthds) == 1 || return method_error

    # look up the method and code instance
    mtypes, msp, m = mthds[1]
    mi = ccall(:jl_specializations_get_linfo, Ref{MethodInstance}, (Any, Any, Any), m, mtypes, msp)
    ci = retrieve_code_info(mi)::CodeInfo

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

    # return the current world age (which is not technically the codegen world age,
    # but works well enough for invalidation purposes)
    push!(new_ci.code, ReturnNode(Base.get_world_counter()))
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
    $(Expr(:meta,
           :generated,
           Expr(:new,
                Core.GeneratedFunctionStub,
                :codegen_world_age_generator,
                Any[:methodinstance, :ft, :tt],
                Any[],
                @__LINE__,
                QuoteNode(Symbol(@__FILE__)),
                true)))
end

end

export codegen_world_age





