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

using Core: MethodInstance
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

tls_world_age() = ccall(:jl_get_tls_world_age, UInt, ())

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

## looking up method instances

export methodinstance

using Core.Compiler: retrieve_code_info, CodeInfo, MethodInstance, SSAValue, SlotNumber, ReturnNode
using Base: _methods_by_ftype

@inline function typed_signature(ft::Type, tt::Type)
    u = Base.unwrap_unionall(tt)
    return Base.rewrap_unionall(Tuple{ft, u.parameters...}, tt)
end

# create a MethodError from a function type
# TODO: fix upstream
function unsafe_function_from_type(ft::Type)
    if isdefined(ft, :instance)
        ft.instance
    else
        # HACK: dealing with a closure or something... let's do somthing really invalid,
        #       which works because MethodError doesn't actually use the function
        Ref{ft}()[]
    end
end
function MethodError(ft::Type, tt::Type, world::Integer=typemax(UInt))
    Base.MethodError(unsafe_function_from_type(ft), tt, world)
end

"""
    methodinstance(ft::Type, tt::Type, [world::UInt])

Look up the method instance that corresponds to invoking the function with type `ft` with
argument typed `tt`. If the `world` argument is specified, the look-up is static and will
always return the same result. If the `world` argument is not specified, the look-up is
dynamic and the returned method instance will automatically be invalidated when a relevant
function is redefined.
"""
function methodinstance(ft::Type, tt::Type, world::Integer=tls_world_age())
    sig = typed_signature(ft, tt)

    # look-up the method
    if VERSION >= v"1.10.0-DEV.65"
        meth = Base._which(sig; world).method
    elseif VERSION >= v"1.7.0-DEV.435"
        meth = Base._which(sig, world).method
    else
        meth = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), sig, world)
        if meth == nothing
            error("no unique matching method found for the specified argument types")
        end
    end

    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                      (Any, Any), sig, meth.sig)::Core.SimpleVector

    meth = Base.func_for_method_checked(meth, ti, env)

    method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                            (Any, Any, Any, UInt), meth, ti, env, world)

    return method_instance
end

Base.@deprecate_binding FunctionSpec methodinstance

