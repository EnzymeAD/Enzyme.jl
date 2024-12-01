function return_type(interp::Core.Compiler.AbstractInterpreter, mi::Core.MethodInstance)::Type
    @static if VERSION < v"1.11.0"
        code = Core.Compiler.get(Core.Compiler.code_cache(interp), mi, nothing)
        if code isa Core.Compiler.CodeInstance
            return code.rettype
        end
        result = Core.Compiler.InferenceResult(mi, Core.Compiler.typeinf_lattice(interp))
        Core.Compiler.typeinf(interp, result, :global)
        Core.Compiler.is_inferred(result) || return Any
        Core.Compiler.widenconst(Core.Compiler.ignorelimited(result.result))
    else
        something(Core.Compiler.typeinf_type(interp, mi), Any)
    end
end

function primal_interp_world(
    @nospecialize(::ReverseMode),
    world::UInt
)
    mode = Enzyme.API.DEM_ReverseModeCombined

    CT = @static if VERSION >= v"1.11.0-DEV.1552"
        EnzymeCacheToken(
            typeof(DefaultCompilerTarget()),
            false,
            GPUCompiler.GLOBAL_METHOD_TABLE, #=job.config.always_inline=#
            EnzymeCompilerParams,
            false,
        )
    else
        Enzyme.Compiler.GLOBAL_REV_CACHE
    end

    Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode)
end

function primal_interp_world(
    @nospecialize(::ForwardMode),
    world::UInt
)
    mode = Enzyme.API.DEM_ForwardMode

    CT = @static if VERSION >= v"1.11.0-DEV.1552"
        EnzymeCacheToken(
            typeof(DefaultCompilerTarget()),
            false,
            GPUCompiler.GLOBAL_METHOD_TABLE, #=job.config.always_inline=#
            EnzymeCompilerParams,
            true,
        )
    else
        Enzyme.Compiler.GLOBAL_FWD_CACHE
    end

    Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode)
end

@inline primal_interp_world(
    @nospecialize(::ReverseModeSplit),
    world::UInt) = primal_interp_world(Reverse, world)

function primal_return_type_world(
    @nospecialize(mode::Mode),
    world::UInt,
    @nospecialize(TT::Type),
)
    Core.Compiler._return_type(primal_interp_world(mode, world), TT)
end

function primal_return_type_world(
    @nospecialize(mode::Mode),
    world::UInt,
    mi::Core.MethodInstance,
)
    interp = primal_interp_world(mode, world)
    return_type(interp, mi)
end

primal_return_type_world(
    @nospecialize(mode::Mode),
    world::UInt,
    @nospecialize(FT::Type),
    @nospecialize(TT::Type),
   ) = primal_return_type_world(mode, world, Tuple{FT, TT.parameters...})

function primal_return_type_generator(world::UInt, source, self, @nospecialize(mode::Type), @nospecialize(ft::Type), @nospecialize(tt::Type))
    @nospecialize
    @assert Core.Compiler.isType(ft) && Core.Compiler.isType(tt)
    @assert mode <: Mode
    mode = mode()
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
    #interp = primal_interp_world(mode, world)
    #method_table = Core.Compiler.method_table(interp)
    method_table = nothing
    mthds = Base._methods_by_ftype(
        sig,
        method_table,
        -1, #=lim=#
        world,
        false, #=ambig=#
        min_world,
        max_world,
        has_ambig,
    )
    stub = Core.GeneratedFunctionStub(
        identity,
        Core.svec(:methodinstance, :mode, :ft, :tt),
        Core.svec(),
    )
    mthds === nothing && return stub(world, source, method_error)
    length(mthds) == 1 || return stub(world, source, method_error)

    # look up the method and code instance
    mtypes, msp, m = mthds[1]
    mi = ccall(
        :jl_specializations_get_linfo,
        Ref{Core.MethodInstance},
        (Any, Any, Any),
        m,
        mtypes,
        msp,
    )
    ci = Core.Compiler.retrieve_code_info(mi, world)::Core.Compiler.CodeInfo

    # prepare a new code info
    new_ci = copy(ci)
    empty!(new_ci.code)
    @static if isdefined(Core, :DebugInfo)
      new_ci.debuginfo = Core.DebugInfo(:none)
    else
      empty!(new_ci.codelocs)
      resize!(new_ci.linetable, 1)                # see note below
    end
    empty!(new_ci.ssaflags)
    new_ci.ssavaluetypes = 0
    new_ci.min_world = min_world[]
    new_ci.max_world = max_world[]
    new_ci.edges = Core.MethodInstance[mi]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :mode, :ft, :tt]
    new_ci.slotflags = UInt8[0x00 for i = 1:4]

    # return the codegen world age
    res = primal_return_type_world(mode, world, mi)
    push!(new_ci.code, Core.Compiler.ReturnNode(res))
    push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    @static if isdefined(Core, :DebugInfo)
    else
      push!(new_ci.codelocs, 1)   # see note below
    end
    new_ci.ssavaluetypes += 1

    # NOTE: we keep the first entry of the original linetable, and use it for location info
    #       on the call to check_cache. we can't not have a codeloc (using 0 causes
    #       corruption of the back trace), and reusing the target function's info
    #       has as advantage that we see the name of the kernel in the backtraces.

    return new_ci
end

@eval Base.@assume_effects :removable :foldable :nothrow @inline function primal_return_type(mode::Mode, ft::Type, tt::Type)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, primal_return_type_generator))
end

