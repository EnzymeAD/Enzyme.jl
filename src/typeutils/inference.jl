function return_type(interp::Core.Compiler.AbstractInterpreter, mi::Core.MethodInstance)::Type
    return @static if VERSION < v"1.11.0"
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
            world,
            false,
            true,
            true
        )
    else
        Enzyme.Compiler.GLOBAL_REV_CACHE
    end

    return Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode, true)
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
            world,
            true,
            false,
            true
        )
    else
        Enzyme.Compiler.GLOBAL_FWD_CACHE
    end

    return Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode, true)
end

@inline primal_interp_world(
    @nospecialize(::ReverseModeSplit),
    world::UInt
) = primal_interp_world(Reverse, world)

function primal_return_type_world(
        @nospecialize(mode::Mode),
        world::UInt,
        @nospecialize(TT::Type),
    )
    return Core.Compiler._return_type(primal_interp_world(mode, world), TT)
end

function primal_return_type_world(
        @nospecialize(mode::Mode),
        world::UInt,
        mi::Core.MethodInstance,
    )
    interp = primal_interp_world(mode, world)
    return return_type(interp, mi)
end

primal_return_type_world(
    @nospecialize(mode::Mode),
    world::UInt,
    @nospecialize(FT::Type),
    @nospecialize(TT::Type),
) = primal_return_type_world(mode, world, Tuple{FT, TT.parameters...})

function primal_return_type end

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
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))

    mi = my_methodinstance(mode, ft, tt, world, min_world, max_world)

    slotnames = Core.svec(Symbol("#self#"), :mode, :ft, :tt)
    stub = Core.GeneratedFunctionStub(
        identity,
        slotnames,
        Core.svec(),
    )
    mi === nothing && return stub(world, source, :(throw(MethodError(ft, tt, $world))))

    result = primal_return_type_world(mode, world, mi)
    code = Any[Core.Compiler.ReturnNode(result)]
    # create an empty CodeInfo to return the result
    ci = create_fresh_codeinfo(primal_return_type, source, world, slotnames, code)
    ci.max_world = max_world[]

    ci.edges = Any[]
    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.
    add_edge!(ci.edges, mi)

    return ci
end

@eval Base.@assume_effects :removable :foldable :nothrow @inline function primal_return_type(mode::Mode, ft::Type, tt::Type)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, primal_return_type_generator))
end
