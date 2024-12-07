import EnzymeCore: Annotation
import EnzymeCore.EnzymeRules: FwdConfig, RevConfig, forward, augmented_primal, inactive, _annotate_tt

function has_frule_from_sig(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(TT), sv::Core.Compiler.AbsIntState)::Bool
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:FwdConfig,<:Annotation{ft},Type{<:Annotation},tt...}
    return isapplicable(interp, forward, TT, sv)
end

function has_rrule_from_sig(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(TT), sv::Core.Compiler.AbsIntState)::Bool
    ft, tt = _annotate_tt(TT)
    TT = Tuple{<:RevConfig,<:Annotation{ft},Type{<:Annotation},tt...}
    return isapplicable(interp, augmented_primal, TT, sv)
end


function is_inactive_from_sig(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(TT), sv::Core.Compiler.AbsIntState)
    return isapplicable(interp, inactive, TT, sv)
end

# `hasmethod` is a precise match using `Core.Compiler.findsup`,
# but here we want the broader query using `Core.Compiler.findall`.
# Also add appropriate backedges to the caller `MethodInstance` if given.
function isapplicable(@nospecialize(interp::Core.Compiler.AbstractInterpreter),
    @nospecialize(f), @nospecialize(TT), sv::Core.Compiler.AbsIntState)::Bool
    tt = Base.to_tuple_type(TT)
    sig = Base.signature_type(f, tt)
    mt = ccall(:jl_method_table_for, Any, (Any,), sig)
    mt isa Core.MethodTable || return false
    result = Core.Compiler.findall(sig, Core.Compiler.method_table(interp); limit=-1)
    (result === nothing || result === missing) && return false
    @static if isdefined(Core.Compiler, :MethodMatchResult)
        (; matches) = result
    else
        matches = result
    end
    # also need an edge to the method table in case something gets
    # added that did not intersect with any existing method
    fullmatch = Core.Compiler._any(match::Core.MethodMatch -> match.fully_covers, matches)
    if !fullmatch
        Core.Compiler.add_mt_backedge!(sv, mt, sig)
    end
    if Core.Compiler.isempty(matches)
        return false
    else
        for i = 1:Core.Compiler.length(matches)
            match = Core.Compiler.getindex(matches, i)::Core.MethodMatch
            edge = Core.Compiler.specialize_method(match)::Core.MethodInstance
            Core.Compiler.add_backedge!(sv, edge)
        end
        return true
    end
end

function rule_backedge_holder_generator(world::UInt, source, self, ft::Type)
    @nospecialize
    ft = functy.parameters[1]
    sig = Tuple{typeof(Base.identity)}
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL) 
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

    ### TODO: backedge from inactive, augmented_primal, forward, reverse
    @show ft
    edges = Any[]

    if ft == typeof(EnzymeRules.augmented_primal)
        sig = Tuple{typeof(EnzymeRules.augmented_primal), <:RevConfig, <:Annotation, Type{<:Annotation},Vararg{<:Annotation}}
        push!(edges, (ccall(:jl_method_table_for, Any, (Any,), sig), sig))
    elseif ft == typeof(EnzymeRules.forward)
        sig = Tuple{typeof(EnzymeRules.forward), <:FwdConfig, <:Annotation, Type{<:Annotation},Vararg{<:Annotation}}
        push!(edges, (ccall(:jl_method_table_for, Any, (Any,), sig), sig))
    else
        sig = Tuple{typeof(EnzymeRules.inactive), Vararg{<:Annotation}}
        push!(edges, (ccall(:jl_method_table_for, Any, (Any,), sig), sig))

        sig = Tuple{typeof(EnzymeRules.inactive_noinl), Vararg{<:Annotation}}
        push!(edges, (ccall(:jl_method_table_for, Any, (Any,), sig), sig))

        sig = Tuple{typeof(EnzymeRules.noalias), Vararg{<:Annotation}}
        push!(edges, (ccall(:jl_method_table_for, Any, (Any,), sig), sig))
    end
    @show edges
    new_ci.edges = edges

    # XXX: setting this edge does not give us proper method invalidation, see
    #      JuliaLang/julia#34962 which demonstrates we also need to "call" the kernel.
    #      invoking `code_llvm` also does the necessary codegen, as does calling the
    #      underlying C methods -- which GPUCompiler does, so everything Just Works.

    # prepare the slots
    new_ci.slotnames = Symbol[Symbol("#self#"), :ft]
    new_ci.slotflags = UInt8[0x00 for i = 1:2]

    # return the codegen world age
    push!(new_ci.code, Core.Compiler.ReturnNode(0))
    push!(new_ci.ssaflags, 0x00)   # Julia's native compilation pipeline (and its verifier) expects `ssaflags` to be the same length as `code`
    @static if isdefined(Core, :DebugInfo)
    else
      push!(new_ci.codelocs, 1)   # see note below
    end
    new_ci.ssavaluetypes += 1

    return new_ci
end

@eval Base.@assume_effects :removable :foldable :nothrow @inline function rule_backedge_holder(ft::Type)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, rule_backedge_holder_generator))
end