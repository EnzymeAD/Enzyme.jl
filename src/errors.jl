const VERBOSE_ERRORS = Ref(false)

"""
    EnzymeError

Common supertype for Enzyme-specific errors.

This type is made public so that downstream packages can add custom [error hints](https://docs.julialang.org/en/v1/base/base/#Base.Experimental.register_error_hint) for the most common exceptions thrown by Enzyme.
"""
abstract type EnzymeError <: Base.Exception end

abstract type CompilationException <: EnzymeError end

function pretty_print_mi(mi, io=stdout; digit_align_width = 1)
    spec = mi.specTypes.parameters
    ft = spec[1]
    arg_types_param = spec[2:end]
    f_is_function = false
    kwargs = []
    if ft === typeof(Core.kwcall) && length(arg_types_param) >= 2 && arg_types_param[1] <: NamedTuple
        ft = arg_types_param[2]
        kwt = arg_types_param[1]
        arg_types_param = arg_types_param[3:end]
        keys = kwt.parameters[1]::Tuple
        kwargs = Any[(keys[i], fieldtype(kwt, i)) for i in eachindex(keys)]
    end

    Base.show_signature_function(io, ft)
    Base.show_tuple_as_call(io, :function, Tuple{arg_types_param...}; hasfirst=false, kwargs = isempty(kwargs) ? nothing : kwargs)

    m = mi.def

    modulecolor = :light_black
    tv, decls, file, line = Base.arg_decl_parts(m)
    #if m.sig <: Tuple{Core.Builtin, Vararg}
    #    file = "none"
    #    line = 0
    #end

    if !(get(io, :compact, false)::Bool) # single-line mode
        println(io)
        digit_align_width += 4
    end

    # module & file, re-using function from errorshow.jl
    Base.print_module_path_file(io, Base.parentmodule(m), string(file), line; modulecolor, digit_align_width)
end

using InteractiveUtils

function code_typed_helper(mi::Core.MethodInstance, world::UInt, mode::Enzyme.API.CDerivativeMode = Enzyme.API.DEM_ReverseModeCombined; interactive::Bool=false, kwargs...)
    CT = @static if VERSION >= v"1.11.0-DEV.1552"
        EnzymeCacheToken(
            typeof(DefaultCompilerTarget()),
            false,
            GPUCompiler.GLOBAL_METHOD_TABLE, #=job.config.always_inline=#
            EnzymeCompilerParams,
            world,
            mode == API.DEM_ForwardMode,
            mode != API.DEM_ForwardMode,
            true
        )
    else
        if mode == API.DEM_ForwardMode
            GLOBAL_FWD_CACHE
        else
            GLOBAL_REV_CACHE
        end
    end

    interp = Enzyme.Compiler.Interpreter.EnzymeInterpreter(CT, nothing, world, mode, true)

    sig = mi.specTypes  # XXX: can we just use the method instance?
    if interactive
        # call Cthulhu without introducing a dependency on Cthulhu
        mod = get(Base.loaded_modules, Cthulhu, nothing)
        mod===nothing && error("Interactive code reflection requires Cthulhu; please install and load this package first.")
        descend_code_typed = getfield(mod, :descend_code_typed)
        descend_code_typed(sig; interp, kwargs...)
    else
        Base.code_typed_by_type(sig; interp, kwargs...)
    end
end

struct EnzymeRuntimeException <: EnzymeError
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeRuntimeException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "EnzymeRuntimeException: Enzyme execution failed.\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeRuntimeExceptionMI <: EnzymeError
    msg::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::EnzymeRuntimeExceptionMI; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

function Base.showerror(io::IO, ece::EnzymeRuntimeExceptionMI)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "EnzymeRuntimeException: Enzyme execution failed within\n")
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": catch this exception as `err` and call `code_typed(err)` to inspect the surrounding code.\n";
        color = :cyan,
    )
    println(io)
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

abstract type CustomRuleError <: Base.Exception end

struct NonConstantKeywordArgException <: CustomRuleError
    backtrace::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::NonConstantKeywordArgException; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

function Base.showerror(io::IO, ece::NonConstantKeywordArgException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "NonConstantKeywordArgException: Custom Rule for method was passed a differentiable keyword argument. Differentiable kwargs cannot currently be specified from within the rule system.\n")
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": Experimental utility Enzyme.EnzymeRules.inactive_kwarg will enable you to mark the keyword arguments as non-differentiable, if that is correct.";
        color = :cyan,
    )
    println(io)
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    Base.println(io, Base.unsafe_string(ece.backtrace))
end

struct CallingConventionMismatchError{ST} <: CustomRuleError
    backtrace::ST
    mi::Core.MethodInstance
    world::UInt
end

function Base.showerror(io::IO, ece::CallingConventionMismatchError)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "CallingConventionMismatchError: Enzyme hit an internal error trying to parse the julia calling convention definition for:\n")
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    println(io)
    if VERSION >= v"1.12"
        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": You are currently on Julia 1.12, which changed its calling convention. Tracking issue for Enzyme adapting to this new calling convention is https://github.com/EnzymeAD/Enzyme.jl/issues/2707.\n";
            color = :cyan,
        )
    else
        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
            color = :cyan,
        )
    end
    println(io)


    if true || VERBOSE_ERRORS[]
        if ece.backtrace isa Cstring
           Base.println(io, Base.unsafe_string(ece.backtrace))
        else
           Base.println(io, ece.backtrace)
        end
    else
        print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    end
end

InteractiveUtils.code_typed(ece::CallingConventionMismatchError; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

struct ForwardRuleReturnError{C, RT, fwd_RT} <: CustomRuleError
    backtrace::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::ForwardRuleReturnError; kwargs...) = code_typed_helper(ece.mi, ece.world, Enzyme.API.DEM_ForwardMode; kwargs...)

function Base.showerror(io::IO, ece::ForwardRuleReturnError{C, RT, fwd_RT}) where {C, RT, fwd_RT}
    ExpRT = EnzymeRules.forward_rule_return_type(C, RT)
    @assert ExpRT != fwd_RT
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end

    RealRt = eltype(RT)

    hint = nothing

    width = EnzymeRules.width(C)

    desc = if EnzymeRules.needs_primal(C) && EnzymeRules.needs_shadow(C)
        if width == 1
            if !(fwd_RT <: Duplicated)
                if fwd_RT <: BatchDuplicated
                    hint = "For width 1, the return type should be a Duplicated, not BatchDuplicated"
                elseif fwd_RT <: RealRt
                    hint = "Both primal and shadow need to be returned"
                else
                    hint = "Return type should be a Duplicated"
                end
            elseif eltype(fwd_RT) <: RealRt
                hint = "Expected the abstract type $RealRt for primal/shadow, you returned $(eltype(fwd_RT)). Even though $(eltype(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
            else
                hint = "The type within your Duplicated $(eltype(fwd_RT)) does not match the primal type $RealRt"
            end
        else
            if !(fwd_RT <: BatchDuplicated)
                if fwd_RT <: BatchDuplicated && EnzymeCore.batch_size(fwd_RT) != width
                    hint = "Mismatched batch size, expected batch size $width, found a BatchDuplicated of width $(EnzymeCore.batch_size(fwd_RT))"
                elseif fwd_RT <: Duplicated
                    hint = "For width $width, the return type should be a BatchDuplicated, not a Duplicated"
                elseif fwd_RT <: RealRt
                    hint = "Both primal and shadow need to be returned"
                else
                    hint = "Return type should be a BatchDuplicated"
                end
            elseif eltype(fwd_RT) <: RealRt
                hint = "Expected the abstract type $RealRt for primal/shadow, you returned $(eltype(fwd_RT)). Even though $(eltype(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
            else
                hint = "The type within your BatchDuplicated $(eltype(fwd_RT)) does not match the primal type $RealRt"
            end
        end
        "primal and shadow configuration"
    elseif EnzymeRules.needs_primal(C) && !EnzymeRules.needs_shadow(C)
        if fwd_RT <: BatchDuplicated || fwd_RT <: Duplicated
            hint = "Shadow was not requested, you should only return the primal"
        elseif fwd_RT <: (NTuple{N, <:RealRt} where N)
            hint = "You appear to be returning a tuple of shadows, but only the primal was requested"
        elseif fwd_RT <: RealRt
            hint = "Expected the abstract type $RealRt for primal, you returned $(fwd_RT). Even though $(fwd_RT) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
        else
            hint = "Your return type does not match the primal type $RealRt"
        end

        "primal-only configuration"
    elseif !EnzymeRules.needs_primal(C) && EnzymeRules.needs_shadow(C)
        if fwd_RT <: BatchDuplicated || fwd_RT <: Duplicated
            hint = "Primal was not requested, you should only return the shadow"
        elseif width == 1
            if fwd_RT <: (NTuple{N, <:RealRt} where N)
                hint = "You look to be returning a tuple of shadows, when the batch size is 1"
            elseif fwd_RT <: RealRt
                hint = "Expected the abstract type $RealRt for shadow, you returned $(fwd_RT). Even though $(fwd_RT) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
            else
                hint = "Your return type does not match the shadow type $RealRt"
            end
        else
            if !(fwd_RT <: NTuple)
                hint = "Configuration required batch size $width, which requires returning a tuple of shadows"
            elseif !(fwd_RT <: NTuple{width, <:Any})
                hint = "Did not return a tuple of shadows of the right size, expected a tuple of size $width"
            elseif eltype(fwd_RT) <: RealRt
                hint = "Expected the abstract type $RealRt for each shadow in the tuple to create $ExpRT, you returned $(eltype(fwd_RT)) as the eltype of your tuple ($fwd_RT). Even though $(eltype(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
            else
                hint = "Your return type does not match the batched shadow type $ExpRT"
            end
        end
        "shadow-only configuration"
    else
        @assert !EnzymeRules.needs_primal(C) && !EnzymeRules.needs_shadow(C)

        if fwd_RT <: BatchDuplicated || fwd_RT <: Duplicated
            hint = "Neither primal nor shadow were requested, you should return nothing, not both the primal and shadow"
        elseif fwd_RT <: (NTuple{N, <:RealRt} where N)
            hint = "You appear to be returning a tuple of shadows, but neither primal nor shadow were requested"
        elseif fwd_RT <: RealRt && width == 1
            hint = "You appear to be returning a primal or shadow, but neither were requested"
        elseif fwd_RT <: RealRt
            hint = "You appear to be returning a primal, but it was not requested"
        else
            hint = "You should return nothing"
        end

        "neither primal nor shadow configuration"
    end

    print(io, "ForwardRuleReturnError: Incorrect return type for $desc of forward custom rule with width $width of a function which returned $(eltype(RealRt)):\n")
    print(io, "  found    : ", fwd_RT, "\n")
    print(io, "  expected : ", ExpRT, "\n")
    println(io)
    print(io, "For more information see `EnzymeRules.forward_rule_return_type`\n")
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": ", hint;
        color = :cyan,
    )
    println(io)
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": if the reason for the return type is unclear, you can catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
        color = :cyan,
    )
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    Base.println(io, Base.unsafe_string(ece.backtrace))
end


struct AugmentedRuleReturnError{C, RT, aug_RT} <: CustomRuleError
    backtrace::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::AugmentedRuleReturnError; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

function Base.showerror(io::IO, ece::AugmentedRuleReturnError{C, RT, fwd_RT}) where {C, RT, fwd_RT}
    ExpRT = EnzymeRules.augmented_rule_return_type(C, RT, Any)
    @assert ExpRT != fwd_RT
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end

    width = EnzymeRules.width(C)

    RealRt = eltype(RT)

    primal_found = nothing
    shadow_found = nothing

    hint = nothing

    desc = if EnzymeRules.needs_primal(C) && EnzymeRules.needs_shadow(C)
        if !(fwd_RT <: EnzymeRules.AugmentedReturn)
            hint = "Return should be a struct of type EnzymeRules.AugmentedReturn"
        elseif fwd_RT isa UnionAll && (fwd_RT.body isa UnionAll || fwd_RT.body.parameters[1] isa TypeVar || fwd_RT.body.parameters[2] isa TypeVar)
            hint = "Return is a UnionAll, not a concrete type, try explicitly returning a single value of type EnzymeRules.AugmentedReturn{PrimalType, ShadowType, CacheType} as follows\n  return EnzymeRules.augmented_rule_return_type(config, RA)(primal, shadow, cache)"
        elseif EnzymeRules.primal_type(fwd_RT) == Nothing
            hint = "Missing primal return"
        elseif EnzymeRules.shadow_type(fwd_RT) == Nothing
            hint = "Missing shadow return"
        elseif EnzymeRules.primal_type(fwd_RT) != RealRt
            if EnzymeRules.primal_type(fwd_RT) <: RealRt
                hint = "Expected the abstract type $RealRt for primal, you returned $(EnzymeRules.primal_type(fwd_RT)). Even though $(EnzymeRules.primal_type(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
            else
                hint = "Mismatched primal type $(EnzymeRules.sprimal_type(fwd_RT)), expected $RealRt"
            end
        elseif EnzymeRules.shadow_type(fwd_RT) != RealRt
            if width == 1
                if EnzymeRules.shadow_type(fwd_RT) <: RealRt
                    hint = "Expected the abstract type $RealRt for shadow, you returned $(EnzymeRules.shadow_type(fwd_RT)). Even though $(EnzymeRules.shadow_type(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
                elseif EnzymeRules.shadow_type(fwd_RT) <: (NTuple{N, <:RealRt} where N)
                    hint = "Batch size was 1, expected a single shadow, not a tuple of shadows."
                else
                    hint = "Mismatched shadow type $(EnzymeRules.shadow_type(fwd_RT)), expected $(EnzymeRules.shadow_type(ExpRT))."
                end
            else
                if EnzymeRules.shadow_type(fwd_RT) <: RealRt
                    hint = "Batch size was $width, expected a tuple of shadows, not a single shadow."
                elseif EnzymeRules.shadow_type(fwd_RT) <: (NTuple{N, <:RealRt} where N)
                    hint = "Expected the abstract type $RealRt for the element shadow type (for a batched shadow type $(EnzymeRules.shadow_type(ExpRT))), you returned $(eltype(EnzymeRules.shadow_type(fwd_RT))) as the element shadow type (batched to become $(EnzymeRules.shadow_type(fwd_RT)). Even though $(eltype(EnzymeRules.shadow_type(fwd_RT))) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
                else
                    hint = "Mismatched shadow type $(EnzymeRules.shadow_type(fwd_RT)), expected $(EnzymeRules.shadow_type(ExpRT))."
                end
            end
        end

        "primal and shadow configuration"
    elseif EnzymeRules.needs_primal(C) && !EnzymeRules.needs_shadow(C)
        if !(fwd_RT <: EnzymeRules.AugmentedReturn)
            hint = "Return should be a struct of type EnzymeRules.AugmentedReturn"
        elseif EnzymeRules.primal_type(fwd_RT) == Nothing
            hint = "Missing primal return"
        elseif EnzymeRules.shadow_type(fwd_RT) != Nothing
            hint = "Shadow return was not requested"
        elseif EnzymeRules.primal_type(fwd_RT) != RealRt
            if EnzymeRules.primal_type(fwd_RT) <: RealRt
                hint = "Expected the abstract type $RealRt for primal, you returned $(EnzymeRules.primal_type(fwd_RT)). Even though $(EnzymeRules.primal_type(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
            else
                hint = "Mismatched primal type $(EnzymeRules.primal_type(fwd_RT)), expected $RealRt"
            end
        end

        "primal-only configuration"
    elseif !EnzymeRules.needs_primal(C) && EnzymeRules.needs_shadow(C)

        if !(fwd_RT <: EnzymeRules.AugmentedReturn)
            hint = "Return should be a struct of type EnzymeRules.AugmentedReturn"
        elseif EnzymeRules.primal_type(fwd_RT) != Nothing
            hint = "Primal was not requested"
        elseif EnzymeRules.shadow_type(fwd_RT) != RealRt
            if width == 1
                if EnzymeRules.shadow_type(fwd_RT) <: RealRt
                    hint = "Expected the abstract type $RealRt for shadow, you returned $(EnzymeRules.shadow_type(fwd_RT)). Even though $(EnzymeRules.shadow_type(fwd_RT)) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
                elseif EnzymeRules.shadow_type(fwd_RT) <: (NTuple{N, <:RealRt} where N)
                    hint = "Batch size was 1, expected a single shadow, not a tuple of shadows."
                else
                    hint = "Mismatched shadow type $(EnzymeRules.shadow_type(fwd_RT)), expected $(EnzymeRules.shadow_type(ExpRT))."
                end
            else
                if EnzymeRules.shadow_type(fwd_RT) <: RealRt
                    hint = "Batch size was $width, expected a tuple of shadows, not a single shadow."
                elseif EnzymeRules.shadow_type(fwd_RT) <: (NTuple{N, <:RealRt} where N)
                    hint = "Expected the abstract type $RealRt for the element shadow type (for a batched shadow type $(EnzymeRules.shadow_type(ExpRT))), you returned $(eltype(EnzymeRules.shadow_type(fwd_RT))) as the element shadow type (batched to become $(EnzymeRules.shadow_type(fwd_RT)). Even though $(eltype(EnzymeRules.shadow_type(fwd_RT))) <: $RealRt, rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
                else
                    hint = "Mismatched shadow type $(EnzymeRules.shadow_type(fwd_RT)), expected $(EnzymeRules.shadow_type(ExpRT))."
                end
            end
        end

        "shadow-only configuration"
    else
        if !(fwd_RT <: EnzymeRules.AugmentedReturn)
            hint = "Return should be a struct of type EnzymeRules.AugmentedReturn"
        elseif EnzymeRules.primal_type(fwd_RT) != Nothing
            hint = "Primal was not requested"
        elseif EnzymeRules.shadow_type(fwd_RT) != Nothing
            hint = "Shadow return was not requested"
        end

        @assert !EnzymeRules.needs_primal(C) && !EnzymeRules.needs_shadow(C)
        "neither primal nor shadow configuration"
    end

    print(io, "AugmentedRuleReturnError: Incorrect return type for $desc of augmented_primal custom rule with width $width of a function which returned $(eltype(RealRt)):\n")
    print(io, "  found    : ", fwd_RT, "\n")
    print(io, "  expected : ", ExpRT, "\n")
    println(io)
    print(io, "For more information see `EnzymeRules.augmented_rule_return_type`\n")
    println(io)
    if hint !== nothing
        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": ", hint;
            color = :cyan,
        )
        println(io)
    end
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": if the reason for the return type is unclear, you can catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
        color = :cyan,
    )
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    Base.println(io, Base.unsafe_string(ece.backtrace))
end


struct ReverseRuleReturnError{C, ArgAct, rev_RT} <: CustomRuleError
    backtrace::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::ReverseRuleReturnError; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

function Base.showerror(io::IO, ece::ReverseRuleReturnError{C, ArgAct, rev_RT}) where {C, ArgAct, rev_RT}
    width = EnzymeRules.width(C)
    Tys = (
        A <: Active ? (width == 1 ? eltype(A) : NTuple{Int(width),eltype(A)}) : Nothing for A in ArgAct.parameters
    )
    ExpRT = Tuple{Tys...}
    @assert ExpRT != rev_RT
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end

    hint = nothing

    if !(rev_RT <: Tuple)
        hint = "Return type should be a tuple with one element for each argument"
    elseif length(rev_RT.parameters) != length(ExpRT.parameters)
        hint = "Returned tuple should have one result for each argument, had $(length(rev_RT.parameters)) elements, expected $(length(ExpRT.parameters))"
    else
        for i in 1:length(ArgAct.parameters)
            if ExpRT.parameters[i] == rev_RT.parameters[i]
                continue
            end
            if ExpRT.parameters[i] === Nothing
                hint = "Tuple return mismatch at index $i, argument of type $(ArgAct.parameters[i]) corresponds to return of nothing (only Active inputs have returns)"
                break
            end

            if rev_RT.parameters[i] === Nothing
                hint = "Tuple return mismatch at index $i, argument of type $(ArgAct.parameters[i]) corresponds to return of $(ExpRT.parameters[i]), found nothing (Active inputs have returns)"
                break
            end

            if width == 1

                if rev_RT.parameters[i] <: (NTuple{N, ExpRT.parameters[i]} where N)
                    hint = "Tuple return mismatch at index $i, returned a tuple of results when expected just one of type $(ExpRT.parameters[i])."
                    break
                end

                if rev_RT.parameters[i] <: ExpRT.parameters[i]
                    hint = "Tuple return mismatch at index $i, expected the abstract type $(ExpRT.parameters[i]), you returned $(rev_RT.parameters[i]). Even though $(rev_RT.parameters[i]) <: $(ExpRT.parameters[i]), rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
                    break
                end

            else

                if !(rev_RT.parameters[i] <: NTuple)
                    hint = "Tuple return mismatch at index $i, returned a single result of type $(rev_RT.parameters[i]) for a batched configuration of width $width, expected an inner tuple for each batch element."
                    break
                end

                if eltype(rev_RT.parameters[i]) <: eltype(ExpRT.parameters[i])
                    hint = "Tuple return mismatch at index $i, expected the abstract type $(eltype(ExpRT.parameters[i])) (here batched to form $(ExpRT.parameters[i])), you returned $(eltype(rev_RT.parameters[i])) (batched to form $(eltype(rev_RT.parameters[i]))). Even though $(eltype(rev_RT.parameters[i])) <: $(eltype(ExpRT.parameters[i])), rules require an exact match (akin to how you cannot substitute Vector{Float64} in a method that takes a Vector{Real})."
                    break
                end
            end

            hint = "Tuple return mismatch at index $i, argument of type $(ArgAct.parameters[i]) corresponds to returning type $(ExpRT.parameters[i]), you returned $(rev_RT.parameters[i])."
            break
        end
    end
    @assert hint !== nothing

    print(io, "ReverseRuleReturnError: Incorrect return type for reverse custom rule with width $(EnzymeRules.width(C)):\n")
    print(io, "  found    : ", rev_RT, "\n")
    print(io, "  expected : ", ExpRT, "\n")
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": ", hint;
        color = :cyan,
    )
    println(io)
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": if the reason for the return type is unclear, you can catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
        color = :cyan,
    )
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    Base.println(io, Base.unsafe_string(ece.backtrace))
end

struct MixedReturnException{RT} <: CustomRuleError
    backtrace::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::MixedReturnException; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

function Base.showerror(io::IO, ece::MixedReturnException{RT}) where RT
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "MixedReturnException: Custom Rule for method returns type $(RT), which has mixed internal activity types. This is not presently supported.\n")
    print(io, "See https://enzyme.mit.edu/julia/stable/faq/#Mixed-activity for more information.\n")
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": if the reason for the return type is unclear, you can catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
        color = :cyan,
    )
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    Base.println(io, Base.unsafe_string(ece.backtrace))
end


struct UnionSretReturnException{RT} <: CustomRuleError
    backtrace::Cstring
    mi::Core.MethodInstance
    world::UInt
end

InteractiveUtils.code_typed(ece::UnionSretReturnException; kwargs...) = code_typed_helper(ece.mi, ece.world; kwargs...)

function Base.showerror(io::IO, ece::UnionSretReturnException{RT}) where RT
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "UnionSretReturnException: Custom Rule for method returns type $(RT), which is a union has an sret layout calling convention. This is not presently supported.\n")
    print(io, "Please open an issue if you hit this.")
    println(io)
    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": if the reason for the return type is unclear, you can catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
        color = :cyan,
    )
    println(io)
    println(io)
    pretty_print_mi(ece.mi, io)
    println(io)
    Base.println(io, Base.unsafe_string(ece.backtrace))
end

struct NonInferredActiveReturn <: CompilationException
    actualRetType::Type
    rettype::Type
end

function Base.showerror(io::IO, ece::NonInferredActiveReturn)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "NonInferredActiveReturn: Enzyme compilation failed.\n")
    println(io, " Called reverse-mode autodiff with return activity $(ece.rettype), which had a different setting of Base.allocatedinline from the actual return type $(ece.actualRetType). This is not presently supported (but open an issue).")

    if ece.actualRetType <: eltype(ece.rettype)
        newRT = if ece.rettype <: Active
            Active{ece.actualRetType}
        elseif ece.rettype <: MixedDuplicated
            MixedDuplicated{ece.actualRetType}
        elseif ece.rettype <: BatchMixedDuplicated
            BatchMixedDuplicated{ece.actualRetType, batch_size(ece.rettype)}
        else
            throw(AssertionError("Unexpected Activity $(ece.rettype)"))
        end

        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": You can avoid this error by explicitly setting the return activity as $newRT";
            color = :cyan,
        )
    end
end

struct NoDerivativeException <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::NoDerivativeException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "NoDerivativeException: Enzyme compilation failed.\n")
    if ece.ir !== nothing
        if VERBOSE_ERRORS[]
            print(io, "Current scope: \n")
            print(io, ece.ir)
        else
            print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
        end
    end
    if occursin("cannot handle unknown binary operator", ece.msg)
      for msg in split(ece.msg, '\n')
        if occursin("cannot handle unknown binary operator", msg)
          print('\n', msg, '\n')
        end
      end
    else
      print(io, '\n', ece.msg, '\n')
    end
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct IllegalTypeAnalysisException <: CompilationException
    msg::String
    mi::Union{Nothing, Core.MethodInstance}
    world::Union{Nothing, UInt}
    sval::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalTypeAnalysisException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "IllegalTypeAnalysisException: Enzyme compilation failed due to illegal type analysis.\n")
    if VERSION >= v"1.12" && VERSION < v"1.12.5"
        printstyled(io, "Hint:"; bold = true, color = :cyan)
        printstyled(
            io,
            ": You are using Julia $(VERSION) which is known as a source of this error. This will be fixed in Julia 1.12.5. Either use Julia 1.10, 1.11, or wait for Julia 1.12.5.\nTo track the release progress, see https://github.com/JuliaLang/julia/pull/60612.";
            color = :cyan,
        )
        print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    else
        print(io, " This usually indicates the use of a Union type, which is not fully supported with Enzyme.API.strictAliasing set to true [the default].\n")
        print(io, " Ideally, remove the union (which will also make your code faster), or try setting Enzyme.API.strictAliasing!(false) before any autodiff call.\n")
        print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
            if ece.mi !== nothing
            print(io, " Failure within method: ", ece.mi, "\n")
            printstyled(io, "Hint"; bold = true, color = :cyan)
            printstyled(
                io,
                ": catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\nIf you have Cthulu.jl loaded you can also use `code_typed(err; interactive = true)` to interactively introspect the code.";
                color = :cyan,
            )
        end
    end

    if VERBOSE_ERRORS[]
        if ece.ir !== nothing
            print(io, "Current scope: \n")
            print(io, ece.ir)
        end
        print(io, "\n Type analysis state: \n")
        write(io, ece.sval)
        print(io, '\n', ece.msg, '\n')
    end
    if ece.bt !== nothing
        print(io, "\nCaused by:")
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

function InteractiveUtils.code_typed(ece::IllegalTypeAnalysisException; kwargs...)
    mi = ece.mi
    if mi === nothing
        throw(AssertionError("code_typed(::IllegalTypeAnalysisException; interactive::Bool=false, kwargs...) not supported for error without mi"))
    end
    world = ece.world::UInt
    mode = Enzyme.API.DEM_ReverseModeCombined
    code_typed_helper(ece.mi, ece.world; kwargs...)
end

struct IllegalFirstPointerException <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::IllegalFirstPointerException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "IllegalFirstPointerException: Enzyme compilation failed due to an internal error (first pointer exception).\n")
    print(io, " Please open an issue with the code to reproduce and full error log on github.com/EnzymeAD/Enzyme.jl\n")
    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    if VERBOSE_ERRORS[]
      if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
      end
    end
    print(io, '\n', ece.msg, '\n')
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct EnzymeInternalError <: CompilationException
    msg::String
    ir::Union{Nothing,String}
    bt::Union{Nothing,Vector{StackTraces.StackFrame}}
end

function Base.showerror(io::IO, ece::EnzymeInternalError)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "EnzymeInternalError: Enzyme compilation failed due to an internal error.\n")
    print(io, " Please open an issue with the code to reproduce and full error log on github.com/EnzymeAD/Enzyme.jl\n")
    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    if VERBOSE_ERRORS[]
      if ece.ir !== nothing
        print(io, "Current scope: \n")
        print(io, ece.ir)
      end
      print(io, '\n', ece.msg, '\n')
    else
      for msg in split(ece.msg, '\n')
        if occursin("Illegal replace ficticious phi for", msg)
          print('\n', msg, '\n')
        end
      end
    end
    if ece.bt !== nothing
        Base.show_backtrace(io, ece.bt)
        println(io)
    end
end

struct EnzymeMutabilityException <: EnzymeError
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeMutabilityException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    msg = Base.unsafe_string(ece.msg)
    print(io, "EnzymeMutabilityException: ", msg, '\n')
end

struct EnzymeRuntimeActivityError{ST, MT,WT} <: EnzymeError
    msg::ST
    mi::MT
    world::WT
end

function Base.showerror(io::IO, ece::EnzymeRuntimeActivityError)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    println(io, "EnzymeRuntimeActivityError: Detected potential need for runtime activity.\n")
    println(io, "Constant memory is stored (or returned) to a differentiable variable and correctness cannot be guaranteed with static activity analysis.")
    println(
        io,
        "This might be due to the use of a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/faq/#faq-runtime-activity).",
    )
    println(
        io,
        "If Enzyme should be able to prove this use non-differentable, open an issue!",
    )
    println(io)
    println(io, "To work around this issue, either:")
    println(
        io,
        "   a) rewrite this variable to not be conditionally active (fastest performance, slower to setup), or",
    )
    println(
        io,
        "   b) set the Enzyme mode to turn on runtime activity (e.g. autodiff(set_runtime_activity(Reverse), ...) ). This will maintain correctness, but may slightly reduce performance.",
    )
    println(io)
    if ece.mi !== nothing
        print(io, "Failure within method:\n")
        println(io)
        pretty_print_mi(ece.mi, io)
        println(io)
        println(io)

        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": catch this exception as `err` and call `code_typed(err)` to inspect the surrounding code.\n";
            color = :cyan,
        )
    end
    println(io)
    msg = if ece.msg isa Cstring
        Base.unsafe_string(ece.msg)
    else
        ece.msg
    end
    print(io, msg, '\n')
end

function InteractiveUtils.code_typed(ece::EnzymeRuntimeActivityError; interactive::Bool=false, kwargs...)
    mi = ece.mi
    if mi === nothing
        throw(AssertionError("code_typed(::EnzymeRuntimeActivityError; interactive::Bool=false, kwargs...) not supported for error without mi"))
    end
    code_typed_helper(ece.mi, ece.world; kwargs...)
end

struct EnzymeNoTypeError{MT,WT} <: EnzymeError
    msg::Cstring
    mi::MT
    world::WT
end

function Base.showerror(io::IO, ece::EnzymeNoTypeError)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "EnzymeNoTypeError: Enzyme cannot statically prove the type of a value being differentiated and risks a correctness error if it gets it wrong.\n")
    print(io, " Generally this shouldn't occur as Enzyme records type information from julia, but may be expected if you, for example, copy untyped data.\n")
    print(io, " or alternatively emit very large sized registers that exceed the maximum size of Enzyme's type analysis. If it seems reasonable to differentiate\n")
    print(io, " this code, open an issue! If the cause of the error is too large of a register, you can request Enzyme increase the size (https://enzyme.mit.edu/julia/dev/api/#Enzyme.API.maxtypeoffset!-Tuple{Any})\n")
    print(io, " or depth (https://enzyme.mit.edu/julia/dev/api/#Enzyme.API.maxtypedepth!-Tuple{Any}) of its type analysis.\n");
    print(io, " Alternatively, you can tell Enzyme to take its best guess from context with (https://enzyme.mit.edu/julia/dev/api/#Enzyme.API.looseTypeAnalysis!-Tuple{Any})\n")
    print(io, " All of these settings are global configurations that need to be set immediately after loading Enzyme, before any differentiation occurs.\n")
    print(io, " To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)\n")
    if VERBOSE_ERRORS[]
        msg = Base.unsafe_string(ece.msg)
        print(io, msg, '\n')
    end
    if ece.mi !== nothing
        print(io, " Failure within method: ", ece.mi, "\n")
        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": catch this exception as `err` and call `code_typed(err)` to inspect the erroneous code.\n";
            color = :cyan,
        )
    end
end

function InteractiveUtils.code_typed(ece::EnzymeNoTypeError; interactive::Bool=false, kwargs...)
    mi = ece.mi
    if mi === nothing
        throw(AssertionError("code_typed(::EnzymeNoTypeError; interactive::Bool=false, kwargs...) not supported for error without mi"))
    end
    code_typed_helper(ece.mi, ece.world; kwargs...)
end

struct EnzymeNoShadowError <: EnzymeError
    msg::Cstring
end

function Base.showerror(io::IO, ece::EnzymeNoShadowError)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    print(io, "EnzymeNoShadowError: Enzyme could not find shadow for value\n")
    msg = Base.unsafe_string(ece.msg)
    print(io, msg, '\n')
end

struct EnzymeNoDerivativeError{MT,WT} <: EnzymeError
    msg::Cstring
    mi::MT
    world::WT
end

function InteractiveUtils.code_typed(ece::EnzymeNoDerivativeError; interactive::Bool=false, kwargs...)
    mi = ece.mi
    if mi === nothing
        throw(AssertionError("code_typed(::EnzymeNoDerivativeError; interactive::Bool=false, kwargs...) not supported for error without mi"))
    end
    code_typed_helper(ece.mi, ece.world; kwargs...)
end

function Base.showerror(io::IO, ece::EnzymeNoDerivativeError)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    msg = Base.unsafe_string(ece.msg)
    print(io, "EnzymeNoDerivativeError: ", msg, '\n')

    if ece.mi !== nothing
        print(io, "Failure within method:\n")
        println(io)
        pretty_print_mi(ece.mi, io)
        println(io)
        println(io)

        printstyled(io, "Hint"; bold = true, color = :cyan)
        printstyled(
            io,
            ": catch this exception as `err` and call `code_typed(err)` to inspect the surrounding code.\n";
            color = :cyan,
        )
    end
end

parent_scope(val::LLVM.Function, depth = 0) = depth == 0 ? LLVM.parent(val) : val
parent_scope(val::LLVM.Module, depth = 0) = val
parent_scope(@nospecialize(val::LLVM.Value), depth = 0) = parent_scope(LLVM.parent(val), depth + 1)
parent_scope(val::LLVM.Argument, depth = 0) =
    parent_scope(LLVM.Function(LLVM.API.LLVMGetParamParent(val)), depth + 1)

function julia_error(
    cstr::Cstring,
    val::LLVM.API.LLVMValueRef,
    errtype::API.ErrorType,
    data::Ptr{Cvoid},
    data2::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
)::LLVM.API.LLVMValueRef
    msg = Base.unsafe_string(cstr)
    julia_error(msg, val, errtype, data, data2, B)
end

function julia_error(
    msg::String,
    val::LLVM.API.LLVMValueRef,
    errtype::API.ErrorType,
    data::Ptr{Cvoid},
    data2::LLVM.API.LLVMValueRef,
    B::LLVM.API.LLVMBuilderRef,
)::LLVM.API.LLVMValueRef
    bt = nothing
    ir = nothing
    if val != C_NULL
        val = LLVM.Value(val)
        if isa(val, LLVM.Instruction)
            dbgval = val
            while !haskey(metadata(dbgval), LLVM.MD_dbg)
                dbgval = LLVM.API.LLVMGetNextInstruction(dbgval)
                if dbgval == C_NULL
                    dbgval = nothing
                    break
                else
                    dbgval = LLVM.Instruction(dbgval)
                end
            end
            if dbgval !== nothing
                bt = GPUCompiler.backtrace(dbgval)
            end
        end
        if isa(val, LLVM.ConstantExpr)
            for u in LLVM.uses(val)
                u = LLVM.user(u)
                if isa(u, LLVM.Instruction)
                    bt = GPUCompiler.backtrace(val)
                end
            end
        elseif val isa LLVM.Function
            # Need to convert function to string, since when the error is going to be printed
            # the module might have been destroyed
            ir = string(val)
        elseif val isa LLVM.GlobalVariable
            # Need to convert global to string, since when the error is going to be printed
            # the module might have been destroyed
            ir = string(val)
        else
            # Need to convert function to string, since when the error is going to be printed
            # the module might have been destroyed
            ir = string(parent_scope(val)::LLVM.Function)
        end
    end

    if errtype == API.ET_NoDerivative
        if occursin("No create nofree of empty function", msg) ||
           occursin("No forward mode derivative found for", msg) ||
           occursin("No augmented forward pass", msg) ||
           occursin("No reverse pass found", msg) ||
           occursin("Runtime Activity not yet implemented for Forward-Mode BLAS", msg)
            ir = nothing
        end
        if B != C_NULL
            B = IRBuilder(B)
            msg2 = sprint() do io
                if ir !== nothing
                    print(io, "Current scope: \n")
                    print(io, ir)
                end
                print(io, '\n', msg, '\n')
                if bt !== nothing
                    Base.show_backtrace(io, bt)
                    println(io)
                end
            end
            if data2 != C_NULL
                        data2 = LLVM.Value(data2)
                if value_type(data2) != LLVM.IntType(1)
                    data2 = nothing
                end
            else
                data2 = nothing
            end

            mi = nothing
            world = nothing

            if isa(val, LLVM.Instruction)
                f = LLVM.parent(LLVM.parent(val))::LLVM.Function
                mi, rt = enzyme_custom_extract_mi(
                    f,
                    false,
                ) #=error=#
                world = enzyme_extract_world(f)
            elseif isa(val, LLVM.Argument)
                f = parent_scope(val)::LLVM.Function
                mi, rt = enzyme_custom_extract_mi(
                    f,
                    false,
                ) #=error=#
                world = enzyme_extract_world(f)
            end
            if mi !== nothing
                emit_error(B, nothing, (msg2, mi, world), EnzymeNoDerivativeError{Core.MethodInstance, UInt}, data2)
            else
                emit_error(B, nothing, msg2, EnzymeNoDerivativeError{Nothing, Nothing}, data2)
            end

            return C_NULL
        end
        throw(NoDerivativeException(msg, ir, bt))
    elseif errtype == API.ET_NoShadow
        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))

        msgN = sprint() do io::IO
            if isa(val, LLVM.Argument)
                fn = parent_scope(val)::LLVM.Function
                ir = string(LLVM.name(fn)) * string(function_type(fn))
                print(io, "Current scope: \n")
                print(io, ir)
            end
            legal, obj = absint(val)
            if legal
                obj0 = obj
                obj = unbind(obj)
                println(io, "\nValue of type: ", Core.Typeof(obj))
                println(io ,  " of value    : ", obj)
                if obj0 isa Core.Binding
                println(io ,  " binding     : ", obj0)
                end
                println(io)
            end
            if !isa(val, LLVM.Argument) && !isa(val, LLVM.GlobalVariable)
                print(io, "\n Inverted pointers: \n")
                ip = API.EnzymeGradientUtilsInvertedPointersToString(gutils)
                sval = Base.unsafe_string(ip)
                write(io, sval)
                API.EnzymeStringFree(ip)
            end
            print(io, '\n', msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(IRBuilder(B), nothing, msgN, EnzymeNoShadowError)
        return LLVM.null(get_shadow_type(gutils, value_type(val))).ref
    elseif errtype == API.ET_IllegalTypeAnalysis
        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)

        mi = nothing
        world = nothing

        if isa(val, LLVM.Instruction)
            f = LLVM.parent(LLVM.parent(val))::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(
                f,
                false,
            ) #=error=#
            world = enzyme_extract_world(f)
        elseif isa(val, LLVM.Argument)
            f = parent_scope(val)::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(
                f,
                false,
            ) #=error=#
            world = enzyme_extract_world(f)
        end
        throw(IllegalTypeAnalysisException(msg, mi, world, sval, ir, bt))
    elseif errtype == API.ET_NoType
        @assert B != C_NULL
        B = IRBuilder(B)

        data = API.EnzymeTypeAnalyzerRef(data)
        ip = API.EnzymeTypeAnalyzerToString(data)
        sval = Base.unsafe_string(ip)
        API.EnzymeStringFree(ip)

        msg2 = sprint() do io::IO
            if !occursin("Cannot deduce single type of store", msg)
                if ir !== nothing
                    print(io, "Current scope: \n")
                    print(io, ir)
                end
                print(io, "\n Type analysis state: \n")
                write(io, sval)
            end
            print(io, '\n', msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
            pscope = parent_scope(val)::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(pscope, false) #=error=#
            if mi !== nothing
                println(io, "within ", mi)
            end
        end

        mi = nothing
        world = nothing

        if isa(val, LLVM.Instruction)
            f = LLVM.parent(LLVM.parent(val))::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(
                f,
                false,
            ) #=error=#
            world = enzyme_extract_world(f)
        elseif isa(val, LLVM.Argument)
            f = parent_scope(val)::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(
                f,
                false,
            ) #=error=#
            world = enzyme_extract_world(f)
        end
        if mi !== nothing
            emit_error(B, nothing, (msg2, mi, world), EnzymeNoTypeError{Core.MethodInstance, UInt})
        else
            emit_error(B, nothing, msg2, EnzymeNoTypeError{Nothing, Nothing})
        end
        return C_NULL
    elseif errtype == API.ET_IllegalFirstPointer
        throw(IllegalFirstPointerException(msg, ir, bt))
    elseif errtype == API.ET_InternalError
        throw(EnzymeInternalError(msg, ir, bt))
    elseif errtype == API.ET_GCRewrite
        data2 = LLVM.Value(data2)
        fn = LLVM.Function(LLVM.API.LLVMGetParamParent(data2::LLVM.Argument))
        @static if VERSION < v"1.11"
            sretkind = LLVM.kind(if LLVM.version().major >= 12
                LLVM.TypeAttribute("sret", LLVM.Int32Type())
            else
                LLVM.EnumAttribute("sret")
            end)
            if occursin("Could not find use of stored value", msg) && length(parameters(fn)) >= 1 && any(LLVM.kind(attr) == sretkind for attr in collect(LLVM.parameter_attributes(fn, 1)))
                return C_NULL
            end
        end
        msgN = sprint() do io::IO
            print(io, msg)
            println(io)
            println(io, "Fn = ", string(fn))
            println(io, "val = ", string(val))
            println(io, "arg = ", string(data2::LLVM.Argument))
            if data !== C_NULL
                data = LLVM.Value(LLVM.API.LLVMValueRef(data))
                println(io, "cur = ", string(data))
            end
        end
        GPUCompiler.@safe_warn msgN
        return C_NULL
    elseif errtype == API.ET_TypeDepthExceeded
        msg2 = sprint() do io
            print(io, msg)
            println(io)

            if val != C_NULL
                println(io, val)
            end

            st = API.EnzymeTypeTreeToString(data)
            println(io, Base.unsafe_string(st))
            API.EnzymeStringFree(st)

            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end
        GPUCompiler.@safe_warn msg2
        return C_NULL
    elseif errtype == API.ET_IllegalReplaceFicticiousPHIs
        data2 = LLVM.Value(data2)
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            println(io, string(LLVM.parent(LLVM.parent(data2))))
            println(io, val)
            println(io, data2)
        end
        throw(EnzymeInternalError(msg2, ir, bt))
    elseif errtype == API.ET_MixedActivityError
        data2 = LLVM.Value(data2)
        badval = nothing
        gutils = GradientUtils(API.EnzymeGradientUtilsRef(data))
        # Ignore mismatched activity if phi/store of ghost
        seen = Dict{LLVM.Value,LLVM.Value}()
        illegal = false
        created = LLVM.Instruction[]
        world = enzyme_extract_world(LLVM.parent(position(IRBuilder(B))))
        width = get_width(gutils)
        function make_batched(@nospecialize(cur::LLVM.Value), B::LLVM.IRBuilder)::LLVM.Value
            if width == 1
                return cur
            else
                shadowres = UndefValue(
                    LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur))),
                )
                for idx = 1:width
                    shadowres = insert_value!(B, shadowres, cur, idx - 1)
                    if isa(shadowres, LLVM.Instruction)
                        push!(created, shadowres)
                    end
                end
                return shadowres
            end
        end

        illegalVal = nothing
        mode = get_mode(gutils)

        function make_replacement(@nospecialize(cur::LLVM.Value), prevbb::LLVM.IRBuilder)::LLVM.Value
            ncur = new_from_original(gutils, cur)
            if cur in keys(seen)
                return seen[cur]
            end

                if isa(cur, LLVM.LoadInst)
                    larg, off = get_base_and_offset(operands(cur)[1])
                    if off == 0 && isa(larg, LLVM.AllocaInst)
                         legal = true
                         for u in LLVM.uses(larg)
                            u = LLVM.user(u)
                            if isa(u, LLVM.LoadInst)
                                continue
                            end
                            if isa(u, LLVM.CallInst) && isa(called_operand(u), LLVM.Function)
                               intr = LLVM.API.LLVMGetIntrinsicID(LLVM.called_operand(u))
                               if intr == LLVM.Intrinsic("llvm.lifetime.start").id || intr == LLVM.Intrinsic("llvm.lifetime.end").id || LLVM.name(called_operand(u)) == "llvm.enzyme.lifetime_end" || LLVM.name(called_operand(u)) ==
 "llvm.enzyme.lifetime_start"
                                    continue
                               end
                            end
                            if isa(u, LLVM.StoreInst)
                                 v = operands(u)[1]
                                 if v == larg
                                    legal = false;
                                    break
                                 end
                                 if v isa ConstantInt && convert(Int, v) == -1
                                    continue
                                 end
                            end
                            legal = false
                            break
                         end
                         if legal
                            return make_batched(ncur, prevbb)
                         end
                    end
                end

            legal, TT, byref = abs_typeof(cur, true)

            if legal
                if guaranteed_const_nongen(TT, world)
                    return make_batched(ncur, prevbb)
                end

                legal2, obj = absint(cur)
                obj0 = obj
                # Only do so for the immediate operand/etc to a phi, since otherwise we will make multiple
                if legal2
                   obj = unbind(obj)
                   if is_memory_instance(obj) || (obj isa Core.SimpleVector && length(obj) == 0)
                        return make_batched(ncur, prevbb)
                   end
                   if active_reg(TT, world) == ActiveState &&
                     ( isa(cur, LLVM.ConstantExpr) || isa(cur, LLVM.GlobalVariable)) &&
                   cur == data2
                    if width == 1
                        if mode == API.DEM_ForwardMode
                            instance = make_zero(obj)
                            return unsafe_to_llvm(prevbb, instance)
                        else
                            res = emit_allocobj!(prevbb, Base.RefValue{TT})
                            T_int8 = LLVM.Int8Type()
                            T_size_t = convert(LLVM.LLVMType, UInt)
                            LLVM.memset!(prevbb, bitcast!(prevbb, res, LLVM.PointerType(T_int8, 10)),  LLVM.ConstantInt(T_int8, 0), LLVM.ConstantInt(T_size_t, sizeof(TT)), 0)
                            push!(created, res)
                            return res
                        end
                    else
                        shadowres = UndefValue(
                            LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur))),
                        )
                        for idx = 1:width
                            res = if mode == API.DEM_ForwardMode
                                instance = make_zero(obj)
                                unsafe_to_llvm(prevbb, instance)
                            else
                                sres = emit_allocobj!(prevbb, Base.RefValue{TT})
                                T_int8 = LLVM.Int8Type()
                                T_size_t = convert(LLVM.LLVMType, UInt)
                                LLVM.memset!(prevbb, bitcast!(prevbb, sres, LLVM.PointerType(T_int8, 10)),  LLVM.ConstantInt(T_int8, 0), LLVM.ConstantInt(T_size_t, sizeof(TT)), 0)
                                push!(created, sres)
                                sres
                            end
                            shadowres = insert_value!(prevbb, shadowres, res, idx - 1)
                            if shadowres isa LLVM.Instruction
                                push!(created, shadowres)
                            end
                        end
                        return shadowres
                    end
                    end

                end

@static if VERSION < v"1.11-"
else
                if isa(cur, LLVM.LoadInst)
                    larg, off = get_base_and_offset(operands(cur)[1])
                    if isa(larg, LLVM.LoadInst)
                        legal2, obj = absint(larg)
                        obj = unbind(obj)
                        if legal2 && is_memory_instance(obj)
                            return make_batched(ncur, prevbb)
                        end
                    end
                end
end

                badval = if legal2
                    sv = string(obj) * " of type" * " " * string(TT)
                    if obj0 isa Core.Binding
                        sv = sv *" binded at "*string(obj0)
                    end
                    sv
                else
                    "Unknown object of type" * " " * string(TT)
                end
                @assert !illegal
                illegalVal = cur
                illegal = true
                return make_batched(ncur, prevbb)
            end

            if isa(cur, LLVM.PointerNull)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.UndefValue)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.PoisonValue)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantAggregateZero)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantAggregate)
                return make_batched(ncur, prevbb)
            end
            if isa(cur, LLVM.ConstantInt)
                if convert(UInt64, cur) == 0
                    return make_batched(ncur, prevbb)
                end
            end
            if isa(cur, LLVM.ConstantFP)
                return make_batched(ConstantFP(value_type(cur), 0), prevbb)
            end
            if isa(cur, LLVM.ConstantDataSequential)
                cvals = LLVM.Value[]
                changed = false
                for v in collect(cur)
                    tmp = make_replacement(v, prevbb)
                    if illegal
                        return ncur
                    end
                    if v != tmp
                        changed = true
                    end
                    push!(cvals, tmp)
                end

                cur2 = if changed
                    @assert !illegal
                    illegalVal = cur
                    illegal = true
                    # TODO replace with correct insertions/splats
                    ncur
                else
                    make_batched(ncur, prevbb)
                end
                return cur2
            end
            if isa(cur, LLVM.ConstantInt)
                if LLVM.width(value_type(cur)) <= sizeof(Int) * 8
                    return make_batched(ncur, prevbb)
                end
                if LLVM.width(value_type(cur)) == sizeof(Int) * 8 &&
                   abs(convert(Int, cur)) < 10000
                    return make_batched(ncur, prevbb)
                end
                # if storing a constant int as a non-pointer, presume it is not a GC'd var and is safe
                # for activity state to mix
                if isa(val, LLVM.StoreInst)
                    operands(val)[1] == cur &&
                        !isa(value_type(operands(val)[1]), LLVM.PointerType)
                    return make_batched(ncur, prevbb)
                end
            end

            if isa(cur, LLVM.SelectInst)
                lhs = make_replacement(operands(cur)[2], prevbb)
                if illegal
                    return ncur
                end
                rhs = make_replacement(operands(cur)[3], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(cur)[2] && rhs == operands(cur)[3]
                    return make_batched(ncur, prevbb)
                end
                if width == 1
                    nv = select!(
                        prevbb,
                        new_from_original(gutils, operands(cur)[1]),
                        lhs,
                        rhs,
                    )
                    push!(created, nv)
                    seen[cur] = nv
                    return nv
                else
                    shadowres = LLVM.UndefValue(value_type(lhs))
                    for idx = 1:width
                        shadowres = insert_value!(
                            prevbb,
                            shadowres,
                            select!(
                                prevbb,
                                new_from_original(gutils, operands(cur)[1]),
                                extract_value!(prevbb, lhs, idx - 1),
                                extract_value!(prevbb, rhs, idx - 1),
                            ),
                            idx - 1,
                        )
                        if isa(shadowres, LLVM.Instruction)
                            push!(created, shadowres)
                        end
                    end
                    return shadowres
                end
            end

            if isa(cur, LLVM.InsertValueInst)
                B2 = IRBuilder()
                position!(B2, LLVM.Instruction(LLVM.API.LLVMGetNextInstruction(ncur)))

                lhs = make_replacement(operands(cur)[1], B2)
                if illegal
                    return ncur
                end
                rhs = make_replacement(operands(cur)[2], B2)
                if illegal
                    return ncur
                end
                if lhs == operands(cur)[1] && rhs == operands(cur)[2]
                    return make_batched(ncur, cur)
                end
                inds = LLVM.API.LLVMGetIndices(cur.ref)
                ninds = LLVM.API.LLVMGetNumIndices(cur.ref)
                jinds = Cuint[unsafe_load(inds, i) for i = 1:ninds]
                if width == 1
                    nv = API.EnzymeInsertValue(B2, lhs, rhs, jinds)
                    push!(created, nv)
                    seen[cur] = nv
                    return nv
                else
                    shadowres = lhs
                    for idx = 1:width
                        jindsv = copy(jinds)
                        pushfirst!(jindsv, idx - 1)
                        shadowres = API.EnzymeInsertValue(
                            B2,
                            shadowres,
                            extract_value!(B2, rhs, idx - 1),
                            jindsv,
                        )
                        if isa(shadowres, LLVM.Instruction)
                            push!(created, shadowres)
                        end
                    end
                    return shadowres
                end
            end

            if isa(cur, LLVM.LoadInst) || isa(cur, LLVM.BitCastInst) || isa(cur, LLVM.AddrSpaceCastInst) || (isa(cur, LLVM.GetElementPtrInst) && all(Base.Fix2(isa, LLVM.ConstantInt), operands(cur)[2:end])) || (isa(cur,LLVM.ConstantExpr) &&  opcode(cur) in (LLVM.API.LLVMBitCast, LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMGetElementPtr))
                lhs = make_replacement(operands(cur)[1], prevbb)
                if illegal
                    return ncur
                end
                if lhs == operands(ncur)[1]
                    return make_batched(ncur, prevbb)
                elseif width != 1 && isa(lhs, LLVM.InsertValueInst) && operands(lhs)[2] == operands(ncur)[1]
                    return make_batched(ncur, prevbb)
                end
            end

            if isa(cur, LLVM.PHIInst)
                Bphi = IRBuilder()
                position!(Bphi, ncur)
                shadowty = LLVM.LLVMType(API.EnzymeGetShadowType(width, value_type(cur)))
                phi2 = phi!(Bphi, shadowty, "tempphi" * LLVM.name(cur))
                seen[cur] = phi2
                changed = false
                recsize = length(created) + 1
                for (v, bb) in LLVM.incoming(cur)
                    B2 = IRBuilder()
                    position!(B2, new_from_original(gutils, last(instructions(bb))))
                    tmp = make_replacement(v, B2)
                    if illegal
                        changed = true
                        break
                    end
                    @assert value_type(tmp) == shadowty
                    if tmp != new_from_original(gutils, v) && v != cur
                        changed = true
                    end
                    push!(LLVM.incoming(phi2), (tmp, new_from_original(gutils, bb)))
                end
                if !changed || illegal
                    LLVM.API.LLVMInstructionEraseFromParent(phi2)
                    seen[cur] = ncur
                    plen = length(created)
                    for i = recsize:plen
                        u = created[i]
                        replace_uses!(u, LLVM.UndefValue(value_type(u)))
                    end
                    for i = recsize:plen
                        u = created[i]
                        LLVM.API.LLVMInstructionEraseFromParent(u)
                    end
                    for i = recsize:plen
                        pop!(created)
                    end
                    return illegal ? ncur : make_batched(ncur, prevbb)
                end
                push!(created, phi2)
                return phi2
            end

            tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, cur))
            st = API.EnzymeTypeTreeToString(tt)
            st2 = Base.unsafe_string(st)
            API.EnzymeStringFree(st)
            if st2 == "{[-1]:Integer}"
                return make_batched(ncur, prevbb)
            end

            if !illegal
                illegal = true
                illegalVal = cur
            end
            return ncur
        end

        b = IRBuilder(B)
        replacement = make_replacement(data2, b)

        if !illegal
            return replacement.ref
        end
        for u in created
            replace_uses!(u, LLVM.UndefValue(value_type(u)))
        end
        for u in created
            LLVM.API.LLVMInstructionEraseFromParent(u)
        end
        if LLVM.API.LLVMIsAReturnInst(val) != C_NULL
            mi, rt = enzyme_custom_extract_mi(
                LLVM.parent(LLVM.parent(val))::LLVM.Function,
                false,
            ) #=error=#
            if mi !== nothing && isghostty(rt)
                return C_NULL
            end
        end
        msg2 = sprint() do io
            print(io, msg)
            println(io)
            if badval !== nothing
                println(io, " Julia value causing error:  " * badval)
            else
                ttval = val
                if isa(ttval, LLVM.StoreInst)
                    ttval = operands(ttval)[1]
                end
                tt = TypeTree(API.EnzymeGradientUtilsAllocAndGetTypeTree(gutils, ttval))
                st = API.EnzymeTypeTreeToString(tt)
                print(io, "Type tree: ")
                println(io, Base.unsafe_string(st))
                API.EnzymeStringFree(st)
            end
            if illegalVal !== nothing
                println(io, " LLVM view of erring value:   " * string(illegalVal))
            end
            if bt !== nothing
                Base.show_backtrace(io, bt)
            end
        end

            mi = nothing
        world = nothing

        if isa(val, LLVM.Instruction)
            f = LLVM.parent(LLVM.parent(val))::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(
                f,
                false,
            ) #=error=#
            world = enzyme_extract_world(f)
        elseif isa(val, LLVM.Argument)
            f = parent_scope(val)::LLVM.Function
            mi, rt = enzyme_custom_extract_mi(
                f,
                false,
            ) #=error=#
            world = enzyme_extract_world(f)
        end
        mode = Enzyme.API.DEM_ReverseModeCombined

        if mi !== nothing
            emit_error(b, nothing, (msg2, mi, world), EnzymeRuntimeActivityError{Cstring, Core.MethodInstance, UInt})
        else
            emit_error(b, nothing, msg2, EnzymeRuntimeActivityError{Cstring, Nothing, Nothing})
        end
        return C_NULL
    elseif errtype == API.ET_GetIndexError
        @assert B != C_NULL
        B = IRBuilder(B)
        msg5 = sprint() do io::IO
            print(io, "Enzyme internal error\n")
            print(io, msg, '\n')
            if bt !== nothing
                print(io, "\nCaused by:")
                Base.show_backtrace(io, bt)
                println(io)
            end
        end
        emit_error(B, nothing, msg5)
        return C_NULL
    end
    throw(AssertionError("Unknown errtype"))
end

struct EnzymeNonScalarReturnException <: EnzymeError
    object
    extra::String
end

function Base.showerror(io::IO, ece::EnzymeNonScalarReturnException)
    if isdefined(Base.Experimental, :show_error_hints)
        Base.Experimental.show_error_hints(io, ece)
    end
    if Enzyme.Compiler.guaranteed_const(Core.Typeof(ece.object))
        println(io, "EnzymeNonScalarReturnException: Return type of active-returning differentiated function was not differentiable, found ", ece.object, " of type ", Core.Typeof(ece.object))
        println(io, "Either rewrite the autodiff call to return Const, or the function being differentiated to return an active type")
    else
        println(io, "EnzymeNonScalarReturnException: Return type of differentiated function was not a scalar as required, found ", ece.object, " of type ", Core.Typeof(ece.object))
        println(io, "If calling Enzyme.autodiff(Reverse, f, Active, ...), try Enzyme.autodiff_thunk(Reverse, f, Duplicated, ....)")
        println(io, "If calling Enzyme.gradient, try Enzyme.jacobian")
    end
    if length(ece.extra) != 0
        print(io, ece.extra)
    end
end

struct ThunkCallError <: Exception
    thunk::Type
    fn::Type
    args::(NTuple{N, Type} where N)
    correct::Type
    hint::String
end

function Base.showerror(io::IO, e::ThunkCallError)
    print(io, "ThunkCallError:\n")
    print(io, "  No method matching:\n    ")
    Base.show_signature_function(io, e.thunk)
    Base.show_tuple_as_call(io, :var"", Tuple{e.fn, e.args...}, hasfirst=false)
    println(io)
    println(io)
    print(io, "  Expected:\n    ")
    Base.show_signature_function(io, e.thunk)
    Base.show_tuple_as_call(io, :function, e.correct; hasfirst=false, kwargs=nothing)
    println(io)
    println(io)

    printstyled(io, "Hint"; bold = true, color = :cyan)
    printstyled(
        io,
        ": " * e.hint * "\n",
        color = :cyan,
    )
end
