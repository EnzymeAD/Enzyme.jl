############################################################################################
### @easy_rule

function has_easy_rule end

function has_easy_rule_from_sig(@nospecialize(TT);
                              world::UInt=Base.get_world_counter(),
                              method_table::Union{Nothing,Core.Compiler.MethodTableView}=nothing,
                              caller::Union{Nothing,Core.MethodInstance,Core.Compiler.MethodLookupResult}=nothing)
    return isapplicable(has_easy_rule, TT; world, method_table, caller)
end


# Note: must be declared before it is used, which is later in this file.
macro strip_linenos(expr)
    return esc(Base.remove_linenums!(expr))
end


"""
    uses_symbol(a, b::Symbol)

Internal function.

Checks if `a` contains a use of the symbol `b`.
"""
function uses_symbol(a, b::Symbol)
    return false
end

function uses_symbol(a::Expr, b::Symbol)
    for arg in a.args
        if uses_symbol(arg, b)
            return true
        end
    end
    return false
end

function uses_symbol(a::Symbol, b::Symbol)
    return a == b
end


"""
    _unconstrain(a)

Internal function.

Turn both `a` and `a::S` into `a`
"""
_unconstrain(arg::Symbol) = arg
function _unconstrain(arg::Expr)
    Meta.isexpr(arg, :(::), 2) && return arg.args[1]  # drop constraint.
    Meta.isexpr(arg, :(...), 1) && return _unconstrain(arg.args[1])
    return error("malformed arguments: $arg")
end


"""
    _constrain_and_name(arg::Expr, _)

Internal function.

Turn both `a` and `::constraint` into `a::Annotation{<:constraint}` etc
"""
function _constrain_and_name(arg::Expr)
    Meta.isexpr(arg, :(::), 2) && return Expr(:(::), Symbol("ann_", arg.args[1]), :(Annotation{<:$(arg.args[2])})) # it is already fine.
    Meta.isexpr(arg, :(::), 1) && return Expr(:(::), Symbol("ann_", gensym()), :(Annotation{<:$(arg.args[1])})) # add name
    # Meta.isexpr(arg, :(...), 1) &&
    #     return Expr(:(...), _constrain_and_name(arg.args[1], :Annotation))
    return error("malformed arguments: $arg")
end
_constrain_and_name(name::Symbol) = Expr(:(::), Symbol("ann_", name), :Annotation)


"""
    _just_name(arg::Expr, _)

Internal function.

Extract `a` from `a::constraint`.
"""
function _just_name(arg::Expr)
    @assert Meta.isexpr(arg, :(::), 2)
    return arg.args[1]
end

"""
    multiply_fwd_into(prev, partial, dx)

Internal function.

Multiply a partial derivative (df/dx) by its shadow input (dx) to form `df`.

Specifically, perform prev + partial * dx, returning the result or re-using prev's memory, where applicable.
"""
function multiply_fwd_into end

"""
    multiply_rev_into(prev, partial, df)

Internal function.

Multiply a partial derivative (df/dx) by its shadow input (df) to form `dx`.

Specifically, perform prev + conj(partial * conj(dx)), returning the result or re-using prev's memory, where applicable.
"""
function multiply_rev_into end

"""
    _normalize_scalarrules_macro_input(call, maybe_setup, partials)

Internal function.

returns (in order) the correctly escaped:
- `call` with out any type constraints
- `setup_stmts`: the content of `@setup` or `[]` if that is not provided,
-  `inputs`: with all args having the constraints removed from call, or
    defaulting to `Number`
- `partials`: which are all `Expr{:tuple,...}`
"""
function _normalize_scalarrules_macro_input(call, maybe_setup, partials)
    # Setup: normalizing input form etc

    if Meta.isexpr(maybe_setup, :macrocall) && maybe_setup.args[1] == Symbol("@setup")
        setup_stmts = Any[ex for ex in maybe_setup.args[3:end]]
    else
        setup_stmts = []
        partials = (maybe_setup, partials...)
    end
    @assert Meta.isexpr(call, :call)

    # Annotate all arguments in the signature as scalars
    normal_inputs = call.args[2:end]
    pre_inputs = _constrain_and_name.(normal_inputs)
    input_names = _just_name.(pre_inputs)
    inputs = esc.(pre_inputs)
    # Remove annotations and escape names for the call
    call.args[2:end] .= _unconstrain.(call.args[2:end])
    call.args = esc.(call.args)

    # For consistency in code that follows we make all partials tuple expressions
    partials = map(partials) do partial
        @assert Meta.isexpr(partial, :tuple)
        partial.args
    end

    return call, setup_stmts, inputs, input_names, normal_inputs, partials
end

function scalar_frule_expr(__source__, f, call, setup_stmts, inputs, input_names, partials)

    call2 = Expr(:call, esc(:f), call.args[2:end]...)

    exprs = Expr[]

    arg_names = Symbol[]
    for sname in input_names
        rname = Symbol(String(sname)[length("ann_")+1:end])
        push!(arg_names, rname)
        push!(exprs, Expr(:(=), rname, :($sname.val)))
    end

    tosum0 = Vector{Tuple{Int, Symbol, Any}}[]

    for (o, partial0) in enumerate(partials)
        @assert partial0 isa Array && length(partial0) == length(inputs)
        tosum = Tuple{Int, Symbol, Any}[]
        push!(tosum0, tosum)
        for (i, (p, sname)) in enumerate(zip(partial0, input_names))
            if Meta.isexpr(p, :macrocall) && p.args[1] == Symbol("@Constant")
                continue
            end
            push!(tosum, (i , sname, p))
        end
    end

    actives = Expr[]
    for ann_name in input_names
        push!(actives, Expr(:if, Expr(:call, <:, esc(ann_name), EnzymeCore.Const), nothing, :(Expr(:(.), Symbol($(String(ann_name))), :(:dval)))))
    end

    N = length(inputs)

    return @strip_linenos quote
        # _ is the input derivative w.r.t. function internals. since we do not
        # allow closures/functors with @easy_rule, it is always ignored
        @generated function EnzymeCore.EnzymeRules.forward($(esc(:config)), $(esc(:fn))::Const{<:$(Core.Typeof)($f)}, ::Type{<:Annotation{$(esc(:RT))}}, $(inputs...)) where $(esc(:RT))
            genexprs = Expr[$(exprs...,)...]
            gensetup = Expr[$(setup_stmts...,)...]

            has_omega = needs_primal(config)
            for expr in gensetup                
                if uses_symbol(expr, :Ω)
                    has_omega = true
                end
            end
            
            tosum0 = $tosum0

            N = $N
            W = width(config)

            actives = Union{Nothing, Expr}[$(actives...)]

            if needs_shadow(config)
                outsyms = Matrix{Symbol}(undef, length(tosum0), W)
                visited = zeros(Bool, length(tosum0), N)
                for (o, tosum) in enumerate(tosum0)
                    for w in 1:W
                        outexpr = Symbol("outsym_", string(o), "_", string(w))
                        outsyms[o, w] = outexpr
                        seen = false
                        for (i, sname, p) in tosum
                            if actives[i] isa Nothing
                                continue
                            end

                            dval = actives[i]
                            if W != 1
                                dval = Expr(:call, getfield, dval, w)
                            end

                            pname = Symbol("partial_", string(o), "_", string(i), "_",  sname)
                            if !visited[o, i]

                                # Descend through the rule to see if any users require the original result, Ω
                                # for now, conservatively assume this is indeed the case
                                if uses_symbol(p, :Ω)
                                    has_omega = true
                                end

                                push!(gensetup, Expr(:(=), pname, p))
                                visited[o, i] = true
                            end

                            if !seen
                                push!(gensetup, Expr(:(=), outexpr, nothing))
                                seen = true
                            end

                            push!(gensetup, Expr(:(=), outexpr, Expr(:call, EnzymeCore.EnzymeRules.multiply_fwd_into, outexpr, pname, dval)))
                        end

                        if !seen
                            KT = $(esc(:RT))
                            inp = :Ω
                            if KT <: Tuple
                                KT = KT.parameters[o]
                                inp = :(Ω[$o])
                            end
                            if KT <: AbstractFloat
                                push!(gensetup, Expr(:(=), outexpr, Expr(:call, Base.zero, KT)))
                            else
                                has_omega = true
                                push!(gensetup, Expr(:(=), outexpr, Expr(:call, EnzymeCore.make_zero, inp)))
                            end
                        end
                    end
                end
                @assert length(outsyms) > 0

                outres = Vector{Symbol}(undef, W)
                for w in 1:W
                    outexpr = Symbol("outres_$w")
                    outres[w] = outexpr
                    if $(esc(:RT)) <: Tuple                    
                        push!(gensetup, Expr(:(=), outexpr, Expr(:tuple, outsyms[:, w]...)))
                    else
                        @assert length(tosum0) == 1
                        push!(gensetup, Expr(:(=), outexpr, outsyms[1, w]))
                    end
                end

                if W == 1
                    push!(gensetup, Expr(:(=), :dΩ, outres[1]))
                else
                    push!(gensetup, Expr(:(=), :dΩ, Expr(:tuple, outres...)))
                end
            end

            genres = if needs_primal(config)
                if needs_shadow(config)
                    if width(config) == 1
                        Expr(:call, Duplicated, :Ω, :dΩ)
                    else
                        Expr(:call, BatchDuplicated, :Ω, :dΩ)
                    end
                else
                    :Ω
                end
            else
                if needs_shadow(config)
                    :dΩ
                else
                    nothing
                end
            end

            if has_omega
                push!(genexprs, Expr(:(=), :Ω, Expr(:call, :f, $arg_names...)))
            end

            return quote
                Base.@_inline_meta
                $($(__source__))
                f = fn.val
                $(genexprs...)
                $(gensetup...)
                return $genres
            end
        end
    end
end

function scalar_rrule_expr(__source__, f, call, setup_stmts, inputs, input_names, partials)

    call2 = Expr(:call, esc(:f), call.args[2:end]...)

    exprs = Expr[]
    revexprs = Expr[]

    ann_names = Symbol[]
    arg_names = Symbol[]
    for (i, sname) in enumerate(input_names)
        rname = Symbol(String(sname)[length("ann_")+1:end])
        push!(ann_names, sname)
        push!(arg_names, rname)
        push!(exprs, Expr(:(=), rname, Expr(:call, getfield, sname, :(:val))))
        push!(revexprs, Expr(:(=), rname,
            Expr(:if,
                Expr(:call, Base.isa, :(cache[($i)]), Nothing),
                Expr(:call, getfield, sname, :(:val)),
                :(cache[($i)])
                )))
    end

    tosum0 = Vector{Tuple{Int, Symbol, Any}}[]

    for (o, partial0) in enumerate(partials)
        @assert partial0 isa Array && length(partial0) == length(inputs)
        tosum = Tuple{Int, Symbol, Any}[]
        push!(tosum0, tosum)
        for (i, (p, sname)) in enumerate(zip(partial0, input_names))
            if Meta.isexpr(p, :macrocall) && p.args[1] == Symbol("@Constant")
                continue
            end
            push!(tosum, (i , sname, p))
        end
    end

    actives = Expr[]
    for ann_name in input_names
        push!(actives, Expr(:if, Expr(:call, <:, esc(ann_name), EnzymeCore.Const), nothing, :(Expr(:(.), Symbol($(String(ann_name))), :(:dval)))))
    end

    N = length(inputs)

    @strip_linenos quote

        # _ is the input derivative w.r.t. function internals. since we do not
        # allow closures/functors with @scalar_rule, it is always ignored
        @generated function EnzymeCore.EnzymeRules.augmented_primal($(esc(:config)), $(esc(:fn))::Const{<:$(Core.Typeof)($f)}, $(esc(:RTA))::Type{<:Annotation{$(esc(:RT))}}, $(inputs...)) where $(esc(:RT))
            genexprs = Expr[$(exprs...,)...]
            gensetup = Expr[]

            has_omega = needs_primal(config)
            
            inp_types = ($(map(esc, ann_names)...),)
            tosum0 = $tosum0

            N = $N
            W = width(config)

            actives = Union{Nothing, Expr}[$(actives...)]

            inp_names = String[($((String.(input_names))...),)...]

            cache_inputs = Vector{Bool}(undef, N)
            fill!(cache_inputs, false)

            if needs_shadow(config)
                outsyms = Matrix{Symbol}(undef, length(tosum0), W)
                for (o, tosum) in enumerate(tosum0)
                    for w in 1:W
                        outexpr = Symbol("outsym_", string(o), "_", string(w))
                        outsyms[o, w] = outexpr

                        KT = $(esc(:RT))
                        inp = :Ω
                        if $(esc(:RT)) <: Tuple
                            KT = KT.parameters[o]
                            inp = :(Ω[$o])
                        end
                        if KT <: AbstractFloat
                            push!(gensetup, Expr(:(=), outexpr, Expr(:call, Base.zero, KT)))
                        else
                            has_omega = true
                            push!(gensetup, Expr(:(=), outexpr, Expr(:call, EnzymeCore.make_zero, inp)))
                        end
                    end
                end

                @assert length(outsyms) > 0

                outres = Vector{Symbol}(undef, W)
                for w in 1:W
                    outexpr = Symbol("outres_$w")
                    outres[w] = outexpr
                    if $(esc(:RT)) <: Tuple                    
                        push!(gensetup, Expr(:(=), outexpr, Expr(:tuple, outsyms[:, w]...)))
                    else
                        @assert length(tosum0) == 1
                        push!(gensetup, Expr(:(=), outexpr, outsyms[1, w]))
                    end
                end

                if W == 1
                    push!(gensetup, Expr(:(=), :dΩ, outres[1]))
                else
                    push!(gensetup, Expr(:(=), :dΩ, Expr(:tuple, outres...)))
                end
            end

            caches = []
            for (inum, sym_name) in enumerate(inp_names)
                if $(esc(:RTA)) <: Const
                    push!(caches, nothing)
                    continue
                end

                if !EnzymeRules.overwritten(config)[inum+1]
                    push!(caches, nothing)
                    continue
                end
                sym = Symbol("cache_$(sym_name)")
                used = nothing
                for (o, tosum) in enumerate(tosum0)
                    for (i, sname, p) in tosum
                        if actives[i] isa Nothing
                            continue
                        end
                        if inp_types[i] <: Const
                            continue
                        end
                        if !uses_symbol(p, Symbol(sym_name))
                            continue
                        end

                        expr = :(!($name <: AbstractFloat || $name <: Integer))

                        if used == nothing
                            used = expr
                        else
                            used = Expr(:|, used, expr)
                        end
                    end
                end

                if used == nothing
                    push!(caches, nothing)
                else
                    push!(caches, Expr(:if,
                        used,
                        Expr(:call, Base.copy, Symbol(sym_name)),
                        nothing
                    ))
                end
            end
            if needs_shadow(config)
                push!(caches, :dΩ)
            end
            push!(gensetup, Expr(:(=), :cache, Expr(:tuple, caches...)))

            genres = if needs_primal(config)
                if needs_shadow(config)
                    if width(config) == 1
                        Expr(:call, EnzymeRules.AugmentedReturn, :Ω, :dΩ, :cache)
                    else
                        Expr(:call, EnzymeRules.AugmentedReturn, :Ω, :dΩ, :cache)
                    end
                else
                    Expr(:call, EnzymeRules.AugmentedReturn, :Ω, nothing, :cache)
                end
            else
                if needs_shadow(config)
                    Expr(:call, EnzymeRules.AugmentedReturn, nothing, :dΩ, :cache)
                else
                    Expr(:call, EnzymeRules.AugmentedReturn, nothing, nothing, :cache)
                end
            end

            if has_omega
                push!(genexprs, Expr(:(=), :Ω, Expr(:call, :f, $arg_names...)))
            end

            quote
                Base.@_inline_meta
                $($(__source__))
                f = fn.val
                $(genexprs...)
                $(gensetup...)
                return $genres
            end
        end

        # _ is the input derivative w.r.t. function internals. since we do not
        # allow closures/functors with @scalar_rule, it is always ignored
        @generated function EnzymeCore.EnzymeRules.reverse($(esc(:config)), $(esc(:fn))::Const{<:$(Core.Typeof)($f)}, $(esc(:RTA)), $(esc(:cache)), $(inputs...))
            genexprs = Expr[$(revexprs...,)...]
            gensetup = Expr[$(setup_stmts...,)...]

            has_omega = false
            for expr in gensetup                
                if uses_symbol(expr, :Ω)
                    has_omega = true
                end
            end

            inp_types = ($(map(esc, ann_names)...),)

            inp_names = String[($((String.(input_names))...),)...]

            tosum0 = $tosum0

            N = $N
            W = width(config)

            RT = if RTA <: Active
                eltype(RTA)
            elseif (RTA <: Tuple && RTA.parameters[1] <: Active)
                eltype(RTA.parameters[1])
            else
                eltype(RTA)
            end

            if RTA <: Active || (RTA <: Tuple && RTA.parameters[1] <: Active)
                if W == 1
                    push!(genexprs, Expr(:(=), :dΩ, Expr(:call, getfield, Symbol("RTA"), 1)))
                else
                    syms = Expr[]
                    for w in 1:W
                        push!(syms, Expr(:call, getfield, Expr(:call, getfield, Symbol("RTA"), w), 1))
                    end
                    push!(genexprs, Expr(:(=), :dΩ, Expr(:tuple, syms...)))
                end
                #if eltype(RTA) <: Complex
                #    push!(genexprs, Expr(:(=), :dΩ, Expr(:call, Base.conj, :dΩ)))
                #end
            elseif RTA <: Type{<:Union{EnzymeCore.DuplicatedNoNeed,EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated, EnzymeCore.BatchDuplicatedNoNeed}}
                push!(genexprs, Expr(:(=), :dΩ, :(cache[end])))
            else
                throw(AssertionError("Easy Rule should never be provided a constant reverse seed"))
            end

            actives = Union{Nothing, Expr}[$(actives...)]

            results = []

            visited = zeros(Bool, length(tosum0), N)

            insyms = Matrix{Symbol}(undef, N, W)

            for (inum, sym_name) in enumerate(inp_names)
                if (RTA <: EnzymeCore.Const)
                    push!(results, nothing)
                    continue
                end

                if inp_types[inum] <: EnzymeCore.Const
                    push!(results, nothing)
                    continue
                end

                seen = false
                for (o, tosum) in enumerate(tosum0)
                    for (i, sname, p) in tosum
                        if actives[i] isa Nothing
                            continue
                        end
                        if i != inum
                            continue
                        end

                        pname = Symbol("partial_", string(o), "_", string(i), "_",  sname)
                        if !visited[o, i]

                            # Descend through the rule to see if any users require the original result, Ω
                            # for now, conservatively assume this is indeed the case
                            if uses_symbol(p, :Ω)
                                has_omega = true
                            end

                            push!(gensetup, Expr(:(=), pname, p))
                        end

                        for w in 1:W

                            dval = :dΩ
                            if W != 1
                                dval = Expr(:call, getfield, dval, w)
                            end

                            if RT <: Tuple        
                                dval = Expr(:call, getfield, dval, o)
                            else
                                @assert length(tosum0) == 1
                            end

                            inexpr = Symbol("insym_", string(i), "_", string(w))
                            insyms[i, w] = inexpr

                            if !seen
                                if inp_types[inum] <: Active
                                    push!(gensetup, Expr(:(=), inexpr, nothing))
                                else
                                    dexpr = Expr(:call, getfield, Symbol(inp_names[inum]), 2)
                                    if W != 1
                                        dexpr = Expr(:call, getfield, inp_names, w)
                                    end
                                    push!(gensetup, Expr(:(=), inexpr, dexpr))
                                end
                            end

                            push!(gensetup, Expr(:(=), inexpr, Expr(:call, multiply_rev_into, inexpr, pname, dval)))
                        end
                        seen = true
                    end
                end



                if !seen && inp_types[inum] <: Active
                    for w in 1:W
                        inexpr = Symbol("insym_", string(inum), "_", string(w))
                        insyms[inum, w] = inexpr

                        push!(gensetup, Expr(:(=), inexpr, Expr(:call, EnzymeCore.make_zero, Expr(:call, getfield, Symbol(inp_names[inum]), 1) )))
                    end
                end

                if !(inp_types[inum] <: Active)
                    push!(results, nothing)
                elseif W == 1
                    push!(results, insyms[inum, 1])
                else
                    push!(results, Expr(:tuple, insyms[inum, :]...))
                end
            end

            genres = Expr(:tuple, results...)

            if has_omega
                push!(genexprs, Expr(:(=), :Ω, Expr(:call, :f, $arg_names...)))
            end

            quote
                Base.@_inline_meta
                $($(__source__))
                f = fn.val
                $(genexprs...)
                $(gensetup...)
                return $genres
            end
        end
    end
end

"""
    @easy_rule(f(x₁::T1, x₂::T2, ...),
                 @setup(statement₁, statement₂, ...),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

A convenience macro that generates simple forward and reverse Enzyme rules using
the provided partial derivatives.

The `easy_rule` macro assumes that the function `f` does not mutate any of its arguments, does not read from global data, and no output aliases with any other output nor input.

The `easy_rule` macro only works if inputs are scalars or arrays of scalars, and the result is a scalar, array, or tuple thereof.

For each output result (a single output is assumed if a scalar is returned), a tuple of partial derivatives is expected. Specifically, each tuple contains one entry for each argument to `f`. This entry should contain the jacobian `∂fi_∂xj` where `i` is the number of the output, and `j` is the number of the input. If both input `i` and output `j` are scalars, `∂fi_∂xj` should be a scalar. If at least one of these are an abstractarray, `∂fi_∂xj` should be a tensor of size `(size(output[j])..., size(input[i])...)`, where scalars are considered zero-dimensional.

The arguments to `f` can either have no type constraints, or specific type constraints.

The result of `f(x₁, x₂, ...)` is automatically bound to `Ω`. This allows the primal result to be referenced (as `Ω`) within the derivative/setup expressions.

At present this does not support defining for closures/functors.

The `@setup` argument can be elided if no setup code is need. In other
words:

```julia
@easy_rule(f(x₁, x₂, ...),
             (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
             (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
             ...)
```

is equivalent to:

```julia
@easy_rule(f(x₁, x₂, ...),
             @setup(nothing),
             (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
             (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
             ...)
```

If a specific argument has no partial derivative, then all corresponding argument values can instead be marked `@Constant`. For example, consider the case where `config` has no derivative.

```julia
@easy_rule(f(config, x, ...),
             @setup(nothing),
             (@Constant, ∂f₁_∂x, ...),
             (@Constant, ∂f₂_∂x, ...),
             ...)
```

"""
macro easy_rule(call, maybe_setup, partials...)
    call, setup_stmts, inputs, input_names, normal_inputs, partials = _normalize_scalarrules_macro_input(
        call, maybe_setup, partials
    )
    f = call.args[1]

    frule_expr = scalar_frule_expr(__source__, f, call, setup_stmts, inputs, input_names, partials)
    rrule_expr = scalar_rrule_expr(__source__, f, call, setup_stmts, inputs, input_names, partials)

    # Final return: building the expression to insert in the place of this macro
    quote
        EnzymeRules.has_easy_rule(::Core.Typeof($f), $(normal_inputs...)) = true
        $(frule_expr)
        $(rrule_expr)
    end
end