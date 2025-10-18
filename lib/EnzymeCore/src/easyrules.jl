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
_constrain_and_name(name::Symbol) = error("malformed input: $arg, must explicitly provide type annotation")


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
    multiply_fwd(partial::AbstractFloat, dx::AbstractFloat)

Internal function.

Multiply a partial derivative (df/dx) by its shadow input (dx) to form `df`.
"""
function multiply_fwd(partial::AbstractFloat, dx::AbstractFloat)
    return partial * dx
end


"""
    add_fwd(partial::AbstractFloat, dx::AbstractFloat)

Internal function.

Add together two partial derivatives.
"""
function add_fwd end


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
        setup_stmts = Any[esc(ex) for ex in maybe_setup.args[3:end]]
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

    tosum0 = Vector{Tuple{Int, Symbol, Expr}}[]

    for (o, partial0) in enumerate(partials)
        @assert partial0 isa Array && length(partial0) == length(inputs)
        tosum = Tuple{Int, Symbol, Expr}[]
        push!(tosum0, tosum)
        for (i, (p, sname)) in enumerate(zip(partial0, input_names))
            if p == :(EnzymeCore.Const)
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
        # allow closures/functors with @scalar_rule, it is always ignored
        @generated function EnzymeCore.EnzymeRules.forward($(esc(:config)), $(esc(:fn))::Annotation{<:$(Core.Typeof)($f)}, ::Type{<:Annotation{$(esc(:RT))}}, $(inputs...)) where $(esc(:RT))
            genexprs = Expr[$(exprs...,)...]
            gensetup = Expr[$(setup_stmts...,)...]

            has_omega = false
            
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

                            msym = Symbol("m_", string(w), "_partial_", string(o), "_", sname)
                            dval = actives[i]::Expr
                            if W != 1
                                dval = Expr(:call, getfield, dval, w)
                            end

                            pname = Symbol("partial_", string(o), "_", string(i), "_",  sname)
                            if !visited[o, i]

                                # Descend through the rule to see if any users require the original result, Ω
                                # for now, conservatively assume this is indeed the case
                                if uses_symbol(p, Ω)
                                    has_omega = true
                                end

                                push!(gensetup, Expr(:(=), pname, p))
                                visited[o, i] = true
                            end
                            push!(gensetup, Expr(:(=), msym, Expr(:call, multiply_fwd, pname, dval)))

                            if !seen
                                push!(gensetup, Expr(:(=), outexpr, msym))
                                seen = true
                                continue
                            end

                            push!(gensetup, Expr(:(=), outexpr, Expr(:call, add_fwd, outexpr, msym)))
                        end

                        if !seen
                            inp = :Ω
                            if $(esc(:RT)) <: Tuple
                                inp = :(Ω[$o])
                            end
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
                        push!(gensetup, Expr(:(=), outexpr, outsyms[:, w]...))
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


"""
    @easy_scalar_rule(f(x₁, x₂, ...),
                 @setup(statement₁, statement₂, ...),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

A convenience macro that generates simple forward (and eventually reverse) Enzyme rules using
the provided partial derivatives.

This macro assumes all inputs are scalars, and all results are scalars, or tuples of scalars. For each output result (a single output is assumed if a scalar is returned), a tuple of partial derivatives is expected. Specifically, each tuple contains one entry for each argument to `f`.

Use of an easy_rule assumes that the function does not mutate any of its arguments, does not read from global data, and no output aliases with any other output nor input.

The arguments to `f` can either have no type constraints, or specific type constraints.

At present this does not support defining for closures/functors.

The `@setup` argument can be elided if no setup code is need. In other
words:

```julia
@easy_scalar_rule(f(x₁, x₂, ...),
             (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
             (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
             ...)
```

is equivalent to:

```julia
@easy_scalar_rule(f(x₁, x₂, ...),
             @setup(nothing),
             (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
             (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
             ...)
```

If a specific argument has no partial derivative, then all corresponding argument values can instead be marked `Enzyme.Const`. For example, consider the case where `config` has no derivative.

```julia
@easy_scalar_rule(f(config, x, ...),
             @setup(nothing),
             (Enzyme.Const, ∂f₁_∂x, ...),
             (Enzyme.Const, ∂f₂_∂x, ...),
             ...)
```

"""
macro easy_scalar_rule(call, maybe_setup, partials...)
    call, setup_stmts, inputs, input_names, normal_inputs, partials = _normalize_scalarrules_macro_input(
        call, maybe_setup, partials
    )
    f = call.args[1]

    frule_expr = scalar_frule_expr(__source__, f, call, setup_stmts, inputs, input_names, partials)

    # Final return: building the expression to insert in the place of this macro
    return quote
        if !($f isa $Type) && $fieldcount($typeof($f)) > 0
            $throw(
                $ArgumentError(
                    "@easy_scalar_rule cannot be used on closures/functors (such as $($f))"
                ),
            )
        end

        has_easy_rule(::Core.Typeof($f), $(normal_inputs...)) = true
        $(frule_expr)
    end
end