module EnzymeChainRulesCoreExt

using ChainRulesCore
using EnzymeCore
using Enzyme


"""
    import_frule(::fn, tys...)

Automatically import a `ChainRulesCore.frule`` as a custom forward mode `EnzymeRule`. When called in batch mode, this
will end up calling the primal multiple times, which may result in incorrect behavior if the function mutates,
and slow code, always. Importing the rule from `ChainRules` is also likely to be slower than writing your own rule,
and may also be slower than not having a rule at all.

Use with caution.

```jldoctest
Enzyme.@import_frule(typeof(Base.sort), Any);

x=[1.0, 2.0, 0.0]; dx=[0.1, 0.2, 0.3]; ddx = [0.01, 0.02, 0.03];

Enzyme.autodiff(Forward, sort, Duplicated, BatchDuplicated(x, (dx,ddx)))
Enzyme.autodiff(Forward, sort, DuplicatedNoNeed, BatchDuplicated(x, (dx,ddx)))
Enzyme.autodiff(Forward, sort, DuplicatedNoNeed, BatchDuplicated(x, (dx,)))
Enzyme.autodiff(Forward, sort, Duplicated, BatchDuplicated(x, (dx,)))

# output

(var"1" = [0.0, 1.0, 2.0], var"2" = (var"1" = [0.3, 0.1, 0.2], var"2" = [0.03, 0.01, 0.02]))
(var"1" = (var"1" = [0.3, 0.1, 0.2], var"2" = [0.03, 0.01, 0.02]),)
(var"1" = [0.3, 0.1, 0.2],)
(var"1" = [0.0, 1.0, 2.0], var"2" = [0.3, 0.1, 0.2])

```
"""
function Enzyme._import_frule(fn, tys...)
    vals = []
    exprs = []
    primals = []
    tangents = []
    tangentsi = []
    anns = []
    for (i, ty) in enumerate(tys)
        val = Symbol("arg_$i")
        TA = Symbol("AN_$i")
        e = :($val::$TA)
        push!(anns, :($TA <: Annotation{<:$(esc(ty))}))
        push!(vals, val)
        push!(exprs, e)
        push!(primals, :($val.val))
        push!(tangents, :($val isa Const ? $ChainRulesCore.NoTangent() : $val.dval))
        push!(tangentsi, :($val isa Const ? $ChainRulesCore.NoTangent() : $val.dval[i]))
    end

    return quote
        function EnzymeRules.forward(config, fn::FA, ::Type{RetAnnotation}, $(exprs...); kwargs...) where {RetAnnotation, FA <: Annotation{<:$(esc(fn))}, $(anns...)}
            batchsize = same_or_one(1, $(vals...))
            return if batchsize == 1
                dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval
                cres = $ChainRulesCore.frule((dfn, $(tangents...)), fn.val, $(primals...); kwargs...)
                if RetAnnotation <: Const
                    if EnzymeRules.needs_primal(config)
                        return cres[1]::eltype(RetAnnotation)
                    else
                        return nothing
                    end
                elseif RetAnnotation <: Duplicated
                    return Duplicated(cres[1], cres[2])
                elseif RetAnnotation <: DuplicatedNoNeed
                    return cres[2]::eltype(RetAnnotation)
                else
                    @assert false
                end
            else
                if RetAnnotation <: Const
                    cres = ntuple(Val(batchsize)) do i
                        Base.@_inline_meta
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...)), fn.val, $(primals...); kwargs...)
                    end
                    if EnzymeRules.needs_primal(config)
                        return cres[1][1]::eltype(RetAnnotation) # nothing
                    else
                        return nothing
                    end
                elseif RetAnnotation <: BatchDuplicated
                    cres1 = begin
                        i = 1
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...)), fn.val, $(primals...); kwargs...)
                    end
                    batches = ntuple(Val(batchsize - 1)) do j
                        Base.@_inline_meta
                        i = j + 1
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...)), fn.val, $(primals...); kwargs...)[2]
                    end
                    return BatchDuplicated(cres1[1], (cres1[2], batches...))
                elseif RetAnnotation <: BatchDuplicatedNoNeed
                    ntuple(Val(batchsize)) do i
                        Base.@_inline_meta
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...)), fn.val, $(primals...); kwargs...)[2]
                    end
                else
                    @assert false
                end
            end
        end
    end # quote
end


"""
    import_rrule(::fn, tys...)

Automatically import a ChainRules.rrule as a custom reverse mode EnzymeRule. When called in batch mode, this
will end up calling the primal multiple times which results in slower code. This macro assumes that the underlying
function to be imported is read-only, and returns a Duplicated or Const object. This macro also assumes that the
inputs permit a .+= operation and that the output has a valid Enzyme.make_zero function defined. It also assumes
that overwritten(x) accurately describes if there is any non-preserved data from forward to reverse, not just
the outermost data structure being overwritten as provided by the specification.

Finally, this macro falls back to almost always caching all of the inputs, even if it may not be needed for the
derivative computation.

As a result, this auto importer is also likely to be slower than writing your own rule, and may also be slower
than not having a rule at all.

Use with caution.

```
Enzyme.@import_rrule(typeof(Base.sort), Any);
```
"""
function Enzyme._import_rrule(fn, tys...)
    vals = []
    valtys = []
    exprs = []
    primals = []
    tangents = []
    tangentsi = []
    anns = []
    nothings = []
    ntys = length(tys)
    act_res = Expr[:(fn isa Active ? res[1] : nothing)]
    invertcomb = Expr[]
    # TODO at one point extend api to support active fn's
    # push!(nothings, :nothing)
    # push!(invertcomb,
    # quote
    #     fn isa Active ? (
    #         (EnzymeRules.width(config) == 1) ? tcomb[1][1] :
    #         ntuple(Val(EnzymeRules.width(config))) do batch_i
    #             Base.@_inline_meta
    #             tcomb[batch_i][1]
    #         end
    #        ) : nothing
    # end)

    ptys = []
    for (i, ty) in enumerate(tys)
        push!(nothings, :(nothing))
        val = Symbol("arg_$i")
        TA = Symbol("AN_$i")
        e = :($val::$TA)
        push!(ptys, :(::$(esc(ty))))
        push!(anns, :($TA <: Annotation{<:$(esc(ty))}))
        push!(vals, val)
        push!(exprs, e)
        primal = Symbol("primcopy_$i")
        push!(primals, primal)
        push!(valtys, :($primal = $(EnzymeRules.overwritten)(config)[$i + 1] ? deepcopy($val.val) : $val.val))
        push!(tangents, :($val isa $Enzyme.Const ? $ChainRulesCore.NoTangent() : $val.dval))
        push!(tangentsi, :($val isa $Enzyme.Const ? $ChainRulesCore.NoTangent() : $val.dval[i]))
        push!(act_res, :($val isa Active ? (res[$i + 1] isa $ChainRulesCore.NoTangent ? zero($val) : $ChainRulesCore.unthunk(res[$i + 1])) : nothing))
        push!(
            invertcomb, quote
                $val isa Active ? (
                        (EnzymeRules.width(config) == 1) ? tcomb[1][$i + 1] :
                        ntuple(Val(EnzymeRules.width(config))) do batch_i
                            Base.@_inline_meta
                            tcomb[batch_i][$i + 1]
                    end
                    ) : nothing
            end
        )
    end

    return quote
        EnzymeRules.has_easy_rule(::$(esc(fn)), $(ptys...)) = true

        function EnzymeRules.augmented_primal(config, fn::FA, ::Type{RetAnnotation}, $(exprs...); kwargs...) where {RetAnnotation, FA <: Annotation{<:$(esc(fn))}, $(anns...)}
            $(valtys...)

            @assert !(RetAnnotation <: Const)
            res, pullback = $ChainRulesCore.rrule(fn.val, $(primals...); kwargs...)

            primal = if EnzymeRules.needs_primal(config)
                res
            else
                nothing
            end

            shadow, byref = if !EnzymeRules.needs_shadow(config)
                nothing, Val(false)
            elseif !Enzyme.Compiler.guaranteed_nonactive(Core.Typeof(res))
                (
                    if EnzymeRules.width(config) == 1
                        Ref(Enzyme.make_zero(res))
                    else
                        ntuple(Val(EnzymeRules.width(config))) do j
                            Base.@_inline_meta
                            Ref(Enzyme.make_zero(res))
                        end
                    end, Val(true),
                )
            else
                (
                    if EnzymeRules.width(config) == 1
                        Enzyme.make_zero(res)
                    else
                        ntuple(Val(EnzymeRules.width(config))) do j
                            Base.@_inline_meta
                            Enzyme.make_zero(res)
                        end
                    end, Val(false),
                )
            end

            cache = (shadow, pullback, byref)
            return EnzymeRules.augmented_rule_return_type(config, RetAnnotation){typeof(cache)}(primal, shadow, cache)
        end

        function EnzymeRules.reverse(config, fn::FA, ::Type{RetAnnotation}, tape::TapeTy, $(exprs...); kwargs...) where {RetAnnotation, TapeTy, FA <: Annotation{<:$(esc(fn))}, $(anns...)}
            if !(RetAnnotation <: Const)
                shadow, pullback, byref = tape

                tcomb = ntuple(Val(EnzymeRules.width(config))) do batch_i
                    Base.@_inline_meta
                    shad = EnzymeRules.width(config) == 1 ? shadow : shadow[batch_i]
                    if byref === Val(true)
                        shad = shad[]
                    end
                    res = pullback(shad)

                    for (cr, en) in zip(res, (fn, $(vals...)))
                        if en isa Const || cr isa $ChainRulesCore.NoTangent
                            continue
                        end
                        if en isa Active
                            continue
                        end
                        if EnzymeRules.width(config) == 1
                            en.dval .+= cr
                        else
                            en.dval[batch_i] .+= cr
                        end
                    end

                    ($(act_res...),)
                end

                return ($(invertcomb...),)
            end

            return ($(nothings...),)
        end

        function EnzymeRules.reverse(config, fn::FA, dval::Active{RetAnnotation}, tape::TapeTy, $(exprs...); kwargs...) where {RetAnnotation, TapeTy, FA <: Annotation{<:$(esc(fn))}, $(anns...)}
            oldshadow, pullback = tape

            shadow = dval.val

            tcomb = ntuple(Val(EnzymeRules.width(config))) do batch_i
                Base.@_inline_meta
                shad = EnzymeRules.width(config) == 1 ? shadow : shadow[batch_i]
                res = pullback(shad)

                for (cr, en) in zip(res, (fn, $(vals...)))
                    if en isa Const || cr isa $ChainRulesCore.NoTangent
                        continue
                    end
                    if en isa Active
                        continue
                    end
                    if EnzymeRules.width(config) == 1
                        en.dval .+= cr
                    else
                        en.dval[batch_i] .+= cr
                    end
                end

                ($(act_res...),)
            end

            return ($(invertcomb...),)
        end
    end
end

end # module
