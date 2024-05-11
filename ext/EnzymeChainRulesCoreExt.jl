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
        push!(anns, :($TA <: Annotation{<:$ty}))
        push!(vals, val)
        push!(exprs, e)
        push!(primals, :($val.val))
        push!(tangents, :($val isa Const ? $ChainRulesCore.NoTangent() : $val.dval))
        push!(tangentsi, :($val isa Const ? $ChainRulesCore.NoTangent() : $val.dval[i]))
    end

    quote
        function EnzymeRules.forward(fn::FA, ::Type{RetAnnotation}, $(exprs...); kwargs...) where {RetAnnotation, FA<:Annotation{<:$(esc(fn))}, $(anns...)}
            batchsize = same_or_one(1, $(vals...))
            if batchsize == 1
                dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval
                cres = $ChainRulesCore.frule((dfn, $(tangents...),), fn.val, $(primals...); kwargs...)
                if RetAnnotation <: Const
                    return nothing
                elseif RetAnnotation <: Duplicated
                    return Duplicated(cres[1], cres[2])
                elseif RetAnnotation <: DuplicatedNoNeed
                    return cres[2]::eltype(RetAnnotation)
                else
                    @assert false
                end
            else
                if RetAnnotation <: Const
                    ntuple(Val(batchsize)) do i
                        Base.@_inline_meta
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)
                    end
                    return nothing
                elseif RetAnnotation <: BatchDuplicated
                    cres1 = begin
                        i = 1
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)
                    end
                    batches = ntuple(Val(batchsize-1)) do j
                        Base.@_inline_meta
                        i = j+1
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)[2]
                    end
                    return BatchDuplicated(cres1[1], (cres1[2], batches...))
                elseif RetAnnotation <: BatchDuplicatedNoNeed
                    ntuple(Val(batchsize)) do i
                        Base.@_inline_meta
                        dfn = fn isa Const ? $ChainRulesCore.NoTangent() : fn.dval[i]
                        $ChainRulesCore.frule((dfn, $(tangentsi...),), fn.val, $(primals...); kwargs...)[2]
                    end
                else
                    @assert false
                end
            end
        end
    end # quote
end


end # module