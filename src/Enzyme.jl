module Enzyme

import EnzymeCore

import EnzymeCore: Forward, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal, ReverseSplitModified, ReverseSplitWidth, ReverseMode, ForwardMode, ReverseHolomorphic, ReverseHolomorphicWithPrimal
export Forward, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal, ReverseSplitModified, ReverseSplitWidth, ReverseMode, ForwardMode, ReverseHolomorphic, ReverseHolomorphicWithPrimal

import EnzymeCore: Annotation, Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed, ABI, DefaultABI, FFIABI, InlineABI, NonGenABI, set_err_if_func_written, clear_err_if_func_written, set_abi
export Annotation, Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed, DefaultABI, FFIABI, InlineABI, NonGenABI, set_err_if_func_written, clear_err_if_func_written, set_abi

import EnzymeCore: BatchDuplicatedFunc
export BatchDuplicatedFunc

import EnzymeCore: MixedDuplicated, BatchMixedDuplicated
export MixedDuplicated, BatchMixedDuplicated

import EnzymeCore: batch_size, get_func 
export batch_size, get_func

import EnzymeCore: autodiff, autodiff_deferred, autodiff_thunk, autodiff_deferred_thunk, tape_type, make_zero, make_zero!
export autodiff, autodiff_deferred, autodiff_thunk, autodiff_deferred_thunk, tape_type, make_zero, make_zero!

export jacobian, gradient, gradient!, hvp, hvp!, hvp_and_gradient!
export markType, batch_size, onehot, chunkedonehot

using LinearAlgebra
import EnzymeCore: ReverseMode, ReverseModeSplit, ForwardMode, Mode

import EnzymeCore: EnzymeRules
export EnzymeRules

# Independent code, must be loaded before "compiler.jl"
include("pmap.jl")

import LLVM
include("api.jl")

Base.convert(::Type{API.CDerivativeMode}, ::ReverseMode) = API.DEM_ReverseModeCombined
Base.convert(::Type{API.CDerivativeMode}, ::ReverseModeSplit) = API.DEM_ReverseModeGradient
Base.convert(::Type{API.CDerivativeMode}, ::ForwardMode) = API.DEM_ForwardMode

function guess_activity end

include("logic.jl")
include("typeanalysis.jl")
include("typetree.jl")
include("gradientutils.jl")
include("utils.jl")
include("compiler.jl")
include("internal_rules.jl")
include("autodiff.jl")
include("diffinterface.jl")

import .Compiler: CompilationException


function _import_frule end # defined in EnzymeChainRulesCoreExt extension

"""
    import_frule(::fn, tys...)

Automatically import a `ChainRulesCore.frule`` as a custom forward mode `EnzymeRule`. When called in batch mode, this
will end up calling the primal multiple times, which may result in incorrect behavior if the function mutates,
and slow code, always. Importing the rule from `ChainRules` is also likely to be slower than writing your own rule,
and may also be slower than not having a rule at all.

Use with caution.

```julia
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
macro import_frule(args...)
    return _import_frule(args...)
end 

function _import_rrule end # defined in EnzymeChainRulesCoreExt extension

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

```julia
Enzyme.@import_rrule(typeof(Base.sort), Any);
```
"""
macro import_rrule(args...)
    return _import_rrule(args...)
end

end # module
