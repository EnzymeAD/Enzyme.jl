module EnzymeDIExt

using ADTypes: ADTypes, AutoEnzyme
using Base: Fix1
import DifferentiationInterface as DI
using EnzymeCore:
    Active,
    Annotation,
    BatchDuplicated,
    BatchDuplicatedNoNeed,
    BatchMixedDuplicated,
    Combined,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    EnzymeCore,
    Forward,
    ForwardMode,
    ForwardWithPrimal,
    MixedDuplicated,
    Mode,
    NoPrimal,
    Reverse,
    ReverseMode,
    ReverseModeSplit,
    ReverseSplitNoPrimal,
    ReverseSplitWidth,
    ReverseSplitWithPrimal,
    ReverseWithPrimal,
    Split,
    WithPrimal
using Enzyme:
    autodiff,
    autodiff_thunk,
    create_shadows,
    gradient,
    gradient!,
    guess_activity,
    hvp,
    hvp!,
    jacobian,
    make_zero,
    make_zero!,
    onehot

include("utils.jl")

include("forward_onearg.jl")
include("forward_twoarg.jl")

include("reverse_onearg.jl")
include("reverse_twoarg.jl")

end # module