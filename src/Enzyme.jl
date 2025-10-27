module Enzyme

import EnzymeCore

import EnzymeCore:
    Forward,
    ForwardWithPrimal,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitNoPrimal,
    ReverseSplitWithPrimal,
    ReverseSplitModified,
    ReverseSplitWidth,
    Mode,
    ReverseMode,
    ReverseModeSplit,
    ForwardMode,
    ReverseHolomorphic,
    ReverseHolomorphicWithPrimal
export Forward,
    ForwardWithPrimal,
    Reverse,
    ReverseWithPrimal,
    ReverseSplitNoPrimal,
    ReverseSplitWithPrimal,
    ReverseSplitModified,
    ReverseSplitWidth,
    Mode,
    ReverseMode,
    ReverseModeSplit,
    ForwardMode,
    ReverseHolomorphic,
    ReverseHolomorphicWithPrimal

import EnzymeCore:
    Annotation,
    Const,
    Active,
    Duplicated,
    DuplicatedNoNeed,
    BatchDuplicated,
    BatchDuplicatedNoNeed,
    ABI,
    DefaultABI,
    FFIABI,
    InlineABI,
    NonGenABI,
    set_err_if_func_written,
    clear_err_if_func_written,
    set_abi,
    set_runtime_activity,
    clear_runtime_activity,
    set_strong_zero,
    clear_strong_zero,
    within_autodiff,
    ignore_derivatives,
    WithPrimal,
    NoPrimal,
    needs_primal,
    runtime_activity,
    strong_zero
export Annotation,
    Const,
    Active,
    Duplicated,
    DuplicatedNoNeed,
    BatchDuplicated,
    BatchDuplicatedNoNeed,
    DefaultABI,
    FFIABI,
    InlineABI,
    NonGenABI,
    set_err_if_func_written,
    clear_err_if_func_written,
    set_abi,
    set_runtime_activity,
    clear_runtime_activity,
    set_strong_zero,
    clear_strong_zero,
    WithPrimal,
    NoPrimal,
    within_autodiff,
    needs_primal,
    runtime_activity,
    strong_zero

import EnzymeCore: BatchDuplicatedFunc
export BatchDuplicatedFunc

import EnzymeCore: MixedDuplicated, BatchMixedDuplicated
export MixedDuplicated, BatchMixedDuplicated

import EnzymeCore: batch_size, get_func
export batch_size, get_func

import EnzymeCore:
    autodiff,
    autodiff_deferred,
    autodiff_thunk,
    autodiff_deferred_thunk,
    tape_type,
    make_zero,
    make_zero!,
    remake_zero!
export autodiff,
    autodiff_deferred,
    autodiff_thunk,
    autodiff_deferred_thunk,
    tape_type,
    make_zero,
    make_zero!,
    remake_zero!

export jacobian, gradient, gradient!, hvp, hvp!, hvp_and_gradient!
export batch_size, onehot, chunkedonehot

using LinearAlgebra
import SparseArrays

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

mutable struct EnzymeContext
end

include("logic.jl")
include("analyses/type.jl")
include("typetree.jl")
include("gradientutils.jl")
include("utils.jl")
include("compiler.jl")
include("internal_rules.jl")

import .Compiler: CompilationException

@inline function falses_from_args(N)
    ntuple(Returns(false), Val(N))
end

@inline function any_active(args::Vararg{Annotation,N}) where {N}
    any(ntuple(Val(N)) do i
        Base.@_inline_meta
        arg = @inbounds args[i]
        if arg isa Active
            return true
        elseif arg isa MixedDuplicated
            return true
        elseif arg isa BatchMixedDuplicated
            return true
        else
            return false
        end
    end)
end

@inline function vaTypeof(args::Vararg{Any,N}) where {N}
    return Tuple{(
        ntuple(Val(N)) do i
            Base.@_inline_meta
            Core.Typeof(args[i])
        end
    )...}
end

@inline function vaEltypeof(args::Vararg{Any,N}) where {N}
    return Tuple{(
        ntuple(Val(N)) do i
            Base.@_inline_meta
            eltype(Core.Typeof(args[i]))
        end
    )...}
end

@inline function vaEltypes(args::Type{Ty}) where {Ty<:Tuple}
    return Tuple{(
        ntuple(Val(length(Ty.parameters))) do i
            Base.@_inline_meta
            eltype(Ty.parameters[i])
        end
    )...}
end

@inline function same_or_one_helper(current, next)
    if current == -1
        return next
    elseif current == next
        return next
    else
        error("Multiple distinct batch sizes")
    end
end

@inline same_or_one_rec(current) = current
@inline same_or_one_rec(current, arg::BatchMixedDuplicated{T,N}, args...) where {T,N} =
    same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(
    current,
    arg::Type{BatchMixedDuplicated{T,N}},
    args...,
) where {T,N} = same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicatedFunc{T,N}, args...) where {T,N} =
    same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicatedFunc{T,N}}, args...) where {T,N} =
    same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicated{T,N}, args...) where {T,N} =
    same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::Type{BatchDuplicated{T,N}}, args...) where {T,N} =
    same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg::BatchDuplicatedNoNeed{T,N}, args...) where {T,N} =
    same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(
    current,
    arg::Type{BatchDuplicatedNoNeed{T,N}},
    args...,
) where {T,N} = same_or_one_rec(same_or_one_helper(current, N), args...)
@inline same_or_one_rec(current, arg, args...) = same_or_one_rec(current, args...)

@inline function same_or_one(defaultVal, args...)
    local_soo_res = same_or_one_rec(-1, args...)
    if local_soo_res == -1
        defaultVal
    else
        local_soo_res
    end
end


@inline function refn_seed(x::T) where {T}
    if T <: Complex
        return conj(x) / 2
    else
        return x
    end
end

@inline function imfn_seed(x::T) where {T}
    if T <: Complex
        return im * conj(x) / 2
    else
        return T(0)
    end
end

@inline function seed_complex_args(
    seen,
    seen2,
    args::Vararg{Annotation,Nargs},
) where {Nargs}
    return ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        arg = args[i]
        if arg isa Const || arg isa Active
            arg
        elseif arg isa Duplicated || arg isa DuplicatedNoNeed
            RT = eltype(Core.Typeof(arg))
            BatchDuplicated(
                arg.val,
                (arg.dval, make_zero(RT, seen, arg.dval), make_zero(RT, seen2, arg.dval)),
            )
        else
            throw(
                ErrorException(
                    "Active Complex return does not yet support batching in combined reverse mode",
                ),
            )
        end
    end
end

@inline function fuse_complex_results(results, args::Vararg{Annotation,Nargs}) where {Nargs}
    ntuple(Val(Nargs)) do i
        Base.@_inline_meta
        if args[i] isa Active
            Compiler.recursive_add(
                Compiler.recursive_add(results[1][i][1], results[1][i][2], refn_seed),
                results[1][i][3],
                imfn_seed,
            )
        else
            results[1][i]
        end
    end
end

"""
    autodiff(::ReverseMode, f, Activity, args::Annotation...)

Auto-differentiate function `f` at arguments `args` using reverse mode.

Limitations:

* `f` may only return a `Real` (of a built-in/primitive type) or `nothing`,
  not an array, struct, `BigFloat`, etc. To handle vector-valued return
  types, use a mutating `f!` that returns `nothing` and stores it's return
  value in one of the arguments, which must be wrapped in a
  [`Duplicated`](@ref).

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in an [`Active`](@ref) (for arguments whose derivative result must be returned
rather than mutated in place, such as primitive types and structs thereof)
or [`Duplicated`](@ref) (for mutable arguments like arrays, `Ref`s and structs
thereof).

`Activity` is the Activity of the return value, it may be `Const` or `Active`.

Example:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(a, b, c, d) = a * √(b[1]^2 + b[2]^2) + c^2 * d^2
∂f_∂a, _, _, ∂f_∂d = autodiff(Reverse, f, Active, Active(a), Duplicated(b, ∂f_∂b), Const(c), Active(d))[1]

# output

(3.966106403010388, nothing, nothing, 54450.0)
```

here, `autodiff` returns a tuple
``(\\partial f/\\partial a, \\partial f/\\partial d)``,
while ``\\partial f/\\partial b`` will be *added to* `∂f_∂b` (but not returned).
`c` will be treated as `Const(c)`.

One can also request the original returned value of the computation.

Example:

```jldoctest
Enzyme.autodiff(ReverseWithPrimal, x->x*x, Active(3.0))

# output

((6.0,), 9.0)
```

!!! note

    Enzyme gradients with respect to integer values are zero.
    [`Active`](@ref) will automatically convert plain integers to floating
    point values, but cannot do so for integer values in tuples and structs.
"""
@inline function autodiff(
    mode::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,RABI,Holomorphic,ErrIfFuncWritten},
    f::FA,
    ::Type{A0},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A0<:Annotation,
    ReturnPrimal,
    RuntimeActivity,
    StrongZero,
    RABI<:ABI,
    Holomorphic,
    Nargs,
    ErrIfFuncWritten,
}
    tt′ = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    ModifiedBetweenT = falses_from_args(Nargs + 1)
    ModifiedBetween = Val(ModifiedBetweenT)

    tt = vaEltypeof(args...)

    FTy = Core.Typeof(f.val)

    rt, A = if A0 isa UnionAll
        rt0 = Compiler.primal_return_type(Reverse, FTy, tt)
        rt0, A0{rt0}
    else
        eltype(A0), A0
    end

    if A0 <: Active
        if (!allocatedinline(rt) || rt isa Union) && rt != Union{}
            forward, adjoint = autodiff_thunk(
                ReverseModeSplit{
                    ReturnPrimal,
                    #=ReturnShadow=#false,
                    RuntimeActivity,
                    StrongZero,
                    width,
                    ModifiedBetweenT,
                    RABI,
                    Holomorphic,
                    ErrIfFuncWritten,
                    #=ShadowInit=#true
                }(),
                FA,
                Duplicated{rt},
                (tt′).parameters...
            )
            res = forward(f, args...)
            tape = res[1]
            if ReturnPrimal
                return (adjoint(f, args..., tape)[1], res[2])
            else
                return adjoint(f, args..., tape)
            end
        end
    elseif A0 <: Duplicated ||
           A0 <: DuplicatedNoNeed ||
           A0 <: BatchDuplicated ||
           A0 <: BatchDuplicatedNoNeed ||
           A0 <: BatchDuplicatedFunc
        throw(ErrorException("Duplicated Returns not yet handled"))
    end

    opt_mi = if RABI <: NonGenABI
        my_methodinstance(Reverse, eltype(FA), tt)
    else
        Val(0)
    end

    if (A0 <: Active && rt <: Complex) && rt != Union{}
        if Holomorphic
            seen = IdDict()
            seen2 = IdDict()

            f = if f isa Const || f isa Active
                f
            elseif f isa Duplicated || f isa DuplicatedNoNeed
                BatchDuplicated(
                    f.val,
                    (
                        f.dval,
                        make_zero(typeof(f), seen, f.dval),
                        make_zero(typeof(f), seen2, f.dval),
                    ),
                )
            else
                throw(
                    ErrorException(
                        "Active Complex return does not yet support batching in combined reverse mode",
                    ),
                )
            end

            width = same_or_one(3, args...)
            args = seed_complex_args(seen, seen2, args...)
            tt′ = vaTypeof(args...)

            thunk = Enzyme.Compiler.thunk(
                opt_mi,
                typeof(f),
                A,
                tt′,
                Val(API.DEM_ReverseModeCombined),
                Val(width),
                ModifiedBetween,
                Val(ReturnPrimal),
                Val(false),
                RABI,
                Val(ErrIfFuncWritten),
                Val(RuntimeActivity),
                Val(StrongZero)
            ) #=ShadowInit=#

            results = thunk(f, args..., (rt(0), rt(1), rt(im)))

            # compute the correct complex derivative in reverse mode by propagating the conjugate return values
            # then subtracting twice the imaginary component to get the correct result

            for (k, v) in seen
                Compiler.recursive_accumulate(k, v, refn_seed)
            end
            for (k, v) in seen2
                Compiler.recursive_accumulate(k, v, imfn_seed)
            end

            fused = fuse_complex_results(results, args...)

            return (fused, results[2:end]...)
        end

        throw(
            ErrorException(
                "Reverse-mode Active Complex return is ambiguous and requires more information to specify the desired result. See https://enzyme.mit.edu/julia/stable/faq/#Complex-numbers for more details.",
            ),
        )
    end

    thunk = Enzyme.Compiler.thunk(
        opt_mi,
        FA,
        A,
        tt′,
        Val(API.DEM_ReverseModeCombined),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        RABI,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#

    if A0 <: Active
        args = (args..., Compiler.default_adjoint(rt))
    end
    thunk(f, args...)
end

"""
    autodiff(mode::Mode, f, ::Type{A}, args::Annotation...)

Like [`autodiff`](@ref) but will try to extend f to an annotation, if needed.
"""
@inline function autodiff(
    mode::CMode,
    f::F,
    args::Vararg{Annotation,Nargs},
) where {F,CMode<:Mode,Nargs}
    autodiff(EnzymeCore.set_err_if_func_written(mode), Const(f), args...)
end
@inline function autodiff(
    mode::CMode,
    f::F,
    ::Type{RT},
    args::Vararg{Annotation,Nargs},
) where {F,RT<:Annotation,CMode<:Mode,Nargs}
    autodiff(EnzymeCore.set_err_if_func_written(mode), Const(f), RT, args...)
end

"""
    autodiff(mode::Mode, f, args...)

Like [`autodiff`](@ref) but will try to guess the activity of the return value.
"""
@inline function autodiff(
    mode::CMode,
    f::FA,
    args::Vararg{Annotation,Nargs},
) where {FA<:Annotation,CMode<:Mode,Nargs}
    tt = vaEltypeof(args...)
    rt = Compiler.primal_return_type(
        mode isa ForwardMode ? Forward : Reverse,
        eltype(FA),
        tt,
    )
    A = guess_activity(rt, mode)
    autodiff(mode, f, A, args...)
end

"""
    autodiff(::ForwardMode, f, Activity, args::Annotation...)

Auto-differentiate function `f` at arguments `args` using forward mode.

`args` may be numbers, arrays, structs of numbers, structs of arrays and so
on. Enzyme will only differentiate in respect to arguments that are wrapped
in a [`Duplicated`](@ref) or similar argument. Unlike reverse mode in
[`autodiff`](@ref), [`Active`](@ref) arguments are not allowed here, since
all derivative results of immutable objects will be returned and should
instead use [`Duplicated`](@ref) or variants like [`DuplicatedNoNeed`](@ref).

`Activity` is the Activity of the return value, it may be:
* `Const` if the return is not to be differentiated with respect to
* `Duplicated`, if the return is being differentiated with respect to
* `BatchDuplicated`, like `Duplicated`, but computing multiple derivatives
  at once. All batch sizes must be the same for all arguments.

Example returning both original return and derivative:

```jldoctest
f(x) = x*x
res, ∂f_∂x = autodiff(ForwardWithPrimal, f, Duplicated, Duplicated(3.14, 1.0))

# output

(6.28, 9.8596)
```

Example returning just the derivative:

```jldoctest
f(x) = x*x
∂f_∂x = autodiff(Forward, f, Duplicated, Duplicated(3.14, 1.0))

# output

(6.28,)
```
"""
@inline function autodiff(
    mode::ForwardMode{ReturnPrimal,RABI,ErrIfFuncWritten,RuntimeActivity,StrongZero},
    f::FA,
    ::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
} where {ReturnPrimal,RABI<:ABI,Nargs,ErrIfFuncWritten,RuntimeActivity,StrongZero}
    if any_active(args...)
        throw(ErrorException("Active arguments not allowed in forward mode"))
    end
    tt′ = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    if A <: DuplicatedNoNeed || A <: BatchDuplicatedNoNeed
        throw(
            ErrorException(
                "`DuplicatedNoNeed` passed in as return activity for Forward Mode AD is no longer returning or avoiding the primal.\nPlease use autodiff(Forward, ...) or autodiff(ForwardWithPrimal, ...)",
            ),
        )
    end
    RT = if A <: Duplicated && width != 1
        if A isa UnionAll
            BatchDuplicated{T,width} where {T}
        else
            BatchDuplicated{eltype(A),width}
        end
    elseif A <: DuplicatedNoNeed && width != 1
        if A isa UnionAll
            BatchDuplicatedNoNeed{T,width} where {T}
        else
            BatchDuplicatedNoNeed{eltype(A),width}
        end
    else
        A
    end

    ModifiedBetween = Val(falses_from_args(Nargs + 1))

    tt = vaEltypeof(args...)

    opt_mi = if RABI <: NonGenABI
        my_methodinstance(Forward, eltype(FA), tt)
    else
        Val(0)
    end

    thunk = Enzyme.Compiler.thunk(
        opt_mi,
        FA,
        RT,
        tt′,
        Val(API.DEM_ForwardMode),
        Val(width), #=Mode=#
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        RABI,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#
    thunk(f, args...)
end

"""
    autodiff_deferred(::ReverseMode, f, Activity, args::Annotation...)

Same as [`autodiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(
    mode::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,RABI,Holomorphic,ErrIfFuncWritten},
    f::FA,
    ::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    Nargs,
    RABI<:ABI,
    Holomorphic,
    ErrIfFuncWritten,
    RuntimeActivity,
    StrongZero
}
    tt′ = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    tt = vaEltypeof(args...)

    FTy = Core.Typeof(f.val)

    A2 = A

    if A isa UnionAll
        rt = Compiler.primal_return_type(Reverse, FTy, tt)
        A2 = A{rt}
        if rt == Union{}
            rt = Nothing
        end
    else
        @assert A isa DataType
        rt = A
        if rt == Union{}
	    throw(ErrorException("Return type inferred to be Union{}. Giving up."))
        end
    end


    ModifiedBetweenT = falses_from_args(Nargs + 1)
    ModifiedBetween = Val(ModifiedBetweenT)

    if A <: Active
        if (!allocatedinline(rt) || rt isa Union) && rt != Union{}
            rs = ReverseModeSplit{
                    ReturnPrimal,
                    #=ReturnShadow=#false,
                    RuntimeActivity,
                    StrongZero,
                    width,
                    ModifiedBetweenT,
                    RABI,
                    Holomorphic,
                    ErrIfFuncWritten,
                    #=ShadowInit=#true
                }()
            TapeType = tape_type(rs, FA, Duplicated{rt},
                (tt′).parameters...)
            forward, adjoint = autodiff_deferred_thunk(
                rs,
                TapeType,
                FA,
                Duplicated{rt},
                (tt′).parameters...
            )
            res = forward(f, args...)
            tape = res[1]
            if ReturnPrimal
                return (adjoint(f, args..., tape)[1], res[2])
            else
                return adjoint(f, args..., tape)
            end
        end
    elseif A <: Duplicated ||
           A <: DuplicatedNoNeed ||
           A <: BatchDuplicated ||
           A <: BatchDuplicatedNoNeed ||
           A <: BatchDuplicatedFunc
        throw(ErrorException("Duplicated Returns not yet handled"))
    end

    if (A <: Active && rt <: Complex) && rt != Union{}
        if Holomorphic
            throw(
                ErrorException(
                    "Reverse-mode Active Holomorphic is not yet implemented in deferred codegen",
                ),
            )
        end

        throw(
            ErrorException(
                "Reverse-mode Active Complex return is ambiguous and requires more information to specify the desired result. See https://enzyme.mit.edu/julia/stable/faq/#Complex-numbers for more details.",
            ),
        )
    end

    adjoint_ptr = Compiler.deferred_codegen(
        FA,
        A,
        tt′,
        Val(API.DEM_ReverseModeCombined),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        UnknownTapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#

    thunk =
        Compiler.CombinedAdjointThunk{Ptr{Cvoid},FA,A2,tt′,width,ReturnPrimal}(adjoint_ptr)
    if A <: Active
        args = (args..., Compiler.default_adjoint(rt))
    elseif A <: Duplicated ||
           A <: DuplicatedNoNeed ||
           A <: BatchDuplicated ||
           A <: BatchDuplicatedNoNeed
        throw(ErrorException("Duplicated Returns not yet handled"))
    end
    thunk(f, args...)
end

"""
    autodiff_deferred(::ForwardMode, f, Activity, args::Annotation...)

Same as `autodiff(::ForwardMode, f, Activity, args...)` but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(
    mode::ForwardMode{ReturnPrimal,RABI,ErrIfFuncWritten,RuntimeActivity,StrongZero},
    f::FA,
    ::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    ReturnPrimal,
    FA<:Annotation,
    A<:Annotation,
    Nargs,
    RABI<:ABI,
    ErrIfFuncWritten,
    RuntimeActivity,
    StrongZero
}
    if any_active(args...)
        throw(ErrorException("Active arguments not allowed in forward mode"))
    end
    tt′ = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A <: DuplicatedNoNeed || A <: BatchDuplicatedNoNeed
        throw(
            ErrorException(
                "Return activity `DuplicatedNoNeed` is no longer now returning or avoiding the primal is passed in for Forward Mode AD.\nPlease use autodiff(Forward, ...) or autodiff(ForwardWithPrimal, ...)",
            ),
        )
    end
    RT = if A <: Duplicated && width != 1
        if A isa UnionAll
            BatchDuplicated{T,width} where {T}
        else
            BatchDuplicated{eltype(A),width}
        end
    elseif A <: DuplicatedNoNeed && width != 1
        if A isa UnionAll
            BatchDuplicatedNoNeed{T,width} where {T}
        else
            BatchDuplicatedNoNeed{eltype(A),width}
        end
    else
        A
    end
    tt = vaEltypeof(args...)

    FT = Core.Typeof(f.val)

    if RT isa UnionAll
        rt = Compiler.primal_return_type(Forward, FT, tt)
	if rt == Union{}
	   rt = Nothing
	end
	rt = RT{rt}
    else
        @assert RT isa DataType
        rt = RT
    end

    if eltype(rt) == Union{}
        error("Return type inferred to be Union{}. Giving up.")
    end

    if RT <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end

    ModifiedBetween = Val(falses_from_args(Nargs + 1))

    adjoint_ptr = Compiler.deferred_codegen(
        Core.Typeof(f),
        rt,
        tt′,
        Val(API.DEM_ForwardMode),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        UnknownTapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#
    thunk = Compiler.ForwardModeThunk{Ptr{Cvoid},FA,rt,tt′,width,ReturnPrimal}(adjoint_ptr)
    thunk(f, args...)
end

"""
    autodiff_thunk(::ReverseModeSplit, ftype, Activity, argtypes::Type{<:Annotation}...)

Provide the split forward and reverse pass functions for annotated function type
ftype when called with args of type `argtypes` when using reverse mode.

`Activity` is the Activity of the return value, it may be `Const`, `Active`,
or `Duplicated` (or its variants `DuplicatedNoNeed`, `BatchDuplicated`, and
`BatchDuplicatedNoNeed`).

The forward function will return a tape, the primal (or nothing if not requested),
and the shadow (or nothing if not a `Duplicated` variant), and tapes the corresponding
type arguements provided.

The reverse function will return the derivative of `Active` arguments, updating the `Duplicated`
arguments in place. The same arguments to the forward pass should be provided, followed by
the adjoint of the return (if the return is active), and finally the tape from the forward pass.

Example:

```jldoctest

A = [2.2]; ∂A = zero(A)
v = 3.3

function f(A, v)
    res = A[1] * v
    A[1] = 0
    res
end

forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{typeof(A)}, Active{typeof(v)})

tape, result, shadow_result  = forward(Const(f), Duplicated(A, ∂A), Active(v))
_, ∂v = reverse(Const(f), Duplicated(A, ∂A), Active(v), 1.0, tape)[1]

result, ∂v, ∂A

# output

(7.26, 2.2, [3.3])
```
"""
@inline function autodiff_thunk(
    mode::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        StrongZero,
        Width,
        ModifiedBetweenT,
        RABI,
        #=Holomorphic=#false,
        ErrIfFuncWritten,
        ShadowInit
    },
    ::Type{FA},
    ::Type{A},
    args::Vararg{Type{<:Annotation},Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    ReturnShadow,
    Width,
    ModifiedBetweenT,
    RABI<:ABI,
    Nargs,
    ErrIfFuncWritten,
    ShadowInit,
    RuntimeActivity,
    StrongZero
}
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Nargs + 1))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    tt = Tuple{map(eltype, args)...}

    tt′ = Tuple{args...}
    opt_mi = if RABI <: NonGenABI
        my_methodinstance(Reverse, eltype(FA), tt)
    else
        Val(0)
    end
    Enzyme.Compiler.thunk(
        opt_mi,
        FA,
        A,
        tt′,
        Val(API.DEM_ReverseModeGradient),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(ShadowInit),
        RABI,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#
end

"""
    autodiff(::Function, ::Mode, args...)

Specialization of [`autodiff`](@ref) to handle do argument closures.

```jldoctest

autodiff(Reverse, Active(3.1)) do x
  return x*x
end

# output
((6.2,),)
```
"""
@inline function autodiff(
    f::Function,
    m::MMode,
    ::Type{A},
    args::Vararg{Annotation,Nargs},
) where {A<:Annotation,Nargs,MMode<:Mode}
    autodiff(m, f, A, args...)
end
@inline function autodiff(
    f::Function,
    m::MMode,
    args::Vararg{Annotation,Nargs},
) where {Nargs,MMode<:Mode}
    autodiff(m, f, args...)
end

"""
    autodiff_thunk(::ForwardMode, ftype, Activity, argtypes::Type{<:Annotation}...)

Provide the thunk forward mode function for annotated function type
ftype when called with args of type `argtypes`.

`Activity` is the Activity of the return value, it may be `Const` or `Duplicated`
(or its variants `DuplicatedNoNeed`, `BatchDuplicated`, and`BatchDuplicatedNoNeed`).

The forward function will return the shadow (or nothing if not a `Duplicated` variant)
and the primal (if requested).

Example returning both the return derivative and original return:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
forward = autodiff_thunk(ForwardWithPrimal, Const{typeof(f)}, Duplicated, Duplicated{Float64})
∂f_∂x, res = forward(Const(f), Duplicated(3.14, 1.0))

# output

(6.28, 9.8596)
```

Example returning just the derivative:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
forward = autodiff_thunk(Forward, Const{typeof(f)}, Duplicated, Duplicated{Float64})
∂f_∂x, = forward(Const(f), Duplicated(3.14, 1.0))

# output

(6.28,)
```
"""
@inline function autodiff_thunk(
    mode::ForwardMode{ReturnPrimal,RABI,ErrIfFuncWritten,RuntimeActivity,StrongZero},
    ::Type{FA},
    ::Type{A},
    args::Vararg{Type{<:Annotation},Nargs},
) where {
    ReturnPrimal,
    FA<:Annotation,
    A<:Annotation,
    RABI<:ABI,
    Nargs,
    ErrIfFuncWritten,
    RuntimeActivity,
    StrongZero
}
    width = same_or_one(1, A, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    if A <: Active
        throw(ErrorException("Active Returns not allowed in forward mode"))
    end
    if A <: DuplicatedNoNeed || A <: BatchDuplicatedNoNeed
        throw(
            ErrorException(
                "Return activity `DuplicatedNoNeed` is no longer now returning or avoiding the primal is passed in for Forward Mode AD.\nPlease use autodiff(Forward, ...) or autodiff(ForwardWithPrimal, ...)",
            ),
        )
    end

    ModifiedBetween = Val(falses_from_args(Nargs + 1))

    tt = Tuple{map(eltype, args)...}

    tt′ = Tuple{args...}
    opt_mi = if RABI <: NonGenABI
        my_methodinstance(Forward, eltype(FA), tt)
    else
        Val(0)
    end
    results = Enzyme.Compiler.thunk(
        opt_mi,
        FA,
        A,
        tt′,
        Val(API.DEM_ForwardMode),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        RABI,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#
end

@inline function tape_type(
    mode::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        StrongZero,
        Width,
        ModifiedBetweenT,
        RABI,
        #=Holomorphic=#false,
        ErrIfFuncWritten,
        ShadowInit,
    },
    ::Type{FA},
    ::Type{A},
    args::Vararg{Type{<:Annotation},Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    ReturnShadow,
    Width,
    ModifiedBetweenT,
    RABI<:ABI,
    Nargs,
    ErrIfFuncWritten,
    RuntimeActivity,
    StrongZero,
    ShadowInit,
}
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Nargs + 1))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    TT = Tuple{args...}

    primal_tt = Tuple{map(eltype, args)...}
    opt_mi = if RABI <: NonGenABI
        my_methodinstance(Forward, eltype(FA), primal_tt)
    else
        Val(0)
    end
    nondef = Enzyme.Compiler.thunk(
        opt_mi,
        FA,
        A,
        TT,
        Val(API.DEM_ReverseModeGradient),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(ShadowInit),
        RABI,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    )
    if nondef[1] isa Enzyme.Compiler.PrimalErrorThunk
        return Nothing
    else
        TapeType = EnzymeRules.tape_type(nondef[1])
        return TapeType
    end
end

const tape_cache = Dict{UInt,Type}()

const tape_cache_lock = ReentrantLock()

import .Compiler: remove_innerty, UnknownTapeType

@inline function tape_type(
    parent_job::Union{GPUCompiler.CompilerJob,Nothing},
    mode::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        StrongZero,
        Width,
        ModifiedBetweenT,
        RABI,
        #=Holomorphic=#false,
        #=ErrIfFuncWritten=#false,
        #=ShadowInit=#false,
    },
    ::Type{FA},
    ::Type{A},
    args::Vararg{Type{<:Annotation},Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    ReturnShadow,
    Width,
    ModifiedBetweenT,
    RABI<:ABI,
    Nargs,
    RuntimeActivity,
    StrongZero,
}
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = falses_from_args(Val(1), args...)
    else
        ModifiedBetween = ModifiedBetweenT
    end

    @assert ReturnShadow
    TT = Tuple{args...}

    primal_tt = Tuple{map(eltype, args)...}

    mi = my_methodinstance(parent_job === nothing ? Reverse : GPUCompiler.get_interpreter(parent_job), eltype(FA), primal_tt)

    target = Compiler.EnzymeTarget()
    params = Compiler.EnzymeCompilerParams(
        Tuple{FA,TT.parameters...},
        API.DEM_ReverseModeGradient,
        width,
        Compiler.remove_innerty(A),
        true,
        false,
        ModifiedBetweenT, #=abiwrap=#
        ReturnPrimal,
        false,
        Compiler.UnknownTapeType,
        RABI,
        false, #=errifwritte=#
        RuntimeActivity,
        StrongZero
    )

    if parent_job !== nothing
        target = GPUCompiler.nest_target(target, parent_job.config.target)
        params = GPUCompiler.nest_params(params, parent_job.config.params)
    end

    job = GPUCompiler.CompilerJob(mi, GPUCompiler.CompilerConfig(target, params; kernel = false))


    key = hash(parent_job, hash(job))

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(tape_cache_lock)

    try
        obj = get(tape_cache, key, nothing)
        # If the tape is not cached, compile it
        if obj === nothing

	    ts_ctx = Compiler.JuliaContext()
	    ctx = Compiler.context(ts_ctx)
	    Compiler.activate(ctx)
            try
                _, meta = GPUCompiler.compile(:llvm, job)
                obj = meta.TapeType
                tape_cache[key] = obj
		obj
    	    finally
                Compiler.deactivate(ctx)
		Compiler.dispose(ts_ctx)
            end
	else
	    obj
        end
    finally
        unlock(tape_cache_lock)
    end
end

"""
    autodiff_deferred_thunk(::ReverseModeSplit, TapeType::Type, ftype::Type{<:Annotation}, Activity::Type{<:Annotation}, argtypes::Type{<:Annotation}...)

Provide the split forward and reverse pass functions for annotated function type
ftype when called with args of type `argtypes` when using reverse mode.

`Activity` is the Activity of the return value, it may be `Const`, `Active`,
or `Duplicated` (or its variants `DuplicatedNoNeed`, `BatchDuplicated`, and
`BatchDuplicatedNoNeed`).

The forward function will return a tape, the primal (or nothing if not requested),
and the shadow (or nothing if not a `Duplicated` variant), and tapes the corresponding
type arguements provided.

The reverse function will return the derivative of `Active` arguments, updating the `Duplicated`
arguments in place. The same arguments to the forward pass should be provided, followed by
the adjoint of the return (if the return is active), and finally the tape from the forward pass.

Example:

```jldoctest

A = [2.2]; ∂A = zero(A)
v = 3.3

function f(A, v)
    res = A[1] * v
    A[1] = 0
    res
end

TapeType = tape_type(ReverseSplitWithPrimal, Const{typeof(f)}, Active, Duplicated{typeof(A)}, Active{typeof(v)})
forward, reverse = autodiff_deferred_thunk(ReverseSplitWithPrimal, TapeType, Const{typeof(f)}, Active{Float64}, Duplicated{typeof(A)}, Active{typeof(v)})

tape, result, shadow_result  = forward(Const(f), Duplicated(A, ∂A), Active(v))
_, ∂v = reverse(Const(f), Duplicated(A, ∂A), Active(v), 1.0, tape)[1]

result, ∂v, ∂A

# output

(7.26, 2.2, [3.3])
```
"""
@inline function autodiff_deferred_thunk(
    mode::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        StrongZero,
        Width,
        ModifiedBetweenT,
        RABI,
        #=Holomorphic=#false,
        ErrIfFuncWritten,
        ShadowInit,
    },
    tt::Type{TapeType},
    fa::Type{FA},
    a2::Type{A2},
    args::Vararg{Type{<:Annotation},Nargs},
) where {
    FA<:Annotation,
    A2<:Annotation,
    TapeType,
    ReturnPrimal,
    ReturnShadow,
    Width,
    ModifiedBetweenT,
    RABI<:ABI,
    Nargs,
    ErrIfFuncWritten,
    RuntimeActivity,
    StrongZero,
    ShadowInit
}
    @assert RABI == FFIABI
    width = if Width == 0
        w = same_or_one(1, args...)
        if w == 0
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end
        w
    else
        Width
    end

    if ModifiedBetweenT === true
        ModifiedBetween = Val(falses_from_args(Nargs + 1))
    else
        ModifiedBetween = Val(ModifiedBetweenT)
    end

    TT = Tuple{args...}

    rt = if A2 isa UnionAll
        primal_tt = Tuple{map(eltype, args)...}
	rt0 = Compiler.primal_return_type(Reverse, eltype(FA), primal_tt)
	A2{rt0}
    else
	A2
    end

    primal_ptr = Compiler.deferred_codegen(
        FA,
        rt,
        TT,
        Val(API.DEM_ReverseModePrimal),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(ShadowInit),
        TapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#
    adjoint_ptr = Compiler.deferred_codegen(
        FA,
        rt,
        TT,
        Val(API.DEM_ReverseModeGradient),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        TapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
        Val(StrongZero)
    ) #=ShadowInit=#

    RT = if A2 <: Duplicated && width != 1
        if A2 isa UnionAll
            BatchDuplicated{T,width} where {T}
        else
            BatchDuplicated{eltype(A2),width}
        end
    elseif A2 <: DuplicatedNoNeed && width != 1
        if A2 isa UnionAll
            BatchDuplicatedNoNeed{T,width} where {T}
        else
            BatchDuplicatedNoNeed{eltype(A2),width}
        end
    elseif A2 <: MixedDuplicated && width != 1
        if A2 isa UnionAll
            BatchMixedDuplicated{T,width} where {T}
        else
            BatchMixedDuplicated{eltype(A2),width}
        end
    else
        A2
    end

    aug_thunk =
        Compiler.AugmentedForwardThunk{Ptr{Cvoid},FA,rt,TT,width,ReturnPrimal,TapeType}(
            primal_ptr,
        )
    adj_thunk = Compiler.AdjointThunk{Ptr{Cvoid},FA,rt,TT,width,TapeType}(adjoint_ptr)
    aug_thunk, adj_thunk
end

include("sugar.jl")

function _import_frule end # defined in EnzymeChainRulesCoreExt extension

"""
    import_frule(::fn, tys...)

Automatically import a `ChainRulesCore.frule` as a custom forward mode `EnzymeRule`. When called in batch mode, this
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

Automatically import a `ChainRules.rrule` as a custom reverse mode EnzymeRule. When called in batch mode, this
will end up calling the primal multiple times which results in slower code. This macro assumes that the underlying
function to be imported is read-only, and returns a Duplicated or Const object. This macro also assumes that the
inputs permit a `.+=` operation and that the output has a valid `Enzyme.make_zero` function defined. It also assumes
that `overwritten(x)` accurately describes if there is any non-preserved data from forward to reverse, not just
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

if VERSION < v"1.12.0"
    include("precompile.jl")
end

function __init__()
    @static if VERSION ≥ v"1.12-"
        if ccall(:jl_generating_output, Cint, ()) == 1
            @warn """
            Enzyme.jl currently doesn't support versions of Julia 1.12 or newer. We are
            actively working on adding support for newer versions of Julia. For the time
            being we recommend using 1.11 or LTS.

            For latest updates, check the status of support for Julia 1.12+ at
            https://github.com/EnzymeAD/Enzyme.jl/issues/2665.
            """ maxlog = 1
        end
    end

    return nothing
end

end # module
