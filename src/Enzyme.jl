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
    ReverseMode,
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
    ReverseMode,
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
    within_autodiff,
    WithPrimal,
    NoPrimal
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
    WithPrimal,
    NoPrimal,
    within_autodiff

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
    make_zero!
export autodiff,
    autodiff_deferred,
    autodiff_thunk,
    autodiff_deferred_thunk,
    tape_type,
    make_zero,
    make_zero!

export jacobian, gradient, gradient!, hvp, hvp!, hvp_and_gradient!
export markType, batch_size, onehot, chunkedonehot

using LinearAlgebra
import SparseArrays
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

import .Compiler: CompilationException

@inline function falses_from_args(N)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        false
    end
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
    rmode::ReverseMode{ReturnPrimal,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten},
    f::FA,
    ::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
    ReturnPrimal,
    RuntimeActivity,
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

    tt = Tuple{map(T -> eltype(Core.Typeof(T)), args)...}

    FTy = Core.Typeof(f.val)

    rt = if A isa UnionAll
        Compiler.primal_return_type(rmode, Val(codegen_world_age(FTy, tt)), FTy, tt)
    else
        eltype(A)
    end

    if A <: Active
        if (!allocatedinline(rt) || rt isa Union) && rt != Union{}
            forward, adjoint = autodiff_thunk(
                ReverseModeSplit{
                    ReturnPrimal,
                    #=ReturnShadow=#false,
                    RuntimeActivity,
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
    elseif A <: Duplicated ||
           A <: DuplicatedNoNeed ||
           A <: BatchDuplicated ||
           A <: BatchDuplicatedNoNeed ||
           A <: BatchDuplicatedFunc
        throw(ErrorException("Duplicated Returns not yet handled"))
    end

    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(FTy, tt))
    end

    if (A <: Active && rt <: Complex) && rt != Union{}
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
    ) #=ShadowInit=#

    if A <: Active
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
    tt = Tuple{map(T -> eltype(Core.Typeof(T)), args)...}
    rt = if mode isa ReverseMode
        Compiler.primal_return_type(
            mode,
            Val(codegen_world_age(eltype(FA), tt)),
            eltype(FA),
            tt,
        )
    else
        Core.Compiler.return_type(f.val, tt)
    end
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
    ::ForwardMode{ReturnPrimal,RABI,ErrIfFuncWritten,RuntimeActivity},
    f::FA,
    ::Type{A},
    args::Vararg{Annotation,Nargs},
) where {
    FA<:Annotation,
    A<:Annotation,
} where {ReturnPrimal,RABI<:ABI,Nargs,ErrIfFuncWritten,RuntimeActivity}
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

    tt = Tuple{map(T -> eltype(Core.Typeof(T)), args)...}

    opt_mi = if RABI <: NonGenABI
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(Core.Typeof(f.val), tt))
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
    ) #=ShadowInit=#
    thunk(f, args...)
end

"""
    autodiff_deferred(::ReverseMode, f, Activity, args::Annotation...)

Same as [`autodiff`](@ref) but uses deferred compilation to support usage in GPU
code, as well as high-order differentiation.
"""
@inline function autodiff_deferred(
    rmode::ReverseMode{ReturnPrimal,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten},
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
}
    tt′ = vaTypeof(args...)
    width = same_or_one(1, args...)
    if width == 0
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end
    tt = Tuple{map(T -> eltype(Core.Typeof(T)), args)...}

    FTy = Core.Typeof(f.val)
    world = codegen_world_age(FTy, tt)

    A2 = A

    if A isa UnionAll
        rt = Compiler.primal_return_type(rmode, Val(world), FTy, tt)
        rt = Core.Compiler.return_type(f.val, tt)
	A2 = A{rt}
	if rt == Union{}
	    throw(ErrorException("Return type inferred to be Union{}. Giving up."))
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
        Val(world),
        FA,
        Val(tt′),
        Val(A),
        Val(API.DEM_ReverseModeCombined),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        UnknownTapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
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
    ::ForwardMode{ReturnPrimal,RABI,ErrIfFuncWritten,RuntimeActivity},
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
    tt = Tuple{map(T -> eltype(Core.Typeof(T)), args)...}

    world = codegen_world_age(Core.Typeof(f.val), tt)

    if RT isa UnionAll
        rt = Core.Compiler.return_type(f.val, tt)
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
        Val(world),
        FA,
        Val(tt′),
        Val(rt),
        Val(API.DEM_ForwardMode),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        UnknownTapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
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
    rs::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
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
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(eltype(FA), tt))
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

The forward function will return the primal (if requested) and the shadow
(or nothing if not a `Duplicated` variant).

Example returning both the return derivative and original return:

```jldoctest
a = 4.2
b = [2.2, 3.3]; ∂f_∂b = zero(b)
c = 55; d = 9

f(x) = x*x
forward = autodiff_thunk(ForwardWithPrimal, Const{typeof(f)}, Duplicated, Duplicated{Float64})
res, ∂f_∂x = forward(Const(f), Duplicated(3.14, 1.0))

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
∂f_∂x = forward(Const(f), Duplicated(3.14, 1.0))

# output

(6.28,)
```
"""
@inline function autodiff_thunk(
    ::ForwardMode{ReturnPrimal,RABI,ErrIfFuncWritten,RuntimeActivity},
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
        Compiler.fspec(eltype(FA), tt′)
    else
        Val(codegen_world_age(eltype(FA), tt))
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
    ) #=ShadowInit=#
end

@inline function tape_type(
    ::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
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
        Compiler.fspec(eltype(FA), TT)
    else
        Val(codegen_world_age(eltype(FA), primal_tt))
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
    ) #=ShadowInit=#
    if nondef[1] isa Enzyme.Compiler.PrimalErrorThunk
        return Nothing
    else
        TapeType = EnzymeRules.tape_type(nondef[1])
        return TapeType
    end
end

const tape_cache = Dict{UInt,Type}()

const tape_cache_lock = ReentrantLock()

import .Compiler: fspec, remove_innerty, UnknownTapeType

@inline function tape_type(
    parent_job::Union{GPUCompiler.CompilerJob,Nothing},
    ::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
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

    world = codegen_world_age(eltype(FA), primal_tt)

    mi = Compiler.fspec(eltype(FA), TT, world)

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
    )
    job = Compiler.CompilerJob(mi, Compiler.CompilerConfig(target, params; kernel = false))


    key = hash(parent_job, hash(job))

    # NOTE: no use of lock(::Function)/@lock/get! to keep stack traces clean
    lock(tape_cache_lock)

    try
        obj = get(tape_cache, key, nothing)
        if obj === nothing

            Compiler.JuliaContext() do ctx
                _, meta = Compiler.codegen(:llvm, job; optimize = false, parent_job)
                obj = meta.TapeType
                tape_cache[key] = obj
            end
        end
        obj
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

    primal_tt = Tuple{map(eltype, args)...}
    world = codegen_world_age(eltype(FA), primal_tt)

    primal_ptr = Compiler.deferred_codegen(
        Val(world),
        FA,
        Val(TT),
        Val(Compiler.remove_innerty(A2)),
        Val(API.DEM_ReverseModePrimal),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(ShadowInit),
        TapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
    ) #=ShadowInit=#
    adjoint_ptr = Compiler.deferred_codegen(
        Val(world),
        FA,
        Val(TT),
        Val(Compiler.remove_innerty(A2)),
        Val(API.DEM_ReverseModeGradient),
        Val(width),
        ModifiedBetween,
        Val(ReturnPrimal),
        Val(false),
        TapeType,
        Val(ErrIfFuncWritten),
        Val(RuntimeActivity),
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

    rt = if RT isa UnionAll
        RT{Core.Compiler.return_type(Tuple{eltype(FA),map(eltype, args)...})}
    else
        @assert RT isa DataType
        RT
    end

    aug_thunk =
        Compiler.AugmentedForwardThunk{Ptr{Cvoid},FA,rt,TT,width,ReturnPrimal,TapeType}(
            primal_ptr,
        )
    adj_thunk = Compiler.AdjointThunk{Ptr{Cvoid},FA,rt,TT,width,TapeType}(adjoint_ptr)
    aug_thunk, adj_thunk
end

# White lie, should be `Core.LLVMPtr{Cvoid, 0}` but that's not supported by ccallable
Base.@ccallable function __enzyme_float(x::Ptr{Cvoid})::Cvoid
    return nothing
end

Base.@ccallable function __enzyme_double(x::Ptr{Cvoid})::Cvoid
    return nothing
end

@inline function markType(::Type{T}, ptr::Ptr{Cvoid}) where {T}
    markType(Base.unsafe_convert(Ptr{T}, ptr))
end

@inline function markType(data::Array{T}) where {T}
    GC.@preserve data markType(pointer(data))
end

# TODO(WM): We record the type of a single index here, we could give it a range
@inline function markType(data::SubArray)
    GC.@preserve data markType(pointer(data))
end

@inline function markType(data::Ptr{Float32})
    @static if sizeof(Int) == sizeof(Int64)
        Base.llvmcall(
            (
                "declare void @__enzyme_float(i8* nocapture) nounwind define void @c(i64 %q) nounwind alwaysinline { %p = inttoptr i64 %q to i8* call void @__enzyme_float(i8* %p) ret void }",
                "c",
            ),
            Cvoid,
            Tuple{Ptr{Float32}},
            data,
        )
    else
        Base.llvmcall(
            (
                "declare void @__enzyme_float(i8* nocapture) nounwind define void @c(i32 %q) nounwind alwaysinline { %p = inttoptr i32 %q to i8* call void @__enzyme_float(i8* %p) ret void }",
                "c",
            ),
            Cvoid,
            Tuple{Ptr{Float32}},
            data,
        )
    end
    nothing
end

@inline function markType(data::Ptr{Float64})
    @static if sizeof(Int) == sizeof(Int64)
        Base.llvmcall(
            (
                "declare void @__enzyme_double(i8* nocapture) nounwind define void @c(i64 %q) nounwind alwaysinline { %p = inttoptr i64 %q to i8* call void @__enzyme_double(i8* %p) ret void }",
                "c",
            ),
            Cvoid,
            Tuple{Ptr{Float64}},
            data,
        )
    else
        Base.llvmcall(
            (
                "declare void @__enzyme_double(i8* nocapture) nounwind define void @c(i32 %q) nounwind alwaysinline { %p = inttoptr i32 %q to i8* call void @__enzyme_double(i8* %p) ret void }",
                "c",
            ),
            Cvoid,
            Tuple{Ptr{Float64}},
            data,
        )
    end
    nothing
end

@inline function onehot(x)
    N = length(x)
    ntuple(Val(N)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx = 1:N
            @inbounds res[idx] = (i == idx) ? 1.0 : 0.0
        end
        return res
    end
end
@inline function onehot(x, start, endl)
    ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        res = similar(x)
        for idx = 1:length(x)
            @inbounds res[idx] = (i + start - 1 == idx) ? 1.0 : 0.0
        end
        return res
    end
end

@inline function onehot(::Type{NTuple{N,T}}) where {T,N}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i == idx) ? 1.0 : 0.0
        end
    end
end
@inline function onehot(x::NTuple{N,T}) where {T,N}
    onehot(NTuple{N,T})
end
@inline function onehot(x::NTuple{N,T}, start, endl) where {T,N}
    ntuple(Val(endl - start + 1)) do i
        Base.@_inline_meta
        ntuple(Val(N)) do idx
            Base.@_inline_meta
            return (i + start - 1 == idx) ? 1.0 : 0.0
        end
    end
end

@inline function onehot(x::AbstractFloat)
    return (one(x),)
end

"""
    gradient(::ReverseMode, f, args...)

Compute the gradient of a real-valued function `f` using reverse mode.
For each differentiable argument, this function will allocate and return new derivative object, returning
a tuple of derivatives for each argument. If an argument is not differentiable, the element of the returned
tuple with be nothing.

In reverse mode (here), the derivatives will be the same type as the original argument.

This is a structure gradient. For a struct `x` it returns another instance of the same type,
whose fields contain the components of the gradient.
In the result, `grad.a` contains `∂f/∂x.a` for any differential `x.a`,
while `grad.c == x.c` for other types.

Examples:

```jldoctest gradient
f(x) = x[1]*x[2]

grad = gradient(Reverse, f, [2.0, 3.0])

# output
([3.0, 2.0],)
```

```jldoctest gradient
grad = gradient(Reverse, only ∘ f, (a = 2.0, b = [3.0], c = "str"))

# output

((a = 3.0, b = [2.0], c = "str"),)
```

```jldoctest gradient
mul(x, y) = x[1]*y[1]

grad = gradient(Reverse, mul, [2.0], [3.0])

# output
([3.0], [2.0])
```

```jldoctest gradient

grad = gradient(Reverse, mul, [2.0], Const([3.0]))

# output
([3.0], nothing)
```

If passing a mode that returns the primal (e.g. ReverseWithPrimal), the return type will instead be
a tuple where the first element contains the derivatives, and the second element contains the result of the original computation.

```jldoctest gradient

grad = gradient(ReverseWithPrimal, f, [2.0, 3.0])

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```
```jldoctest gradient

grad = gradient(ReverseWithPrimal, mul, [2.0], [3.0])

# output
(derivs = ([3.0], [2.0]), val = 6.0)
```

```jldoctest gradient
grad = gradient(ReverseWithPrimal, mul, [2.0], Const([3.0]))

# output
(derivs = ([3.0], nothing), val = 6.0)
```

"""
@generated function gradient(
    rm::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten},
    f::F,
    x::ty_0,
    args::Vararg{Any,N},
) where {F,ty_0,ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten,N}
    toemit = Expr[quote
        act_0 =
            !(x isa Enzyme.Const) &&
            Compiler.active_reg_inner(Core.Typeof(x), (), nothing, Val(true)) ==
            Compiler.ActiveState #=justActive=#
    end]
    rargs = Union{Symbol,Expr}[:x]
    acts = Symbol[Symbol("act_0")]

    for i = 1:N
        argidx = quote
            args[$i]
        end
        push!(rargs, argidx)
        sym = Symbol("act_$i")
        push!(acts, sym)
        push!(
            toemit,
            quote
                $sym =
                    !($argidx isa Enzyme.Const) &&
                    Compiler.active_reg_inner(
                        Core.Typeof($argidx),
                        (),
                        nothing,
                        Val(true),
                    ) == Compiler.ActiveState #=justActive=#
            end,
        )
    end

    idx = 0
    shadows = Symbol[]
    enz_args = Expr[]
    resargs = Expr[]
    for (arg, act) in zip(rargs, acts)
        shad = Symbol("shad_$idx")
        push!(shadows, shad)
        push!(toemit, quote
            $shad = if $arg isa Enzyme.Const
                nothing
            elseif $act
                Ref(make_zero($arg))
            else
                make_zero($arg)
            end
        end)
        push!(enz_args, quote
            if $arg isa Enzyme.Const
                $arg
            elseif $act
                MixedDuplicated($arg, $shad)
            else
                Duplicated($arg, $shad)
            end
        end)
        push!(resargs, quote
            if $arg isa Enzyme.Const
                nothing
            elseif $act
                $shad[]
            else
                $shad
            end
        end)
        idx += 1
    end
    push!(toemit, quote
        res = autodiff(rm, f, Active, $(enz_args...))
    end)

    if ReturnPrimal
        return quote
            Base.@_inline_meta
            $(toemit...)
            (; derivs = ($(resargs...),), val = res[2])
        end
    else
        return quote
            Base.@_inline_meta
            $(toemit...)
            ($(resargs...),)
        end
    end
end

"""
    gradient!(::ReverseMode, dx, f, x)

Compute the gradient of an array-input function `f` using reverse mode,
storing the derivative result in an existing array `dx`.
Both `x` and `dx` must be `Array`s of the same type.

Example:

```jldoctest gradip
f(x) = x[1]*x[2]

dx = [0.0, 0.0]
gradient!(Reverse, dx, f, [2.0, 3.0])

# output
([3.0, 2.0],)
```

```jldoctest gradip
dx = [0.0, 0.0]
gradient!(ReverseWithPrimal, dx, f, [2.0, 3.0])

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```
"""
@inline function gradient!(
    rm::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten},
    dx::X,
    f::F,
    x::X,
) where {X<:Array,F,ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}
    make_zero!(dx)
    res = autodiff(rm, f, Active, Duplicated(x, dx))
    return if ReturnPrimal
        (; derivs = (dx,), val = res[2])
    else
        (dx,)
    end
end

@inline function chunkedonehot(x, ::Val{chunk}) where {chunk}
    sz = length(x)
    num = ((sz + chunk - 1) ÷ chunk)
    ntuple(Val(num)) do i
        Base.@_inline_meta
        onehot(x, (i - 1) * chunk + 1, i == num ? sz : (i * chunk))
    end
end

@inline function chunkedonehot(x::AbstractFloat, ::Val{chunk}) where {chunk}
    return ((one(x),),)
end

@inline tupleconcat(x) = x
@inline tupleconcat(x, y) = (x..., y...)
@inline tupleconcat(x, y, z...) = (x..., tupleconcat(y, z...)...)

function create_shadows(::Nothing, x)
    return (onehot(x),)
end

function create_shadows(::Val{1}, x)
    return (onehot(x),)
end

function create_shadows(::Val{chunk}, x) where {chunk}
    return (chunkedonehot(x, Val(chunk)),)
end

struct TupleArray{T,Shape,Length,N} <: AbstractArray{T,N}
    data::NTuple{Length,T}
end
TupleArray(data::NTuple{Length,T}, Shape) where {Length,T} =
    TupleArray{T,Shape,Length,length(Shape)}(data)

@inline Base.eltype(::TupleArray{T}) where {T} = T
@inline Base.eltype(::Type{<:TupleArray{T}}) where {T} = T
@inline Base.size(::TupleArray{<:Any,Shape}) where {Shape} = Shape
@inline Base.ndims(::TupleArray{<:Any,<:Any,<:Any,N}) where {N} = N

function Base.convert(
    ::Type{Array{T,N}},
    X::TupleArray{T,Shape,Length,N},
) where {T,Shape,Length,N}
    vals = Array{T,N}(undef, Shape...)
    for i = 1:Length
        @inbounds val[i] = X.data[i]
    end
    return vals
end

function Base.getindex(a::TupleArray, args::Vararg{Int,N}) where {N}
    start = 0
    for i = 1:N
        start *= size(a, N - i + 1)
        start += (args[N-i+1] - 1)
    end
    start += 1
    return a.data[start]
end

@inline function tupstack(x, inshape, outshape)
    st = Base.stack(x)
    if length(outshape) == 1
        st
    else
        reshape(st, (inshape..., outshape...))
    end
end

"""
    gradient(::ForwardMode, f, x; shadows=onehot(x), chunk=nothing)

Compute the gradient of an array-input function `f` using forward mode. The
optional keyword argument `shadow` is a vector of one-hot vectors of type `x`
which are used to forward-propagate into the return. For performance reasons,
this should be computed once, outside the call to `gradient`, rather than
within this call.

Example:

```jldoctest gradfwd
f(x) = x[1]*x[2]

gradient(Forward, f, [2.0, 3.0])

# output

([3.0, 2.0],)
```

```jldoctest gradfwd
gradient(ForwardWithPrimal, f, [2.0, 3.0])

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```

```jldoctest gradfwd
gradient(Forward, f, [2.0, 3.0]; chunk=Val(1))

# output

([3.0, 2.0],)
```

```jldoctest gradfwd
gradient(ForwardWithPrimal, f, [2.0, 3.0]; chunk=Val(1))

# output
(derivs = ([3.0, 2.0],), val = 6.0)
```

For functions which return an AbstractArray or scalar, this function will return an AbstracttArray
whose shape is `(size(output)..., size(input)...)`. No guarantees are presently made
about the type of the AbstractArray returned by this function (which may or may not be the same
as the input AbstractArray if provided).

For functions who return other types, this function will retun an AbstractArray
of shape `size(input)` of values of the output type. 
```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = gradient(Forward, f, [2.0, 3.0, 4.0])

# output
([3.0 2.0 0.0; 0.0 1.0 1.0],)
```
"""
@inline function gradient(
    fm::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity},
    f,
    x;
    chunk::CS = nothing,
    shadows = create_shadows(chunk, x),
) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,CS}
    if length(shadows[1]) == 0
        return if ReturnPrimal
            (; derivs = (x,), val = f(x.val))
        else
            (x,)
        end
    end
    if chunk == Val(0)
        throw(ErrorException("Cannot differentiate with a batch size of 0"))
    end

    gradtup = if chunk == nothing
        resp = autodiff(fm, f, BatchDuplicated, BatchDuplicated(x, shadows[1]))

        res = values(resp[1])
        dres = if x isa AbstractFloat
            res[1]
        else
            res
        end
        if ReturnPrimal
            ((dres,), resp[2])
        else
            (dres,)
        end
    elseif chunk == Val(1)
        if ReturnPrimal
            rp = autodiff(fm, f, Duplicated, Duplicated(x, shadows[1][1]))
            dres1 = rp[1]
            fm2 = ForwardMode{false,ABI,ErrIfFuncWritten,RuntimeActivity}() #=ReturnPrimal=#

            res = ntuple(length(shadows[1]) - 1) do i
                autodiff(fm2, f, Duplicated, Duplicated(x, shadows[1][i+1]))[1]
            end
            gres = if x isa AbstractFloat
                dres1[1]
            else
                (dres1, res...)
            end
            ((gres,), rp[2])
        else
            res = ntuple(length(shadows[1])) do i
                autodiff(fm, f, Duplicated, Duplicated(x, shadows[1][i]))[1]
            end
            (if x isa AbstractFloat
                res[1]
            else
                res
            end,)
        end
    else
        if ReturnPrimal
            rp = autodiff(fm, f, BatchDuplicated, BatchDuplicated(x, shadows[1][1]))
            dres1 = values(rp[1])
            gres = if x isa AbstractFloat
                dres1[1]
            else
                fm2 = ForwardMode{false,ABI,ErrIfFuncWritten,RuntimeActivity}() #=ReturnPrimal=#
                tmp = ntuple(length(shadows[1]) - 1) do i
                    values(
                        autodiff(
                            fm2,
                            f,
                            BatchDuplicated,
                            BatchDuplicated(x, shadows[1][i+1]),
                        )[1],
                    )
                end
                tupleconcat(dres1, tmp...)
            end
            ((gres,), rp[2])
        else
            tmp = ntuple(length(shadows[1])) do i
                values(autodiff(fm, f, BatchDuplicated, BatchDuplicated(x, shadows[1][i]))[1])
            end
            res = tupleconcat(tmp...)
            (if x isa AbstractFloat
                res[1]
            else
                res
            end,)
        end
    end

    cols = if ReturnPrimal
        gradtup[1][1]
    else
        gradtup[1]
    end
    res = if x isa AbstractFloat
        cols
    elseif length(cols) > 0 && cols[1] isa AbstractArray && x isa AbstractArray
        inshape = size(x)
        outshape = size(cols[1])
        # st : outshape x total inputs
        tupstack(cols, outshape, inshape)
    elseif x isa AbstractArray
        TupleArray(cols, size(x))
    else
        cols
    end
    if ReturnPrimal
        (; derivs = (res,), val = gradtup[2])
    else
        (res,)
    end
end

"""
    jacobian(::ForwardMode, args...; kwargs...)

Equivalent to gradient(::ForwardMode, args...; kwargs...)
"""
@inline function jacobian(fm::ForwardMode, args...; kwargs...)
    gradient(fm, args...; kwargs...)
end

"""
    jacobian(::ReverseMode, f, x; n_outs=nothing, chunk=nothing)
    jacobian(::ReverseMode, f, x)

Compute the jacobian of a array-output function `f` using (potentially vector)
reverse mode. The `chunk` argument optionally denotes the chunk size to use and
`n_outs` optionally denotes the shape of the array returned by `f` (e.g `size(f(x))`).

Example:

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

jacobian(Reverse, f, [2.0, 3.0, 4.0])

# output
([3.0 2.0 0.0; 0.0 1.0 1.0],)
```

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(ReverseWithPrimal, f, [2.0, 3.0, 4.0])

# output
(derivs = ([3.0 2.0 0.0; 0.0 1.0 1.0],), val = [6.0, 7.0])
```

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(Reverse, f, [2.0, 3.0, 4.0], n_outs=Val((2,)))

# output
([3.0 2.0 0.0; 0.0 1.0 1.0],)
```

```jldoctest
f(x) = [ x[1] * x[2], x[2] + x[3] ]

grad = jacobian(ReverseWithPrimal, f, [2.0, 3.0, 4.0], n_outs=Val((2,)))

# output
(derivs = ([3.0 2.0 0.0; 0.0 1.0 1.0],), val = [6.0, 7.0])
```

This function will return an AbstractArray whose shape is `(size(output)..., size(input)...)`.
No guarantees are presently made about the type of the AbstractArray returned by this function
(which may or may not be the same as the input AbstractArray if provided).

In the future, when this function is extended to handle non-array return types, 
this function will retun an AbstractArray of shape `size(output)` of values of the input type. 
```
"""
@inline function jacobian(
    ::ReverseMode{ReturnPrimal,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten},
    f::F,
    x::X;
    n_outs::OutType = nothing,
    chunk::CT = nothing,
) where {ReturnPrimal,F,X,RABI<:ABI,ErrIfFuncWritten,RuntimeActivity,OutType,CT,Holomorphic}

    if n_outs == nothing
        res = if f isa Const
            f.val(x)
        else
            f(x)
        end
        jac = if res isa AbstractArray
            jacobian(
                ReverseMode{false,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten}(),
                f,
                x;
                n_outs = Val(size(res)),
                chunk,
            )
        elseif res isa AbstractFloat
            gradient(
                ReverseMode{false,RuntimeActivity,RABI,Holomorphic,ErrIfFuncWritten}(),
                f,
                x,
            )
        else
            throw(
                AssertionError(
                    "Unsupported return type of function for reverse-mode jacobian, $(Core.Typeof(res))",
                ),
            )
        end

        return if ReturnPrimal
            (; derivs = jac, val = res)
        else
            jac
        end
    else
        n_out_val = if length(Compiler.element(n_outs)) == 0
            0
        else
            prod(Compiler.element(n_outs))
        end

        if chunk == Val(0)
            throw(ErrorException("Cannot differentiate with a batch size of 0"))
        end

        XT = Core.Typeof(x)
        MD = Compiler.active_reg_inner(XT, (), nothing, Val(true)) == Compiler.ActiveState #=justActive=#
        tt = Tuple{XT}
        rt = if f isa Const
            Core.Compiler.return_type(f.val, tt)
        else
            Core.Compiler.return_type(f, tt)
        end

        ModifiedBetweenT = (false, false)
        FRT = Core.Typeof(f)
        FA = Const{FRT}

        if chunk == Val(1) || chunk == nothing
            primal, adjoint = autodiff_thunk(
                ReverseModeSplit{
                    #=ReturnPrimal=#false,
                    #=ReturnShadow=#true,
                    RuntimeActivity,
                    #=width=#1,
                    ModifiedBetweenT,
                    RABI,
                    Holomorphic,
                    ErrIfFuncWritten,
                    #=ShadowInit=#false
                }(),
                FA,
                DuplicatedNoNeed{rt},
                MD ? MixedDuplicated{XT} : Duplicated{XT}
            )
            tmp = ntuple(Val(n_out_val)) do i
                Base.@_inline_meta
                z = make_zero(x)
                dx = MD ? Ref(z) : z
                res = primal(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx))
                tape = res[1]
                @inbounds res[3][i] += Compiler.default_adjoint(eltype(typeof(res[3])))
                adjoint(Const(f), MD ? MixedDuplicated(x, dx) : Duplicated(x, dx), tape)
                return MD ? dx[] : dx, (i == 1 ? size(res[3]) : nothing)
            end
            rows = map(first, tmp)
            outshape = tmp[1][2]
            rows, outshape
        else
            chunksize = Compiler.element(chunk)
            primal, adjoint = autodiff_thunk(
                ReverseModeSplit{
                    #=ReturnPrimal=#false,
                    #=ReturnShadow=#true,
                    RuntimeActivity,
                    chunksize,
                    ModifiedBetweenT,
                    RABI,
                    Holomorphic,
                    ErrIfFuncWritten,
                    #=ShadowInit=#false
                }(),
                FA,
                BatchDuplicatedNoNeed{rt, chunksize},
                MD ? BatchMixedDuplicated{XT, chunksize} : BatchDuplicated{XT, chunksize}
            )

            num = ((n_out_val + chunksize - 1) ÷ chunksize)

            if num * chunksize == n_out_val
                last_size = chunksize
                primal2, adjoint2 = primal, adjoint
            else
                last_size = n_out_val - (num - 1) * chunksize
                tt′ = Tuple{BatchDuplicated{Core.Typeof(x),last_size}}
                primal2, adjoint2 = autodiff_thunk(
                    ReverseModeSplit{
                        #=ReturnPrimal=#false,
                        #=ReturnShadow=#true,
                        RuntimeActivity,
                        last_size,
                        ModifiedBetweenT,
                        RABI,
                        Holomorphic,
                        ErrIfFuncWritten,
                        #=ShadowInit=#false
                    }(),
                    FA,
                    BatchDuplicatedNoNeed{rt, last_size},
                    MD ? BatchMixedDuplicated{XT, last_size} : BatchDuplicated{XT, last_size}
                )
            end

            tmp = ntuple(num) do i
                Base.@_inline_meta
                dx = ntuple(Val(i == num ? last_size : chunksize)) do idx
                    Base.@_inline_meta
                    z = make_zero(x)
                    MD ? Ref(z) : z
                end
                res = (i == num ? primal2 : primal)(
                    Const(f),
                    MD ? BatchMixedDuplicated(x, dx) : BatchDuplicated(x, dx),
                )
                tape = res[1]
                j = 0
                for shadow in res[3]
                    j += 1
                    @inbounds shadow[(i-1)*chunksize+j] +=
                        Compiler.default_adjoint(eltype(typeof(shadow)))
                end
                (i == num ? adjoint2 : adjoint)(
                    Const(f),
                    MD ? BatchMixedDuplicated(x, dx) : BatchDuplicated(x, dx),
                    tape,
                )
                return MD ? (
                    ntuple(Val(i == num ? last_size : chunksize)) do idx
                        Base.@_inline_meta
                        dx[idx][]
                    end
                ) : dx,
                (i == 1 ? size(res[3][1]) : nothing)
            end
            rows = tupleconcat(map(first, tmp)...)
            outshape = tmp[1][2]
            rows, outshape
        end
        res = if x isa AbstractArray
            inshape = size(x)
            st2 = tupstack(rows, inshape, outshape)

            st3 = if length(outshape) == 1 && length(inshape) == 1
                transpose(st2)
            else
                transp = (
                    ((length(inshape)+1):(length(inshape)+length(outshape)))...,
                    (1:length(inshape))...,
                )
                PermutedDimsArray(st2, transp)
            end

            st3
        else
            reshape(collect(rows), outshape)
        end
        if ReturnPrimal
            # TODO optimize away redundant fwd pass
            (; derivs = (res,), val = if f isa Enzyme.Const
                f.val(x)
            else
                f(x)
            end)
        else
            (res,)
        end
    end
end

"""
    hvp(f::F, x::X, v::X) where {F, X}

Compute the Hessian-vector product of an array-input scalar-output function `f`, as evaluated at `x` times the vector `v`.

In other words, compute hessian(f)(x) * v

See [`hvp!`](@ref) for a version which stores the result in an existing buffer and also [`hvp_and_gradient!`](@ref) for a function to compute both the hvp and the gradient in a single call.

Example:

```jldoctest hvp; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
f(x) = sin(x[1] * x[2])

hvp(f, [2.0, 3.0], [5.0, 2.7])

# output
2-element Vector{Float64}:
 19.6926882637302
 16.201003759768003
```
"""
@inline function hvp(f::F, x::X, v::X) where {F,X}
    res = make_zero(x)
    hvp!(res, f, x, v)
    return res
end


"""
    hvp!(res::X, f::F, x::X, v::X) where {F, X}

Compute an in-place Hessian-vector product of an array-input scalar-output function `f`, as evaluated at `x` times the vector `v`.
The result will be stored into `res`. The function still allocates and zero's a buffer to store the intermediate gradient, which is
not returned to the user.

In other words, compute res .= hessian(f)(x) * v

See [`hvp_and_gradient!`](@ref) for a function to compute both the hvp and the gradient in a single call.

Example:

```jldoctest hvpip; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
f(x) = sin(x[1] * x[2])

res = Vector{Float64}(undef, 2)
hvp!(res, f, [2.0, 3.0], [5.0, 2.7])

res
# output
2-element Vector{Float64}:
 19.6926882637302
 16.201003759768003
```
"""
@inline function hvp!(res::X, f::F, x::X, v::X) where {F,X}
    grad = make_zero(x)
    Enzyme.autodiff(
        Forward,
        gradient!,
        Const(Reverse),
        DuplicatedNoNeed(grad, res),
        Const(f),
        Duplicated(x, v),
    )
    return nothing
end



"""
    hvp_and_gradient!(res::X, grad::X, f::F, x::X, v::X) where {F, X}

Compute an in-place Hessian-vector product of an array-input scalar-output function `f`, as evaluated at `x` times the vector `v` as well as
the gradient, storing the gradient into `grad`. Both the hessian vector product and the gradient can be computed together more efficiently
than computing them separately.

The result will be stored into `res`. The gradient will be stored into `grad`.

In other words, compute res .= hessian(f)(x) * v  and grad .= gradient(Reverse, f)(x)

Example:

```jldoctest hvp_and_gradient; filter = r"([0-9]+\\.[0-9]{8})[0-9]+" => s"\\1***"
f(x) = sin(x[1] * x[2])

res = Vector{Float64}(undef, 2)
grad = Vector{Float64}(undef, 2)
hvp_and_gradient!(res, grad, f, [2.0, 3.0], [5.0, 2.7])

res
grad
# output
2-element Vector{Float64}:
 2.880510859951098
 1.920340573300732
```
"""
@inline function hvp_and_gradient!(res::X, grad::X, f::F, x::X, v::X) where {F,X}
    Enzyme.autodiff(
        Forward,
        gradient!,
        Const(Reverse),
        Duplicated(grad, res),
        Const(f),
        Duplicated(x, v),
    )
    return nothing
end


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

"""
   within_autodiff()

Returns true if within autodiff, otherwise false.
"""
@inline EnzymeCore.within_autodiff() = false

end # module
