module EnzymeCore

export Forward, ForwardWithPrimal, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal
export ReverseSplitModified, ReverseSplitWidth, ReverseHolomorphic, ReverseHolomorphicWithPrimal
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed, Annotation
export MixedDuplicated, BatchMixedDuplicated
export Seed, BatchSeed
export DefaultABI, FFIABI, InlineABI, NonGenABI
export BatchDuplicatedFunc
export within_autodiff
export needs_primal

function batch_size end

"""
    abstract type Annotation{T}

Abstract type for [`autodiff`](@ref Enzyme.autodiff) function argument wrappers like
[`Const`](@ref), [`Active`](@ref) and [`Duplicated`](@ref).
"""
abstract type Annotation{T} end
Base.eltype(::Type{<:Annotation{T}}) where T = T

"""
    Const(x)

Mark a function argument `x` of [`autodiff`](@ref Enzyme.autodiff) as constant,
Enzyme will not auto-differentiate in respect `Const` arguments.
"""
struct Const{T} <: Annotation{T}
    val::T
end

# To deal with Const(Int) and prevent it to go to `Const{DataType}(T)`
Const(::Type{T}) where T = Const{Type{T}}(T)

"""
    Active(x)

Mark a function argument `x` of [`autodiff`](@ref Enzyme.autodiff) as active,
Enzyme will auto-differentiate in respect `Active` arguments.

!!! note

    Enzyme gradients with respect to integer values are zero.
    [`Active`](@ref) will automatically convert plain integers to floating
    point values, but cannot do so for integer values in tuples and structs.
"""
struct Active{T} <: Annotation{T}
    val::T
    @inline Active(x::T1) where {T1} = new{T1}(x)
    @inline Active(x::T1) where {T1 <: Array} = error("Unsupported Active{"*string(T1)*"}, consider Duplicated or Const")
end

Active(i::Integer) = Active(float(i))
Active(ci::Complex{T}) where T <: Integer = Active(float(ci))

"""
    Duplicated(x, ∂f_∂x)

Mark a function argument `x` of [`autodiff`](@ref Enzyme.autodiff) as duplicated, Enzyme will
auto-differentiate in respect to such arguments, with `dx` acting as an
accumulator for gradients (so ``\\partial f / \\partial x`` will be *added to*)
`∂f_∂x`.
"""
struct Duplicated{T} <: Annotation{T}
    val::T
    dval::T
    @inline Duplicated(x::T1, dx::T1, check::Bool=true) where {T1} = new{T1}(x, dx)
    @inline function Duplicated(x::T1, dx::T1, check::Bool=true) where {T1 <: SubArray}
        if check
            @assert x.indices == dx.indices
            @assert x.offset1 == dx.offset1
            @assert x.stride1 == dx.stride1
        end
        new{T1}(x, dx)
    end
end

"""
    DuplicatedNoNeed(x, ∂f_∂x)

Like [`Duplicated`](@ref), except also specifies that Enzyme may avoid computing
the original result and only compute the derivative values. This creates opportunities
for improved performance.

```julia

function square_byref(out, v)
    out[] = v * v
    nothing
end

out = Ref(0.0)
dout = Ref(1.0)
Enzyme.autodiff(Reverse, square_byref, DuplicatedNoNeed(out, dout), Active(1.0))
dout[]

# output
0.0
```

For example, marking the out variable as `DuplicatedNoNeed` instead of `Duplicated` allows
Enzyme to avoid computing `v * v` (while still computing its derivative).

This should only be used if `x` is a write-only variable. Otherwise, if the differentiated
function stores values in `x` and reads them back in subsequent computations, using
`DuplicatedNoNeed` may result in incorrect derivatives. In particular, `DuplicatedNoNeed`
should not be used for preallocated workspace, even if the user might not care about its
final value, as marking a variable as NoNeed means that reads from the variable are now
undefined.
"""
struct DuplicatedNoNeed{T} <: Annotation{T}
    val::T
    dval::T
    @inline DuplicatedNoNeed(x::T1, dx::T1, check::Bool=true) where {T1} = new{T1}(x, dx)
    @inline function DuplicatedNoNeed(x::T1, dx::T1, check::Bool=true) where {T1 <: SubArray}
        if check
            @assert x.indices == dx.indices
            @assert x.offset1 == dx.offset1
            @assert x.stride1 == dx.stride1
        end
        new{T1}(x, dx)
    end
end

"""
    BatchDuplicated(x, ∂f_∂xs)

Like [`Duplicated`](@ref), except contains several shadows to compute derivatives
for all at once. Argument `∂f_∂xs` should be a tuple of the several values of type `x`.
"""
struct BatchDuplicated{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,T}
    @inline BatchDuplicated(x::T1, dx::NTuple{N,T1}, check::Bool=true) where {T1, N} = new{T1, N}(x, dx)
    @inline function DuplicatedNoNeed(x::T1, dx::NTuple{N,T1}, check::Bool=true) where {T1 <: SubArray, N}
        if check
            for dxi in dx
                @assert x.indices == dxi.indices
                @assert x.offset1 == dxi.offset1
                @assert x.stride1 == dxi.stride1
            end
        end
        new{T1, N}(x, dx)
    end
end

struct BatchDuplicatedFunc{T,N,Func} <: Annotation{T}
    val::T
end
get_func(::BatchDuplicatedFunc{T,N,Func}) where {T,N,Func} = Func
get_func(::Type{BatchDuplicatedFunc{T,N,Func}}) where {T,N,Func} = Func

"""
    BatchDuplicatedNoNeed(x, ∂f_∂xs)

Like [`DuplicatedNoNeed`](@ref), except contains several shadows to compute derivatives
for all at once. Argument `∂f_∂xs` should be a tuple of the several values of type `x`.
"""
struct BatchDuplicatedNoNeed{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,T}
    @inline BatchDuplicatedNoNeed(x::T1, dx::NTuple{N,T1}, check::Bool=true) where {T1, N} = new{T1, N}(x, dx)
    @inline function DuplicatedNoNeed(x::T1, dx::NTuple{N,T1}, check::Bool=true) where {T1 <: SubArray, N}
        if check
            for dxi in dx
                @assert x.indices == dxi.indices
                @assert x.offset1 == dxi.offset1
                @assert x.stride1 == dxi.stride1
            end
        end
        new{T1, N}(x, dx)
    end
end
@inline batch_size(::BatchDuplicated{T,N}) where {T,N} = N
@inline batch_size(::BatchDuplicatedFunc{T,N}) where {T,N} = N
@inline batch_size(::BatchDuplicatedNoNeed{T,N}) where {T,N} = N
@inline batch_size(::Type{BatchDuplicated{T,N}}) where {T,N} = N
@inline batch_size(::Type{BatchDuplicatedFunc{T,N}}) where {T,N} = N
@inline batch_size(::Type{BatchDuplicatedNoNeed{T,N}}) where {T,N} = N


"""
    MixedDuplicated(x, ∂f_∂x)

Like [`Duplicated`](@ref), except x may contain both active [immutable] and duplicated [mutable]
data which is differentiable. Only used within custom rules.
"""
struct MixedDuplicated{T} <: Annotation{T}
    val::T
    dval::Base.RefValue{T}
    @inline MixedDuplicated(x::T1, dx::Base.RefValue{T1}, check::Bool=true) where {T1} = new{T1}(x, dx)
end

"""
    BatchMixedDuplicated(x, ∂f_∂xs)

Like [`MixedDuplicated`](@ref), except contains several shadows to compute derivatives
for all at once. Only used within custom rules.
"""
struct BatchMixedDuplicated{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,Base.RefValue{T}}
    @inline BatchMixedDuplicated(x::T1, dx::NTuple{N,Base.RefValue{T1}}, check::Bool=true) where {T1, N} = new{T1, N}(x, dx)
end
@inline batch_size(::BatchMixedDuplicated{T,N}) where {T,N} = N
@inline batch_size(::Type{BatchMixedDuplicated{T,N}}) where {T,N} = N

"""
    Seed(dy)

Wrapper for a single adjoint to the return value in reverse mode.
"""
struct Seed{T}
    dval::T
end

"""
    BatchSeed(dys::NTuple)

Wrapper for a tuple of adjoints to the return value in reverse mode.
"""
struct BatchSeed{N, T}
    dvals::NTuple{N, T}
end

"""
    abstract type ABI

Abstract type for what ABI  will be used.

# Subtypes

- [`FFIABI`](@ref) (the default)
- [`InlineABI`](@ref)
- [`NonGenABI`](@ref)
"""
abstract type ABI end

"""
    struct FFIABI <: ABI

Foreign function call [`ABI`](@ref). JIT the differentiated function, then inttoptr call the address.
"""
struct FFIABI <: ABI end

"""
    struct InlineABI <: ABI

Inlining function call [`ABI`](@ref). 
"""
struct InlineABI <: ABI end

"""
    struct NonGenABI <: ABI

Non-generated function [`ABI`](@ref). 
"""
struct NonGenABI <: ABI end

const DefaultABI = FFIABI

"""
    abstract type Mode{ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}

Abstract type for which differentiation mode will be used.

# Subtypes

- [`ForwardMode`](@ref)
- [`ReverseMode`](@ref)
- [`ReverseModeSplit`](@ref)

# Type parameters

- `ABI`: what runtime [`ABI`](@ref) to use
- `ErrIfFuncWritten`: whether to error when the function differentiated is a closure and written to.
- `RuntimeActivity`: whether to enable runtime activity (default off). Runtime Activity is required is the differentiability of all mutable variables cannot be determined statically. For a deeper explanation see the [FAQ](@ref faq-runtime-activity)
- `StrongZero`: whether to enforce that propagating a zero derivative input always ends up in zero derivative outputs. This is required to avoid nan's if one of the arguments may be infinite or nan. For a deeper explanation see the [FAQ](@ref faq-strong-zero)

!!! warning
    The type parameters of `Mode` are not part of the public API and can change without notice.
    You can modify them with the following helper functions:
    - [`WithPrimal`](@ref) / [`NoPrimal`](@ref)
    - [`set_err_if_func_written`](@ref) / [`clear_err_if_func_written`](@ref)
    - [`set_runtime_activity`](@ref) / [`clear_runtime_activity`](@ref)
    - [`set_strong_zero`](@ref) / [`clear_strong_zero`](@ref)
    - [`set_abi`](@ref)
"""
abstract type Mode{ABI, ErrIfFuncWritten, RuntimeActivity, StrongZero} end

"""
    runtime_activity(::Mode)
    strong_zero(::Type{<:Mode})

Returns whether the given mode has runtime activity set. For a deeper explanation of what strong zero is see the [FAQ](@ref faq-runtime-activity)
"""
runtime_activity(::Mode{<:Any, <:Any, RuntimeActivity}) where RuntimeActivity = RuntimeActivity
runtime_activity(::Type{<:Mode{<:Any, <:Any, RuntimeActivity}}) where RuntimeActivity = RuntimeActivity

"""
    strong_zero(::Mode)
    strong_zero(::Type{<:Mode})

Returns whether the given mode has strong zero set. For a deeper explanation of what strong zero is see the [FAQ](@ref faq-strong-zero)
"""
strong_zero(::Mode{<:Any, <:Any, <:Any, StrongZero}) where StrongZero = StrongZero
strong_zero(::Type{<:Mode{<:Any, <:Any, <:Any, StrongZero}}) where StrongZero = StrongZero

"""
    struct ReverseMode{
        ReturnPrimal,
        RuntimeActivity,
        StrongZero,
        ABI,
        Holomorphic,
        ErrIfFuncWritten
    } <: Mode{ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}

Subtype of [`Mode`](@ref) for reverse mode differentiation.

# Type parameters

- `ReturnPrimal`: whether to return the primal return value from the augmented-forward pass.
- `Holomorphic`: Whether the complex result function is holomorphic and we should compute `d/dz`
- other parameters: see [`Mode`](@ref)

!!! warning
    The type parameters of `ReverseMode` are not part of the public API and can change without notice.
    Please use one of the following concrete instantiations instead:
    - [`Reverse`](@ref)
    - [`ReverseWithPrimal`](@ref)
    - [`ReverseHolomorphic`](@ref)
    - [`ReverseHolomorphicWithPrimal`](@ref)
    You can modify them with the following helper functions:
    - [`WithPrimal`](@ref) / [`NoPrimal`](@ref)
    - [`set_err_if_func_written`](@ref) / [`clear_err_if_func_written`](@ref)
    - [`set_runtime_activity`](@ref) / [`clear_runtime_activity`](@ref)
    - [`set_strong_zero`](@ref) / [`clear_strong_zero`](@ref)
    - [`set_abi`](@ref)
"""
struct ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} <: Mode{ABI, ErrIfFuncWritten,RuntimeActivity,StrongZero} end

"""
    const Reverse

Default instance of [`ReverseMode`](@ref) that doesn't return the primal
"""
const Reverse = ReverseMode{false,false, false, DefaultABI, false, false}()

"""
    const ReverseWithPrimal

Default instance of [`ReverseMode`](@ref) that also returns the primal.
"""
const ReverseWithPrimal = ReverseMode{true,false,false,DefaultABI, false, false}()

"""
    const ReverseHolomorphic

Holomorphic instance of [`ReverseMode`](@ref) that doesn't return the primal
"""
const ReverseHolomorphic = ReverseMode{false,false,false,DefaultABI, true, false}()

"""
    const ReverseHolomorphicWithPrimal

Holomorphic instance of [`ReverseMode`](@ref) that also returns the primal
"""
const ReverseHolomorphicWithPrimal = ReverseMode{true,false,false,DefaultABI, true, false}()

@inline set_err_if_func_written(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,true}()
@inline clear_err_if_func_written(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,false}()

@inline set_abi(::Type{ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,OldABI,Holomorphic,ErrIfFuncWritten}}, ::Type{NewABI}) where {ReturnPrimal,RuntimeActivity,StrongZero,OldABI,Holomorphic,ErrIfFuncWritten,NewABI<:ABI} = ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,NewABI,Holomorphic,ErrIfFuncWritten}
@inline set_abi(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,OldABI,Holomorphic,ErrIfFuncWritten}, ::Type{NewABI}) where {ReturnPrimal,RuntimeActivity,StrongZero,OldABI,Holomorphic,ErrIfFuncWritten,NewABI<:ABI} = ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,NewABI,Holomorphic,ErrIfFuncWritten}()

@inline set_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,true,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}()
@inline set_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}, rt::Bool) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,rt,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}()
@inline set_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}, ::Mode{<:Any, <:Any, RT2}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten, RT2} = ReverseMode{ReturnPrimal,RT2,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}()
@inline clear_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,false,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}()


@inline set_strong_zero(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,true,ABI,Holomorphic,ErrIfFuncWritten}()
@inline set_strong_zero(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}, rt::Bool) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,rt,ABI,Holomorphic,ErrIfFuncWritten}()
@inline set_strong_zero(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}, ::Mode{<:Any, <:Any, <:Any, SZ2}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten, SZ2} = ReverseMode{ReturnPrimal,RuntimeActivity,SZ2,ABI,Holomorphic,ErrIfFuncWritten}()
@inline clear_strong_zero(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,false,ABI,Holomorphic,ErrIfFuncWritten}()

"""
    WithPrimal(::Mode)

Return a new mode which includes the primal value.
"""
@inline WithPrimal(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{true,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}()

"""
    NoPrimal(::Mode)

Return a new mode which excludes the primal value.
"""
@inline NoPrimal(::ReverseMode{ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{false,RuntimeActivity,StrongZero,ABI,Holomorphic,ErrIfFuncWritten}()

"""
    needs_primal(::Mode)
    needs_primal(::Type{Mode})

Returns `true` if the mode needs the primal value, otherwise `false`.
"""
@inline needs_primal(::ReverseMode{ReturnPrimal}) where {ReturnPrimal} = ReturnPrimal
@inline needs_primal(::Type{<:ReverseMode{ReturnPrimal}}) where {ReturnPrimal} = ReturnPrimal

"""
    struct ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        Width,
        RuntimeActivity,
        StrongZero,
        ModifiedBetween,
        ABI,
        ErrFuncIfWritten
    } <: Mode{ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}
        WithPrimal(::Enzyme.Mode)

Subtype of [`Mode`](@ref) for split reverse mode differentiation, to use in [`autodiff_thunk`](@ref) and variants.

# Type parameters

- `ReturnShadow`: whether to return the shadow return value from the augmented-forward.
- `Width`: batch size (pick `0` to derive it automatically)
- `ModifiedBetween`: `Tuple` of each argument's "modified between" state (pick `true` to derive it automatically).
- other parameters: see [`ReverseMode`](@ref)

!!! warning
    The type parameters of `ReverseModeSplit` are not part of the public API and can change without notice.
    Please use one of the following concrete instantiations instead: 
    - [`ReverseSplitNoPrimal`](@ref)
    - [`ReverseSplitWithPrimal`](@ref)
    You can modify them with the following helper functions:
    - [`WithPrimal`](@ref) / [`NoPrimal`](@ref)
    - [`set_err_if_func_written`](@ref) / [`clear_err_if_func_written`](@ref)
    - [`set_runtime_activity`](@ref) / [`clear_runtime_activity`](@ref)
    - [`set_strong_zero`](@ref) / [`clear_strong_zero`](@ref)
    - [`set_abi`](@ref)
    - [`ReverseSplitModified`](@ref), [`ReverseSplitWidth`](@ref)
"""
struct ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI,Holomorphic,ErrIfFuncWritten,ShadowInit} <: Mode{ABI, ErrIfFuncWritten,RuntimeActivity,StrongZero} end

"""
    const ReverseSplitNoPrimal

Default instance of [`ReverseModeSplit`](@ref) that doesn't return the primal
"""
const ReverseSplitNoPrimal = ReverseModeSplit{false, true, false, false, 0, true,DefaultABI, false, false, false}()

"""
    const ReverseSplitWithPrimal

Default instance of [`ReverseModeSplit`](@ref) that also returns the primal
"""
const ReverseSplitWithPrimal = ReverseModeSplit{true, true, false, false, 0, true,DefaultABI, false, false, false}()

"""
    ReverseSplitModified(::ReverseModeSplit, ::Val{MB})

Return a new instance of [`ReverseModeSplit`](@ref) mode where `ModifiedBetween` is set to `MB`. 
"""
@inline ReverseSplitModified(::ReverseModeSplit{ReturnPrimal, ReturnShadow, RuntimeActivity, StrongZero, Width, MBO, ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, ::Val{MB}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,MB,MBO,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity, StrongZero, Width,MB,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

"""
    ReverseSplitWidth(::ReverseModeSplit, ::Val{W})

Return a new instance of [`ReverseModeSplit`](@ref) mode where `Width` is set to `W`. 
"""
@inline ReverseSplitWidth(::ReverseModeSplit{ReturnPrimal, ReturnShadow, RuntimeActivity, StrongZero, WidthO, MB, ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, ::Val{Width}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero, Width,MB,WidthO,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,MB,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

@inline set_err_if_func_written(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, true, ShadowInit}()
@inline clear_err_if_func_written(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI, Holomorphic, false, ShadowInit}()

@inline set_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,true,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline set_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, rt::Bool) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,rt,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline set_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, ::Mode{<:Any, <:Any, RT2}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit, RT2} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RT2,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline clear_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,false,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()


@inline set_strong_zero(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,true,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline set_strong_zero(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, rt::Bool) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,rt,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline set_strong_zero(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, ::Mode{<:Any, <:Any, <:Any, SZ2}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit, SZ2} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,SZ2,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline clear_strong_zero(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,false,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

@inline set_abi(::Type{ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,OldABI,Holomorphic,ErrIfFuncWritten,ShadowInit}}, ::Type{NewABI}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,OldABI,Holomorphic,ErrIfFuncWritten,ShadowInit,NewABI<:ABI} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,NewABI,Holomorphic,ErrIfFuncWritten,ShadowInit}
@inline set_abi(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,OldABI,Holomorphic,ErrIfFuncWritten,ShadowInit}, ::Type{NewABI}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,OldABI,Holomorphic,ErrIfFuncWritten,ShadowInit,NewABI<:ABI} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,NewABI,Holomorphic,ErrIfFuncWritten,ShadowInit}()

@inline WithPrimal(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{true,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline NoPrimal(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{false,ReturnShadow,RuntimeActivity,StrongZero,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

@inline needs_primal(::ReverseModeSplit{ReturnPrimal}) where {ReturnPrimal} = ReturnPrimal
@inline needs_primal(::Type{<:ReverseModeSplit{ReturnPrimal}}) where {ReturnPrimal} = ReturnPrimal

"""
    struct ForwardMode{
        ReturnPrimal,
        ABI,
        ErrIfFuncWritten,
        RuntimeActivity,
        StrongZero
    } <: Mode{ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}

Subtype of [`Mode`](@ref) for forward mode differentiation.

# Type parameters

- `ReturnPrimal`: whether to return the primal return value from the augmented-forward.
- other parameters: see [`Mode`](@ref)

!!! warning
    The type parameters of `ForwardMode` are not part of the public API and can change without notice.
    Please use one of the following concrete instantiations instead:
    - [`Forward`](@ref)
    - [`ForwardWithPrimal`](@ref)
    You can modify them with the following helper functions:
    - [`WithPrimal`](@ref) / [`NoPrimal`](@ref)
    - [`set_err_if_func_written`](@ref) / [`clear_err_if_func_written`](@ref)
    - [`set_runtime_activity`](@ref) / [`clear_runtime_activity`](@ref)
    - [`set_abi`](@ref)
"""
struct ForwardMode{ReturnPrimal, ABI, ErrIfFuncWritten,RuntimeActivity,StrongZero} <: Mode{ABI, ErrIfFuncWritten, RuntimeActivity,StrongZero}
end

"""
    const Forward

Default instance of [`ForwardMode`](@ref) that doesn't return the primal
"""
const Forward = ForwardMode{false, DefaultABI, false, false, false}()

"""
    const ForwardWithPrimal

Default instance of [`ForwardMode`](@ref) that also returns the primal
"""
const ForwardWithPrimal = ForwardMode{true, DefaultABI, false, false, false}()

@inline set_err_if_func_written(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,true,RuntimeActivity,StrongZero}()
@inline clear_err_if_func_written(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,false,RuntimeActivity,StrongZero}()

@inline set_abi(::Type{ForwardMode{ReturnPrimal,OldABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}}, ::Type{NewABI}) where {ReturnPrimal,OldABI,ErrIfFuncWritten,RuntimeActivity,StrongZero,NewABI<:ABI} = ForwardMode{ReturnPrimal,NewABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}
@inline set_abi(::ForwardMode{ReturnPrimal,OldABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}, ::Type{NewABI}) where {ReturnPrimal,OldABI,ErrIfFuncWritten,RuntimeActivity,StrongZero,NewABI<:ABI} = ForwardMode{ReturnPrimal,NewABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}()

@inline set_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,true,StrongZero}()
@inline set_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}, rt::Bool) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,rt,StrongZero}()
@inline set_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}, ::Mode{<:Any, <:Any, RT2}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero, RT2} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RT2,StrongZero}()
@inline clear_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,false,StrongZero}()

@inline set_strong_zero(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,true}()
@inline set_strong_zero(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}, rt::Bool) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,rt}()()
@inline set_strong_zero(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}, ::Mode{<:Any, <:Any, <:Any, SZ2}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero,SZ2} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,SZ2}()
@inline clear_strong_zero(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,false}()

@inline WithPrimal(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{true,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}()
@inline NoPrimal(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero} = ForwardMode{false,ABI,ErrIfFuncWritten,RuntimeActivity,StrongZero}()

@inline needs_primal(::ForwardMode{ReturnPrimal}) where {ReturnPrimal} = ReturnPrimal
@inline needs_primal(::Type{<:ForwardMode{ReturnPrimal}}) where {ReturnPrimal} = ReturnPrimal

function autodiff end
function autodiff_deferred end
function autodiff_thunk end
function autodiff_deferred_thunk end

"""
    make_zero(::Type{T}, seen::IdDict, prev::T, ::Val{copy_if_inactive}=Val(false))::T

Recursively make a zero'd copy of the value `prev` of type `T`. The argument `copy_if_inactive` specifies
what to do if the type `T` is guaranteed to be inactive, use the primal (the default) or still copy the value. 
"""
function make_zero end

"""
    make_zero!(val::T, seen::IdSet{Any}=IdSet())::Nothing

Recursively set a variables differentiable fields to zero.

!!! warn
    Only applicable for mutable types `T`.
"""
function make_zero! end

"""
    remake_zero!(val::T, seen::IdSet{Any}=IdSet())::Nothing

Recursively set a variables differentiable fields to zero.

!!! warn
    This assumes that the input value was previously generated by make_zero. Otherwise, this may not zero the immutable fields of a struct.
"""
function remake_zero! end

"""
    make_zero(prev::T)

Helper function to recursively make zero.
"""
@inline function make_zero(prev::T, ::Val{copy_if_inactive}=Val(false)) where {T, copy_if_inactive}
    make_zero(Core.Typeof(prev), IdDict(), prev, Val(copy_if_inactive))
end

function tape_type end

"""
    compiler_job_from_backend(::KernelAbstractions.Backend, F::Type, TT:Type)::GPUCompiler.CompilerJob

Returns a GPUCompiler CompilerJob from a backend as specified by the first argument to the function.

For example, in CUDA one would do:

```julia
function EnzymeCore.compiler_job_from_backend(::CUDABackend, @nospecialize(F::Type), @nospecialize(TT::Type))
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, CUDA.compiler_config(CUDA.device()))
end
```
"""
function compiler_job_from_backend end

include("rules.jl")

if !isdefined(Base, :get_extension)
    include("../ext/AdaptExt.jl")
end

"""
    within_autodiff()

Returns true if within autodiff, otherwise false.
"""
function within_autodiff end

"""
    set_err_if_func_written(::Mode)

Return a new mode which throws an error for any attempt to write into an unannotated function object.
"""
function set_err_if_func_written end

"""
    clear_err_if_func_written(::Mode)

Return a new mode which doesn't throw an error for attempts to write into an unannotated function object.
"""
function clear_err_if_func_written end

"""
    set_runtime_activity(::Mode)
    set_runtime_activity(::Mode, activity::Bool)
    set_runtime_activity(::Mode, config::Union{FwdConfig,RevConfig})
    set_runtime_activity(::Mode, prev::Mode)

Return a new mode where runtime activity analysis is activated / set to the desired value. See the [FAQ](@ref faq-runtime-activity) for more information.
"""
function set_runtime_activity end

"""
    clear_runtime_activity(::Mode)

Return a new mode where runtime activity analysis is deactivated. See [Enzyme.Mode](@ref) for more information on runtime activity.
"""
function clear_runtime_activity end

"""
    set_strong_zero(::Mode)
    set_strong_zero(::Mode, activity::Bool)
    set_strong_zero(::Mode, config::Union{FwdConfig,RevConfig})
    set_strong_zero(::Mode, prev::Mode)

Return a new mode where strong zero is activated / set to the desired value. See the [FAQ](@ref faq-strong-zero) for more information.
"""
function set_strong_zero end

"""
    clear_strong_zero(::Mode)

Return a new mode where strong_zero is deactivated. See [Enzyme.Mode](@ref) for more information on strong zero.
"""
function clear_strong_zero end

"""
    set_abi(::Mode, ::Type{ABI})

Return a new mode with its [`ABI`](@ref) set to the chosen type.
"""
function set_abi end

"""
    Split(
        ::ReverseMode, [::Val{ReturnShadow}, ::Val{Width}, ::Val{ModifiedBetween}, ::Val{ShadowInit}]
    )

Turn a [`ReverseMode`](@ref) object into a [`ReverseModeSplit`](@ref) object while preserving as many of the settings as possible.
The rest of the settings can be configured with optional positional arguments of `Val` type.

This function acts as the identity on a [`ReverseModeSplit`](@ref).

See also [`Combined`](@ref).
"""
function Split(
    ::ReverseMode{
        ReturnPrimal,
        RuntimeActivity,
        StrongZero,
        ABI,
        Holomorphic,
        ErrIfFuncWritten
    },
    ::Val{ReturnShadow}=Val(true),
    ::Val{Width}=Val(0),
    ::Val{ModifiedBetween}=Val(true),
    ::Val{ShadowInit}=Val(false),
) where {
    ReturnPrimal,
    ReturnShadow,
    RuntimeActivity,
    StrongZero,
    Width,
    ModifiedBetween,
    ABI,
    Holomorphic,
    ErrIfFuncWritten,
    ShadowInit
}
    mode_split = ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        StrongZero,
        Width,
        ModifiedBetween,
        ABI,
        Holomorphic,
        ErrIfFuncWritten,
        ShadowInit
    }()
    return mode_split
end

Split(mode::ReverseModeSplit, args...) = mode

"""
    Combined(::ReverseMode)

Turn a [`ReverseModeSplit`](@ref) object into a [`ReverseMode`](@ref) object while preserving as many of the settings as possible.

This function acts as the identity on a [`ReverseMode`](@ref).

See also [`Split`](@ref).
"""
function Combined(
    ::ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        RuntimeActivity,
        StrongZero,
        Width,
        ModifiedBetween,
        ABI,
        Holomorphic,
        ErrIfFuncWritten,
        ShadowInit
    }
) where {
    ReturnPrimal,
    ReturnShadow,
    RuntimeActivity,
    StrongZero,
    Width,
    ModifiedBetween,
    ABI,
    Holomorphic,
    ErrIfFuncWritten,
    ShadowInit
}
    mode_unsplit = ReverseMode{
        ReturnPrimal,
        RuntimeActivity,
        StrongZero,
        ABI,
        Holomorphic,
        ErrIfFuncWritten
    }()
    return mode_unsplit
end

Combined(mode::ReverseMode) = mode

end # module EnzymeCore
