module EnzymeCore

export Forward, ForwardWithPrimal, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal
export ReverseSplitModified, ReverseSplitWidth, ReverseHolomorphic, ReverseHolomorphicWithPrimal
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed, Annotation
export MixedDuplicated, BatchMixedDuplicated
export DefaultABI, FFIABI, InlineABI, NonGenABI
export BatchDuplicatedFunc
export within_autodiff

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
    abstract type Mode{ABI,ErrIfFuncWritten,RuntimeActivity}

Abstract type for which differentiation mode will be used.

# Subtypes

- [`ForwardMode`](@ref)
- [`ReverseMode`](@ref)
- [`ReverseModeSplit`](@ref)

# Type parameters

- `ABI`: what runtime [`ABI`](@ref) to use
- `ErrIfFuncWritten`: whether to error when the function differentiated is a closure and written to.
- `RuntimeActivity`: whether to enable runtime activity (default off)

!!! warning
    The type parameters of `Mode` are not part of the public API and can change without notice.
    You can modify them with the following helper functions:
    - [`WithPrimal`](@ref) / [`NoPrimal`](@ref)
    - [`set_err_if_func_written`](@ref) / [`clear_err_if_func_written`](@ref)
    - [`set_runtime_activity`](@ref) / [`clear_runtime_activity`](@ref)
    - [`set_abi`](@ref)
"""
abstract type Mode{ABI, ErrIfFuncWritten, RuntimeActivity} end

"""
    struct ReverseMode{
        ReturnPrimal,
        RuntimeActivity,
        ABI,
        Holomorphic,
        ErrIfFuncWritten
    } <: Mode{ABI,ErrIfFuncWritten,RuntimeActivity}

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
    - [`set_abi`](@ref)
"""
struct ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} <: Mode{ABI, ErrIfFuncWritten,RuntimeActivity} end

"""
    const Reverse

Default instance of [`ReverseMode`](@ref) that doesn't return the primal
"""
const Reverse = ReverseMode{false,false,DefaultABI, false, false}()

"""
    const ReverseWithPrimal

Default instance of [`ReverseMode`](@ref) that also returns the primal.
"""
const ReverseWithPrimal = ReverseMode{true,false,DefaultABI, false, false}()

"""
    const ReverseHolomorphic

Holomorphic instance of [`ReverseMode`](@ref) that doesn't return the primal
"""
const ReverseHolomorphic = ReverseMode{false,false,DefaultABI, true, false}()

"""
    const ReverseHolomorphicWithPrimal

Holomorphic instance of [`ReverseMode`](@ref) that also returns the primal
"""
const ReverseHolomorphicWithPrimal = ReverseMode{true,false,DefaultABI, true, false}()

@inline set_err_if_func_written(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,true}()
@inline clear_err_if_func_written(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,false}()

@inline set_abi(::ReverseMode{ReturnPrimal,RuntimeActivity,OldABI,Holomorphic,ErrIfFuncWritten}, ::Type{NewABI}) where {ReturnPrimal,RuntimeActivity,OldABI,Holomorphic,ErrIfFuncWritten,NewABI<:ABI} = ReverseMode{ReturnPrimal,RuntimeActivity,NewABI,Holomorphic,ErrIfFuncWritten}()

@inline set_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,true,ABI,Holomorphic,ErrIfFuncWritten}()
@inline set_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}, rt::Bool) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,rt,ABI,Holomorphic,ErrIfFuncWritten}()
@inline clear_runtime_activity(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{ReturnPrimal,false,ABI,Holomorphic,ErrIfFuncWritten}()

"""
    WithPrimal(::Mode)

Return a new mode which includes the primal value.
"""
@inline WithPrimal(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{true,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}()

"""
    NoPrimal(::Mode)

Return a new mode which excludes the primal value.
"""
@inline NoPrimal(::ReverseMode{ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}) where {ReturnPrimal,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten} = ReverseMode{false,RuntimeActivity,ABI,Holomorphic,ErrIfFuncWritten}()

"""
    struct ReverseModeSplit{
        ReturnPrimal,
        ReturnShadow,
        Width,
        RuntimeActivity,
        ModifiedBetween,
        ABI,
        ErrFuncIfWritten
    } <: Mode{ABI,ErrIfFuncWritten,RuntimeActivity}
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
    - [`set_abi`](@ref)
    - [`ReverseSplitModified`](@ref), [`ReverseSplitWidth`](@ref)
"""
struct ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,RuntimeActivity,ModifiedBetween,ABI,Holomorphic,ErrIfFuncWritten,ShadowInit} <: Mode{ABI, ErrIfFuncWritten,RuntimeActivity} end

"""
    const ReverseSplitNoPrimal

Default instance of [`ReverseModeSplit`](@ref) that doesn't return the primal
"""
const ReverseSplitNoPrimal = ReverseModeSplit{false, true, false, 0, true,DefaultABI, false, false, false}()

"""
    const ReverseSplitWithPrimal

Default instance of [`ReverseModeSplit`](@ref) that also returns the primal
"""
const ReverseSplitWithPrimal = ReverseModeSplit{true, true, false, 0, true,DefaultABI, false, false, false}()

"""
    ReverseSplitModified(::ReverseModeSplit, ::Val{MB})

Return a new instance of [`ReverseModeSplit`](@ref) mode where `ModifiedBetween` is set to `MB`. 
"""
@inline ReverseSplitModified(::ReverseModeSplit{ReturnPrimal, ReturnShadow, RuntimeActivity, Width, MBO, ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, ::Val{MB}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,MB,MBO,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity, Width,MB,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

"""
    ReverseSplitWidth(::ReverseModeSplit, ::Val{W})

Return a new instance of [`ReverseModeSplit`](@ref) mode where `Width` is set to `W`. 
"""
@inline ReverseSplitWidth(::ReverseModeSplit{ReturnPrimal, ReturnShadow, RuntimeActivity, WidthO, MB, ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, ::Val{Width}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,MB,WidthO,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,MB,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

@inline set_err_if_func_written(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, true, ShadowInit}()
@inline clear_err_if_func_written(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI, Holomorphic, false, ShadowInit}()

@inline set_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,true,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline set_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}, rt::Bool) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,rt,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline clear_runtime_activity(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{ReturnPrimal,ReturnShadow,false,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()

@inline WithPrimal(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{true,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()
@inline NoPrimal(::ReverseModeSplit{ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}) where {ReturnPrimal,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit} = ReverseModeSplit{false,ReturnShadow,RuntimeActivity,Width,ModifiedBetween,ABI, Holomorphic, ErrIfFuncWritten, ShadowInit}()


"""
    struct ForwardMode{
        ReturnPrimal,
        ABI,
        ErrIfFuncWritten,
        RuntimeActivity
    } <: Mode{ABI,ErrIfFuncWritten,RuntimeActivity}

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
struct ForwardMode{ReturnPrimal, ABI, ErrIfFuncWritten,RuntimeActivity} <: Mode{ABI, ErrIfFuncWritten, RuntimeActivity}
end

"""
    const Forward

Default instance of [`ForwardMode`](@ref) that doesn't return the primal
"""
const Forward = ForwardMode{false, DefaultABI, false, false}()

"""
    const ForwardWithPrimal

Default instance of [`ForwardMode`](@ref) that also returns the primal
"""
const ForwardWithPrimal = ForwardMode{true, DefaultABI, false, false}()

@inline set_err_if_func_written(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{ReturnPrimal,ABI,true,RuntimeActivity}()
@inline clear_err_if_func_written(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{ReturnPrimal,ABI,false,RuntimeActivity}()

@inline set_abi(::ForwardMode{ReturnPrimal,OldABI,ErrIfFuncWritten,RuntimeActivity}, ::Type{NewABI}) where {ReturnPrimal,OldABI,ErrIfFuncWritten,RuntimeActivity,NewABI<:ABI} = ForwardMode{ReturnPrimal,NewABI,ErrIfFuncWritten,RuntimeActivity}()

@inline set_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,true}()
@inline set_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}, rt::Bool) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,rt}()
@inline clear_runtime_activity(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,false}()

@inline WithPrimal(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{true,ABI,ErrIfFuncWritten,RuntimeActivity}()
@inline NoPrimal(::ForwardMode{ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity}) where {ReturnPrimal,ABI,ErrIfFuncWritten,RuntimeActivity} = ForwardMode{false,ABI,ErrIfFuncWritten,RuntimeActivity}()


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

Recursively set a variables differentiable fields to zero. Only applicable for mutable types `T`.
"""
function make_zero! end

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
    set_runtime_activity(::Mode, activitiy::Bool)
    set_runtime_activity(::Mode, config::Union{FwdConfig,RevConfig})

Return a new mode where runtime activity analysis is activated / set to the desired value.
"""
function set_runtime_activity end

"""
    clear_runtime_activity(::Mode)

Return a new mode where runtime activity analysis is deactivated.
"""
function clear_runtime_activity end

"""
    set_abi(::Mode, ::Type{ABI})

Return a new mode with its [`ABI`](@ref) set to the chosen type.
"""
function set_abi end

end # module EnzymeCore
