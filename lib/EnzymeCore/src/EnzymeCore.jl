module EnzymeCore

export Forward, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal
export ReverseSplitModified, ReverseSplitWidth, ReverseHolomorphic, ReverseHolomorphicWithPrimal
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed
export DefaultABI, FFIABI, InlineABI
export BatchDuplicatedFunc

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
the original result and only compute the derivative values.
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
    abstract type ABI

Abstract type for what ABI  will be used.
"""
abstract type ABI end

"""
    struct FFIABI <: ABI

Foreign function call ABI. JIT the differentiated function, then inttoptr call the address.
"""
struct FFIABI <: ABI end
"""
    struct InlineABI <: ABI

Inlining function call ABI. 
"""
struct InlineABI <: ABI end
const DefaultABI = FFIABI

"""
    abstract type Mode

Abstract type for what differentiation mode will be used.
"""
abstract type Mode{ABI} end

"""
    struct ReverseMode{ReturnPrimal,ABI,Holomorphic} <: Mode{ABI}

Reverse mode differentiation.
- `ReturnPrimal`: Should Enzyme return the primal return value from the augmented-forward.
- `ABI`: What runtime ABI to use
- `Holomorphic`: Whether the complex result function is holomorphic and we should compute d/dz
"""
struct ReverseMode{ReturnPrimal,ABI,Holomorphic} <: Mode{ABI} end
const Reverse = ReverseMode{false,DefaultABI, false}()
const ReverseWithPrimal = ReverseMode{true,DefaultABI, false}()
const ReverseHolomorphic = ReverseMode{false,DefaultABI, true}()
const ReverseHolomorphicWithPrimal = ReverseMode{true,DefaultABI, true}()

"""
    struct ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI} <: Mode{ABI}

Reverse mode differentiation.
- `ReturnPrimal`: Should Enzyme return the primal return value from the augmented-forward.
- `ReturnShadow`: Should Enzyme return the shadow return value from the augmented-forward.
- `Width`: Batch Size (0 if to be automatically derived)
- `ModifiedBetween`: Tuple of each argument's modified between state (true if to be automatically derived).
"""
struct ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetween,ABI} <: Mode{ABI} end
const ReverseSplitNoPrimal = ReverseModeSplit{false, true, 0, true,DefaultABI}()
const ReverseSplitWithPrimal = ReverseModeSplit{true, true, 0, true,DefaultABI}()
@inline ReverseSplitModified(::ReverseModeSplit{ReturnPrimal, ReturnShadow, Width, MBO, ABI}, ::Val{MB}) where {ReturnPrimal,ReturnShadow,Width,MB,MBO,ABI} = ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,MB,ABI}()
@inline ReverseSplitWidth(::ReverseModeSplit{ReturnPrimal, ReturnShadow, WidthO, MB, ABI}, ::Val{Width}) where {ReturnPrimal,ReturnShadow,Width,MB,WidthO,ABI} = ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,MB,ABI}()
"""
    struct Forward <: Mode

Forward mode differentiation
"""
struct ForwardMode{ABI} <: Mode{ABI}
end
const Forward = ForwardMode{DefaultABI}()

function autodiff end
function autodiff_deferred end
function autodiff_thunk end
function autodiff_deferred_thunk end

"""
    make_zero(::Type{T}, seen::IdDict, prev::T, ::Val{copy_if_inactive}=Val(false))::T

    Recursively make a zero'd copy of the value `prev` of type `T`. The argument `copy_if_inactive` specifies
    what to do if the type `T` is guaranteed to be inactive, use the primal (the default) or still copy the value. 
"""
function make_zero

"""
    make_zero!(prev::T)::T

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

end # module EnzymeCore
