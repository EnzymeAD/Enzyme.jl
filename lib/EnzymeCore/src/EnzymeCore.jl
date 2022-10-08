module EnzymeCore

using Adapt

export Forward, Reverse
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed

function batch_size end

"""
    abstract type Annotation{T}

Abstract type for [`autodiff`](@ref) function argument wrappers like
[`Const`](@ref), [`Active`](@ref) and [`Duplicated`](@ref).
"""
abstract type Annotation{T} end
Base.eltype(::Type{<:Annotation{T}}) where T = T

"""
    Const(x)

Mark a function argument `x` of [`autodiff`](@ref) as constant,
Enzyme will not auto-differentiate in respect `Const` arguments.
"""
struct Const{T} <: Annotation{T}
    val::T
end
Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))

# To deal with Const(Int) and prevent it to go to `Const{DataType}(T)`
Const(::Type{T}) where T = Const{Type{T}}(T)

"""
    Active(x)

Mark a function argument `x` of [`autodiff`](@ref) as active,
Enzyme will auto-differentiate in respect `Active` arguments.

!!! note

    Enzyme gradients with respect to integer values are zero.
    [`Active`](@ref) will automatically convert plain integers to floating
    point values, but cannot do so for integer values in tuples and structs.
"""
struct Active{T} <: Annotation{T}
    val::T
end
Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))

Active(i::Integer) = Active(float(i))

"""
    Duplicated(x, ∂f_∂x)

Mark a function argument `x` of [`autodiff`](@ref) as duplicated, Enzyme will
auto-differentiate in respect to such arguments, with `dx` acting as an
accumulator for gradients (so ``\\partial f / \\partial x`` will be *added to*)
`∂f_∂x`.
"""
struct Duplicated{T} <: Annotation{T}
    val::T
    dval::T
end
Adapt.adapt_structure(to, x::Duplicated) = Duplicated(adapt(to, x.val), adapt(to, x.dval))

"""
    DuplicatedNoNeed(x, ∂f_∂x)

Like [`Duplicated`](@ref), except also specifies that Enzyme may avoid computing
the original result and only compute the derivative values.
"""
struct DuplicatedNoNeed{T} <: Annotation{T}
    val::T
    dval::T
end
Adapt.adapt_structure(to, x::DuplicatedNoNeed) = DuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))

"""
    BatchDuplicated(x, ∂f_∂xs)

Like [`Duplicated`](@ref), except contains several shadows to compute derivatives
for all at once. Argument `∂f_∂xs` should be a tuple of the several values of type `x`.
"""
struct BatchDuplicated{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,T}
end
Adapt.adapt_structure(to, x::BatchDuplicated) = BatchDuplicated(adapt(to, x.val), adapt(to, x.dval))

"""
    BatchDuplicatedNoNeed(x, ∂f_∂xs)

Like [`DuplicatedNoNeed`](@ref), except contains several shadows to compute derivatives
for all at once. Argument `∂f_∂xs` should be a tuple of the several values of type `x`.
"""
struct BatchDuplicatedNoNeed{T,N} <: Annotation{T}
    val::T
    dval::NTuple{N,T}
end
batch_size(::BatchDuplicated{T,N}) where {T,N} = N
batch_size(::BatchDuplicatedNoNeed{T,N}) where {T,N} = N
Adapt.adapt_structure(to, x::BatchDuplicatedNoNeed) = BatchDuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))

"""
    abstract type Mode

Abstract type for what differentiation mode will be used.
"""
abstract type Mode end

"""
    struct Reverse <: Mode

Reverse mode differentiation
"""
struct ReverseMode <: Mode
end
const Reverse = ReverseMode()

"""
    struct Forward <: Mode

Forward mode differentiation
"""
struct ForwardMode <: Mode
end
const Forward = ForwardMode()

end # module EnzymeCore
