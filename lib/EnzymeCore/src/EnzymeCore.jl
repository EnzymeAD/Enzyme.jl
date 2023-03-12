module EnzymeCore

using Adapt
using Preferences

export Forward, Reverse, ReverseWithPrimal, ReverseSplitNoPrimal, ReverseSplitWithPrimal
export ReverseSplitModified, ReverseSplitWidth
export Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed

const structure_check = parse(Bool, @load_preference("structure_check", "false"))

"""
    structure_check!(flag)

Toggle the default setting for congruence/structure checking.
"""
function structure_check!(flag)
    @set_preferences!("structure_check" => flag)
    @info("structure_check toggled, restart your Julia session for this change to take effect!")

    if VERSION <= v"1.6.5" || VERSION == v"1.7.0"
        @warn """
        Due to a bug in Julia (until 1.6.5 and 1.7.1), setting preferences in transitive dependencies
        is broken (https://github.com/JuliaPackaging/Preferences.jl/issues/24). To fix this either update
        your version of Julia, or add EnzyemCore as a direct dependency to your project.
        """
    end
    return nothing
end

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
Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))

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
end
Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))

Active(i::Integer) = Active(float(i))

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
    function Duplicated(val::T, dval::T; checked=structure_check) where T
        checked && check_congruence(val, dval)
        new{T}(val, dval)
    end
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
    function DuplicatedNoNeed(val::T, dval::T; checked=structure_check) where T
        checked && check_congruence(val, dval)
        new{T}(val, dval)
    end
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
    function BatchDuplicated(val::T, dval::NTuple{N,T}; checked=structure_check) where {T, N}
        checked && check_congruence(val, dval)
        new{T, N}(val, dval)
    end
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
    function BatchDuplicatedNoNeed(val::T, dval::NTuple{N,T}; checked=structure_check) where {T, N}
        checked && check_congruence(val, dval)
        new{T, N}(val, dval)
    end
end
batch_size(::BatchDuplicated{T,N}) where {T,N} = N
batch_size(::BatchDuplicatedNoNeed{T,N}) where {T,N} = N
Adapt.adapt_structure(to, x::BatchDuplicatedNoNeed) = BatchDuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))

"""
    congruent(a::T, b::T)::T

Defines values to be congruent, e.g. structurally equivalent.
"""
function congruent end

congruent(a, b) = false
congruent(a::T, b::T) where T<:Number = true
function congruent(a::T, b::T) where T<:DenseArray
    axes(a) == axes(b) && all(congruent, zip(a, b))
end
congruent(a::T, b::T) where T<:Ref = congruent(a[], b[])
congruent(a::T, b::T) where T<:Tuple = all(congruent, zip(a, b))
congruent(a::T, b::T) where T<:NamedTuple = all(congruent, zip(a, b))

congruent(tup::Tuple{T, T}) where T = congruent(tup...)

function check_congruence(a::T, b::T) where T
    # TODO: Use once hasmethod is static
    # if !hasmethod(congruent, Tuple{T, T})
    #     error("""
    #     Implement EnzymeCore.congruent(a, b) for your type $T
    #     """)
    # end
    if !congruent(a, b)
        error("""
        Your values are not congruent, structural equivalence is
        requirement for the correctness of the adjoint pass.

        You may need to implement EnzymeCore.congruent(a, b) for your type $T
        """)
    end
end

function check_congruence(a::T, b::NTuple{N, T}) where {N, T}
    ntuple(Val(N)) do i
        check_congruence(a, b[i])
    end
end

"""
    abstract type Mode

Abstract type for what differentiation mode will be used.
"""
abstract type Mode end

"""
    struct ReverseMode{ReturnPrimal} <: Mode

Reverse mode differentiation.
- `ReturnPrimal`: Should Enzyme return the primal return value from the augmented-forward.
"""
struct ReverseMode{ReturnPrimal} <: Mode end
const Reverse = ReverseMode{false}()
const ReverseWithPrimal = ReverseMode{true}()

"""
    struct ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetween} <: Mode

Reverse mode differentiation.
- `ReturnPrimal`: Should Enzyme return the primal return value from the augmented-forward.
- `ReturnShadow`: Should Enzyme return the shadow return value from the augmented-forward.
- `Width`: Batch Size (0 if to be automatically derived)
- `ModifiedBetween`: Tuple of each argument's modified between state (true if to be automatically derived).
"""
struct ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,ModifiedBetween} <: Mode end
const ReverseSplitNoPrimal = ReverseModeSplit{false, true, 0, true}()
const ReverseSplitWithPrimal = ReverseModeSplit{true, true, 0, true}()
@inline ReverseSplitModified(::ReverseModeSplit{ReturnPrimal, ReturnShadow, Width, MBO}, ::Val{MB}) where {ReturnPrimal,ReturnShadow,Width,MB,MBO} = ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,MB}()
@inline ReverseSplitWidth(::ReverseModeSplit{ReturnPrimal, ReturnShadow, WidthO, MB}, ::Val{Width}) where {ReturnPrimal,ReturnShadow,Width,MB,WidthO} = ReverseModeSplit{ReturnPrimal,ReturnShadow,Width,MB}()
"""
    struct Forward <: Mode

Forward mode differentiation
"""
struct ForwardMode <: Mode
end
const Forward = ForwardMode()

function autodiff end
function autodiff_deferred end
function autodiff_thunk end

include("rules.jl")

end # module EnzymeCore
