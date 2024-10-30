module RecursiveMapTests

using Enzyme
using JLArrays
using Logging
using StaticArrays
using Test

# Universal getters/setters for built-in and custom containers/wrappers
getx(w::Base.RefValue) = w[]
getx(w::Core.Box) = w.contents
getx(w::JLArray) = JLArrays.@allowscalar first(w)
gety(w::JLArray) = JLArrays.@allowscalar last(w)
getx(w) = first(w)
gety(w) = last(w)

setx!(w::Base.RefValue, x) = (w[] = x)
setx!(w::Core.Box, x) = (w.contents = x)
setx!(w, x) = (w[begin] = x)
sety!(w, y) = (w[end] = y)

# non-isbits MArray doesn't support setindex!, so requires a little hack
function setx!(w::MArray{S,T}, x) where {S,T}
    if isbitstype(T)
        w[begin] = x
    else
        w.data = (x, Base.tail(w.data)...)
    end
    return x
end

function sety!(w::MArray{S,T}, y) where {S,T}
    if isbitstype(T)
        w[end] = y
    else
        w.data = (Base.front(w.data)..., y)
    end
    return y
end

struct Empty end

mutable struct MutableEmpty end

Base.:(==)(::MutableEmpty, ::MutableEmpty) = true

struct Wrapper{T}
    x::T
end

Base.:(==)(a::Wrapper, b::Wrapper) = (a === b) || (a.x == b.x)
getx(a::Wrapper) = a.x

mutable struct MutableWrapper{T}
    x::T
end

Base.:(==)(a::MutableWrapper, b::MutableWrapper) = (a === b) || (a.x == b.x)

getx(a::MutableWrapper) = a.x
setx!(a::MutableWrapper, x) = (a.x = x)

struct DualWrapper{Tx,Ty}
    x::Tx
    y::Ty
end

DualWrapper{T}(x::T, y) where {T} = DualWrapper{T,typeof(y)}(x, y)

function Base.:(==)(a::DualWrapper, b::DualWrapper)
    return (a === b) || ((a.x == b.x) && (a.y == b.y))
end

getx(a::DualWrapper) = a.x
gety(a::DualWrapper) = a.y

mutable struct MutableDualWrapper{Tx,Ty}
    x::Tx
    y::Ty
end

MutableDualWrapper{T}(x::T, y) where {T} = MutableDualWrapper{T,typeof(y)}(x, y)

function Base.:(==)(a::MutableDualWrapper, b::MutableDualWrapper)
    return (a === b) || ((a.x == b.x) && (a.y == b.y))
end

getx(a::MutableDualWrapper) = a.x
gety(a::MutableDualWrapper) = a.y

setx!(a::MutableDualWrapper, x) = (a.x = x)
sety!(a::MutableDualWrapper, y) = (a.y = y)

struct Incomplete{T,U}
    s::String
    x::Float64
    w::T
    y::U  # possibly not initialized
    z  # not initialized
    Incomplete(s, x, w) = new{typeof(w),Any}(s, x, w)
    Incomplete(s, x, w, y) = new{typeof(w),typeof(y)}(s, x, w, y)
end

function Base.:(==)(a::Incomplete, b::Incomplete)
    (a === b) && return true
    ((a.s == b.s) && (a.x == b.x) && (a.w == b.w)) || return false
    if isdefined(a, :y) && isdefined(b, :y)
        (a.w == b.w) || return false
        if isdefined(a, :z) && isdefined(b, :z)
            (a.z == b.z) || return false
        elseif isdefined(a, :z) || isdefined(b, :z)
            return false
        end
    elseif isdefined(a, :y) || isdefined(b, :y)
        return false
    end
    return true
end

mutable struct MutableIncomplete{T}
    s::String
    const x::Float64
    y::Float64
    z  # not initialized
    w::T
    function MutableIncomplete(s, x, y, w)
        ret = new{typeof(w)}(s, x, y)
        ret.w = w
        return ret
    end
end

function Base.:(==)(a::MutableIncomplete, b::MutableIncomplete)
    (a === b) && return true
    if (a.s != b.s) || (a.x != b.x) || (a.y != b.y) || (a.w != b.w)
        return false
    end
    if isdefined(a, :z) && isdefined(b, :z)
        (a.z == b.z) || return false
    elseif isdefined(a, :z) || isdefined(b, :z)
        return false
    end
    return true
end

mutable struct CustomVector{T} <: AbstractVector{T}
    data::Vector{T}
end

Base.:(==)(a::CustomVector, b::CustomVector) = (a === b) || (a.data == b.data)

function Enzyme.EnzymeCore.isvectortype(::Type{CustomVector{T}}) where {T}
    return Enzyme.EnzymeCore.isscalartype(T)
end

function Enzyme.EnzymeCore.make_zero(prev::CV) where {CV<:CustomVector{<:AbstractFloat}}
    @info "make_zero(::CustomVector)"
    return CustomVector(zero(prev.data))::CV
end

function Enzyme.EnzymeCore.make_zero!(prev::CustomVector{<:AbstractFloat})
    @info "make_zero!(::CustomVector)"
    fill!(prev.data, false)
    return nothing
end

struct WithIO{F}  # issue 2091
    v::Vector{Float64}
    callback::F
    function WithIO(v, io)
        callback() = println(io, "hello")
        return new{typeof(callback)}(v, callback)
    end
end

macro test_noerr(expr)
    return quote
        @test_nowarn try
            # catch errors to get failed test instead of "exception outside of a @test"
            $(esc(expr))
        catch e
            showerror(stderr, e)
        end
    end
end

const scalartypes = [Float32, ComplexF32, Float64, ComplexF64, BigFloat, Complex{BigFloat}]

const inactivebits = (1, Empty())
const inactivetup = (inactivebits, "a", MutableEmpty())
const inactivearr = [inactivetup]

const wrappers = [
    (name="Tuple{X}",                     f=tuple,                                           N=1, mutable=false, typed=true,      bitsonly=false),
    (name="@NamedTuple{x::X}",            f=(NamedTuple{(:x,)} ∘ tuple),                     N=1, mutable=false, typed=true,      bitsonly=false),
    (name="struct{X}",                    f=Wrapper,                                         N=1, mutable=false, typed=true,      bitsonly=false),

    (name="@NamedTuple{x}",               f=(@NamedTuple{x} ∘ tuple),                        N=1, mutable=false, typed=false,     bitsonly=false),
    (name="struct{Any}",                  f=Wrapper{Any},                                    N=1, mutable=false, typed=false,     bitsonly=false),

    (name="Array{X}",                     f=(x -> [x]),                                      N=1, mutable=true,  typed=true,      bitsonly=false),
    (name="Base.RefValue{X}",             f=Ref,                                             N=1, mutable=true,  typed=true,      bitsonly=false),
    (name="mutable struct{X}",            f=MutableWrapper,                                  N=1, mutable=true,  typed=true,      bitsonly=false),

    (name="Array{Any}",                   f=(x -> Any[x]),                                   N=1, mutable=true,  typed=false,     bitsonly=false),
    (name="Base.RefValue{Any}",           f=Ref{Any},                                        N=1, mutable=true,  typed=false,     bitsonly=false),
    (name="Core.Box",                     f=Core.Box,                                        N=1, mutable=true,  typed=false,     bitsonly=false),
    (name="mutable struct{Any}",          f=MutableWrapper{Any},                             N=1, mutable=true,  typed=false,     bitsonly=false),

    (name="Tuple{X,Y}",                   f=tuple,                                           N=2, mutable=false, typed=true,      bitsonly=false),
    (name="@NamedTuple{x::X,y::Y}",       f=(NamedTuple{(:x, :y)} ∘ tuple),                  N=2, mutable=false, typed=true,      bitsonly=false),
    (name="struct{X,Y}",                  f=DualWrapper,                                     N=2, mutable=false, typed=true,      bitsonly=false),

    (name="@NamedTuple{x,y::Y}",          f=((x, y) -> @NamedTuple{x,y::typeof(y)}((x, y))), N=2, mutable=false, typed=:partial,  bitsonly=false),
    (name="struct{Any,Y}",                f=DualWrapper{Any},                                N=2, mutable=false, typed=:partial,  bitsonly=false),

    (name="@NamedTuple{x,y}",             f=(@NamedTuple{x,y} ∘ tuple),                      N=2, mutable=false, typed=false,     bitsonly=false),
    (name="struct{Any}",                  f=DualWrapper{Any,Any},                            N=2, mutable=false, typed=false,     bitsonly=false),

    (name="mutable struct{X,Y}",          f=MutableDualWrapper,                              N=2, mutable=true,  typed=true,      bitsonly=false),

    (name="Array{promote_type(X,Y)}",     f=((x, y) -> [x, y]),                              N=2, mutable=true,  typed=:promoted, bitsonly=false),
    (name="mutable struct{Any,Y}",        f=MutableDualWrapper{Any},                         N=2, mutable=true,  typed=:partial,  bitsonly=false),

    (name="Array{Any}",                   f=((x, y) -> Any[x, y]),                           N=2, mutable=true,  typed=false,     bitsonly=false),
    (name="mutable struct{Any,Any}",      f=MutableDualWrapper{Any,Any},                     N=2, mutable=true,  typed=false,     bitsonly=false),

    # StaticArrays extension
    (name="SVector{1,X}",                 f=(SVector{1} ∘ tuple),                            N=1, mutable=false, typed=true,      bitsonly=false),
    (name="SVector{1,Any}",               f=(SVector{1,Any} ∘ tuple),                        N=1, mutable=false, typed=false,     bitsonly=false),
    (name="MVector{1,X}",                 f=(MVector{1} ∘ tuple),                            N=1, mutable=true,  typed=true,      bitsonly=false),
    (name="MVector{1,Any}",               f=(MVector{1,Any} ∘ tuple),                        N=1, mutable=true,  typed=false,     bitsonly=false),
    (name="SVector{2,promote_type(X,Y)}", f=(SVector{2} ∘ tuple),                            N=2, mutable=false, typed=:promoted, bitsonly=false),
    (name="SVector{2,Any}",               f=(SVector{2,Any} ∘ tuple),                        N=2, mutable=false, typed=false,     bitsonly=false),
    (name="MVector{2,promote_type(X,Y)}", f=(MVector{2} ∘ tuple),                            N=2, mutable=true,  typed=:promoted, bitsonly=false),
    (name="MVector{2,Any}",               f=(MVector{2,Any} ∘ tuple),                        N=2, mutable=true,  typed=false,     bitsonly=false),

    # GPUArrays extension
    (name="JLArray{X}",                     f=(x -> JLArray([x])),                           N=1, mutable=true,  typed=true,      bitsonly=true),
    (name="JLArray{promote_type(X,Y)}",     f=((x, y) -> JLArray([x, y])),                   N=2, mutable=true,  typed=:promoted, bitsonly=true),
]

@static if VERSION < v"1.11-"
else
_memory(x::Vector) = Memory{eltype(x)}(x)
push!(
    wrappers,
    (name="Memory{X}",                    f=(x -> _memory([x])),                             N=1, mutable=true,  typed=true,      bitsonly=false),
    (name="Memory{Any}",                  f=(x -> _memory(Any[x])),                          N=1, mutable=true,  typed=false,     bitsonly=false),
    (name="Memory{promote_type(X,Y)}",    f=((x, y) -> _memory([x, y])),                     N=2, mutable=true,  typed=:promoted, bitsonly=false),
    (name="Memory{Any}",                  f=((x, y) -> _memory(Any[x, y])),                  N=2, mutable=true,  typed=false,     bitsonly=false),
)
end

function test_make_zero()
    @testset "scalars" begin
        @testset "$T" for T in scalartypes
            x = oneunit(T)
            x_makez = make_zero(x)
            @test typeof(x_makez) === T  # correct type
            @test x_makez == zero(T)     # correct value
            @test x == oneunit(T)        # no mutation of original (relevant for BigFloat)
        end
    end
    @testset "nested types" begin
        @testset "$T in $(wrapper.name)" for T in scalartypes, wrapper in filter(
            w -> (w.N == 1), wrappers
        )
            (!wrapper.bitsonly || isbitstype(T)) || continue
            x = oneunit(T)
            w = wrapper.f(x)
            w_makez = make_zero(w)
            @test typeof(w_makez) === typeof(w)  # correct type
            @test typeof(getx(w_makez)) === T    # correct type
            @test getx(w_makez) == zero(T)       # correct value
            @test getx(w) === x                  # no mutation of original
            @test x == oneunit(T)                # no mutation of original (relevant for BigFloat)
            @testset "doubly included in $(dualwrapper.name)" for dualwrapper in filter(
                w -> (w.N == 2), wrappers
            )
                (!dualwrapper.bitsonly || isbitstype(T)) || continue
                w_inner = wrapper.f(x)
                if !dualwrapper.bitsonly || isbits(w_inner)
                    d_outer = dualwrapper.f(w_inner, w_inner)
                    d_outer_makez = make_zero(d_outer)
                    @test typeof(d_outer_makez) === typeof(d_outer)        # correct type
                    @test typeof(getx(d_outer_makez)) === typeof(w_inner)  # correct type
                    @test typeof(getx(getx(d_outer_makez))) === T          # correct type
                    @test getx(d_outer_makez) === gety(d_outer_makez)      # correct layout
                    @test getx(getx(d_outer_makez)) == zero(T)             # correct value
                    @test getx(d_outer) === gety(d_outer)                  # no mutation of original
                    @test getx(d_outer) === w_inner                        # no mutation of original
                    @test getx(w_inner) === x                              # no mutation of original
                    @test x == oneunit(T)                                  # no mutation of original (relevant for BigFloat)
                end
                d_inner = dualwrapper.f(x, x)
                if !wrapper.bitsonly || isbits(d_inner)
                    w_outer = wrapper.f(d_inner)
                    w_outer_makez = make_zero(w_outer)
                    @test typeof(w_outer_makez) === typeof(w_outer)               # correct type
                    @test typeof(getx(w_outer_makez)) === typeof(d_inner)         # correct type
                    @test typeof(getx(getx(w_outer_makez))) === T                 # correct type
                    @test getx(getx(w_outer_makez)) == gety(getx(w_outer_makez))  # correct layout
                    @test getx(getx(w_outer_makez)) == zero(T)                    # correct value
                    @test getx(w_outer) === d_inner                               # no mutation of original
                    @test getx(d_inner) === gety(d_inner)                         # no mutation of original
                    @test getx(d_inner) === x                                     # no mutation of original
                    @test x == oneunit(T)                                         # no mutation of original (relevant for BigFloat)
                end
                if wrapper.mutable && !dualwrapper.mutable && !dualwrapper.bitsonly
                    # some code paths can only be hit with three layers of wrapping:
                    # mutable(immutable(mutable(scalar)))
                    @testset "all wrapped in $(outerwrapper.name)" for outerwrapper in filter(
                        w -> ((w.N == 1) && w.mutable && !w.bitsonly), wrappers
                    )
                        w_inner = wrapper.f(x)
                        d_middle = dualwrapper.f(w_inner, w_inner)
                        w_outer = outerwrapper.f(d_middle)
                        w_outer_makez = make_zero(w_outer)
                        @test typeof(w_outer_makez) === typeof(w_outer)                 # correct type
                        @test typeof(getx(w_outer_makez)) === typeof(d_middle)          # correct type
                        @test typeof(getx(getx(w_outer_makez))) === typeof(w_inner)     # correct type
                        @test typeof(getx(getx(getx(w_outer_makez)))) === T             # correct type
                        @test getx(getx(w_outer_makez)) === gety(getx(w_outer_makez))   # correct layout
                        @test getx(getx(getx(w_outer_makez))) == zero(T)                # correct value
                        @test getx(w_outer) === d_middle                                # no mutation of original
                        @test getx(d_middle) === gety(d_middle)                         # no mutation of original
                        @test getx(d_middle) === w_inner                                # no mutation of original
                        @test getx(w_inner) === x                                       # no mutation of original
                        @test x == oneunit(T)                                           # no mutation of original (relevant for BigFloat)
                    end
                end
            end
        end
    end
    @testset "inactive" begin
        @testset "in $(wrapper.name)" for wrapper in wrappers
            if wrapper.N == 1
                for (inactive, condition) in [
                    (inactivebits, true),
                    (inactivearr, !wrapper.bitsonly),
                ]
                    condition || continue
                    w = wrapper.f(inactive)
                    w_makez = make_zero(w)
                    if wrapper.typed in (true, :promoted)
                        if w isa JLArray  # needs JLArray activity
                            @test_broken w_makez === w
                        else
                            @test w_makez === w               # preserved wrapper identity if guaranteed const
                        end
                    end
                    @test typeof(w_makez) === typeof(w)       # correct type
                    @test getx(w_makez) === inactive          # preserved identity
                    @test getx(w) === inactive                # no mutation of original
                    if inactive === inactivearr
                        @test inactivearr[1] === inactivetup  # preserved value
                    end
                end
                @testset "mixed" begin
                    for (inactive, mixed, condition) in [
                        (inactivebits, (1.0, inactivebits), true),
                        (inactivearr, [1.0, inactivearr], !wrapper.bitsonly),
                    ]
                        condition || continue
                        w = wrapper.f(mixed)
                        w_makez = make_zero(w)
                        @test typeof(w_makez) === typeof(w)            # correct type
                        @test typeof(getx(w_makez)) === typeof(mixed)  # correct type
                        @test getx(w_makez)[1] === 0.0                 # correct value
                        @test getx(w_makez)[2] === inactive            # preserved inactive identity
                        @test getx(w) === mixed                        # no mutation of original
                        if inactive === inactivearr
                            @test inactivearr[1] === inactivetup       # preserved inactive value
                            @test mixed[1] === 1.0                     # no mutation of original
                            @test mixed[2] === inactivearr             # no mutation of original
                        end
                    end
                end
            else  # wrapper.N == 2
                @testset "multiple references" begin
                    for (inactive, condition) in [
                        (inactivebits, true),
                        (inactivearr, !wrapper.bitsonly),
                    ]
                        condition || continue
                        w = wrapper.f(inactive, inactive)
                        w_makez = make_zero(w)
                        if wrapper.typed in (true, :promoted)
                            if w isa JLArray  # needs JLArray activity
                                @test_broken w_makez === w
                            else
                                @test w_makez === w                # preserved wrapper identity if guaranteed const
                            end
                        end
                        @test typeof(w_makez) === typeof(w)        # correct type
                        @test getx(w_makez) === gety(w_makez)      # preserved layout
                        @test getx(w_makez) === inactive           # preserved identity
                        @test getx(w) === gety(w)                  # no mutation of original
                        @test getx(w) === inactive                 # no mutation of original
                        if inactive === inactive
                            @test inactivearr[1] === inactivetup   # preserved value
                        end
                    end
                end
                if !wrapper.bitsonly
                    @testset "mixed" begin
                        a = [1.0]
                        w = wrapper.f(a, inactivearr)
                        w_makez = make_zero(w)
                        @test typeof(w_makez) === typeof(w)        # correct type
                        @test typeof(getx(w_makez)) === typeof(a)  # correct type
                        @test getx(w_makez) == [0.0]               # correct value
                        @test gety(w_makez) === inactivearr        # preserved inactive identity
                        @test inactivearr[1] === inactivetup       # preserved inactive value
                        @test getx(w) === a                        # no mutation of original
                        @test a[1] === 1.0                         # no mutation of original
                        @test gety(w) === inactivearr              # no mutation of original
                        if wrapper.typed == :partial
                            # above: untyped active   / typed inactive
                            # below: untyped inactive / typed active
                            w = wrapper.f(inactivearr, a)
                            w_makez = make_zero(w)
                            @test typeof(w_makez) === typeof(w)        # correct type
                            @test getx(w_makez) === inactivearr        # preserved inactive identity
                            @test inactivearr[1] === inactivetup       # preserved inactive value
                            @test typeof(gety(w_makez)) === typeof(a)  # correct type
                            @test gety(w_makez) == [0.0]               # correct value
                            @test getx(w) === inactivearr              # no mutation of original
                            @test gety(w) === a                        # no mutation of original
                            @test a[1] === 1.0                         # no mutation of original
                        end
                    end
                end
            end
        end
        @testset "copy_if_inactive $value" for (value, args) in [
            ("unspecified", ()),
            ("= false",     (Val(false),)),
            ("= true",      (Val(true),)),
        ]
            a = [1.0]
            w = Any[a, inactivearr, inactivearr]
            w_makez = make_zero(w, args...)
            @test typeof(w_makez) === typeof(w)                  # correct type
            @test typeof(w_makez[1]) === typeof(a)               # correct type
            @test w_makez[1] == [0.0]                            # correct value
            @test w_makez[2] === w_makez[3]                      # correct layout (layout should propagate even when copy_if_inactive = Val(true))
            @test w[1] === a                                     # no mutation of original
            @test a[1] === 1.0                                   # no mutation of original
            @test w[2] === w[3]                                  # no mutation of original
            @test w[2] === inactivearr                           # no mutation of original
            @test inactivearr[1] === inactivetup                 # no mutation of original
            if args == (Val(true),)
                @test typeof(w_makez[2]) === typeof(inactivearr)  # correct type
                @test w_makez[2] == inactivearr                   # correct value
                @test w_makez[2][1] !== inactivetup               # correct identity
            else
                @test w_makez[2] === inactivearr                  # correct value/type/identity
            end
        end
    end
    @testset "heterogeneous containers" begin
        scalars, scalarsz = oneunit.(scalartypes), zero.(scalartypes)
        wraps, wrapsz = Wrapper.(scalars), Wrapper.(scalarsz)
        mwraps, mwrapsz = MutableWrapper.(scalars), MutableWrapper.(scalarsz)
        items = (inactivetup..., scalars..., wraps..., mwraps...)
        itemsz = (inactivetup..., scalarsz..., wrapsz..., mwrapsz...)
        labels = Symbol.("i" .* string.(1:length(items)))
        @testset "$name" for (name, c, cz) in [
            ("Tuple",      Tuple(items),                 Tuple(itemsz)),
            ("NamedTuple", NamedTuple(labels .=> items), NamedTuple(labels .=> itemsz)),
            ("Array",      collect(items),               collect(itemsz)),
        ]
            c_makez = make_zero(c)
            @test typeof(c_makez) === typeof(c)                                     # correct type
            @test all(typeof(czj) === typeof(cj) for (czj, cj) in zip(c_makez, c))  # correct type
            @test c_makez == cz                                                     # correct value
            @test all(czj === inj for (czj, inj) in zip(c_makez, inactivetup))      # preserved inactive identities
            @test all(cj === itj for (cj, itj) in zip(c, items))                    # no mutation of original
            @test all(m.x == oneunit(m.x) for m in mwraps)                          # no mutation of original
        end
    end
    @testset "heterogeneous float arrays" begin
        b1r, b2r = big"1.0", big"2.0"
        b1i, b2i = big"1.0" * im, big"2.0" * im
        ar = AbstractFloat[1.0f0, 1.0, b1r, b1r, b2r]
        ai = Complex{<:AbstractFloat}[1.0f0im, 1.0im, b1i, b1i, b2i]
        for (a, btype) in [(ar, typeof(b1r)), (ai, typeof(b1i))]
            a_makez = make_zero(a)
            @test a_makez[1] === zero(a[1])
            @test a_makez[2] === zero(a[2])
            @test typeof(a_makez[3]) === btype
            @test a_makez[3] == 0
            @test a_makez[4] === a_makez[3]
            @test typeof(a_makez[5]) === btype
            @test a_makez[5] == 0
            @test a_makez[5] !== a_makez[3]
        end
    end
    @testset "circular references" begin
        @testset "$(wrapper.name)" for wrapper in filter(
            w -> (w.mutable && (w.typed in (:partial, false))), wrappers
        )
            a = [1.0]
            if wrapper.N == 1
                w = wrapper.f(nothing)
                setx!(w, (w, a))
            else
                w = wrapper.f(nothing, a)
                setx!(w, w)
            end
            w_makez = @test_noerr make_zero(w)
            if wrapper.N == 1
                xz, yz = getx(w_makez)
                x, y = getx(w)
            else
                xz, yz = getx(w_makez), gety(w_makez)
                x, y = getx(w), gety(w)
            end
            @test typeof(w_makez) === typeof(w)  # correct type
            @test typeof(xz) === typeof(w)       # correct type
            @test typeof(yz) === typeof(a)       # correct type
            @test xz === w_makez                 # correct self-reference
            @test yz == [0.0]                    # correct value
            @test x === w                        # no mutation of original
            @test y === a                        # no mutation of original
            @test a[1] === 1.0                   # no mutation of original
        end
    end
    @testset "bring your own IdDict" begin
        a = [1.0]
        seen = IdDict()
        a_makez = make_zero(typeof(a), seen, a)
        @test typeof(a_makez) === typeof(a)  # correct type
        @test a_makez == [0.0]               # correct value
        @test a[1] === 1.0                   # no mutation of original
        @test haskey(seen, a)                # original added to IdDict
        @test seen[a] === a_makez            # original points to zeroed value
    end
    @testset "custom leaf type" begin
        a = [1.0]
        v = CustomVector(a)
        # include optional arg Val(false) to avoid calling the custom method directly;
        # it should still be invoked
        v_makez = @test_logs (:info, "make_zero(::CustomVector)") make_zero(v, Val(false))
        @test typeof(v_makez) === typeof(v)       # correct type
        @test typeof(v_makez.data) === typeof(a)  # correct type
        @test v_makez == CustomVector([0.0])      # correct value
        @test v.data === a                        # no mutation of original
        @test a[1] === 1.0                        # no mutation of original
    end
    @testset "runtime inactive" begin
        a = [1.0]
        v = CustomVector(a)
        with_logger(SimpleLogger(Warn)) do  # silence @info "make_zero(::CustomVector)"
            # ensure compile-time methods are evaluated while CustomVector is considered active
            @assert !EnzymeRules.inactive_type(CustomVector)
            v_makez = make_zero(v, Val(false), Val(false))
            @assert v_makez == CustomVector([0.0])

            # verify that runtime methods also see CustomVector as active
            v_makez = make_zero(v, Val(false), Val(true))
            @test v_makez == CustomVector([0.0])

            # mark CustomVector as inactive
            @eval @inline EnzymeRules.inactive_type(::Type{<:CustomVector}) = true

            # runtime_inactive == false => redefined inactive_type should have no effect
            v_makez = @invokelatest make_zero(v, Val(false), Val(false))
            @test v_makez == CustomVector([0.0])

            # runtime_inactive == true => redefined inactive_type should take effect:
            # CustomVector considered inactive and won't be zeroed, but
            # shared/copied according to copy_if_inactive instead
            v_makez = @invokelatest make_zero(v, Val(false), Val(true))
            @test v_makez === v
            v_makez = @invokelatest make_zero(v, Val(true), Val(true))
            @test v_makez !== v
            @test v_makez == CustomVector([1.0])

            # mark CustomVector as active again
            @eval @inline EnzymeRules.inactive_type(::Type{<:CustomVector}) = false

            # verify that both compile-time and runtime methods see CustomVector as active
            v_makez = @invokelatest make_zero(v, Val(false), Val(false))
            @test v_makez == CustomVector([0.0])
            v_makez = @invokelatest make_zero(v, Val(false), Val(true))
            @test v_makez == CustomVector([0.0])
        end
    end
    @testset "undefined fields/unassigned elements" begin
        @testset "array w inactive/active/mutable/unassigned" begin
            a = [1.0]
            values = ("a", 1.0, a)
            arr = Vector{Any}(undef, 4)
            arr[1:3] .= values
            arr_makez = make_zero(arr)
            @views begin
                @test typeof(arr_makez) === typeof(arr)                  # correct type
                @test all(typeof.(arr_makez[1:3]) .=== typeof.(values))  # correct type
                @test arr_makez[1:3] == ["a", 0.0, [0.0]]                # correct value
                @test !isassigned(arr_makez, 4)                          # propagated undefined
                @test all(arr[1:3] .=== values)                          # no mutation of original
                @test !isassigned(arr, 4)                                # no mutation of original
                @test a[1] === 1.0                                       # no mutation of original
            end
        end
        @testset "struct w inactive/active/mutable/undefined" begin
            a = [1.0]
            @testset "single undefined" begin
                incomplete = Incomplete("a", 1.0, a, nothing)
                incomplete_makez = make_zero(incomplete)
                @test typeof(incomplete_makez) === typeof(incomplete)           # correct type
                @test typeof(incomplete_makez.w) === typeof(a)                  # correct type
                @test incomplete_makez == Incomplete("a", 0.0, [0.0], nothing)  # correct value, propagated undefined
                @test a[1] === 1.0                                              # no mutation of original
            end
            @testset "multiple undefined" begin
                incomplete = Incomplete("a", 1.0, a)
                incomplete_makez = make_zero(incomplete)
                @test typeof(incomplete_makez) === typeof(incomplete)  # correct type
                @test typeof(incomplete_makez.w) === typeof(a)         # correct type
                @test incomplete_makez == Incomplete("a", 0.0, [0.0])  # correct value, propagated undefined
                @test a[1] === 1.0                                     # no mutation of original
            end
        end
        @testset "mutable struct w inactive/const active/active/mutable/undefined" begin
            a = [1.0]
            incomplete = MutableIncomplete("a", #=const=#1.0, 1.0, a)
            incomplete_makez = make_zero(incomplete)
            @test typeof(incomplete_makez) === typeof(incomplete)              # correct type
            @test typeof(incomplete_makez.w) === typeof(a)                     # correct type
            @test incomplete_makez == MutableIncomplete("a", 0.0, 0.0, [0.0])  # correct value, propagated undefined
            @test incomplete == MutableIncomplete("a", 1.0, 1.0, a)            # no mutation of original
            @test incomplete.w === a                                           # no mutation of original
            @test a[1] === 1.0                                                 # no mutation of original
        end
    end
    @testset "containing IO" begin  # issue #2091
        f = WithIO([1.0, 2.0], stdout)
        df = @test_noerr make_zero(f)
        @test df.v == [0.0, 0.0]
        @test df.callback === f.callback
    end
    return nothing
end

function test_make_zero!()
    @testset "nested types" begin
        @testset "$T in $(wrapper.name)" for T in scalartypes, wrapper in filter(
            w -> (w.N == 1), wrappers
        )
            (!wrapper.bitsonly || isbitstype(T)) || continue
            x = oneunit(T)
            if wrapper.mutable
                w = wrapper.f(x)
                make_zero!(w)
                @test typeof(getx(w)) === T  # preserved type
                @test getx(w) == zero(T)     # correct value
                @test x == oneunit(T)        # no mutation of scalar (relevant for BigFloat)
            end
            @testset "doubly included in $(dualwrapper.name)" for dualwrapper in (
                filter(w -> ((w.N == 2) && (w.mutable || wrapper.mutable)), wrappers)
            )
                (!dualwrapper.bitsonly || isbitstype(T)) || continue
                w_inner = wrapper.f(x)
                if !dualwrapper.bitsonly || isbits(w_inner)
                    d_outer = dualwrapper.f(w_inner, w_inner)
                    make_zero!(d_outer)
                    @test typeof(getx(d_outer)) === typeof(w_inner)  # preserved type
                    @test typeof(getx(getx(d_outer))) === T          # preserved type
                    @test getx(getx(d_outer)) == zero(T)             # correct value
                    @test getx(d_outer) === gety(d_outer)            # preserved layout
                    @test x == oneunit(T)                            # no mutation of scalar (relevant for BigFloat)
                    if wrapper.mutable
                        @test getx(d_outer) === w_inner              # preserved identity
                    end
                end
                d_inner = dualwrapper.f(x, x)
                if !wrapper.bitsonly || isbits(d_inner)
                    w_outer = wrapper.f(d_inner)
                    make_zero!(w_outer)
                    @test typeof(getx(w_outer)) === typeof(d_inner)    # preserved type
                    @test typeof(getx(getx(w_outer))) === T            # preserved type
                    @test getx(getx(w_outer)) == zero(T)               # correct value
                    @test getx(getx(w_outer)) === gety(getx(w_outer))  # preserved layout
                    @test x == oneunit(T)                              # no mutation of scalar (relevant for BigFloat)
                    if dualwrapper.mutable
                        @test getx(w_outer) === d_inner                # preserved identity
                    end
                end
                if wrapper.mutable && !dualwrapper.mutable && !dualwrapper.bitsonly
                    # some code paths can only be hit with three layers of wrapping:
                    # mutable(immutable(mutable(scalar)))
                    @testset "all wrapped in $(outerwrapper.name)" for outerwrapper in filter(
                        w -> ((w.N == 1) && w.mutable && !w.bitsonly), wrappers
                    )
                        w_inner = wrapper.f(x)
                        d_middle = dualwrapper.f(w_inner, w_inner)
                        w_outer = outerwrapper.f(d_middle)
                        make_zero!(w_outer)
                        @test typeof(getx(w_outer)) === typeof(d_middle)       # preserved type
                        @test typeof(getx(getx(w_outer))) === typeof(w_inner)  # preserved type
                        @test typeof(getx(getx(getx(w_outer)))) === T          # preserved type
                        @test getx(getx(getx(w_outer))) == zero(T)             # correct value
                        @test getx(getx(w_outer)) === gety(getx(w_outer))      # preserved layout
                        @test getx(getx(w_outer)) === w_inner                  # preserved identity
                        @test x == oneunit(T)                                  # no mutation of scalar (relevant for BigFloat)
                    end
                end
            end
        end
    end
    @testset "inactive" begin
        @testset "in $(wrapper.name)" for wrapper in filter(
            w -> (w.mutable || (w.typed == true)), wrappers
        )
            if wrapper.N == 1
                for (inactive, condition) in [
                    (inactivebits, true),
                    (inactivearr, !wrapper.bitsonly),
                ]
                    condition || continue
                    w = wrapper.f(inactive)
                    make_zero!(w)
                    @test getx(w) === inactive                # preserved identity
                    if inactive === inactivearr
                        @test inactivearr[1] === inactivetup  # preserved value
                    end
                end
                @testset "mixed" begin
                    for (inactive, mixed, condition) in [
                        (inactivebits, (1.0, inactivebits), wrapper.mutable),
                        (inactivearr, [1.0, inactivearr], !wrapper.bitsonly),
                    ]
                        condition || continue
                        w = wrapper.f(mixed)
                        make_zero!(w)
                        @test getx(w)[1] === 0.0
                        @test getx(w)[2] === inactive
                        if inactive === inactivearr
                            @test getx(w) === mixed               # preserved identity
                            @test inactivearr[1] === inactivetup  # preserved value
                        end
                    end
                end
            else  # wrapper.N == 2
                @testset "multiple references" begin
                    for (inactive, condition) in [
                        (inactivebits, true),
                        (inactivearr, !wrapper.bitsonly),
                    ]
                        condition || continue
                        w = wrapper.f(inactive, inactive)
                        make_zero!(w)
                        @test getx(w) === gety(w)                 # preserved layout
                        @test getx(w) === inactive                # preserved identity
                        if inactive === inactivearr
                            @test inactivearr[1] === inactivetup  # preserved value
                        end
                    end
                end
                if !wrapper.bitsonly
                    @testset "mixed" begin
                        a = [1.0]
                        w = wrapper.f(a, inactivearr)
                        make_zero!(w)
                        @test getx(w) === a                   # preserved identity
                        @test a[1] === 0.0                    # correct value
                        @test gety(w) === inactivearr         # preserved inactive identity
                        @test inactivearr[1] === inactivetup  # preserved inactive value
                    end
                end
            end
        end
    end
    @testset "heterogeneous containers" begin
        mwraps = MutableWrapper.(oneunit.(scalartypes))
        mwrapsz = MutableWrapper.(zero.(scalartypes))
        items = (inactivetup..., mwraps...)
        itemsz = (inactivetup..., mwrapsz...)
        labels = Symbol.("i" .* string.(1:length(items)))
        @testset "$name" for (name, c, cz) in [
            ("Tuple",      Tuple(items),                 Tuple(itemsz)),
            ("NamedTuple", NamedTuple(labels .=> items), NamedTuple(labels .=> itemsz)),
            ("Array",      collect(items),               collect(itemsz)),
        ]
            make_zero!(c)
            @test all(cj === itj for (cj, itj) in zip(c, items))  # preserved identities
            @test c == cz                                         # correct value
        end
    end
    @testset "heterogeneous float arrays" begin
        b1r, b2r = big"1.0", big"2.0"
        b1i, b2i = big"1.0" * im, big"2.0" * im
        ar = AbstractFloat[1.0f0, 1.0, b1r, b1r, b2r]
        ai = Complex{<:AbstractFloat}[1.0f0im, 1.0im, b1i, b1i, b2i]
        for (a, btype) in [(ar, typeof(b1r)), (ai, typeof(b1i))]
            a1, a2 = a[1], a[2]
            make_zero!(a)
            @test a[1] === zero(a1)
            @test a[2] === zero(a2)
            @test typeof(a[3]) === btype
            @test a[3] == 0
            @test a[4] === a[3]
            @test typeof(a[5]) === btype
            @test a[5] == 0
            @test a[5] !== a[3]
        end
    end
    @testset "circular references" begin
        @testset "$(wrapper.name)" for wrapper in filter(
            w -> (w.mutable && (w.typed in (:partial, false))), wrappers
        )
            a = [1.0]
            if wrapper.N == 1
                w = wrapper.f(nothing)
                setx!(w, (w, a))
            else
                w = wrapper.f(nothing, a)
                setx!(w, w)
            end
            @test_noerr make_zero!(w)
            if wrapper.N == 1
                x, y = getx(w)
            else
                x, y = getx(w), gety(w)
            end
            @test x === w       # preserved self-referential identity
            @test y === a       # preserved identity
            @test a[1] === 0.0  # correct value
        end
    end
    @testset "bring your own IdDict" begin
        a = [1.0]
        seen = IdDict()
        make_zero!(a, seen)
        @test a[1] === 0.0     # correct value
        @test haskey(seen, a)  # object added to IdDict
        @test seen[a] === a    # object points to zeroed value, i.e., itself
    end
    @testset "custom leaf type" begin
        a = [1.0]
        v = CustomVector(a)
        # bringing own IdDict to avoid calling the custom method directly;
        # it should still be invoked
        @test_logs (:info, "make_zero!(::CustomVector)") make_zero!(v, IdDict())
        @test v.data === a           # preserved identity
        @test a[1] === 0.0           # correct value
    end
    @testset "runtime inactive" begin
        a = [1.0]
        v = CustomVector(a)
        with_logger(SimpleLogger(Warn)) do  # silence @info "make_zero!(::CustomVector)"
            # ensure compile-time methods are evaluated while CustomVector is considered active
            @assert !EnzymeRules.inactive_type(CustomVector)
            make_zero!(v, Val(false))
            @assert v == CustomVector([0.0])

            # verify that runtime methods also see CustomVector as active
            v.data[1] = 1.0
            make_zero!(v, Val(true))
            @test v == CustomVector([0.0])

            # mark CustomVector as inactive
            @eval @inline EnzymeRules.inactive_type(::Type{<:CustomVector}) = true

            # runtime_inactive == false => compile-time methods still used, redefined
            # inactive_type should have no effect
            v.data[1] = 1.0
            @invokelatest make_zero!(v, Val(false))
            @test v == CustomVector([0.0])

            # runtime_inactive == true => redefined inactive_type should take effect
            # CustomVector considered inactive and won't be zeroed
            v.data[1] = 1.0
            @invokelatest make_zero!(v, Val(true))
            @test v == CustomVector([1.0])

            # mark CustomVector as active again
            @eval @inline EnzymeRules.inactive_type(::Type{<:CustomVector}) = false

            # verify that both compile-time and runtime methods see CustomVector as active
            v.data[1] = 1.0
            @invokelatest make_zero!(v, Val(false))
            @test v == CustomVector([0.0])
            v.data[1] = 1.0
            @invokelatest make_zero!(v, Val(true))
            @test v == CustomVector([0.0])
        end
    end
    @testset "undefined fields/unassigned elements" begin
        @testset "array w inactive/active/mutable/unassigned" begin
            a = [1.0]
            values = ("a", 1.0, a)
            arr = Vector{Any}(undef, 4)
            arr[1:3] .= values
            make_zero!(arr)
            @views begin
                @test all(typeof.(arr[1:3]) .=== typeof.(values))  # preserved types
                @test arr[1:3] == ["a", 0.0, [0.0]]                # correct value
                @test arr[3] === a                                 # preserved identity
                @test !isassigned(arr, 4)                          # preserved unassigned
            end
        end
        @testset "struct w inactive/active/mutable/undefined" begin
            a = [1.0]
            incompletearr = [Incomplete("a", 1.0, a)]
            make_zero!(incompletearr)
            @test incompletearr == [Incomplete("a", 0.0, [0.0])]  # correct value, preserved undefined
            @test incompletearr[1].w === a                        # preserved identity
        end
        @testset "mutable struct w inactive/const active/active/mutable/undefined" begin
            a = [1.0]
            incomplete = MutableIncomplete("a", #=const=#1.0, 1.0, a)
            make_zero!(incomplete)
            @test incomplete == MutableIncomplete("a", 0.0, 0.0, [0.0])  # correct value, preserved undefined
            @test incomplete.w === a                                     # preserved identity
        end
        @testset "Array{Tuple{struct w undefined}} (issue #1935)" begin
            # old implementation of make_zero! triggered #1935
            # new implementation would work regardless due to limited use of justActive
            a = [1.0]
            incomplete = Incomplete("a", 1.0, a)
            incompletetuparr = [(incomplete,)]
            make_zero!(incompletetuparr)
            @test typeof(incompletetuparr[1]) === typeof((incomplete,))  # preserved type
            @test incompletetuparr == [(Incomplete("a", 0.0, [0.0]),)]   # correct value
            @test incompletetuparr[1][1].w === a                         # preserved identity
        end
    end
    @testset "active/mixed type error" begin
        @test_throws ArgumentError make_zero!((1.0,))
        @test_throws ArgumentError make_zero!((1.0, [1.0]))
        @test_throws ArgumentError make_zero!((Incomplete("a", 1.0, 1.0im),))  # issue #1935
    end
    @testset "containing IO" begin  # issue #2091
        f = WithIO([1.0, 2.0], stdout)
        fwrapped = [f]
        @test_noerr make_zero!(fwrapped)
        @test fwrapped[1] === f
        @test fwrapped[1].v == [0.0, 0.0]
    end
    return nothing
end

end  # module RecursiveMapTests

@testset "make_zero" RecursiveMapTests.test_make_zero()
@testset "make_zero!" RecursiveMapTests.test_make_zero!()
