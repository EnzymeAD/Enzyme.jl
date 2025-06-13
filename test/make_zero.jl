module MakeZeroTests

using Enzyme
using StaticArrays
using Test

# Universal getters/setters for built-in and custom containers/wrappers
getx(w::Base.RefValue) = w[]
getx(w::Core.Box) = w.contents
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

struct Incomplete{T}
    s::String
    x::Float64
    w::T
    z  # not initialized
    Incomplete(s, x, w) = new{typeof(w)}(s, x, w)
end

function Base.:(==)(a::Incomplete, b::Incomplete)
    (a === b) && return true
    ((a.s == b.s) && (a.x == b.x) && (a.w == b.w)) || return false
    if isdefined(a, :z) && isdefined(b, :z)
        (a.z == b.z) || return false
    elseif isdefined(a, :z) || isdefined(b, :z)
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

function Enzyme.EnzymeCore.make_zero(
    ::Type{CV}, seen::IdDict, prev::CV, ::Val{copy_if_inactive}
) where {CV<:CustomVector{<:AbstractFloat},copy_if_inactive}
    @info "make_zero(::CustomVector)"
    if haskey(seen, prev)
        return seen[prev]
    end
    new = CustomVector(zero(prev.data))::CV
    seen[prev] = new
    return new
end

function Enzyme.EnzymeCore.make_zero!(prev::CustomVector{<:AbstractFloat}, seen)::Nothing
    @info "make_zero!(::CustomVector)"
    if !isnothing(seen)
        if prev in seen
            return nothing
        end
        push!(seen, prev)
    end
    fill!(prev.data, false)
    return nothing
end

function Enzyme.EnzymeCore.make_zero!(prev::CustomVector{<:AbstractFloat})
    return Enzyme.EnzymeCore.make_zero!(prev, nothing)
end

function Enzyme.EnzymeCore.remake_zero!(prev::CustomVector{<:AbstractFloat}, seen)::Nothing
    @info "make_zero!(::CustomVector)"
    if !isnothing(seen)
        if prev in seen
            return nothing
        end
        push!(seen, prev)
    end
    fill!(prev.data, false)
    return nothing
end

function Enzyme.EnzymeCore.remake_zero!(prev::CustomVector{<:AbstractFloat})
    return Enzyme.EnzymeCore.remake_zero!(prev, nothing)
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

const scalartypes = [Float32, ComplexF32, Float64, ComplexF64]

const inactivetup = ("a", Empty(), MutableEmpty())
const inactivearr = [inactivetup]

const wrappers = [
    (name="Tuple{X}",                     f=tuple,                                           N=1, mutable=false, typed=true),
    (name="@NamedTuple{x::X}",            f=(NamedTuple{(:x,)} ∘ tuple),                     N=1, mutable=false, typed=true),
    (name="struct{X}",                    f=Wrapper,                                         N=1, mutable=false, typed=true),

    (name="@NamedTuple{x}",               f=(@NamedTuple{x} ∘ tuple),                        N=1, mutable=false, typed=false),
    (name="struct{Any}",                  f=Wrapper{Any},                                    N=1, mutable=false, typed=false),

    (name="Array{X}",                     f=(x -> [x]),                                      N=1, mutable=true,  typed=true),
    (name="Base.RefValue{X}",             f=Ref,                                             N=1, mutable=true,  typed=true),
    (name="mutable struct{X}",            f=MutableWrapper,                                  N=1, mutable=true,  typed=true),

    (name="Array{Any}",                   f=(x -> Any[x]),                                   N=1, mutable=true,  typed=false),
    (name="Base.RefValue{Any}",           f=Ref{Any},                                        N=1, mutable=true,  typed=false),
    (name="Core.Box",                     f=Core.Box,                                        N=1, mutable=true,  typed=false),
    (name="mutable struct{Any}",          f=MutableWrapper{Any},                             N=1, mutable=true,  typed=false),

    (name="Tuple{X,Y}",                   f=tuple,                                           N=2, mutable=false, typed=true),
    (name="@NamedTuple{x::X,y::Y}",       f=(NamedTuple{(:x, :y)} ∘ tuple),                  N=2, mutable=false, typed=true),
    (name="struct{X,Y}",                  f=DualWrapper,                                     N=2, mutable=false, typed=true),

    (name="@NamedTuple{x,y::Y}",          f=((x, y) -> @NamedTuple{x,y::typeof(y)}((x, y))), N=2, mutable=false, typed=:partial),
    (name="struct{Any,Y}",                f=DualWrapper{Any},                                N=2, mutable=false, typed=:partial),

    (name="@NamedTuple{x,y}",             f=@NamedTuple{x,y} ∘ tuple,                        N=2, mutable=false, typed=false),
    (name="struct{Any}",                  f=DualWrapper{Any,Any},                            N=2, mutable=false, typed=false),

    (name="mutable struct{X,Y}",          f=MutableDualWrapper,                              N=2, mutable=true,  typed=true),

    (name="Array{promote_type(X,Y)}",     f=((x, y) -> [x, y]),                              N=2, mutable=true,  typed=:promoted),
    (name="mutable struct{Any,Y}",        f=MutableDualWrapper{Any},                         N=2, mutable=true,  typed=:partial),

    (name="Array{Any}",                   f=((x, y) -> Any[x, y]),                           N=2, mutable=true,  typed=false),
    (name="mutable struct{Any,Any}",      f=MutableDualWrapper{Any,Any},                     N=2, mutable=true,  typed=false),

    # StaticArrays extension
    (name="SVector{1,X}",                 f=SVector{1} ∘ tuple,                              N=1, mutable=false, typed=true),
    (name="SVector{1,Any}",               f=SVector{1,Any} ∘ tuple,                          N=1, mutable=false, typed=false),
    (name="MVector{1,X}",                 f=MVector{1} ∘ tuple,                              N=1, mutable=true,  typed=true),
    (name="MVector{1,Any}",               f=MVector{1,Any} ∘ tuple,                          N=1, mutable=true,  typed=false),
    (name="SVector{2,promote_type(X,Y)}", f=SVector{2} ∘ tuple,                              N=2, mutable=false, typed=:promoted),
    (name="SVector{2,Any}",               f=SVector{2,Any} ∘ tuple,                          N=2, mutable=false, typed=false),
    (name="MVector{2,promote_type(X,Y)}", f=MVector{2} ∘ tuple,                              N=2, mutable=true,  typed=:promoted),
    (name="MVector{2,Any}",               f=MVector{2,Any} ∘ tuple,                          N=2, mutable=true,  typed=false),
]

@static if VERSION < v"1.11-"
else
_memory(x::Vector) = Memory{eltype(x)}(x)
push!(
    wrappers,
    (name="Memory{X}",                    f=(x -> _memory([x])),                             N=1, mutable=true,  typed=true),
    (name="Memory{Any}",                  f=(x -> _memory(Any[x])),                          N=1, mutable=true,  typed=false),
    (name="Memory{promote_type(X,Y)}",    f=((x, y) -> _memory([x, y])),                     N=2, mutable=true,  typed=:promoted),
    (name="Memory{Any}",                  f=((x, y) -> _memory(Any[x, y])),                  N=2, mutable=true,  typed=false),
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
        @testset "$T in $(wrapper.name)" for
                T in scalartypes, wrapper in filter(w -> (w.N == 1), wrappers)
            x = oneunit(T)
            w = wrapper.f(x)
            w_makez = make_zero(w)
            @test typeof(w_makez) === typeof(w)  # correct type
            @test typeof(getx(w_makez)) === T    # correct type
            @test getx(w_makez) == zero(T)       # correct value
            @test getx(w) === x                  # no mutation of original
            @test x == oneunit(T)                # no mutation of original (relevant for BigFloat)
            @testset "doubly included in $(dualwrapper.name)" for
                    dualwrapper in filter(w -> (w.N == 2), wrappers)
                w_inner = wrapper.f(x)
                d_outer = dualwrapper.f(w_inner, w_inner)
                d_outer_makez = make_zero(d_outer)
                @test typeof(d_outer_makez) === typeof(d_outer)        # correct type
                @test typeof(getx(d_outer_makez)) === typeof(w_inner)  # correct type
                @test typeof(getx(getx(d_outer_makez))) === T          # correct type
                @test getx(d_outer_makez) === gety(d_outer_makez)      # correct topology
                @test getx(getx(d_outer_makez)) == zero(T)             # correct value
                @test getx(d_outer) === gety(d_outer)                  # no mutation of original
                @test getx(d_outer) === w_inner                        # no mutation of original
                @test getx(w_inner) === x                              # no mutation of original
                @test x == oneunit(T)                                  # no mutation of original (relevant for BigFloat)
                d_inner = dualwrapper.f(x, x)
                w_outer = wrapper.f(d_inner)
                w_outer_makez = make_zero(w_outer)
                @test typeof(w_outer_makez) === typeof(w_outer)               # correct type
                @test typeof(getx(w_outer_makez)) === typeof(d_inner)         # correct type
                @test typeof(getx(getx(w_outer_makez))) === T                 # correct type
                @test getx(getx(w_outer_makez)) == gety(getx(w_outer_makez))  # correct topology
                @test getx(getx(w_outer_makez)) == zero(T)                    # correct value
                @test getx(w_outer) === d_inner                               # no mutation of original
                @test getx(d_inner) === gety(d_inner)                         # no mutation of original
                @test getx(d_inner) === x                                     # no mutation of original
                @test x == oneunit(T)                                         # no mutation of original (relevant for BigFloat)
                if wrapper.mutable && !dualwrapper.mutable
                    # some code paths can only be hit with three layers of wrapping:
                    # mutable(immutable(mutable(scalar)))
                    @testset "all wrapped in $(outerwrapper.name)" for
                            outerwrapper in filter(w -> ((w.N == 1) && w.mutable), wrappers)
                        w_inner = wrapper.f(x)
                        d_middle = dualwrapper.f(w_inner, w_inner)
                        w_outer = outerwrapper.f(d_middle)
                        w_outer_makez = make_zero(w_outer)
                        @test typeof(w_outer_makez) === typeof(w_outer)                 # correct type
                        @test typeof(getx(w_outer_makez)) === typeof(d_middle)          # correct type
                        @test typeof(getx(getx(w_outer_makez))) === typeof(w_inner)     # correct type
                        @test typeof(getx(getx(getx(w_outer_makez)))) === T             # correct type
                        @test getx(getx(w_outer_makez)) === gety(getx(w_outer_makez))   # correct topology
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
                w = wrapper.f(inactivearr)
                w_makez = make_zero(w)
                if wrapper.typed == true
                    @test w_makez === w               # preserved wrapper identity if guaranteed const
                end
                @test typeof(w_makez) === typeof(w)   # correct type
                @test getx(w_makez) === inactivearr   # preserved identity
                @test inactivearr[1] === inactivetup  # preserved value
                @test getx(w) === inactivearr         # no mutation of original
            else  # wrapper.N == 2
                @testset "multiple references" begin
                    w = wrapper.f(inactivearr, inactivearr)
                    w_makez = make_zero(w)
                    if wrapper.typed == true
                        @test w_makez === w                # preserved wrapper identity if guaranteed const
                    end
                    @test typeof(w_makez) === typeof(w)    # correct type
                    @test getx(w_makez) === gety(w_makez)  # preserved topology
                    @test getx(w_makez) === inactivearr    # preserved identity
                    @test inactivearr[1] === inactivetup   # preserved value
                    @test getx(w) === gety(w)              # no mutation of original
                    @test getx(w) === inactivearr          # no mutation of original
                end
                @testset "alongside active" begin
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
            @test w_makez[2] === w_makez[3]                      # correct topology (topology should propagate even when copy_if_inactive = Val(true))
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
    @testset "circular references" begin
        @testset "$(wrapper.name)" for wrapper in (
            filter(w -> (w.mutable && (w.typed in (:partial, false))), wrappers)
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
            incomplete = Incomplete("a", 1.0, a)
            incomplete_makez = make_zero(incomplete)
            @test typeof(incomplete_makez) === typeof(incomplete)  # correct type
            @test typeof(incomplete_makez.w) === typeof(a)         # correct type
            @test incomplete_makez == Incomplete("a", 0.0, [0.0])  # correct value, propagated undefined
            @test a[1] === 1.0                                     # no mutation of original
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

function test_make_zero!(make_zero! = Enzyme.make_zero!)
    @testset "nested types" begin
        @testset "$T in $(wrapper.name)" for
                T in scalartypes, wrapper in filter(w -> (w.N == 1), wrappers)
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
                w_inner = wrapper.f(x)
                d_outer = dualwrapper.f(w_inner, w_inner)
                make_zero!(d_outer)
                @test typeof(getx(d_outer)) === typeof(w_inner)  # preserved type
                @test typeof(getx(getx(d_outer))) === T          # preserved type
                @test getx(getx(d_outer)) == zero(T)             # correct value
                @test getx(d_outer) === gety(d_outer)            # preserved topology
                @test x == oneunit(T)                            # no mutation of scalar (relevant for BigFloat)
                if wrapper.mutable
                    @test getx(d_outer) === w_inner              # preserved identity
                end
                d_inner = dualwrapper.f(x, x)
                w_outer = wrapper.f(d_inner)
                make_zero!(w_outer)
                @test typeof(getx(w_outer)) === typeof(d_inner)    # preserved type
                @test typeof(getx(getx(w_outer))) === T            # preserved type
                @test getx(getx(w_outer)) == zero(T)               # correct value
                @test getx(getx(w_outer)) === gety(getx(w_outer))  # preserved topology
                @test x == oneunit(T)                              # no mutation of scalar (relevant for BigFloat)
                if dualwrapper.mutable
                    @test getx(w_outer) === d_inner                # preserved identity
                end
                if wrapper.mutable && !dualwrapper.mutable
                    # some code paths can only be hit with three layers of wrapping:
                    # mutable(immutable(mutable(scalar)))
                    @assert !dualwrapper.mutable  # sanity check
                    @testset "all wrapped in $(outerwrapper.name)" for
                            outerwrapper in filter(w -> ((w.N == 1) && w.mutable), wrappers)
                        w_inner = wrapper.f(x)
                        d_middle = dualwrapper.f(w_inner, w_inner)
                        w_outer = outerwrapper.f(d_middle)
                        make_zero!(w_outer)
                        @test typeof(getx(w_outer)) === typeof(d_middle)       # preserved type
                        @test typeof(getx(getx(w_outer))) === typeof(w_inner)  # preserved type
                        @test typeof(getx(getx(getx(w_outer)))) === T          # preserved type
                        @test getx(getx(getx(w_outer))) == zero(T)             # correct value
                        @test getx(getx(w_outer)) === gety(getx(w_outer))      # preserved topology
                        @test getx(getx(w_outer)) === w_inner                  # preserved identity
                        @test x == oneunit(T)                                  # no mutation of scalar (relevant for BigFloat)
                    end
                end
            end
        end
    end
    @testset "inactive" begin
        @testset "in $(wrapper.name)" for
            wrapper in filter(w -> (w.mutable || (w.typed == true)), wrappers)
            if wrapper.N == 1
                w = wrapper.f(inactivearr)
                make_zero!(w)
                @test getx(w) === inactivearr         # preserved identity
                @test inactivearr[1] === inactivetup  # preserved value
            else  # wrapper.N == 2
                @testset "multiple references" begin
                    w = wrapper.f(inactivearr, inactivearr)
                    make_zero!(w)
                    @test getx(w) === gety(w)             # preserved topology
                    @test getx(w) === inactivearr         # preserved identity
                    @test inactivearr[1] === inactivetup  # preserved value
                end
                @testset "alongside active" begin
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
    @testset "circular references" begin
        @testset "$(wrapper.name)" for wrapper in (
            filter(w -> (w.mutable && (w.typed in (:partial, false))), wrappers)
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
    @testset "bring your own IdSet" begin
        a = [1.0]
        seen = Base.IdSet()
        make_zero!(a, seen)
        @test a[1] === 0.0  # correct value
        @test (a in seen)   # object added to IdSet
    end
    @testset "custom leaf type" begin
        a = [1.0]
        v = CustomVector(a)
        # bringing own IdSet to avoid calling the custom method directly;
        # it should still be invoked
        @test_logs (:info, "make_zero!(::CustomVector)") make_zero!(v, Base.IdSet())
        @test v.data === a           # preserved identity
        @test a[1] === 0.0           # correct value
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
            # old implementation triggered #1935
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
    if make_zero! == Enzyme.make_zero!
        @testset "active/mixed type error" begin
            @test_throws ArgumentError make_zero!((1.0,))
            @test_throws ArgumentError make_zero!((1.0, [1.0]))
            @test_throws ArgumentError make_zero!((Incomplete("a", 1.0, 1.0im),))  # issue #1935
        end
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

function test_remake_zero!()
    test_make_zero!(Enzyme.remake_zero!)

    @testset "Immutable" begin
        x = (0.0, [4.5])
        Enzyme.remake_zero!(x)
        @test x[1] == 0.0
        @test x[2][1] == 0.0

        x = (2.0, [4.5])
        Enzyme.remake_zero!(x)
        @test x[1] == 2.0
        @test x[2][1] == 0.0
    end
end
@testset "make_zero" test_make_zero()
@testset "make_zero!" test_make_zero!()
@testset "remake_zero!" test_remake_zero!()

end  # module MakeZeroTests
