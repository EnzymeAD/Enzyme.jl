using Enzyme
using Test

mutable struct MutableWrapper{T}
    x::T
end

Base.:(==)(a::MutableWrapper, b::MutableWrapper) = (a === b) || isequal(a.x, b.x)

struct Incomplete{T}
    s::String
    x::Float64
    w::T
    z  # not initialized
    Incomplete(s, x, w) = new{typeof(w)}(s, x, w)
end

function Base.:(==)(a::Incomplete, b::Incomplete)
    (a === b) && return true
    (isequal(a.s, b.s) && isequal(a.x, b.x) && isequal(a.w, b.w)) || return false
    if isdefined(a, :z) && isdefined(b, :z)
        isequal(a.z, b.z) || return false
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
    if !isequal(a.s, b.s) || !isequal(a.x, b.x) || !isequal(a.y, b.y) || !isequal(a.w, b.w)
        return false
    end
    if isdefined(a, :z) && isdefined(b, :z)
        isequal(a.z, b.z) || return false
    elseif isdefined(a, :z) || isdefined(b, :z)
        return false
    end
    return true
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

@testset "make_zero" begin
    # floats
    @test make_zero(1.0) == 0.0
    @test make_zero(1.0im) == 0.0im

    # float arrays + multiple references
    rearr = [1.0]
    imarr = [1.0im]
    rearr0 = make_zero(rearr)
    imarr0 = make_zero(imarr)
    @test typeof(rearr0) === typeof(rearr)
    @test typeof(imarr0) === typeof(imarr)
    @test rearr == [1.0]    # no mutation
    @test imarr == [1.0im]  # no mutation
    @test rearr0 == [0.0]
    @test imarr0 == [0.0im]
    rearrs0 = make_zero((rearr, rearr))
    imarrs0 = make_zero((imarr, imarr))
    @test typeof(rearrs0) === typeof((rearr, rearr))
    @test typeof(imarrs0) === typeof((imarr, imarr))
    @test rearr == [1.0]    # no mutation
    @test imarr == [1.0im]  # no mutation
    @test rearrs0[1] === rearrs0[2]
    @test imarrs0[1] === imarrs0[2]
    @test rearrs0[1] == [0.0]
    @test imarrs0[1] == [0.0im]

    # floats in structs
    rewrapped = MutableWrapper(1.0)
    imwrapped = MutableWrapper(1.0im)
    rewrapped0 = make_zero(rewrapped)
    imwrapped0 = make_zero(imwrapped)
    @test typeof(rewrapped0) === typeof(rewrapped)
    @test typeof(imwrapped0) === typeof(imwrapped)
    @test rewrapped == MutableWrapper(1.0)    # no mutation
    @test imwrapped == MutableWrapper(1.0im)  # no mutation
    @test rewrapped0 == MutableWrapper(0.0)
    @test imwrapped0 == MutableWrapper(0.0im)

    # generic array + multiple references
    wrapped = MutableWrapper(1.0)
    mixarr = ["a", 1.0, wrapped]
    mixarr0 = make_zero(mixarr)
    @test typeof(mixarr0) === typeof(mixarr)
    @test view(mixarr, 1:2) == ["a", 1.0]  # no mutation
    @test mixarr[3] === wrapped            # no mutation
    @test mixarr0 == ["a", 0.0, MutableWrapper(0.0)]
    mixarrs0 = make_zero((mixarr, mixarr))
    @test typeof(mixarrs0) === typeof((mixarr, mixarr))
    @test view(mixarr, 1:2) == ["a", 1.0]  # no mutation
    @test mixarr[3] === wrapped            # no mutation
    @test mixarrs0[1] === mixarrs0[2]
    @test mixarrs0[1] == ["a", 0.0, MutableWrapper(0.0)]

    # non-differentiable array + copy_if_inactive
    constarr = ["a"]
    constarr0 = make_zero(constarr)
    @test typeof(constarr0) === typeof(constarr)
    @test constarr == ["a"]  # no mutation
    @test constarr0 === constarr
    constarr0copy = make_zero(constarr, #=copy_if_inactive=#Val(true))
    @test typeof(constarr0copy) === typeof(constarr0)
    @test constarr == ["a"]  # no mutation
    @test constarr0copy !== constarr
    @test constarr0copy == constarr

    # Tuple
    tup = ("a", 1.0, MutableWrapper(1.0))
    tup0 = make_zero(tup)
    @test typeof(tup0) === typeof(tup)
    @test tup == ("a", 1.0, MutableWrapper(1.0))  # no mutation
    @test tup0 == ("a", 0.0, MutableWrapper(0.0))

    # NamedTuple
    ntup = (a="a", b=1.0, c=MutableWrapper(1.0))
    ntup0 = make_zero(ntup)
    @test typeof(ntup0) === typeof(ntup)
    @test ntup == (a="a", b=1.0, c=MutableWrapper(1.0))  # no mutation
    @test ntup0 == (a="a", b=0.0, c=MutableWrapper(0.0))

    # Box + multiple references
    box = Core.Box(1.0)
    box0 = make_zero(box)
    @test typeof(box0) === typeof(box)
    @test box.contents == 1.0  # no mutation
    @test box0.contents == 0.0
    boxes0 = make_zero((box, box))
    @test typeof(boxes0) === typeof((box, box))
    @test box.contents == 1.0  # no mutation
    @test boxes0[1] === boxes0[2]
    @test boxes0[1].contents == 0.0

    # differentiable custom type + multiple references
    wrapped = MutableWrapper(1.0)
    wrapped0 = make_zero(wrapped)
    @test typeof(wrapped0) === typeof(wrapped)
    @test wrapped == MutableWrapper(1.0)  # no mutation
    @test wrapped0 == MutableWrapper(0.0)
    wrappeds0 = make_zero((wrapped, wrapped))
    @test typeof(wrappeds0) === typeof((wrapped, wrapped))
    @test wrapped == MutableWrapper(1.0)  # no mutation
    @test wrappeds0[1] === wrappeds0[2]
    @test wrappeds0[1] == MutableWrapper(0.0)

    # non-differentiable custom type + copy_if_inactive
    constwrapped = MutableWrapper("a")
    constwrapped0 = make_zero(constwrapped)
    @test typeof(constwrapped0) === typeof(constwrapped)
    @test constwrapped == MutableWrapper("a")  # no mutation
    @test constwrapped0 === constwrapped
    constwrapped0copy = make_zero(constwrapped, #=copy_if_inactive=#Val(true))
    @test typeof(constwrapped0copy) === typeof(constwrapped0)
    @test constwrapped == MutableWrapper("a")  # no mutation
    @test constwrapped0copy !== constwrapped
    @test constwrapped0copy == constwrapped

    # immutable struct with active, mutable, inactive and undefined fields
    incomplete = Incomplete("a", 1.0, MutableWrapper(1.0))
    incomplete0 = make_zero(incomplete)
    @test typeof(incomplete0) === typeof(incomplete)
    @test incomplete == Incomplete("a", 1.0, MutableWrapper(1.0))  # no mutation
    @test incomplete0 == Incomplete("a", 0.0, MutableWrapper(0.0))

    # mutable struct with inactive, active, undefined, and mutable fields
    # + multiple references
    incompletemut = MutableIncomplete("a", 1.0, 1.0, MutableWrapper(1.0))
    incompletemut0 = make_zero(incompletemut)
    @test typeof(incompletemut0) === typeof(incompletemut)
    @test incompletemut == MutableIncomplete("a", 1.0, 1.0, MutableWrapper(1.0))  # no mutation
    @test incompletemut0 == MutableIncomplete("a", 0.0, 0.0, MutableWrapper(0.0))
    incompletemuts0 = make_zero((incompletemut, incompletemut))
    @test typeof(incompletemuts0) === typeof((incompletemut, incompletemut))
    @test incompletemut == MutableIncomplete("a", 1.0, 1.0, MutableWrapper(1.0))  # no mutation
    @test incompletemuts0[1] === incompletemuts0[2]
    @test incompletemuts0[1] == MutableIncomplete("a", 0.0, 0.0, MutableWrapper(0.0))

    # containing IO (issue #2091)
    f = WithIO([1.0, 2.0], stdout)
    df = @test_noerr make_zero(f)
    @test df.v == [0.0, 0.0]
    @test df.callback === f.callback
end

@testset "make_zero!" begin
    # floats in mutable struct
    rewrapped, imwrapped = MutableWrapper(1.0), MutableWrapper(1.0im)
    make_zero!(rewrapped)
    make_zero!(imwrapped)
    @test rewrapped == MutableWrapper(0.0)
    @test imwrapped == MutableWrapper(0.0im)

    # mixed tuple in mutable container
    wrapped = MutableWrapper(1.0)
    tuparr = [(1.0, wrapped)]
    make_zero!(tuparr)
    @test tuparr[1] === (0.0, wrapped)
    @test wrapped == MutableWrapper(0.0)

    # mixed namedtuple in mutable container
    wrapped = MutableWrapper(1.0)
    ntuparr = [(a=1.0, b=wrapped)]
    make_zero!(ntuparr)
    @test ntuparr[1] === (a=0.0, b=wrapped)
    @test wrapped == MutableWrapper(0.0)

    # immutable struct with active, mutable, inactive and undefined fields in mutable container
    wrapped = MutableWrapper(1.0)
    incompletearr = [Incomplete("a", 1.0, wrapped)]
    make_zero!(incompletearr)
    @test incompletearr[1] == Incomplete("a", 0.0, wrapped)
    @test wrapped == MutableWrapper(0.0)

    # floats in Ref
    reref, imref = Ref(1.0), Ref(1.0im)
    make_zero!(reref)
    make_zero!(imref)
    @test reref[] == 0.0
    @test imref[] == 0.0im

    # float arrays
    rearr, imarr = [1.0], [1.0im]
    make_zero!(rearr)
    make_zero!(imarr)
    @test rearr[1] == 0.0
    @test imarr[1] == 0.0im

    # non-differentiable array
    constarr = ["a"]
    make_zero!(constarr)
    @test constarr[1] == "a"

    # array with active, mutable, inactive and unassigned elements + multiple references
    wrapped = MutableWrapper(1.0)
    genericarr = Vector(undef, 4)
    genericarr[1:3] .= ("a", 1.0, wrapped)
    genericarrs = [genericarr, genericarr]
    make_zero!(genericarrs)
    @test genericarrs[1] === genericarrs[2]
    @test genericarrs[1] === genericarr
    @test view(genericarr, 1:2) == ["a", 0.0]
    @test genericarr[3] === wrapped
    @test wrapped == MutableWrapper(0.0)
    @test !isassigned(genericarr, 4)

    # Ref with multiple references
    genericref = Ref((1.0,))
    genericrefs = [genericref, genericref]
    make_zero!(genericrefs)
    @test genericrefs[1] === genericrefs[2]
    @test genericrefs[1] === genericref
    @test genericref[] == (0.0,)

    # Ref with mutable value
    wrapped = MutableWrapper(1.0)
    mutref = Ref(wrapped)
    make_zero!(mutref)
    @test mutref[] === wrapped
    @test wrapped == MutableWrapper(0.0)

    # Ref with non-differentiable value
    constref = Ref("a")
    make_zero!(constref)
    @test constref[] == "a"

    # Box with multiple references
    box = Core.Box(1.0)
    boxes = [box, box]
    make_zero!(boxes)
    @test boxes[1] === boxes[2]
    @test boxes[1] === box
    @test box.contents == 0.0

    # Box with mutable value
    wrapped = MutableWrapper(1.0)
    mutbox = Core.Box(wrapped)
    make_zero!(mutbox)
    @test mutbox.contents === wrapped
    @test wrapped == MutableWrapper(0.0)

    # Box with non-differentiable value
    constbox = Core.Box("a")
    make_zero!(constbox)
    @test constbox.contents == "a"

    # mutable struct with inactive, active, const active, undefined, and mutable fields
    # + multiple references
    wrapped = MutableWrapper(1.0)
    incompletemut = MutableIncomplete("a", #=const=#1.0, 1.0, wrapped)
    incompletemuts = [incompletemut, incompletemut]
    make_zero!(incompletemuts)
    @test incompletemuts[1] === incompletemuts[2]
    @test incompletemuts[1] === incompletemut
    @test incompletemut == MutableIncomplete("a", #=const=#0.0, 0.0, MutableWrapper(0.0))
    @test incompletemut.w === wrapped

    # wrapped differentiable array
    arr = [1.0]
    arrwrapped = MutableWrapper(arr)
    make_zero!(arrwrapped)
    @test arrwrapped.x === arr
    @test arr == [0.0]

    # early error on active/mixed type
    @test_throws ErrorException make_zero!(1.0)
    @test_throws ErrorException make_zero!((1.0, MutableWrapper(1.0)))

    # immutable struct with both active and undefined fields in immutable container
    # (currently fails due to #1935)
    wrapped = MutableWrapper(1.0)
    incompletetuparr = [(Incomplete("a", 1.0, wrapped),)]
    make_zero!(incompletetuparr)
    @test incompletetuparr[1][1] == Incomplete("a", 0.0, MutableWrapper(0.0))
    @test incompletetuparr[1][1].w === wrapped

    # containing IO (issue #2091)
    f = WithIO([1.0, 2.0], stdout)
    fwrapped = [f]
    @test_noerr make_zero!(fwrapped)
    @test fwrapped[1] === f
    @test fwrapped[1].v == [0.0, 0.0]
end
