using Enzyme, Test

concat() = ()
concat(a) = a
concat(a, b) = (a..., b...)
concat(a, b, c...) = concat(concat(a, b), c...)

metaconcat(x) = concat(x...)

metaconcat2(x, y) = concat(x..., y...)

midconcat(x, y) = (x, concat(y...)...)

metaconcat3(x, y, z) = concat(x..., y..., z...)

function metasumsq(f, args...) 
	res = 0.0
	x = f(args...)
	for v in x
		v = v::Float64
		res += v*v
	end
	return res
end

function metasumsq2(f, args...) 
	res = 0.0
	x = f(args...)
	for v in x
		for v2 in v
			v2 = v2::Float64
			res += v*v
		end
	end
	return res
end


function metasumsq3(f, args...) 
	res = 0.0
	x = f(args...)
	for v in x
		v = v
		res += v*v
	end
	return res
end

function metasumsq4(f, args...) 
	res = 0.0
	x = f(args...)
	for v in x
		for v2 in v
			v2 = v2
			res += v*v
		end
	end
	return res
end

function make_byref(out, fn, args...)
	out[] = fn(args...)
	nothing
end

function tupapprox(a, b)
	if a isa Tuple && b isa Tuple
		if length(a) != length(b)
			return false
		end
		for (aa, bb) in zip(a, b)
			if !tupapprox(aa, bb)
				return false
			end
		end
		return true
	end
	if a isa Array && b isa Array
		if size(a) != size(b)
			return false
		end
		for i in length(a)
			if !tupapprox(a[i], b[i])
				return false
			end
		end
		return true
	end
	return a ≈ b
end

@testset "Const Apply iterate" begin
    function extiter() 
        vals = Any[3,]
        extracted = Tuple(vals)
        return extracted
    end

    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(extiter)}, Duplicated)

    tape, res, dres = fwd(Const(extiter))
    @test res == (3,)
    @test dres == (3,)
end

@testset "Reverse Apply iterate" begin
    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(0.0, 0.0), (0.0, 0.0)]
    res = Enzyme.autodiff(Reverse, metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])

    dx = [(0.0, 0.0), (0.0, 0.0)]
    res = Enzyme.autodiff(ReverseWithPrimal, metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @test res[2] ≈ 200.84999999999997
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])

    x = [[2.0, 3.0], [7.9, 11.2]]
    dx = [[0.0, 0.0], [0.0, 0.0]]

    res = Enzyme.autodiff(Reverse, metasumsq2, Active, Const(metaconcat), Duplicated(x, dx))
    @test dx ≈ [[4.0, 6.0], [15.8, 22.4]]

    dx = [[0.0, 0.0], [0.0, 0.0]]

    res = Enzyme.autodiff(ReverseWithPrimal, metasumsq2, Active, Const(metaconcat), Duplicated(x, dx))

    @test res[2] ≈ 200.84999999999997
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])


    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(0.0, 0.0), (0.0, 0.0)]

    y = [(13, 17), (25, 31)]
    res = Enzyme.autodiff(Reverse, metasumsq3, Active, Const(metaconcat2), Duplicated(x, dx), Const(y))
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])


    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(0.0, 0.0), (0.0, 0.0)]
    y = [(13, 17), (25, 31)]
    dy = [(0, 0), (0, 0)]
    res = Enzyme.autodiff(Reverse, metasumsq3, Active, Const(metaconcat2), Duplicated(x, dx), Duplicated(y, dy))
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])



    x = [[2.0, 3.0], [7.9, 11.2]]
    dx = [[0.0, 0.0], [0.0, 0.0]]
    y = [[13, 17], [25, 31]]
    res = Enzyme.autodiff(Reverse, metasumsq4, Active, Const(metaconcat2), Duplicated(x, dx), Const(y))
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])


    x = [[2.0, 3.0], [7.9, 11.2]]
    dx = [[0.0, 0.0], [0.0, 0.0]]
    y = [[13, 17], [25, 31]]
    dy = [[0, 0], [0, 0]]
    res = Enzyme.autodiff(Reverse, metasumsq4, Active, Const(metaconcat2), Duplicated(x, dx), Duplicated(y, dy))
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])
end

@testset "BatchReverse Apply iterate" begin
    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(0.0, 0.0), (0.0, 0.0)]
    dx2 = [(0.0, 0.0), (0.0, 0.0)]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])
    @test tupapprox(dx2, [(3*4.0, 3*6.0), (3*15.8, 3*22.4)])

    dx = [(0.0, 0.0), (0.0, 0.0)]
    dx2 = [(0.0, 0.0), (0.0, 0.0)]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test out[] ≈ 200.84999999999997
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])
    @test tupapprox(dx2, [(3*4.0, 3*6.0), (3*15.8, 3*22.4)])

    x = [[2.0, 3.0], [7.9, 11.2]]
    dx = [[0.0, 0.0], [0.0, 0.0]]
    dx2 = [[0.0, 0.0], [0.0, 0.0]]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)

    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(metasumsq2), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test dx ≈ [[4.0, 6.0], [15.8, 22.4]]
    @test dx2 ≈ [[3*4.0, 3*6.0], [3*15.8, 3*22.4]]

    dx = [[0.0, 0.0], [0.0, 0.0]]
    dx2 = [[0.0, 0.0], [0.0, 0.0]]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(metasumsq2), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))

    @test out[] ≈ 200.84999999999997
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])
    @test tupapprox(dx2, [[3*4.0, 3*6.0], [3*15.8, 3*22.4]])


    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(0.0, 0.0), (0.0, 0.0)]
    dx2 = [(0.0, 0.0), (0.0, 0.0)]

    y = [(13, 17), (25, 31)]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(metasumsq3), Const(metaconcat2), BatchDuplicated(x, (dx, dx2)), Const(y))
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])
    @test tupapprox(dx2, [(3*4.0, 3*6.0), (3*15.8, 3*22.4)])


    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(0.0, 0.0), (0.0, 0.0)]
    dx2 = [(0.0, 0.0), (0.0, 0.0)]
    y = [(13, 17), (25, 31)]
    dy = [(0, 0), (0, 0)]
    dy2 = [(0, 0), (0, 0)]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(metasumsq3),Const(metaconcat2), BatchDuplicated(x, (dx, dx2)), BatchDuplicated(y, (dy, dy2)))
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])
    @test tupapprox(dx2, [(3*4.0, 3*6.0), (3*15.8, 3*22.4)])


    x = [[2.0, 3.0], [7.9, 11.2]]
    dx = [[0.0, 0.0], [0.0, 0.0]]
    dx2 = [[0.0, 0.0], [0.0, 0.0]]
    y = [[13, 17], [25, 31]]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(metasumsq4),  Const(metaconcat2), BatchDuplicated(x, (dx, dx2)), Const(y))
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])
    @test tupapprox(dx2, [[3*4.0, 3*6.0], [3*15.8, 3*22.4]])

    x = [[2.0, 3.0], [7.9, 11.2]]
    dx = [[0.0, 0.0], [0.0, 0.0]]
    dx2 = [[0.0, 0.0], [0.0, 0.0]]
    y = [[13, 17], [25, 31]]
    dy = [[0, 0], [0, 0]]
    dy2 = [[0, 0], [0, 0]]
    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(metasumsq4), Const(metaconcat2), BatchDuplicated(x, (dx, dx2)), BatchDuplicated(y, (dy, dy2)))
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])
    @test tupapprox(dx2, [[3*4.0, 3*6.0], [3*15.8, 3*22.4]])
end

@testset "Forward Apply iterate" begin
    x = [(2.0, 3.0), (7.9, 11.2)]
    dx = [(13.7, 15.2), (100.02, 304.1)]

    dres, = Enzyme.autodiff(Forward, metaconcat, Duplicated(x, dx))
    @test length(dres) == 4
    @test dres[1] ≈ 13.7
    @test dres[2] ≈ 15.2
    @test dres[3] ≈ 100.02
    @test dres[4] ≈ 304.1

    res, dres = Enzyme.autodiff(Forward, metaconcat, Duplicated, Duplicated(x, dx))
    @test length(res) == 4
    @test res[1] ≈ 2.0
    @test res[2] ≈ 3.0
    @test res[3] ≈ 7.9
    @test res[4] ≈ 11.2
    @test length(dres) == 4
    @test dres[1] ≈ 13.7
    @test dres[2] ≈ 15.2
    @test dres[3] ≈ 100.02
    @test dres[4] ≈ 304.1


    a = [("a", "b"), ("c", "d")]
    da = [("e", "f"), ("g", "h")]

    dres, = Enzyme.autodiff(Forward, metaconcat, Duplicated(a, da))
    @test length(dres) == 4
    @test dres[1] == "a"
    @test dres[2] == "b"
    @test dres[3] == "c"
    @test dres[4] == "d"

    res, dres = Enzyme.autodiff(Forward, metaconcat, Duplicated, Duplicated(a, da))
    @test length(res) == 4
    @test res[1] == "a"
    @test res[2] == "b"
    @test res[3] == "c"
    @test res[4] == "d"
    @test length(dres) == 4
    @test dres[1] == "a"
    @test dres[2] == "b"
    @test dres[3] == "c"
    @test dres[4] == "d"


    Enzyme.autodiff(Forward, metaconcat, Const(a))

@static if VERSION ≥ v"1.7-" 
    dres, = Enzyme.autodiff(Forward, midconcat, Duplicated(1.0, 7.0), Duplicated(a, da))
    @test length(dres) == 5
    @test dres[1] ≈ 7.0
    @test dres[2] == "a"
    @test dres[3] == "b"
    @test dres[4] == "c"
    @test dres[5] == "d"

    res, dres = Enzyme.autodiff(Forward, midconcat, Duplicated, Duplicated(1.0, 7.0), Duplicated(a, da))
    @test length(res) == 5
    @test res[1] ≈ 1.0
    @test res[2] == "a"
    @test res[3] == "b"
    @test res[4] == "c"
    @test res[5] == "d"

    @test length(dres) == 5
    @test dres[1] ≈ 7.0
    @test dres[2] == "a"
    @test dres[3] == "b"
    @test dres[4] == "c"
    @test dres[5] == "d"


    dres, = Enzyme.autodiff(Forward, midconcat, Duplicated(1.0, 7.0), Const(a))
    @test length(dres) == 5
    @test dres[1] ≈ 7.0
    @test dres[2] == "a"
    @test dres[3] == "b"
    @test dres[4] == "c"
    @test dres[5] == "d"

    res, dres = Enzyme.autodiff(Forward, midconcat, Duplicated, Duplicated(1.0, 7.0), Const(a))
    @test length(res) == 5
    @test res[1] ≈ 1.0
    @test res[2] == "a"
    @test res[3] == "b"
    @test res[4] == "c"
    @test res[5] == "d"
    @test length(dres) == 5
    @test dres[1] ≈ 7.0
    @test dres[2] == "a"
    @test dres[3] == "b"
    @test dres[4] == "c"
    @test dres[5] == "d"
end

    y = [(-92.0, -93.0), (-97.9, -911.2)]
    dy = [(-913.7, -915.2), (-9100.02, -9304.1)]

    dres, = Enzyme.autodiff(Forward, metaconcat2, Duplicated(x, dx), Duplicated(y, dy))
    @test length(dres) == 8
    @test dres[1] ≈ 13.7
    @test dres[2] ≈ 15.2
    @test dres[3] ≈ 100.02
    @test dres[4] ≈ 304.1
    @test dres[5] ≈ -913.7
    @test dres[6] ≈ -915.2
    @test dres[7] ≈ -9100.02
    @test dres[8] ≈ -9304.1

    res, dres = Enzyme.autodiff(Forward, metaconcat2, Duplicated, Duplicated(x, dx), Duplicated(y, dy))
    @test length(res) == 8
    @test res[1] ≈ 2.0
    @test res[2] ≈ 3.0
    @test res[3] ≈ 7.9
    @test res[4] ≈ 11.2
    @test res[5] ≈ -92.0
    @test res[6] ≈ -93.0
    @test res[7] ≈ -97.9
    @test res[8] ≈ -911.2
    @test length(dres) == 8
    @test dres[1] ≈ 13.7
    @test dres[2] ≈ 15.2
    @test dres[3] ≈ 100.02
    @test dres[4] ≈ 304.1
    @test dres[5] ≈ -913.7
    @test dres[6] ≈ -915.2
    @test dres[7] ≈ -9100.02
    @test dres[8] ≈ -9304.1


    dres, = Enzyme.autodiff(Forward, metaconcat3, Duplicated(x, dx), Const(a), Duplicated(y, dy))
    @test length(dres) == 12
    @test dres[1] ≈ 13.7
    @test dres[2] ≈ 15.2
    @test dres[3] ≈ 100.02
    @test dres[4] ≈ 304.1

    @test dres[5] == "a"
    @test dres[6] == "b"
    @test dres[7] == "c"
    @test dres[8] == "d"

    @test dres[9] ≈ -913.7
    @test dres[10] ≈ -915.2
    @test dres[11] ≈ -9100.02
    @test dres[12] ≈ -9304.1

    res, dres = Enzyme.autodiff(Forward, metaconcat3, Duplicated, Duplicated(x, dx), Const(a), Duplicated(y, dy))
    @test length(res) == 12
    @test res[1] ≈ 2.0
    @test res[2] ≈ 3.0
    @test res[3] ≈ 7.9
    @test res[4] ≈ 11.2

    @test res[5] == "a"
    @test res[6] == "b"
    @test res[7] == "c"
    @test res[8] == "d"

    @test res[9] ≈ -92.0
    @test res[10] ≈ -93.0
    @test res[11] ≈ -97.9
    @test res[12] ≈ -911.2

    @test length(dres) == 12
    @test dres[1] ≈ 13.7
    @test dres[2] ≈ 15.2
    @test dres[3] ≈ 100.02
    @test dres[4] ≈ 304.1

    @test dres[5] == "a"
    @test dres[6] == "b"
    @test dres[7] == "c"
    @test dres[8] == "d"

    @test dres[9] ≈ -913.7
    @test dres[10] ≈ -915.2
    @test dres[11] ≈ -9100.02
    @test dres[12] ≈ -9304.1


    dres, = Enzyme.autodiff(Forward, metaconcat, BatchDuplicated(x, (dx, dy)))
    @test length(dres[1]) == 4
    @test dres[1][1] ≈ 13.7
    @test dres[1][2] ≈ 15.2
    @test dres[1][3] ≈ 100.02
    @test dres[1][4] ≈ 304.1
    @test length(dres[2]) == 4
    @test dres[2][1] ≈ -913.7
    @test dres[2][2] ≈ -915.2
    @test dres[2][3] ≈ -9100.02
    @test dres[2][4] ≈ -9304.1

    res, dres = Enzyme.autodiff(Forward, metaconcat, Duplicated, BatchDuplicated(x, (dx, dy)))
    @test length(res) == 4
    @test res[1] ≈ 2.0
    @test res[2] ≈ 3.0
    @test res[3] ≈ 7.9
    @test res[4] ≈ 11.2
    @test length(dres[1]) == 4
    @test dres[1][1] ≈ 13.7
    @test dres[1][2] ≈ 15.2
    @test dres[1][3] ≈ 100.02
    @test dres[1][4] ≈ 304.1
    @test length(dres[2]) == 4
    @test dres[2][1] ≈ -913.7
    @test dres[2][2] ≈ -915.2
    @test dres[2][3] ≈ -9100.02
    @test dres[2][4] ≈ -9304.1
end

@testset "legacy reverse apply iterate" begin
    function mktup(v)
        tup = tuple(v...)
        return tup[1][1] * tup[3][1]
    end

    data = [[3.0], nothing, [2.0]]
    ddata = [[0.0], nothing, [0.0]]

    Enzyme.autodiff(Reverse, mktup, Duplicated(data, ddata))
    @test ddata[1][1] ≈ 2.0
    @test ddata[3][1] ≈ 3.0

    function mktup2(v)
        tup = tuple(v...)
        return (tup[1][1] * tup[3])::Float64
    end

    data = [[3.0], nothing, 2.0]
    ddata = [[0.0], nothing, 0.0]

    @test_throws AssertionError Enzyme.autodiff(Reverse, mktup2, Duplicated(data, ddata))

    function mktup3(v)
        tup = tuple(v..., v...)
        return tup[1][1] * tup[1][1]
    end

    data = [[3.0]]
    ddata = [[0.0]]

    Enzyme.autodiff(Reverse, mktup3, Duplicated(data, ddata))
    @test ddata[1][1] ≈ 6.0
end

include("mixedapplyiter.jl")