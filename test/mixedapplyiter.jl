using Enzyme, Test

concat() = ()
concat(a) = a
concat(a, b) = (a..., b...)
concat(a, b, c...) = concat(concat(a, b), c...)

metaconcat(x) = concat(x...)

metaconcat2(x, y) = concat(x..., y...)

midconcat(x, y) = (x, concat(y...)...)

metaconcat3(x, y, z) = concat(x..., y..., z...)

function mixed_metasumsq(f, args...) 
	res = 0.0
	x = f(args...)
	for v in x
		v = v::Tuple{Float64, Vector{Float64}}
		res += v[1]*v[1] + v[2][1] * v[2][1]
	end
	return res
end

function mixed_metasumsq3(f, args...) 
	res = 0.0
	x = f(args...)
	for v in x
		v = v
		res += v*v
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

@testset "Mixed Reverse Apply iterate (tuple)" begin
    x = [((2.0, [2.7]), (3.0, [3.14])), ((7.9, [47.0]), (11.2, [56.0]))]
    primal = 5562.9996
    @testset "$label" for (label, dx_pre, dx_post) in [
        (
            "dx == 0",
            [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))],
            [((4.0, [5.4]), (6.0, [6.28])), ((15.8, [94.0]), (22.4, [112.0]))],
        ),
        (
            "dx != 0",
            [((1.0, [-2.0]), (-3.0, [4.0])), ((5.0, [-6.0]), (-7.0, [8.0]))],
            [((5.0, [3.4]), (3.0, [10.28])), ((20.8, [88.0]), (15.4, [120.0]))],
        ),
    ]
        dx = deepcopy(dx_pre)
        Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
        @test tupapprox(dx, dx_post)

        dx = deepcopy(dx_pre)
        res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
        @test res[2] ≈ primal
        @test tupapprox(dx, dx_post)
    end
end

@testset "BatchMixed Reverse Apply iterate (tuple)" begin
    x = [((2.0, [2.7]), (3.0, [3.14])), ((7.9, [47.0]), (11.2, [56.0]))]
    primal = 5562.9996
    out_pre, dout_pre, dout2_pre = 0.0, 1.0, 3.0
    @testset "$label" for (label, dx_pre, dx_post, dx2_post) in [
        (
            "dx == 0",
            [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))],
            [((4.0, [5.4]), (6.0, [6.28])), ((15.8, [94.0]), (22.4, [112.0]))],
            [((3 * 4.0, [3 * 5.4]), (3 * 6.0, [3 * 6.28])), ((3 * 15.8, [3 * 94.0]), (3 * 22.4, [3 * 112.0]))],
        ),
        (
            "dx != 0",
            [((1.0, [-2.0]), (-3.0, [4.0])), ((5.0, [-6.0]), (-7.0, [8.0]))],
            [((5.0, [3.4]), (3.0, [10.28])), ((20.8, [88.0]), (15.4, [120.0]))],
            [((1.0 + 3 * 4.0, [-2.0 + 3 * 5.4]), (-3.0 + 3 * 6.0, [4.0 + 3 * 6.28])), ((5.0 + 3 * 15.8, [-6.0 + 3 * 94.0]), (-7.0 + 3 * 22.4, [8.0 + 3 * 112.0]))],
        ),
    ]
        out, dout, dout2 = Ref.((out_pre, dout_pre, dout2_pre))
        dx, dx2 = deepcopy.((dx_pre, dx_pre))
        Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
        @test dout[] ≈ 0
        @test dout2[] ≈ 0
        @test tupapprox(dx, dx_post)
        @test tupapprox(dx2, dx2_post)

        out, dout, dout2 = Ref.((out_pre, dout_pre, dout2_pre))
        dx, dx2 = deepcopy.((dx_pre, dx_pre))
        Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
        @test out[] ≈ primal
        @test dout[] ≈ 0
        @test dout2[] ≈ 0
        @test tupapprox(dx, dx_post)
        @test tupapprox(dx2, dx2_post)
    end
end

@testset "Mixed Reverse Apply iterate (list)" begin
    x = [[(2.0, [2.7]), (3.0, [3.14])], [(7.9, [47.0]), (11.2, [56.0])]]
    primal = 5562.9996
    @testset "$label" for (label, dx_pre, dx_post) in [
        (
            "dx == 0",
            [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]],
            [[(4.0, [5.4]), (6.0, [6.28])], [(15.8, [94.0]), (22.4, [112.0])]],
        ),
        (
            "dx != 0",
            [[(1.0, [-2.0]), (-3.0, [4.0])], [(5.0, [-6.0]), (-7.0, [8.0])]],
            [[(5.0, [3.4]), (3.0, [10.28])], [(20.8, [88.0]), (15.4, [120.0])]],
        ),
    ]
        dx = deepcopy(dx_pre)
        Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
        @test tupapprox(dx, dx_post)

        dx = deepcopy(dx_pre)
        res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
        @test res[2] ≈ primal
        @test tupapprox(dx, dx_post)
    end
end

@testset "BatchMixed Reverse Apply iterate (list)" begin
    x = [[(2.0, [2.7]), (3.0, [3.14])], [(7.9, [47.0]), (11.2, [56.0])]]
    primal = 5562.9996
    out_pre, dout_pre, dout2_pre = 0.0, 1.0, 3.0
    @testset "$label" for (label, dx_pre, dx_post, dx2_post) in [
        (
            "dx == 0",
            [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]],
            [[(4.0, [5.4]), (6.0, [6.28])], [(15.8, [94.0]), (22.4, [112.0])]],
            [[(3 * 4.0, [3 * 5.4]), (3 * 6.0, [3 * 6.28])], [(3 * 15.8, [3 * 94.0]), (3 * 22.4, [3 * 112.0])]],
        ),
        (
            "dx != 0",
            [[(1.0, [-2.0]), (-3.0, [4.0])], [(5.0, [-6.0]), (-7.0, [8.0])]],
            [[(5.0, [3.4]), (3.0, [10.28])], [(20.8, [88.0]), (15.4, [120.0])]],
            [[(1.0 + 3 * 4.0, [-2.0 + 3 * 5.4]), (-3.0 + 3 * 6.0, [4.0 + 3 * 6.28])], [(5.0 + 3 * 15.8, [-6.0 + 3 * 94.0]), (-7.0 + 3 * 22.4, [8.0 + 3 * 112.0])]],
        ),
    ]
        out, dout, dout2 = Ref.((out_pre, dout_pre, dout2_pre))
        dx, dx2 = deepcopy.((dx_pre, dx_pre))
        Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
        @test dout[] ≈ 0
        @test dout2[] ≈ 0
        @test tupapprox(dx, dx_post)
        @test tupapprox(dx2, dx2_post)

        out, dout, dout2 = Ref.((out_pre, dout_pre, dout2_pre))
        dx, dx2 = deepcopy.((dx_pre, dx_pre))
        Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
        @test out[] ≈ primal
        @test dout[] ≈ 0
        @test dout2[] ≈ 0
        @test tupapprox(dx, dx_post)
        @test tupapprox(dx2, dx2_post)
    end
end

struct MyRectilinearGrid5{FT,FZ}
    x :: FT
    z :: FZ
end


@inline flatten_tuple(a::Tuple) = @inbounds a[2:end]
@inline flatten_tuple(a::Tuple{<:Any}) = tuple() #inner_flatten_tuple(a[1])...)

function myupdate_state!(model)
    tupled = Base.inferencebarrier((model,model))
    flatten_tuple(tupled)
    return nothing
end

@testset "Abstract type allocation" begin
    model = MyRectilinearGrid5{Float64, Vector{Float64}}(0.0, [0.0])
    dmodel = MyRectilinearGrid5{Float64, Vector{Float64}}(0.0, [0.0])
    autodiff(Enzyme.Reverse,
                 myupdate_state!,
                 MixedDuplicated(model, Ref(dmodel)))
end
