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
    dx = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]
    res = Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @test tupapprox(dx, [((4.0, [5.4]), (6.0, [6.28])), ((15.8, [94.0]), (22.4, [112.0]))])

    x = [((2.0, [2.7]), (3.0, [3.14])), ((7.9, [47.0]), (11.2, [56.0]))]

    dx = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]
    res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @test res[2] ≈ 5562.9996
    @test tupapprox(dx, [((4.0, [5.4]), (6.0, [6.28])), ((15.8, [94.0]), (22.4, [112.0]))])
end

@testset "BatchMixed Reverse Apply iterate (tuple)" begin
    x = [((2.0, [2.7]), (3.0, [3.14])), ((7.9, [47.0]), (11.2, [56.0]))]
    dx = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]
    dx2 = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test tupapprox(dx, [((4.0, [5.4]), (6.0, [6.28])), ((15.8, [94.0]), (22.4, [112.0]))])
    @test tupapprox(dx2, [((3*4.0, [3*5.4]), (3*6.0, [3*6.28])), ((3*15.8, [3*94.0]), (3*22.4, [3*112.0]))])

    x = [((2.0, [2.7]), (3.0, [3.14])), ((7.9, [47.0]), (11.2, [56.0]))]
    dx = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]
    dx2 = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    res = Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test out[] ≈ 5562.9996
    @test tupapprox(dx, [((4.0, [5.4]), (6.0, [6.28])), ((15.8, [94.0]), (22.4, [112.0]))])
    @test tupapprox(dx2, [((3*4.0, [3*5.4]), (3*6.0, [3*6.28])), ((3*15.8, [3*94.0]), (3*22.4, [3*112.0]))])
end


@testset "Mixed Reverse Apply iterate (list)" begin
    x = [[(2.0, [2.7]), (3.0, [3.14])], [(7.9, [47.0]), (11.2, [56.0])]]
    dx = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]

    res = Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @test tupapprox(dx, [[(4.0, [5.4]), (6.0, [6.28])], [(15.8, [94.0]), (22.4, [112.0])]])

    dx = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]

    res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @test res[2] ≈ 5562.9996
    @test tupapprox(dx, [[(4.0, [5.4]), (6.0, [6.28])], [(15.8, [94.0]), (22.4, [112.0])]])
end

@testset "BatchMixed Reverse Apply iterate (list)" begin
    x = [[(2.0, [2.7]), (3.0, [3.14])], [(7.9, [47.0]), (11.2, [56.0])]]
    dx = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]
    dx2 = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicatedNoNeed(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test tupapprox(dx, [[(4.0, [5.4]), (6.0, [6.28])], [(15.8, [94.0]), (22.4, [112.0])]])
    @test tupapprox(dx2, [[(3*4.0, [3*5.4]), (3*6.0, [3*6.28])], [(3*15.8, [3*94.0]), (3*22.4, [3*112.0])]])

    x = [[(2.0, [2.7]), (3.0, [3.14])], [(7.9, [47.0]), (11.2, [56.0])]]
    dx = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]
    dx2 = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]

    out = Ref(0.0)
    dout = Ref(1.0)
    dout2 = Ref(3.0)
    res = Enzyme.autodiff(Reverse, make_byref, Const, BatchDuplicated(out, (dout, dout2)), Const(mixed_metasumsq), Const(metaconcat), BatchDuplicated(x, (dx, dx2)))
    @test out[] ≈ 5562.9996
    @test tupapprox(dx, [[(4.0, [5.4]), (6.0, [6.28])], [(15.8, [94.0]), (22.4, [112.0])]])
    @test tupapprox(dx2, [[(3*4.0, [3*5.4]), (3*6.0, [3*6.28])], [(3*15.8, [3*94.0]), (3*22.4, [3*112.0])]])
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
