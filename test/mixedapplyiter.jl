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


@testset "Mixed Reverse Apply iterate" begin
    x = [((2.0, [2.7]), (3.0, [3.14])), ((7.9, [47.0]), (11.2, [56.0]))]
    dx = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]
    res = Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @show dx
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])

    dx = [((0.0, [0.0]), (0.0, [0.0])), ((0.0, [0.0]), (0.0, [0.0]))]
    res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @show dx
    @show res
    @test res[2] ≈ 200.84999999999997
    @test tupapprox(dx, [(4.0, 6.0), (15.8, 22.4)])

    x = [[(2.0, [2.7]), (3.0, [3.14])], [(7.9, [47.0]), (11.2, [56.0])]]
    dx = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]

    res = Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))
    @show dx
    @test dx ≈ [[4.0, 6.0], [15.8, 22.4]]

    dx = [[(0.0, [0.0]), (0.0, [0.0])], [(0.0, [0.0]), (0.0, [0.0])]]

    res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(metaconcat), Duplicated(x, dx))

    @test res[2] ≈ 200.84999999999997
    @test tupapprox(dx, [[4.0, 6.0], [15.8, 22.4]])
end
