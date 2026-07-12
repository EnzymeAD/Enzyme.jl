using Enzyme, Test

mixed_concat() = ()
mixed_concat(a) = a
mixed_concat(a, b) = (a..., b...)
mixed_concat(a, b, c...) = mixed_concat(mixed_concat(a, b), c...)

mixed_metaconcat(x) = mixed_concat(x...)

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

function mixed_make_byref(out, fn, args...)
	out[] = fn(args...)
	nothing
end

function mixed_tupapprox(a, b)
    if a isa Tuple && b isa Tuple
        if length(a) != length(b)
            return false
        end
        for (aa, bb) in zip(a, b)
            if !mixed_tupapprox(aa, bb)
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
            if !mixed_tupapprox(a[i], b[i])
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
        Enzyme.autodiff(Reverse, mixed_metasumsq, Active, Const(mixed_metaconcat), Duplicated(x, dx))
        @test mixed_tupapprox(dx, dx_post)

        dx = deepcopy(dx_pre)
        res = Enzyme.autodiff(ReverseWithPrimal, mixed_metasumsq, Active, Const(mixed_metaconcat), Duplicated(x, dx))
        @test res[2] ≈ primal
        @test mixed_tupapprox(dx, dx_post)
    end
end
