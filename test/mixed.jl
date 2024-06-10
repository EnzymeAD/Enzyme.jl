using Enzyme, Test

@noinline function mixedmul(tup::T) where T
    return tup[1] * tup[2][1]
end

function outmixedmul(x::Float64)
    vec = [x]
    tup = (x, vec)
    Base.inferencebarrier(mixedmul)(tup)::Float64
end

function outmixedmul2(res, x::Float64)
    vec = [x]
    tup = (x, vec)
    res[] = Base.inferencebarrier(mixedmul)(tup)::Float64
end

@testset "Basic Mixed Activity" begin
	@test 6.2 ≈ Enzyme.autodiff(Reverse, outmixedmul, Active, Active(3.1))[1][1]
end

@testset "Byref Mixed Activity" begin
	res = Ref(4.7)
	dres = Ref(1.0)
	@test 6.2 ≈ Enzyme.autodiff(Reverse, outmixedmul2, Const, Duplicated(res, dres), Active(3.1))[1][2]
end

@static if VERSION >= v"1.8-"
@testset "Batched Byref Mixed Activity" begin
	res = Ref(4.7)
	dres = Ref(1.0)
	dres2 = Ref(3.0)
	sig = Enzyme.autodiff(Reverse, outmixedmul2, Const, BatchDuplicated(res, (dres, dres2)), Active(3.1))
	@test 6.2 ≈ sig[1][2][1]
	@test 3*6.2 ≈ sig[1][2][2]
end
end

function tupmixedmul(x::Float64)
    vec = [x]
    tup = (x, Base.inferencebarrier(vec))
    Base.inferencebarrier(mixedmul)(tup)::Float64
end

@testset "Tuple Mixed Activity" begin
	@test 6.2 ≈ Enzyme.autodiff(Reverse, tupmixedmul, Active, Active(3.1))[1][1]
end

function outtupmixedmul(res, x::Float64)
    vec = [x]
    tup = (x, Base.inferencebarrier(vec))
    res[] = Base.inferencebarrier(mixedmul)(tup)::Float64
end

@testset "Byref Tuple Mixed Activity" begin
	res = Ref(4.7)
	dres = Ref(1.0)
	@test 6.2 ≈ Enzyme.autodiff(Reverse, outtupmixedmul, Const, Duplicated(res, dres), Active(3.1))[1][2]
end

@static if VERSION >= v"1.8-"
@testset "Batched Byref Tuple Mixed Activity" begin
	res = Ref(4.7)
	dres = Ref(1.0)
	dres2 = Ref(3.0)
	sig = Enzyme.autodiff(Reverse, outtupmixedmul, Const, BatchDuplicated(res, (dres, dres2)), Active(3.1))
	@test 6.2 ≈ sig[1][2][1]
	@test 3*6.2 ≈ sig[1][2][2]
end
end
