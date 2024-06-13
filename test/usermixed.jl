using Enzyme
using Test

function user_mixfnc(tup)
    return tup[1] * tup[2][1]
end

@testset "MixedDuplicated struct call" begin
    tup = (2.7, [3.14])
    dtup = Ref((0.0, [0.0]))

    res = autodiff(Reverse, user_mixfnc, Active, MixedDuplicated(tup, dtup))
    @test dtup[][1] ≈ 3.14
    @test dtup[][2] ≈ [2.7]
end


function user_mixfnc_byref(out, tup)
    out[] = tup[1] * tup[2][1]
    return nothing
end

@testset "Batch MixedDuplicated struct call" begin
    tup = (2.7, [3.14])
    dtup = (Ref((0.0, [0.0])), Ref((0.0, [0.0])))
    out = Ref(0.0)
    dout = (Ref(1.0), Ref(3.0))
    res = autodiff(Reverse, user_mixfnc_byref, Const, BatchDuplicated(out, dout), BatchMixedDuplicated(tup, dtup))
    @test dtup[1][][1] ≈ 3.14
    @test dtup[1][][2] ≈ [2.7]
    @test dtup[2][][1] ≈ 3*3.14
    @test dtup[2][][2] ≈ [3*2.7]
end

function mix_square(x)
    return x * x
end

@testset "MixedDuplicated float64 call" begin
    tup = 2.7
    dtup = Ref(0.0)
    res = autodiff(Reverse, mix_square, Active, MixedDuplicated(tup, dtup))[1]
    @test res[1] == (nothing,)
    @test dtup[] ≈ 2 * 2.7
end


function mix_square_byref(out, x)
    out[] = x * x
    return nothing
end

@testset "BatchMixedDuplicated float64 call" begin
    tup = 2.7
    dtup = (Ref(0.0), Ref(0.0))
    out = Ref(0.0)
    dout = (Ref(1.0), Ref(3.0))
    res = autodiff(Reverse, mix_square, Const, BatchDuplicated(out, dout), BatchMixedDuplicated(tup, dtup))[1]
    @test res[1] == (nothing,)
    @test dtup[1][] ≈ 2 * 2.7
    @test dtup[2][] ≈ 3 * 2 * 2.7
end

function mix_ar(x)
    return x[1] * x[2]
end

@testset "MixedDuplicated vector{float64} call" begin
    tup = [2.7, 3.14]
    dtup = Ref([0.0, 0.0])
    res = autodiff(Reverse, mix_ar, Active, MixedDuplicated(tup, dtup))
    @test res[1] == (nothing,)
    @test dtup[] ≈ [3.14, 2.7]
end


function mix_ar_byref(out, x)
    out[] = x[1] * x[2]
    return nothing
end

@testset "BatchMixedDuplicated vector{float64} call" begin
    tup = [2.7, 3.14]
    dtup = (Ref([0.0, 0.0]), Ref([0.0, 0.0]))
    out = Ref(0.0)
    dout = (Ref(1.0), Ref(3.0))
    res = autodiff(Reverse, mix_ar, Const, BatchDuplicated(out, dout), BatchMixedDuplicated(tup, dtup))
    @test res[1] == (nothing,)
    @test dtup[1][] ≈ [3.14, 2.7]
    @test dtup[2][] ≈ [3*3.14, 3*2.7]
end
