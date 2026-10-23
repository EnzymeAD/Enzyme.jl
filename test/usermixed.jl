using Enzyme
using Test

########## MixedDuplicated of Return

function user_mixret(x, y)
    return (x, y)
end

@testset "MixedDuplicated struct return" begin
    x = 2.7
    y = [3.14]
    dy = [0.0]

    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(user_mixret)}, MixedDuplicated, Active{Float64}, Duplicated{Vector{Float64}})

    tape, res, dres = fwd(Const(user_mixret), Active(x), Duplicated(y, dy))

    @test res[1] ≈ x
    @test res[2] === y

    @test dres[][1] ≈ 0.0
    @test dres[][2] === dy

    outs = rev(Const(user_mixret), Active(x), Duplicated(y, dy), (47.56, dy), tape)

    @test outs[1][1] ≈ 47.56
end

@testset "BatchMixedDuplicated struct return" begin
    x = 2.7
    y = [3.14]
    dy = [0.0]
    dy2 = [0.0]

    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(user_mixret)}, BatchMixedDuplicated, Active{Float64}, BatchDuplicated{Vector{Float64}, 2})

    tape, res, dres = fwd(Const(user_mixret), Active(x), BatchDuplicated(y, (dy, dy2)))

    @test res[1] ≈ x
    @test res[2] === y

    @test dres[1][][1] ≈ 0.0
    @test dres[1][][2] === dy
    @test dres[2][][1] ≈ 0.0
    @test dres[2][][2] === dy2

    outs = rev(Const(user_mixret), Active(x), BatchDuplicated(y, (dy, dy2)), ((47.0, dy), (56.0, dy)), tape)

    @test outs[1][1][1] ≈ 47.0
    @test outs[1][1][2] ≈ 56.0
end


function user_fltret(x, y)
    return x
end

@testset "MixedDuplicated float return" begin
    x = 2.7

    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(identity)}, MixedDuplicated, Active{Float64})

    tape, res, dres = fwd(Const(identity), Active(x))

    @test res ≈ x
    @test dres[] ≈ 0.0

    outs = rev(Const(identity), Active(x), 47.56, tape)

    @test outs[1][1] ≈ 47.56
end

@testset "BatchMixedDuplicated float return" begin
    x = 2.7
    y = [3.14]
    dy = [0.0]
    dy2 = [0.0]

    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(user_fltret)}, BatchMixedDuplicated, Active{Float64}, BatchDuplicated{Vector{Float64}, 2})

    tape, res, dres = fwd(Const(user_fltret), Active(x), BatchDuplicated(y, (dy, dy2)))

    @test res ≈ x

    @test dres[1][] ≈ 0.0
    @test dres[2][] ≈ 0.0

    outs = rev(Const(user_fltret), Active(x), BatchDuplicated(y, (dy, dy2)), (47.0, 56.0), tape)

    @test outs[1][1][1] ≈ 47.0
    @test outs[1][1][2] ≈ 56.0
end

function vecsq(x)
    x[2] = x[1] * x[1]
    return x
end

@testset "MixedDuplicated vector return" begin
    y = [3.14, 0.0]
    dy = [0.0, 2.7]

    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(vecsq)}, MixedDuplicated, Duplicated{Vector{Float64}})

    tape, res, dres = fwd(Const(vecsq), Duplicated(y, dy))

    @test res === y

    @test dres[] === dy

    outs = rev(Const(vecsq), Duplicated(y, dy), dy, tape)

    @test dy ≈ [3.14 * 2.7 * 2, 0.0]
end


########## MixedDuplicated of Argument

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
    @test dtup[2][][1] ≈ 3 * 3.14
    @test dtup[2][][2] ≈ [3 * 2.7]
end

function mix_square(x)
    return x * x
end

@testset "MixedDuplicated float64 call" begin
    tup = 2.7
    dtup = Ref(0.0)
    res = autodiff(Reverse, mix_square, Active, MixedDuplicated(tup, dtup))
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
    res = autodiff(Reverse, mix_square_byref, Const, BatchDuplicated(out, dout), BatchMixedDuplicated(tup, dtup))
    @test res[1] == (nothing, nothing)
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
    res = autodiff(Reverse, mix_ar_byref, Const, BatchDuplicated(out, dout), BatchMixedDuplicated(tup, dtup))
    @test res[1] == (nothing, nothing)
    @test dtup[1][] ≈ [3.14, 2.7]
    @test dtup[2][] ≈ [3 * 3.14, 3 * 2.7]
end
