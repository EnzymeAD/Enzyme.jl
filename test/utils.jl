using Enzyme: Active, Duplicated, Forward, pick_batchsize
using Test

mode = Forward
ftype = typeof(sum)
argtypes = typeof.((Duplicated(ones(1), zeros(1)),))
@test pick_batchsize(1, mode, ftype, Active, argtypes...) == 1

argtypes = typeof.((Duplicated(ones(100), zeros(100)),))
@test pick_batchsize(100, mode, ftype, Active, argtypes...) == 16
