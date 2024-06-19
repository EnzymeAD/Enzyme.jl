using Enzyme: pick_chunksize
using EnzymeCore: Forward
using Test

mode = Forward
ftype = typeof(sum)
argtypes = typeof.((Duplicated(ones(1), zeros(1)),))
@test pick_chunksize(Val(1), mode, ftype, argtypes...) == 1

argtypes = typeof.((Duplicated(ones(100), zeros(100)),))
@test pick_chunksize(Val(100), mode, ftype, argtypes...) == 16
