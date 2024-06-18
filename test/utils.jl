using Test

@test Enzyme.pick_chunksize(zeros(1)) == Val(16)
@test Enzyme.pick_chunksize(zeros(100)) == Val(16)
