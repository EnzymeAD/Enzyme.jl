using Test

@test Enzyme.pick_chunksize(zeros(4)) == 4
@test Enzyme.pick_chunksize(zeros(4); threshold=2) == 2
@test Enzyme.pick_chunksize(zeros(100)) == 16
