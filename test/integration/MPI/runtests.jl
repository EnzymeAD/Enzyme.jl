using MPI
using Enzyme
using Test

# Current MPI support (needs to be tested from Julia)
# - MPI_Ssend
# - MPI_Waitall
# - MPI_Barrier/MPI_Probe
# - MPI_Allreduce
# - MPI_Bcast
# - MPI_Reduce
# - MPI_Gather/MPI_Scatter
# - MPI_Allgather

# Query functions MPI_Comm_size/MPI_Comm_rank
@testset "queries" for np in (1, 2, 4)
    run(`$(mpiexec()) -n $np $(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, "queries.jl"))`)
end

# Test MPI_Recv/MPI_Send with a blocking ring communication pattern
@testset "blocking_ring" for np in (1, 2, 4)
    run(`$(mpiexec()) -n $np $(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, "blocking_ring.jl"))`)
end

# Test MPI_Irecv/MPI_Isend/MPI_Wait with a non-blocking halo exchange pattern
VERSION >= v"1.11.0" && @testset "nonblocking_halo" for np in (1, 2, 4)
    run(`$(mpiexec()) -n $np $(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, "nonblocking_halo.jl"))`)
end
