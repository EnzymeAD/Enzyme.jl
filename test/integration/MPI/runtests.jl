using MPI
using Enzyme

# Current MPI support (needs to be tested from Julia)
# - MPI_Recv/MPI_Send
# - MPI_Ssend
# - MPI_Irecv/MPI_Isend with MPI_Wait/MPI_Waitall
# - MPI_Barrier/MPI_Probe
# - MPI_Allreduce
# - MPI_Bcast
# - MPI_Reduce
# - MPI_Gather/MPI_Scatter
# - MPI_Allgather

@testset "blocking_ring" for np in (1, 2, 4)
    run(`$(mpiexec()) -n $np $(Base.julia_cmd()) blocking_ring.jl`)
end

@testset "nonblocking_halo" for np in (1, 2, 4)
    run(`$(mpiexec()) -n $np $(Base.julia_cmd()) nonblocking_halo.jl`)
end

