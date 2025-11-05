using MPI
using Enzyme
using Test

# Query functions MPI_Comm_size/MPI_Comm_rank
@testset "queries" for np in (1, 2, 4)
    run(`$(mpiexec()) -n $np $(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, "queries.jl"))`)
end
