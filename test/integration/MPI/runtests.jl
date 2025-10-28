using MPI
using Enzyme

include("mpi.jl")

run(`$(mpiexec()) -n 2 $(Base.julia_cmd()) mpi.jl`)
