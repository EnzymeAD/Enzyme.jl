using MPI
using Enzyme
using Test

MPI.Init()

comm = MPI.COMM_WORLD

@test autodiff(ForwardWithPrimal, MPI.Comm_size, Const(comm)) == (MPI.Comm_size(comm),)
@test autodiff(ForwardWithPrimal, MPI.Comm_rank, Const(comm)) == (MPI.Comm_rank(comm),)

@test autodiff(ReverseWithPrimal, MPI.Comm_size, Const(comm)) == ((nothing,), MPI.Comm_size(comm))
@test autodiff(ReverseWithPrimal, MPI.Comm_rank, Const(comm)) == ((nothing,), MPI.Comm_rank(comm))
