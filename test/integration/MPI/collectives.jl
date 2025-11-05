using MPI
using Enzyme
using Test

MPI.Init()

@show Base.get_extension(Enzyme, :EnzymeMPIExt)

buff = Ref(3.0)
comm = MPI.COMM_WORLD

MPI.Allreduce!(buff, MPI.SUM, comm)

@test buff[] == MPI.Comm_size(comm) * 3.0

buff[] = 3.0
dbuff = Ref(0.0)

if MPI.Comm_rank(comm) == 0
    dbuff[] = 1.0
end

autodiff(ForwardWithPrimal, MPI.Allreduce!, Duplicated(buff, dbuff), Const(MPI.SUM), Const(comm))

@test buff[] == MPI.Comm_size(comm) * 3.0
@test dbuff[] == 1.0
