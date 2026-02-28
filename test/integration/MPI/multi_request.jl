using MPI
using Enzyme
using Test


function halo(reqs, x)
    np = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    requests = Vector{MPI.Request}()
    if rank != 0
        buf = @view x[1:1]
        MPI.Isend(x[2:2], MPI.COMM_WORLD, reqs[1]; dest = rank - 1, tag = 0)
        MPI.Irecv!(buf, MPI.COMM_WORLD, reqs[2]; source = rank - 1, tag = 0)
    end
    if rank != np - 1
        buf = @view x[end:end]
        MPI.Isend(x[(end - 1):(end - 1)], MPI.COMM_WORLD, reqs[3]; dest = rank + 1, tag = 0)
        MPI.Irecv!(buf, MPI.COMM_WORLD, reqs[4]; source = rank + 1, tag = 0)
    end
    for req in requests
        MPI.Wait(req) # TODO: Check MPI.Waitall
    end
    return nothing
end

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nl = rank == 0 ? 0 : 2
nr = rank == np - 1 ? 0 : 2
nlocal = nr + nl + 1

reqs = MPI.UnsafeMultiRequest(4)
x = zeros(nlocal)
fill!(x, Float64(rank))
halo(reqs, x)
MPI.Barrier(MPI.COMM_WORLD)

@test x[nl + 1] == Float64(rank)      # Local
if rank != 0
    @test x[1] == Float64(rank - 1)   # Recv
    @test x[2] == Float64(rank)       # Send
end
if rank != np - 1
    @test x[end] == Float64(rank + 1) # Recv
    @test x[end - 1] == Float64(rank) # Send
end

reqs = MPI.UnsafeMultiRequest(4)
dreqs = MPI.UnsafeMultiRequest(4)
dx = zeros(nlocal)
fill!(dx, Float64(rank))
autodiff(Reverse, halo, Duplicated(reqs, dreqs), Duplicated(x, dx))
MPI.Barrier(MPI.COMM_WORLD)

@test dx[nl + 1] == Float64(rank)                  # Local -> no change
if rank != 0
    @test dx[1] == 0.0                             # Recv -> Send & zero'd
    @test dx[2] == Float64(rank + rank - 1)        # Send -> += Recv
end
if rank != np - 1
    @test dx[end] == 0.0                           # Recv -> Send & zero'd
    @test dx[end - 1] == Float64(rank + rank + 1)  # Send -> += Recv
end

reqs = MPI.UnsafeMultiRequest(4)
dreqs = MPI.UnsafeMultiRequest(4)
fill!(dx, Float64(rank))
autodiff(Forward, halo, Duplicated(reqs, dreqs), Duplicated(x, dx))
MPI.Barrier(MPI.COMM_WORLD)

@test dx[nl + 1] == Float64(rank)
if rank != 0
    @test dx[1] == Float64(rank - 1)
    @test dx[2] == Float64(rank)
end
if rank != np - 1
    @test dx[end] == Float64(rank + 1)
    @test dx[end - 1] == Float64(rank)
end
