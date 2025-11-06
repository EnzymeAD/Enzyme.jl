using MPI
using Enzyme
using Test


function halo(x)
    np = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    requests = Vector{MPI.Request}()
    if rank != 0
        buf = @view x[1:1]
        push!(requests, MPI.Isend(x[2:2], MPI.COMM_WORLD; dest = rank - 1, tag = 0))
        push!(requests, MPI.Irecv!(buf, MPI.COMM_WORLD; source = rank - 1, tag = 0))
    end
    if rank != np - 1
        buf = @view x[end:end]
        push!(requests, MPI.Isend(x[(end - 1):(end - 1)], MPI.COMM_WORLD; dest = rank + 1, tag = 0))
        push!(requests, MPI.Irecv!(buf, MPI.COMM_WORLD; source = rank + 1, tag = 0))
    end
    for request in requests
        MPI.Wait(request)
    end
    return nothing
end

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nl = rank == 0 ? 0 : 2
nr = rank == np - 1 ? 0 : 2
nlocal = nr + nl + 1

x = zeros(nlocal)
fill!(x, Float64(rank))
halo(x)
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

dx = zeros(nlocal)
fill!(dx, Float64(rank))
autodiff(Reverse, halo, Duplicated(x, dx))
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

fill!(dx, Float64(rank))
autodiff(Forward, halo, Duplicated(x, dx))
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
