using MPI
using Enzyme
using Test

struct Context
    x::Vector{Float64}
end

function halo(context)
    x = context.x
    np = MPI.Comm_size(MPI.COMM_WORLD)
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    requests = Vector{MPI.Request}()
    if rank != 0
        buf =  @view x[1:1]
        push!(requests, MPI.Isend(x[2:2], MPI.COMM_WORLD; dest=rank-1, tag=0))
        push!(requests, MPI.Irecv!(buf, MPI.COMM_WORLD; source=rank-1, tag=0))
    end
    if rank != np-1
        buf =  @view x[end:end]
        push!(requests, MPI.Isend(x[end-1:end-1], MPI.COMM_WORLD; dest=rank+1, tag=0))
        push!(requests, MPI.Irecv!(buf, MPI.COMM_WORLD; source=rank+1, tag=0))
    end
    for request in requests
        MPI.Wait(request)
    end
    return nothing
end

MPI.Init()
np = MPI.Comm_size(MPI.COMM_WORLD)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
n = np*10
n1 = Int(round(rank / np * (n+np))) - rank
n2 = Int(round((rank + 1) / np * (n+np))) - rank
nl = rank == 0 ? n1+1 : n1
nr = rank == np-1 ? n2-1 : n2
nlocal = nr-nl+1
context = Context(zeros(nlocal))
fill!(context.x, Float64(rank))
halo(context)
if rank != 0
    @test context.x[1] == Float64(rank-1)
end
if rank != np-1
    @test context.x[end] == Float64(rank+1)
end

dcontext = Context(zeros(nlocal))
fill!(dcontext.x, Float64(rank))
autodiff(Reverse, halo, Duplicated(context, dcontext))
MPI.Barrier(MPI.COMM_WORLD)
if rank != 0
    @test dcontext.x[2] == Float64(rank + rank - 1)
end
if rank != np-1
    @test dcontext.x[end-1] == Float64(rank + rank + 1)
end
if !isinteractive()
    MPI.Finalize()
end
