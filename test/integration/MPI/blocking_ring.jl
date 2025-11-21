using MPI
using Enzyme
using Test

function ring(token, comm)
    rank = MPI.Comm_rank(comm)
    N = MPI.Comm_size(comm)

    @assert N >= 1
    if N == 1
        return token
    end

    buf = Ref(token)
    if rank != 0
        MPI.Recv!(buf, comm; source = rank - 1)
    end
    MPI.Send(buf, comm; dest = mod(rank + 1, N))

    # Now rank 0 can receive the token
    if rank == 0
        MPI.Recv!(buf, comm; source = N - 1)
    end
    return buf[]
end

if !MPI.Initialized()
    MPI.Init()
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

token = rank == 0 ? 42.0 : NaN
token = ring(token, comm)
@test token == 42.0

function dring_fwd(token, dtoken, comm)
    return autodiff(ForwardWithPrimal, ring, Duplicated(token, dtoken), Const(comm))
end

token = rank == 0 ? 42.0 : NaN
dtoken = rank == 0 ? 1.0 : NaN
dtoken, token = dring_fwd(token, dtoken, comm)
@test token == 42.0
@test dtoken == 1.0

# function dring_rev(token, comm)
#     return autodiff(ReverseWithPrimal, ring, Active(token), Const(comm))
# end
#
# token = rank == 0 ? 42.0 : NaN
# dtoken, token = dring_rev(token, comm)
