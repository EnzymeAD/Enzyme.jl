using MPI
using Enzyme
using Test

@testset "collectives" for np in (1, 2, 4)
        run(`$(mpiexec()) -n $np $(Base.julia_cmd()) --project=$(@__DIR__) $(joinpath(@__DIR__, "collectives.jl"))`)
end
