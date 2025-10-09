using Test

@testset "Threads" begin
    cmd = `$(Base.julia_cmd()) --threads=1 --startup-file=no threads.jl`
    @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
    cmd = `$(Base.julia_cmd()) --threads=2 --startup-file=no threads.jl`
    @test success(pipeline(cmd, stderr = stderr, stdout = stdout))
end
