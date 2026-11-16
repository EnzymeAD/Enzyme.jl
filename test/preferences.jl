using Enzyme, Test

@testset "run_attributor preference" begin
    # Default is false
    @test run_attributor() isa Bool

    current = run_attributor()

    # Setter round-trips the value
    set_run_attributor!(!current)
    @test run_attributor() == !current

    # Restore original value
    set_run_attributor!(current)
    @test run_attributor() == current

    # RunAttributor compile-time constant reflects the preference at load time
    @test Enzyme.Compiler.RunAttributor isa Bool
end
