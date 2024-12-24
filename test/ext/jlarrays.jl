using Enzyme, Test, JLArrays

function jlres(x)
    2 * collect(x)
end

@testset "JLArrays" begin
    Enzyme.jacobian(Forward, jlres, JLArray([3.0, 5.0]))
    Enzyme.jacobian(Reverse, jlres, JLArray([3.0, 5.0]))
end
