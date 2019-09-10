using Enzyme
using Test
using ReverseDiff

@testset "Taylor series tests" begin

# Taylor series for `-log(1-x)`
# eval at -log(1-1/2) = -log(1/2)
function euroad(f::T) where T
    g = zero(T)
    for i in 1:10^7
        g += f^i / i
    end
    return g
end
euroad′(x) = autodiff(euroad, x)

@test euroad(0.5) ≈ -log(0.5)
@test euroad′(0.5) ≈ ReverseDiff.gradient(euroad, 0.5)

end