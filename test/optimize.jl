using Enzyme, LinearAlgebra, Test

function gcloaded_fixup(dest, src)
    N = size(src)
    dat = src.data
    len = N[1]

    i = 1
    while true
        j = 1
        while true
            ld = @inbounds if i <= j
                dat[(i-1) * 2 + j]
            else
                dat[(j-1) * 2 + i]
            end
            @inbounds dest[(i-1) * 2 + j] = ld
            if j == len
                break
            end
            j += 1
        end
        if i == len
            break
        end
        i += 1
    end
    return nothing
end

@testset "GCLoaded fixup" begin
	H = Hermitian(Matrix([4.0 1.0; 2.0 5.0]))
	dest = Matrix{Float64}(undef, 2, 2)

	Enzyme.autodiff(
	    ForwardWithPrimal,
	    gcloaded_fixup,
	    Const,
	    Const(dest),
	    Const(H),
	)[1]
    @test dest ≈ [4.0 2.0; 2.0 5.0]
    dest = Matrix{Float64}(undef, 2, 2)
    gcloaded_fixup(dest, H)
    @test dest ≈ [4.0 2.0; 2.0 5.0]
end
