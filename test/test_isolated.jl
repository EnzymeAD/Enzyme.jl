using Enzyme, LinearAlgebra, Test

function run_test()
    T = UpperTriangular
    TE = ComplexF64
    sizeB = (3,)
    n = sizeB[1]
    M = rand(TE, n, n)
    B = rand(TE, sizeB...)
    Y = zeros(TE, sizeB...)
    A = T(M)
    _A = T(A)
    f!(Y, A, B, ::T) where {T} = ldiv!(Y, T(A), B)
    
    dY = (zeros(TE, sizeB...), zeros(TE, sizeB...))
    dM = (zeros(TE, n, n), zeros(TE, n, n))
    
    activities = (
        Const(f!),
        BatchDuplicated(Y, dY),
        BatchDuplicated(M, dM),
        Const(B),
        Const(_A)
    )
    
    forward, reverse = autodiff_thunk(
        ReverseSplitWithPrimal,
        Const{typeof(f!)},
        BatchDuplicated, # return activity
        map(typeof, Base.tail(activities))... # argument activities
    )
    
    println("Running targeted thunk forward...")
    tape, y_ad, shadow_result = forward(Const(f!), Base.tail(activities)...)
    
    ȳ = (ones(TE, sizeB...), ones(TE, sizeB...))
    for (sr, dy) in zip(shadow_result, ȳ)
        copyto!(sr, dy)
    end
    
    println("Running targeted thunk reverse...")
    reverse(Const(f!), Base.tail(activities)..., tape)
end

@testset "Isolated custom test" begin
    run_test()
end
