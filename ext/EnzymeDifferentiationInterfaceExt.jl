module EnzymeDifferentiationInterfaceExt

function __init__()
    @warn """You are using DifferentiationInterface!"
             DifferentiationInterface introduces interstitial wrappers that may limit the scope of input programs and add overhead."
             This can cause derivatives to be slower, or fail to differentiate with default settings when they work with Enzyme directly (e.g. Enzyme.gradient instead of DI.gradient)."
             If you find issues, please report at https://github.com/EnzymeAD/Enzyme.jl/issues/new and try Enzyme directly in the interim."""
end

end # module
