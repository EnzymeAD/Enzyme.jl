module EnzymeDynamicPPLExt

using ADTypes
using DynamicPPL

@static if isdefined(DynamicPPL, :is_supported)
  DynamicPPL.is_supported(::ADTypes.AutoEnzyme) = true
end

end # module
