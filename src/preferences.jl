using Preferences

"""
    run_attributor() -> Bool

Return whether the Enzyme attributor pass is enabled. This is a compile-time
preference that requires restarting Julia to take effect after being changed.

The attributor pass can improve differentiation quality but may cause issues on
Julia 1.12 and later. Defaults to `false`.

See also [`set_run_attributor!`](@ref).
"""
run_attributor() = @load_preference("run_attributor", false)

"""
    set_run_attributor!(val::Bool)

Set whether the Enzyme attributor pass is enabled. This is a compile-time
preference stored via `Preferences.jl`; a restart of Julia is required for
the change to take effect.

See also [`run_attributor`](@ref).
"""
set_run_attributor!(val::Bool) = @set_preferences!("run_attributor" => val)
