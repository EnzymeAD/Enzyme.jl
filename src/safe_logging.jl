# Logging from code that runs inside generated functions.
#
# The generators of `thunk`, `runtime_thunk`, `primal_return_type` and Enzyme's other
# generated functions run Enzyme's inference and compilation in the world their method was
# defined in, where task switches are not allowed. GPUCompiler's `@safe_*` macros avoid the
# task switch by logging synchronously to `Core.stderr`, but they ask the global logger for
# its level. A logger type whose methods were defined after Enzyme, such as Test's
# `TestLogger` when Test is loaded after Enzyme, has no methods in that world, so on Julia
# 1.10 a global logger of such a type made every derivative fail with a `MethodError`. These
# macros read the level the global log state recorded when that logger was set instead.

"""
    safe_min_level()

The minimum level of the global logger, as the global log state recorded it when the logger
was set, without calling any method of the logger.
"""
safe_min_level() = Base.CoreLogging._global_logstate.min_enabled_level

for level in (:debug, :info, :warn, :error)
    @eval begin
        @doc $"""
            @safe_$level args...

        `@$level`, written synchronously to `Core.stderr` at the global logger's level, for
        code that may run inside a generated function.
        """
        macro $(Symbol(:safe_, level))(ex...)
            macrocall = :(@placeholder $(ex...) _file = $(String(__source__.file)) _line = $(__source__.line))
            macrocall.args[1] = Symbol($"@$level")
            # `with_logger` would create a closure, which generated code cannot hold, so this
            # sets the log state as `with_logger` does.
            return quote
                safe_logstate = Base.CoreLogging.LogState(
                    Base.CoreLogging.SimpleLogger(Core.stderr, safe_min_level())
                )
                @static if VERSION < v"1.11-"
                    t = current_task()
                    old_logstate = t.logstate
                    try
                        t.logstate = safe_logstate
                        $(esc(macrocall))
                    finally
                        t.logstate = old_logstate
                    end
                else
                    Base.ScopedValues.@with(
                        Base.CoreLogging.CURRENT_LOGSTATE => safe_logstate, $(esc(macrocall))
                    )
                end
            end
        end
    end
end
