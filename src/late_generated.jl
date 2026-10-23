# Generated-function entry points whose generators run the EnzymeInterpreter (or method
# lookups through it). They are defined here, after every other method in Enzyme, on purpose:
#
# `jl_code_for_staged` runs a generator with `ct->world_age = def->primary_world`, i.e. the
# world in which the generated *wrapper* method was defined. Everything the generator makes
# the runtime compile on the fly -- in particular `Base.Compiler.typeinf_local(::EnzymeInterpreter, …)`
# and friends -- is therefore inferred at that world. If the wrapper is defined early in the
# module, those CodeInstances get a bounded `max_world` (later Enzyme definitions intersect
# their edges) even though they are created during the precompile workload. Julia 1.12 drops
# such external CodeInstances from the pkgimage (`queue_external_cis` requires
# `max_world == typemax`; fixed on 1.13 by JuliaLang/julia#59436), which costs ~3.4 s of JIT
# on the first `autodiff` of every session.
#
# Defining the wrappers last makes their `primary_world` postdate all other Enzyme methods, so
# the code compiled from inside the generators is valid in all later worlds and precompiles.
# Only the wrapper *methods* live here; the generators stay next to the code they belong to.
#
# Constraint: because these methods are the newest in the module, they must not be called at
# *generation time* from a generated function defined earlier (its generator runs in that
# earlier `primary_world`, where these methods do not exist yet, and would fail with a
# "method too new" MethodError whenever Enzyme is loaded without a pkgimage). Calling them
# from ordinary functions, or from the code a generator *returns*, is fine. `prevmethodinstance`
# deliberately stays in utils.jl for this reason (`onehot_internal`'s generator uses it).

@eval Base.@assume_effects :removable :foldable :nothrow @inline function Compiler.primal_return_type(mode::Mode, ft::Type, tt::Type)
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, Compiler.primal_return_type_generator))
end

@eval @inline function Compiler.thunk(
        fakeworld::Val{0},
        fa::Type{FA},
        a::Type{A},
        tt::Type{TT},
        mode::Val{Mode},
        width::Val{Width},
        modifiedbetween::Val{ModifiedBetween},
        returnprimal::Val{ReturnPrimal},
        shadowinit::Val{ShadowInit},
        abi::Type{ABI},
        erriffuncwritten::Val{ErrIfFuncWritten},
        runtimeactivity::Val{RuntimeActivity},
        strongzero::Val{StrongZero}
    ) where {
        FA <: Annotation,
        A <: Annotation,
        TT,
        Mode,
        Width,
        ModifiedBetween,
        ReturnPrimal,
        ShadowInit,
        ABI,
        ErrIfFuncWritten,
        RuntimeActivity,
        StrongZero,
    }
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, Compiler.thunk_generator))
end

@eval @inline function Compiler.deferred_id_codegen(
        fa::Type{FA},
        a::Type{A},
        tt::Type{TT},
        mode::Val{Mode},
        width::Val{Width},
        modifiedbetween::Val{ModifiedBetween},
        returnprimal::Val{ReturnPrimal},
        shadowinit::Val{ShadowInit},
        expectedtapetype::Type{ExpectedTapeType},
        erriffuncwritten::Val{ErrIfFuncWritten},
        runtimeactivity::Val{RuntimeActivity},
        strongzero::Val{StrongZero}
    ) where {
        FA <: Annotation,
        A <: Annotation,
        TT,
        Mode,
        Width,
        ModifiedBetween,
        ReturnPrimal,
        ShadowInit,
        ExpectedTapeType,
        ErrIfFuncWritten,
        RuntimeActivity,
        StrongZero,
    }
    $(Expr(:meta, :generated_only))
    $(Expr(:meta, :generated, Compiler.deferred_id_generator))
end
