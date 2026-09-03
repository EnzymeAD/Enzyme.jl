using Enzyme
using Test
using Enzyme.EnzymeRules

# How a rule reaches the calling module follows its inlining annotation. `@inline` rules
# are emitted into the calling module and always-inlined. `@noinline` rules are called
# through their natively compiled entry point. Unannotated rules follow Julia's inlining
# heuristics.

cube_inline(x) = x^3
cube_noinline(x) = x^3
cube_default(x) = x^3
cube_big(x) = x^3

@inline function EnzymeRules.forward(config, ::Const{typeof(cube_inline)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(cube_inline(x.val), 10 * 3 * x.val^2 * x.dval)
end

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_noinline)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(cube_noinline(x.val), 10 * 3 * x.val^2 * x.dval)
end

function EnzymeRules.forward(config, ::Const{typeof(cube_default)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(cube_default(x.val), 10 * 3 * x.val^2 * x.dval)
end

# This rule is too large for Julia's inlining heuristics.
function EnzymeRules.forward(config, ::Const{typeof(cube_big)}, ::Type{<:Duplicated}, x::Duplicated)
    acc = 0.0
    for i in 1:64
        acc += sin(x.val + i) * cos(x.val - i) + exp(-abs(x.val)) * log1p(abs(x.val) + i)
        acc += atan(x.val, i) + sqrt(abs(x.val) + i) + tanh(x.val * i) - expm1(-abs(x.val) - i)
    end
    return Duplicated(cube_big(x.val) + 0 * acc, 10 * 3 * x.val^2 * x.dval)
end

@noinline function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(cube_noinline)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (x.val,))
end

@noinline function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(cube_noinline)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (10 * 3 * xv^2 * dret.val,)
end

# This argument holds both a tracked pointer and data. On 1.12+ it has inline roots.
struct Scale
    v::Vector{Float64}
    c::Float64
end

scale_dot(p::Scale, x) = p.c * p.v[1] * x

@noinline function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig, func::Const{typeof(scale_dot)}, ::Type{<:Active}, p::Const{Scale}, x::Active
    )
    primal = EnzymeRules.needs_primal(config) ? func.val(p.val, x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (p.val.c * p.val.v[1],))
end

@noinline function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig, func::Const{typeof(scale_dot)}, dret::Active, tape, p::Const{Scale}, x::Active
    )
    return (nothing, 10 * tape[1] * dret.val)
end

function scale_twice(p, x)
    y = scale_dot(p, x)
    p.v[1] = 5.0
    return y + scale_dot(p, x)
end

# In a loop, the roots of the overwritten argument come back from the tape.
function scale_loop(p, x)
    s = 0.0
    for i in 1:3
        s += scale_dot(p, x)
        p.v[1] += 1.0
    end
    return s
end

@testset "rule inlining annotations" begin
    for f in (cube_inline, cube_noinline, cube_default, cube_big)
        @test autodiff(ForwardWithPrimal, f, Duplicated(2.0, 1.0)) == (120.0, 8.0)
    end
    @test autodiff(Reverse, cube_noinline, Active(2.0))[1][1] == 120.0
    # 10 * (2 * 3) + 10 * (2 * 5). The argument is overwritten between the two calls.
    @test autodiff(Reverse, scale_twice, Const(Scale([3.0], 2.0)), Active(1.5))[1][2] == 160.0
    # 10 * 2 * (3 + 4 + 5)
    @test autodiff(Reverse, scale_loop, Const(Scale([3.0], 2.0)), Active(1.5))[1][2] == 240.0

    @static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
        world = Base.get_world_counter()
        C = EnzymeRules.FwdConfig{true, true, 1, false, false}
        function convention(f)
            TT = Tuple{C, Const{typeof(f)}, Type{Duplicated{Float64}}, Duplicated{Float64}}
            mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(EnzymeRules.forward), TT, world)
            ci = Enzyme.Compiler.codeinst(mi, world)
            return Enzyme.Compiler.call_convention(mi, ci)
        end
        @test convention(cube_inline) === :inline
        @test convention(cube_noinline) === :call
        @test convention(cube_default) === :inline
        @test convention(cube_big) === :call
    end
end

# A natively called rule is compiled like any other Julia code. `ignore_derivatives`
# is the identity there, and `within_autodiff()` is false. A rule emitted into the
# differentiated function is inferred by Enzyme's interpreter, where
# `within_autodiff()` is true.

cube_ignore(x) = x^3
cube_within(x) = x^3

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_ignore)}, ::Type{<:Duplicated}, x::Duplicated)
    xv = Enzyme.ignore_derivatives(x.val)
    return Duplicated(xv^3, 10 * 3 * xv^2 * x.dval)
end

@noinline function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(cube_ignore)}, ::Type{<:Active}, x::Active)
    xv = Enzyme.ignore_derivatives(x.val)
    primal = EnzymeRules.needs_primal(config) ? func.val(xv) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (xv,))
end

@noinline function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(cube_ignore)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (10 * 3 * xv^2 * Enzyme.ignore_derivatives(dret.val),)
end

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_within)}, ::Type{<:Duplicated}, x::Duplicated)
    sign = Enzyme.within_autodiff() ? 1.0 : -1.0
    return Duplicated(x.val^3, sign * 10 * 3 * x.val^2 * x.dval)
end

@testset "natively called rules are ordinary Julia code" begin
    @static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
        # An emitted rule cannot lower `ignore_derivatives`: its extern is
        # left for the differentiated code, which never includes the rule.
        # So these only work through the native path.
        @test autodiff(ForwardWithPrimal, cube_ignore, Duplicated(2.0, 1.0)) == (120.0, 8.0)
        @test autodiff(Reverse, cube_ignore, Active(2.0))[1][1] == 120.0
        @test autodiff(ForwardWithPrimal, cube_within, Duplicated(2.0, 1.0)) == (-120.0, 8.0)
    else
        @test autodiff(ForwardWithPrimal, cube_within, Duplicated(2.0, 1.0)) == (120.0, 8.0)
    end
end

# Signature shapes that Julia's specsig treats specially. Each shape has an
# `@inline` rule, which `check_specsig` compares against Julia's own codegen
# of the rule, and a `@noinline` rule, which is called natively through the
# derived signature.

# A struct with an uninitialized field is boxed by Julia's codegen even though
# `jl_type_to_llvm` gives it a struct type.
struct SUninit
    x::Float64
    v::Vector{Float64}
    SUninit(x) = new(x)
end

# Small integers go in registers, extended to the register width.
for (name, inl, tape, expected) in (
        (:abi_uninit_inline, :inline, :(SUninit(x.val)), :(10 * 3 * tape.x^2 * dret.val)),
        (:abi_uninit_noinline, :noinline, :(SUninit(x.val)), :(10 * 3 * tape.x^2 * dret.val)),
        (:abi_u8_inline, :inline, :(UInt8(200)), :(Float64(tape) * dret.val)),
        (:abi_u8_noinline, :noinline, :(UInt8(200)), :(Float64(tape) * dret.val)),
        (:abi_i8_inline, :inline, :(Int8(-5)), :(Float64(tape) * dret.val)),
        (:abi_i8_noinline, :noinline, :(Int8(-5)), :(Float64(tape) * dret.val)),
        (:abi_bool_inline, :inline, :(true), :(Float64(tape) * dret.val)),
        (:abi_bool_noinline, :noinline, :(true), :(Float64(tape) * dret.val)),
    )
    aug = :(
        function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof($name)}, ::Type{<:Active}, x::Active)
            primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
            return EnzymeRules.AugmentedReturn(primal, nothing, $tape)
        end
    )
    rev = :(
        function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof($name)}, dret::Active, tape, x::Active)
            return ($expected,)
        end
    )
    mac = Symbol("@", inl)
    line = LineNumberNode(@__LINE__, @__FILE__)
    @eval begin
        $name(x) = x^3
        $(Expr(:macrocall, mac, line, aug))
        $(Expr(:macrocall, mac, line, rev))
    end
end

# A `Union` of ghosts is returned as its selector byte only.
ghost_union_inline(x) = x > 0 ? nothing : missing
ghost_union_noinline(x) = x > 0 ? nothing : missing
@inline function EnzymeRules.forward(config, ::Const{typeof(ghost_union_inline)}, ::Type{<:Const}, x::Duplicated)
    return x.val > 0 ? nothing : missing
end
@noinline function EnzymeRules.forward(config, ::Const{typeof(ghost_union_noinline)}, ::Type{<:Const}, x::Duplicated)
    return x.val > 0 ? nothing : missing
end
use_ghost_inline(x) = ghost_union_inline(x) === nothing ? x * x : -x * x
use_ghost_noinline(x) = ghost_union_noinline(x) === nothing ? x * x : -x * x

@testset "specsig shapes" begin
    for f in (abi_uninit_inline, abi_uninit_noinline)
        @test autodiff(Reverse, f, Active(2.0))[1][1] == 120.0
    end
    for f in (abi_u8_inline, abi_u8_noinline)
        @test autodiff(Reverse, f, Active(2.0))[1][1] == 200.0
    end
    for f in (abi_i8_inline, abi_i8_noinline)
        @test autodiff(Reverse, f, Active(2.0))[1][1] == -5.0
    end
    for f in (abi_bool_inline, abi_bool_noinline)
        @test autodiff(Reverse, f, Active(2.0))[1][1] == 1.0
    end
    for f in (use_ghost_inline, use_ghost_noinline)
        @test autodiff(ForwardWithPrimal, f, Duplicated(2.0, 1.0)) == (4.0, 4.0)
        @test autodiff(ForwardWithPrimal, f, Duplicated(-2.0, 1.0)) == (4.0, -4.0)
    end
end

# A derivative that is differentiated again must differentiate through the
# rules it calls, natively called ones included. These rules compute the primal
# themselves, so the outer differentiation sees plain arithmetic in them: they
# give 30 x^2, and the second derivative is 60 x.

cube_nested(x) = x^3

@noinline function EnzymeRules.forward(config, ::Const{typeof(cube_nested)}, ::Type{<:Duplicated}, x::Duplicated)
    return Duplicated(x.val^3, 10 * 3 * x.val^2 * x.dval)
end

@noinline function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(cube_nested)}, ::Type{<:Active}, x::Active)
    primal = EnzymeRules.needs_primal(config) ? x.val^3 : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, (x.val,))
end

@noinline function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(cube_nested)}, dret::Active, tape, x::Active)
    (xv,) = tape
    return (10 * 3 * xv^2 * dret.val,)
end

fwd_over(x) = autodiff(ForwardWithPrimal, Const(cube_nested), Duplicated(x, 1.0))[1]
rev_over(x) = autodiff_deferred(Reverse, Const(cube_nested), Active, Active(x))[1][1]

@testset "nested differentiation through natively called rules" begin
    @test fwd_over(2.0) == 120.0
    @test rev_over(2.0) == 120.0
    # forward over forward, and reverse over forward
    @test autodiff(Forward, fwd_over, Duplicated(2.0, 1.0))[1] ≈ 120.0
    @test autodiff(Reverse, fwd_over, Active(2.0))[1][1] ≈ 120.0
    # forward over reverse
    @test autodiff(Forward, rev_over, Duplicated(2.0, 1.0))[1] ≈ 120.0
end

# Arguments Julia's codegen boxes: abstract types, mutable types and unions.
# The annotations of a rule are immutable structs, so a rule always has an
# unboxed argument, and Julia always gives it a specialized entry point.
# Boxing shows up in the tape, in `@nospecialize` arguments, and in the payload
# of an annotation.

mutable struct MParam
    c::Float64
end

boxed_tape_vector(x) = x^3
boxed_tape_any(x) = x^3
boxed_tape_union(x) = x^3
boxed_tape_union2(x) = x^3
boxed_tape_union3(x) = x^3
boxed_nospec_fwd(x) = x^3
boxed_nospec_rev(v) = v[1]^3
boxed_mutable_payload(p::MParam, x) = p.c * x^3

for inl in (:inline, :noinline)
    mac = Symbol("@", inl)
    line = LineNumberNode(@__LINE__, @__FILE__)
    defs = Any[]
    # A `Vector` tape is a boxed argument of the reverse rule.
    push!(
        defs, :(
            function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_vector)}, ::Type{<:Active}, x::Active)
                primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
                return EnzymeRules.AugmentedReturn(primal, nothing, [x.val])
            end
        )
    )
    push!(
        defs, :(
            function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_vector)}, dret::Active, tape, x::Active)
                return (10 * 3 * tape[1]^2 * dret.val,)
            end
        )
    )
    # An `Any` tape is boxed, and so are `@nospecialize` arguments.
    push!(
        defs, :(
            function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_any)}, ::Type{<:Active}, x::Active)
                primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
                return EnzymeRules.AugmentedReturn{typeof(primal), Nothing, Any}(primal, nothing, x.val)
            end
        )
    )
    push!(
        defs, :(
            function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_any)}, @nospecialize(dret::Active), @nospecialize(tape), @nospecialize(x::Active))
                xv = tape::Float64
                return (10 * 3 * xv^2 * dret.val::Float64,)
            end
        )
    )
    # A `Union` tape is boxed.
    push!(
        defs, :(
            function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_union)}, ::Type{<:Active}, x::Active)
                primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
                return EnzymeRules.AugmentedReturn(primal, nothing, x.val > 0 ? x.val : nothing)
            end
        )
    )
    push!(
        defs, :(
            function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_union)}, dret::Active, tape, x::Active)
                xv = tape === nothing ? 0.0 : tape
                return (10 * 3 * xv^2 * dret.val,)
            end
        )
    )
    # A `Union` tape with two members that carry data. The reverse rule
    # tells them apart, so the selector must survive.
    push!(
        defs, :(
            function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_union2)}, ::Type{<:Active}, x::Active)
                primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
                return EnzymeRules.AugmentedReturn(primal, nothing, x.val > 0 ? x.val : Float32(x.val))
            end
        )
    )
    push!(
        defs, :(
            function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_union2)}, dret::Active, tape, x::Active)
                scale = tape isa Float32 ? 1 : 10
                return (scale * 3 * Float64(tape)^2 * dret.val,)
            end
        )
    )
    # The same tape, with the augmented return typed explicitly, so the union
    # is one field laid out inline: payload, then selector.
    push!(
        defs, :(
            function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_union3)}, ::Type{<:Active}, x::Active)
                primal = EnzymeRules.needs_primal(config) ? func.val(x.val) : nothing
                tape = x.val > 0 ? x.val : nothing
                return EnzymeRules.AugmentedReturn{typeof(primal), Nothing, Union{Nothing, Float64}}(primal, nothing, tape)
            end
        )
    )
    push!(
        defs, :(
            function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_tape_union3)}, dret::Active, tape, x::Active)
                xv = tape === nothing ? 0.0 : tape
                return (10 * 3 * xv^2 * dret.val,)
            end
        )
    )
    # A `@nospecialize` annotation is boxed.
    push!(
        defs, :(
            function EnzymeRules.forward(config, ::Const{typeof(boxed_nospec_fwd)}, ::Type{<:Duplicated}, @nospecialize(x::Duplicated))
                xv = x.val::Float64
                return Duplicated(xv^3, 10 * 3 * xv^2 * x.dval::Float64)
            end
        )
    )
    # Every non-ghost argument boxed, and a singleton return.
    push!(
        defs, :(
            function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_nospec_rev)}, ::Type{<:Active}, @nospecialize(v::Duplicated))
                val = (v.val::Vector{Float64})[1]
                primal = EnzymeRules.needs_primal(config) ? val^3 : nothing
                return EnzymeRules.AugmentedReturn(primal, nothing, val)
            end
        )
    )
    push!(
        defs, :(
            function EnzymeRules.reverse(config::EnzymeRules.RevConfig, func::Const{typeof(boxed_nospec_rev)}, @nospecialize(dret::Active), @nospecialize(tape), @nospecialize(v::Duplicated))
                xv = tape::Float64
                (v.dval::Vector{Float64})[1] += 10 * 3 * xv^2 * dret.val::Float64
                return (nothing,)
            end
        )
    )
    # A mutable payload inside an annotation. The annotation itself goes by
    # reference.
    push!(
        defs, :(
            function EnzymeRules.forward(config, ::Const{typeof(boxed_mutable_payload)}, ::Type{<:Duplicated}, p::Const{MParam}, x::Duplicated)
                return Duplicated(p.val.c * x.val^3, 10 * p.val.c * 3 * x.val^2 * x.dval)
            end
        )
    )
    for def in defs
        @eval $(Expr(:macrocall, mac, line, def))
    end
    @eval @testset $("boxed arguments, $inl rules") begin
        @test autodiff(Reverse, boxed_tape_vector, Active(2.0))[1][1] == 120.0
        @test autodiff(Reverse, boxed_tape_any, Active(2.0))[1][1] == 120.0
        @test autodiff(Reverse, boxed_tape_union, Active(2.0))[1][1] == 120.0
        @test autodiff(Reverse, boxed_tape_union, Active(-2.0))[1][1] == 0.0
        @test autodiff(Reverse, boxed_tape_union2, Active(2.0))[1][1] == 120.0
        @test autodiff(Reverse, boxed_tape_union2, Active(-2.0))[1][1] == 12.0
        @test autodiff(Reverse, boxed_tape_union3, Active(2.0))[1][1] == 120.0
        @test autodiff(Reverse, boxed_tape_union3, Active(-2.0))[1][1] == 0.0
        @test autodiff(ForwardWithPrimal, boxed_nospec_fwd, Duplicated(2.0, 1.0)) == (120.0, 8.0)
        v = [2.0]
        dv = [0.0]
        autodiff(Reverse, boxed_nospec_rev, Active, Duplicated(v, dv))
        @test dv == [120.0]
        @test autodiff(ForwardWithPrimal, boxed_mutable_payload, Const(MParam(2.0)), Duplicated(2.0, 1.0)) == (240.0, 16.0)
    end
end

# The signature derivation, and the fallback for code without a specialized
# entry point.
@static if Enzyme.Compiler.Interpreter.HAS_INVOKE_RULES
    # Julia compiles a function with the boxed `jl_fptr_args` ABI when every
    # argument is boxed and so is the return. No rule has that shape, so use a
    # plain function to exercise the error.
    @noinline all_boxed(a, b, c, d) = a

    @testset "signature derivation" begin
        world = Base.get_world_counter()
        LLVM = Enzyme.Compiler.LLVM
        LLVM.Context() do ctx
            kind = Enzyme.Compiler.arg_kind
            @test kind(Nothing) === :ghost
            @test kind(Type{Float64}) === :ghost
            @test kind(Const{typeof(sin)}) === :ghost
            @test kind(Float64) === :byval
            @test kind(UInt8) === :byval
            @test kind(Bool) === :byval
            @test kind(Ptr{Float64}) === :byval
            @test kind(Duplicated{Float64}) === :byref
            @test kind(Tuple{Float64, Float64}) === :byref
            @test kind(Duplicated{Vector{Float64}}) === :byref
            @test kind(Any) === :boxed
            @test kind(Vector{Float64}) === :boxed
            @test kind(MParam) === :boxed
            @test kind(Union{Nothing, Float64}) === :boxed
            @test kind(SUninit) === :boxed
            @test kind(Duplicated) === :boxed

            mod = LLVM.Module("test")
            LLVM.triple!(mod, Sys.MACHINE)
            @test Enzyme.Compiler.native_invoke_available(mod)

            mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(all_boxed), Tuple{Any, Any, Any, Any}, world)
            ci = Enzyme.Compiler.codeinst(mi, world)
            specptr, invoke = Enzyme.Compiler.Interpreter.codeinst_entry(ci)
            @test specptr == C_NULL
            @test invoke != C_NULL
            @test Enzyme.Compiler.call_convention(mi, ci) === :call
            @test_throws Enzyme.Compiler.CallingConventionMismatchError Enzyme.Compiler.native_codeinst(mod, mi, world)

            C = EnzymeRules.FwdConfig{true, true, 1, false, false}
            TT = Tuple{C, Const{typeof(cube_noinline)}, Type{Duplicated{Float64}}, Duplicated{Float64}}
            mi = Enzyme.Compiler.my_methodinstance(Forward, typeof(EnzymeRules.forward), TT, world)
            native = Enzyme.Compiler.native_codeinst(mod, mi, world)
            @test native !== nothing
            @test native[2] != C_NULL

            # The `pgcstack` parameter carries the `swiftself` attribute only
            # where Julia's codegen uses the swift calling convention, and the
            # `gcstack` attribute always, so `gcstack_arg_index` finds it on
            # either target. `check_specsig` reads the parameter back from that
            # mark, so it accepts the declaration it derived.
            RT = native[1].rettype
            decl = Enzyme.Compiler.specsig_function!(mod, mi, RT, "test_specsig_gcstack", world)
            @test (Enzyme.Compiler.gcstack_arg_index(decl) != 0) == Enzyme.Compiler.jit_gcstack_arg()
            @test Enzyme.Compiler.has_swiftself(decl) ==
                (Enzyme.Compiler.jit_gcstack_arg() && Enzyme.Compiler.jit_uses_swiftcc())
            @test Enzyme.Compiler.check_specsig(decl, mi, RT) === nothing
        end
    end
end
