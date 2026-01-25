function seeded_autodiff_thunk(
    rmode::ReverseModeSplit{ReturnPrimal},
    dresult,
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,FA<:Annotation,RA<:Annotation,N}
    forward, reverse = autodiff_thunk(rmode, FA, RA, typeof.(args)...)
    tape, result, shadow_result = forward(f, args...)
    if RA <: Active
        dresult_righttype = convert(typeof(result), dresult)
        dinputs = only(reverse(f, args..., dresult_righttype, tape))
    else
        shadow_result .+= dresult  # TODO: generalize beyond arrays
        dinputs = only(reverse(f, args..., tape))
    end
    if ReturnPrimal
        return (dinputs, result)
    else
        return (dinputs,)
    end
end

function batch_seeded_autodiff_thunk(
    rmode::ReverseModeSplit{ReturnPrimal},
    dresults::NTuple{B},
    f::FA,
    ::Type{RA},
    args::Vararg{Annotation,N},
) where {ReturnPrimal,B,FA<:Annotation,RA<:Annotation,N}
    rmode_rightwidth = ReverseSplitWidth(rmode, Val(B))
    forward, reverse = autodiff_thunk(rmode_rightwidth, FA, RA, typeof.(args)...)
    tape, result, shadow_results = forward(f, args...)
    if RA <: Active
        dresults_righttype = map(Fix1(convert, typeof(result)), dresults)
        dinputs = only(reverse(f, args..., dresults_righttype, tape))
    else
        foreach(shadow_results, dresults) do d0, d
            d0 .+= d  # use recursive_add here?
        end
        dinputs = only(reverse(f, args..., tape))
    end
    if ReturnPrimal
        return (dinputs, result)
    else
        return (dinputs,)
    end
end

## Pullback

struct EnzymeReverseOneArgPullbackPrep{Y} <: DI.PullbackPrep
    y_example::Y  # useful to create return activity
end

function DI.prepare_pullback(
    f::F,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    y = f(x, map(DI.unwrap, contexts)...)
    return EnzymeReverseOneArgPullbackPrep(y)
end

### Out-of-place

function DI.value_and_pullback(
    f::F,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df(f, backend, mode))
    IA = guess_activity(typeof(x), mode)
    RA = guess_activity(typeof(prep.y_example), mode)
    dx = make_zero(x)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dinputs, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, annotate(IA, x, dx), annotated_contexts...
    )
    new_dx = first(dinputs)
    if isnothing(new_dx)
        return result, (dx,)
    else
        return result, (new_dx,)
    end
end

function DI.value_and_pullback(
    f::F,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df(f, backend, mode, Val(B)))
    IA = batchify_activity(guess_activity(typeof(x), mode), Val(B))
    RA = batchify_activity(guess_activity(typeof(prep.y_example), mode), Val(B))
    tx = ntuple(_ -> make_zero(x), Val(B))
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    dinputs, result = batch_seeded_autodiff_thunk(
        mode, ty, f_and_df, RA, annotate(IA, x, tx), annotated_contexts...
    )
    new_tx = values(first(dinputs))
    if isnothing(new_tx)
        return result, tx
    else
        return result, new_tx
    end
end

function DI.pullback(
    f::F,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return last(DI.value_and_pullback(f, prep, backend, x, ty, contexts...))
end

### In-place

function DI.value_and_pullback!(
    f::F,
    tx::NTuple{1},
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df(f, backend, mode))
    RA = guess_activity(typeof(prep.y_example), mode)
    dx_righttype = convert(typeof(x), only(tx))
    make_zero!(dx_righttype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    _, result = seeded_autodiff_thunk(
        mode, only(ty), f_and_df, RA, Duplicated(x, dx_righttype), annotated_contexts...
    )
    copyto_if_different_addresses!(only(tx), dx_righttype)
    return result, tx
end

function DI.value_and_pullback!(
    f::F,
    tx::NTuple{B},
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    mode = reverse_split_withprimal(backend)
    f_and_df = force_annotation(get_f_and_df(f, backend, mode, Val(B)))
    RA = batchify_activity(guess_activity(typeof(prep.y_example), mode), Val(B))
    tx_righttype = map(Fix1(convert, typeof(x)), tx)
    make_zero!(tx_righttype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    _, result = batch_seeded_autodiff_thunk(
        mode, ty, f_and_df, RA, BatchDuplicated(x, tx_righttype), annotated_contexts...
    )
    foreach(copyto_if_different_addresses!, tx, tx_righttype)
    return result, tx
end

function DI.pullback!(
    f::F,
    tx::NTuple,
    prep::EnzymeReverseOneArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return last(DI.value_and_pullback!(f, tx, prep, backend, x, ty, contexts...))
end

## Gradient

### Without preparation

function DI.gradient(
    f::F,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = reverse_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    IA = guess_activity(typeof(x), mode)
    grad = make_zero(x)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dinputs = only(
        autodiff(mode, f_and_df, Active, annotate(IA, x, grad), annotated_contexts...)
    )
    new_grad = first(dinputs)
    if isnothing(new_grad)
        return grad
    else
        return new_grad
    end
end

function DI.value_and_gradient(
    f::F,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = reverse_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    IA = guess_activity(typeof(x), mode)
    grad = make_zero(x)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dinputs, result = autodiff(
        mode, f_and_df, Active, annotate(IA, x, grad), annotated_contexts...
    )
    new_grad = first(dinputs)
    if isnothing(new_grad)
        return result, grad
    else
        return result, new_grad
    end
end

### With preparation

struct EnzymeGradientPrep{G} <: DI.GradientPrep
    grad_righttype::G
end

function DI.prepare_gradient(
    f::F, ::AutoEnzyme{<:Union{ReverseMode,Nothing}}, x, contexts::Vararg{DI.Context,C}
) where {F,C}
    grad_righttype = make_zero(x)
    return EnzymeGradientPrep(grad_righttype)
end

function DI.gradient(
    f::F,
    ::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.gradient(f, backend, x, contexts...)
end

function DI.gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = reverse_noprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    grad_righttype = grad isa typeof(x) ? grad : prep.grad_righttype
    make_zero!(grad_righttype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f_and_df, Active, Duplicated(x, grad_righttype), annotated_contexts...)
    copyto_if_different_addresses!(grad, grad_righttype)
    return grad
end

function DI.value_and_gradient(
    f::F,
    ::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.value_and_gradient(f, backend, x, contexts...)
end

function DI.value_and_gradient!(
    f::F,
    grad,
    prep::EnzymeGradientPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = reverse_withprimal(backend)
    f_and_df = get_f_and_df(f, backend, mode)
    grad_righttype = grad isa typeof(x) ? grad : prep.grad_righttype
    make_zero!(grad_righttype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    _, y = autodiff(
        mode, f_and_df, Active, Duplicated(x, grad_righttype), annotated_contexts...
    )
    copyto_if_different_addresses!(grad, grad_righttype)
    return y, grad
end
