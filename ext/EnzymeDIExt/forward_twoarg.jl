## Pushforward

function DI.prepare_pushforward(
    f!::F,
    y,
    ::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return DI.NoPushforwardPrep()
end

function DI.value_and_pushforward(
    f!::F,
    y,
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dx_sametype = convert(typeof(x), only(tx))
    dy_sametype = make_zero(y)
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dy_sametype,)
end

function DI.value_and_pushforward(
    f!::F,
    y,
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    ty_sametype = ntuple(_ -> make_zero(y), Val(B))
    x_and_tx = BatchDuplicated(x, tx_sametype)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, ty_sametype
end

function DI.pushforward(
    f!::F,
    y,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    _, ty = DI.value_and_pushforward(f!, y, prep, backend, x, tx, contexts...)
    return ty
end

function DI.value_and_pushforward!(
    f!::F,
    y,
    ty::NTuple{B},
    ::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    mode = forward_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    ty_sametype = map(Fix1(convert, typeof(y)), ty)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    foreach(copyto_if_different_addresses!, ty, ty_sametype)
    return y, ty
end

function DI.pushforward!(
    f!::F,
    y,
    ty::NTuple,
    prep::DI.NoPushforwardPrep,
    backend::AutoEnzyme{<:Union{ForwardMode,Nothing}},
    x,
    tx::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    DI.value_and_pushforward!(f!, y, ty, prep, backend, x, tx, contexts...)
    return ty
end
