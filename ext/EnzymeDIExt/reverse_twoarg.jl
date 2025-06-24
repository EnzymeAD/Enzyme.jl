## Pullback

struct EnzymeReverseTwoArgPullbackPrep{TY} <: DI.PullbackPrep
    ty_copy::TY
end

function DI.prepare_pullback(
    f!::F,
    y,
    ::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple,
    contexts::Vararg{DI.Context,C},
) where {F,C}
    return EnzymeReverseTwoArgPullbackPrep(map(copy, ty))
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    copyto!(only(prep.ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dy_sametype = convert(typeof(y), only(prep.ty_copy))
    y_and_dy = Duplicated(y, dy_sametype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    dinputs = only(
        autodiff(mode, f!_and_df!, Const, y_and_dy, Active(x), annotated_contexts...)
    )
    dx = dinputs[2]
    return y, (dx,)
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x::Number,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    foreach(copyto!, prep.ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    ty_sametype = map(Fix1(convert, typeof(y)), prep.ty_copy)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    dinputs = only(
        autodiff(mode, f!_and_df!, Const, y_and_ty, Active(x), annotated_contexts...)
    )
    tx = values(dinputs[2])
    return y, tx
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    copyto!(only(prep.ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dx_sametype = make_zero(x)  # allocates
    dy_sametype = convert(typeof(y), only(prep.ty_copy))
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    return y, (dx_sametype,)
end

function DI.value_and_pullback(
    f!::F,
    y,
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    foreach(copyto!, prep.ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    tx_sametype = ntuple(_ -> make_zero(x), Val(B))  # allocates
    ty_sametype = map(Fix1(convert, typeof(y)), prep.ty_copy)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    return y, tx_sametype
end

function DI.value_and_pullback!(
    f!::F,
    y,
    tx::NTuple{1},
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{1},
    contexts::Vararg{DI.Context,C},
) where {F,C}
    copyto!(only(prep.ty_copy), only(ty))
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode)
    dx_sametype = convert(typeof(x), only(tx))
    make_zero!(dx_sametype)
    dy_sametype = convert(typeof(y), only(prep.ty_copy))
    x_and_dx = Duplicated(x, dx_sametype)
    y_and_dy = Duplicated(y, dy_sametype)
    annotated_contexts = translate(backend, mode, Val(1), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_dy, x_and_dx, annotated_contexts...)
    copyto_if_different_addresses!(only(tx), dx_sametype)
    return y, (dx_sametype,)
end

function DI.value_and_pullback!(
    f!::F,
    y,
    tx::NTuple{B},
    prep::EnzymeReverseTwoArgPullbackPrep,
    backend::AutoEnzyme{<:Union{ReverseMode,Nothing}},
    x,
    ty::NTuple{B},
    contexts::Vararg{DI.Context,C},
) where {F,B,C}
    foreach(copyto!, prep.ty_copy, ty)
    mode = reverse_noprimal(backend)
    f!_and_df! = get_f_and_df(f!, backend, mode, Val(B))
    tx_sametype = map(Fix1(convert, typeof(x)), tx)
    make_zero!(tx_sametype)
    ty_sametype = map(Fix1(convert, typeof(y)), prep.ty_copy)
    x_and_tx = BatchDuplicated(x, tx_sametype)
    y_and_ty = BatchDuplicated(y, ty_sametype)
    annotated_contexts = translate(backend, mode, Val(B), contexts...)
    autodiff(mode, f!_and_df!, Const, y_and_ty, x_and_tx, annotated_contexts...)
    foreach(copyto_if_different_addresses!, tx, tx_sametype)
    return y, tx_sametype
end
