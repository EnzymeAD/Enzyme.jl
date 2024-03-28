module AdaptExt
    isdefined(Base, :get_extension) ? (using Adapt) : (using ..Adapt)
    isdefined(Base, :get_extension) ? (using EnzymeCore) : (using ..EnzymeCore)

	Adapt.adapt_structure(to, x::Const) = Const(adapt(to, x.val))
	Adapt.adapt_structure(to, x::Active) = Active(adapt(to, x.val))
	Adapt.adapt_structure(to, x::Duplicated) = Duplicated(adapt(to, x.val), adapt(to, x.dval))
	Adapt.adapt_structure(to, x::DuplicatedNoNeed) = DuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
	Adapt.adapt_structure(to, x::BatchDuplicated) = BatchDuplicated(adapt(to, x.val), adapt(to, x.dval))
	Adapt.adapt_structure(to, x::BatchDuplicatedNoNeed) = BatchDuplicatedNoNeed(adapt(to, x.val), adapt(to, x.dval))
end
