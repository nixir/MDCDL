function stepSparseCoding(ihtsc::Type{SparseCoders.IHT}, options, cb::DT, x::AbstractArray; shape::Shapes.AbstractShape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget}
    ts = createTransform(cb, shape)
    # number of non-zero coefficients
    iht = ihtsc(x, t->synthesize(ts, t), t->analyze(ts, t); options...)

    # initial sparse vector y0
    y0 = analyze(ts, x)
    y_opt, loss_iht = iht(y0, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end
