function stepSparseCoding(ihtsc::Type{SparseCoders.IHT}, options, cb::DT, x::AbstractArray; shape::Shapes.AbstractShape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget}
    ts = createTransform(cb, shape)
    # number of non-zero coefficients
    iht = ihtsc(x, t->synthesize(ts, t), t->analyze(ts, t); options...)

    # initial sparse vector y0
    y0 = analyze(ts, x)
    y_opt, loss_iht = iht(y0, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

function stepSparseCoding(istasc::Type{TI}, options, cb::DT, x::AbstractArray; shape::Shapes.AbstractShape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {TI<:Union{SparseCoders.ISTA,SparseCoders.FISTA},DT<:LearningTarget}
    ts = createTransform(cb, shape)
    # number of non-zero coefficients
    ∇f = (_y) -> begin
        -real.(analyze(ts, x - synthesize(ts, _y)))
    end
    ista = istasc(∇f; options...)

    # initial sparse vector y0
    y0 = analyze(ts, x)
    y_opt, _dummy = ista(y0)

    loss_ista_x = norm(x - synthesize(ts, y_opt))^2/2
    return (y_opt, loss_ista_x)
end
