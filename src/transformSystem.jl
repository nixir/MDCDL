abstract type AbstractOperator end

struct TransformSystem{OP} <: AbstractOperator
    shape::Shapes.AbstractShape
    operator::OP
    options::Base.Iterators.Pairs

    function TransformSystem(operator::OP, shape=Shapes.Separated(); options...) where {OP<:FilterBank}
        new{OP}(shape, operator, options)
    end

    function TransformSystem(ts::TransformSystem, shape=ts.shape)
        TransformSystem(deepcopy(ts.operator), shape; ts.options...)
    end
end

decimations(tfs::TransformSystem) = decimations(tfs.operator)
orders(tfs::TransformSystem) = orders(tfs.operator)
nchannels(tfs::TransformSystem) = nchannels(tfs.operator)

createTransform(ns::FilterBank, args...; kwargs...) = TransformSystem(ns, args...; kwargs...)

struct JoinedTransformSystems{T} <: AbstractOperator
    shape::Shapes.AbstractShape
    transforms::Array

    JoinedTransformSystems(ts::Tuple{TS}, args...; kwargs...) where{TS<:TransformSystem} = JoinedTransformSystems(Multiscale(ts...), args...; kwargs...)
    function JoinedTransformSystems(mst::MS, shape=Shapes.Separated()) where {TS<:TransformSystem,MS<:Multiscale}
        new{MS}(shape, collect(mst.filterbanks))
    end
end

function createTransform(ms::MS, shape::S=Shapes.Separated()) where {MS<:Multiscale,S<:Shapes.AbstractShape}
    opsarr = map(1:length(ms.filterbanks)) do lv
        sp = if isfixedsize(S)
            S(fld.(shape.insize, decimations(ms.filterbanks[lv]).^(lv-1)))
        else
            S()
        end
        TransformSystem(ms.filterbanks[lv], sp)
    end
    JoinedTransformSystems(MS(opsarr...), shape)
end
