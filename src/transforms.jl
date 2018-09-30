function reshape_polyvec(::Shapes.Default, ::AbstractNsolt, pvy::PolyphaseVector)
    [ reshape(pvy.data[p,:], pvy.nBlocks) for p in 1:size(pvy.data,1) ]
end

function reshape_polyvec(::Shapes.Arrayed, ::AbstractNsolt, pvy::PolyphaseVector)
    polyphase2mdarray(pvy)
end

function reshape_polyvec(::Shapes.Vec, ::AbstractNsolt, pvy::PolyphaseVector)
    vec(transpose(pvy.data))
end

function reshape_polyvec(::Shapes.Default, ::ParallelFilters, y::AbstractArray)
    y
end

function reshape_polyvec(::Shapes.Arrayed, ::ParallelFilters{TF,D}, y::AbstractArray{TY,D}) where {TF,TY,D}
    cat(D+1, y...)
end

function reshape_polyvec(::Shapes.Vec, ::ParallelFilters, y::AbstractArray)
    vcat(vec.(y)...)
end


function reshape_coefs(::Shapes.Default, ::AbstractNsolt, y::AbstractArray)
    PolyphaseVector(hcat(vec.(y)...) |> transpose |> Matrix, size(y[1]))
end

function reshape_coefs(::Shapes.Arrayed, ::AbstractNsolt, y::AbstractArray)
    mdarray2polyphase(y)
end

function reshape_coefs(sv::Shapes.Vec, nsop::AbstractNsolt, y::AbstractArray)
    szout = fld.(sv.insize, decimations(nsop))
    ty = reshape(y, szout..., nchannels(nsop))
    mdarray2polyphase(ty)
end

function reshape_coefs(::Shapes.Default, ::ParallelFilters, y::AbstractArray)
    y
end

function reshape_coefs(::Shapes.Arrayed, co::ParallelFilters{T,D}, y::AbstractArray) where {T,D}
    [ y[fill(:,D)..., p] for p in 1:nchannels(co)]
end

function reshape_coefs(sv::Shapes.Vec, co::ParallelFilters{T,D}, y::AbstractArray) where {T,D}
    ry = reshape(y, fld.(sv.insize, decimations(co))..., nchannels(co))
    [ ry[fill(:,D)..., p] for p in 1:nchannels(co) ]
end
