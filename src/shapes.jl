module Shapes
    abstract type AbstractShape end

    struct Separated <: AbstractShape
        Separated(sz...) = new()
    end
    struct Vectorized <: AbstractShape
        insize::Tuple
        Vectorized(sz::Integer...) = new(sz)
        Vectorized(sz::AbstractVector) = Vectorized(sz...)
        Vectorized(sz::Tuple) = Vectorized(sz...)
    end
    struct Combined <: AbstractShape
        Combined(sz...) = new()
    end

    Vec = Vectorized
end

isfixedsize(::A) where {A<:Shapes.AbstractShape} = isfixedsize(A)
isfixedsize(::Type{S}) where {S<:Shapes.AbstractShape}= false
isfixedsize(::Type{Shapes.Vec}) = true

function reshape_polyvec(::Shapes.Separated, ::AbstractNsolt, pvy::PolyphaseVector)
    [ reshape(@view(pvy.data[p,:]), pvy.nBlocks) for p in 1:size(pvy.data,1) ]
end

function reshape_polyvec(::Shapes.Combined, ::AbstractNsolt, pvy::PolyphaseVector)
    reshape(transpose(pvy.data), pvy.nBlocks..., size(pvy.data, 1))
end

function reshape_polyvec(::Shapes.Vec, ::AbstractNsolt, pvy::PolyphaseVector)
    vec(transpose(pvy.data))
end

function reshape_polyvec(::Shapes.Separated, ::ParallelFilters, y::AbstractArray)
    y
end

function reshape_polyvec(::Shapes.Combined, ::ParallelFilters{TF,D}, y::AbstractArray{TY,D}) where {TF,TY,D}
    cat(D+1, y...)
end

function reshape_polyvec(::Shapes.Vec, ::ParallelFilters, y::AbstractArray)
    vcat(vec.(y)...)
end


function reshape_coefs(::Shapes.Separated, ::AbstractNsolt, y::AbstractArray)
    PolyphaseVector(hcat(vec.(y)...) |> transpose |> Matrix, size(y[1]))
end

function reshape_coefs(::Shapes.Combined, ::AbstractNsolt, y::AbstractArray{T,D}) where {T,D}
    nBlocks = size(y)[1:D-1]
    outdata = transpose(reshape(y, prod(nBlocks), size(y,D)))
    PolyphaseVector(outdata, nBlocks)
end

function reshape_coefs(sv::Shapes.Vec, nsop::AbstractNsolt, y::AbstractArray)
    szout = fld.(sv.insize, decimations(nsop))
    ty = reshape(y, prod(szout), nchannels(nsop)) |> transpose
    PolyphaseVector(ty, szout)
end

function reshape_coefs(::Shapes.Separated, ::ParallelFilters, y::AbstractArray)
    y
end

function reshape_coefs(::Shapes.Combined, co::ParallelFilters{T,D}, y::AbstractArray) where {T,D}
    [ y[fill(:,D)..., p] for p in 1:nchannels(co)]
end

function reshape_coefs(sv::Shapes.Vec, co::ParallelFilters{T,D}, y::AbstractArray) where {T,D}
    ry = reshape(y, fld.(sv.insize, decimations(co))..., nchannels(co))
    [ ry[fill(:,D)..., p] for p in 1:nchannels(co) ]
end
